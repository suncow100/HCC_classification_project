"""
=====================================================================
HCC vs Benign 이진분류 파이프라인 (멀티페이즈, 환자 단위 분할)
=====================================================================
전체 흐름:
  1. 데이터 로드 + phase(동맥기/문맥기/지연기) 원-핫 인코딩
  2. 환자 단위 train/test 분할 (같은 환자의 여러 phase가 한쪽에만 들어가도록)
  3. 정규화 (MinMaxScaler, train 기준 fit)
  4. 다중공선성 분석 및 제거 (상관계수 0.9 초과 쌍 중 중복성 높은 쪽 제거)
     -- RFE 대신 이 단계로 1차 축소. 라디오믹스 피처는 서로 파생 관계가 많아
        RFE보다 상관관계 필터가 먼저 걸러주는 게 표준적인 순서
  5. LASSO(L1 로지스틱회귀)로 2차 피처 선택
     -- LassoCV(회귀용, MSE 기준)가 아니라 LogisticRegressionCV(L1, roc_auc 기준)
        사용 -- 분류 문제이므로 AUC를 직접 최적화하는 정규화 강도를 탐색해야 함
  6. 선택된 피처로 최종 분류기(LR/RF) 학습 -- GridSearchCV + StratifiedGroupKFold
     (환자 그룹을 인식하는 CV라 train 내부에서도 데이터 누수 없음)
  7. Test set 평가 (환자 단위로 완전히 분리된 holdout)
=====================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedGroupKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, accuracy_score,
    f1_score, classification_report
)

FEATURES_CSV = "/home/jovyan/gcubme/SW_BAEK/HCC/radiomics_features_multiphase.csv"
OUT_DIR = "/home/jovyan/gcubme/SW_BAEK/HCC"

# =====================================================================
# 1. 데이터 로드 + phase 원-핫 인코딩
# =====================================================================
df = pd.read_csv(FEATURES_CSV)
meta_cols = ["patient_id", "category", "class", "binary_label", "phase", "series_path"]
radiomics_cols = [c for c in df.columns if c not in meta_cols]

phase_dummies = pd.get_dummies(df["phase"], prefix="phase")
df = pd.concat([df, phase_dummies], axis=1)
feature_cols = radiomics_cols + list(phase_dummies.columns)

print(f"전체 시리즈(행) 수: {len(df)}, 환자 수: {df['patient_id'].nunique()}, 피처 수: {len(feature_cols)}")

# =====================================================================
# 2. 환자 단위 train/test 분할
# =====================================================================
patient_labels = df[["patient_id", "binary_label"]].drop_duplicates("patient_id")

label_consistency = df.groupby("patient_id")["binary_label"].nunique()
assert (label_consistency == 1).all(), "같은 환자인데 binary_label이 다른 시리즈가 있습니다"

train_patients, test_patients = train_test_split(
    patient_labels["patient_id"], test_size=0.25,
    stratify=patient_labels["binary_label"], random_state=42
)
train_df = df[df["patient_id"].isin(train_patients)].reset_index(drop=True)
test_df = df[df["patient_id"].isin(test_patients)].reset_index(drop=True)

print(f"train: 환자 {train_df['patient_id'].nunique()}명 / 시리즈 {len(train_df)}행")
print(f"test:  환자 {test_df['patient_id'].nunique()}명 / 시리즈 {len(test_df)}행")

X_train_raw = train_df[feature_cols].values
y_train = train_df["binary_label"].values
groups_train = train_df["patient_id"].values
X_test_raw = test_df[feature_cols].values
y_test = test_df["binary_label"].values

# =====================================================================
# 3. 정규화 (train 기준 fit, test는 transform만)
# =====================================================================
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# =====================================================================
# 4. 다중공선성 분석 및 제거
# =====================================================================
def remove_correlated_features(X_train_df: pd.DataFrame, threshold: float = 0.9):
    """
    상관계수가 threshold를 넘는 피처 쌍이 있으면,
    전체 피처들과의 평균 절대상관계수가 더 높은(=더 중복성이 큰) 쪽을 제거
    """
    corr_matrix = X_train_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = set()
    for col in upper.columns:
        high_corr = upper.index[upper[col] > threshold].tolist()
        for other in high_corr:
            if other in to_drop or col in to_drop:
                continue
            if corr_matrix[col].mean() >= corr_matrix[other].mean():
                to_drop.add(col)
            else:
                to_drop.add(other)

    kept_features = [c for c in X_train_df.columns if c not in to_drop]
    return kept_features, sorted(to_drop), corr_matrix

X_train_df = pd.DataFrame(X_train_scaled, columns=feature_cols)
kept_features, dropped_features, corr_matrix = remove_correlated_features(X_train_df, threshold=0.9)

print(f"\n[다중공선성 필터] 전체 {len(feature_cols)}개 중 제거: {len(dropped_features)}개, 남은 피처: {len(kept_features)}개")
print("제거된 피처 목록:")
for f in dropped_features:
    print(" -", f)

top_var_features = X_train_df.var().sort_values(ascending=False).head(30).index
plt.figure(figsize=(14, 12))
sns.heatmap(X_train_df[top_var_features].corr(), cmap="coolwarm", center=0,
            xticklabels=True, yticklabels=True, square=True)
plt.title("Feature Correlation Heatmap (top 30 by variance)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/feature_correlation_heatmap.png", dpi=150)
plt.close()
print(f"-> feature_correlation_heatmap.png 저장 완료")

kept_idx = [feature_cols.index(f) for f in kept_features]
X_train_filtered = X_train_scaled[:, kept_idx]
X_test_filtered = X_test_scaled[:, kept_idx]

# =====================================================================
# 5. LASSO(L1 로지스틱회귀)로 2차 피처 선택
#    -- roc_auc를 직접 최적화하는 정규화 강도(C)를 CV로 탐색
# =====================================================================
cv_group = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
cv_splits = list(cv_group.split(X_train_filtered, y_train, groups=groups_train))

lasso_logit = LogisticRegressionCV(
    Cs=20,                    # C(정규화 강도의 역수) 20개 후보 자동 탐색
    cv=cv_splits,
    penalty="l1",
    solver="liblinear",
    scoring="roc_auc",        # 회귀 손실이 아니라 분류 성능(AUC) 기준으로 C 선택
    max_iter=5000,
    random_state=42,
)
lasso_logit.fit(X_train_filtered, y_train)

selected_mask = lasso_logit.coef_[0] != 0
selected_features = [f for f, m in zip(kept_features, selected_mask) if m]

print(f"\n[LASSO] best C: {lasso_logit.C_[0]:.5f}")
print(f"선택된 피처 {len(selected_features)}개:")
for f, coef in zip(kept_features, lasso_logit.coef_[0]):
    if coef != 0:
        print(f"  {f}: {coef:.4f}")

X_train_sel = X_train_filtered[:, selected_mask]
X_test_sel = X_test_filtered[:, selected_mask]

# =====================================================================
# 6. 최종 분류기 학습 (선택된 피처 기준, 환자 그룹 인식 CV)
# =====================================================================
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

param_grids = {
    "LogisticRegression": (
        LogisticRegression(max_iter=5000, random_state=42),
        {"C": [0.01, 0.1, 1, 10, 100]}
    ),
    "RandomForest": (
        RandomForestClassifier(random_state=42),
        {"n_estimators": [100, 300], "max_depth": [3, 5, None]}
    ),
}

# =====================================================================
# 7. Test set 평가
# =====================================================================
for name, (estimator, grid) in param_grids.items():
    gs = GridSearchCV(estimator, grid, cv=cv, scoring="roc_auc", n_jobs=-1)
    gs.fit(X_train_sel, y_train, groups=groups_train)

    best_model = gs.best_estimator_
    y_pred = best_model.predict(X_test_sel)
    y_proba = best_model.predict_proba(X_test_sel)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_val = roc_auc_score(y_test, y_proba)

    print(f"\n=== {name} (다중공선성필터+LASSO 선택 피처 기준) ===")
    print(f"best params: {gs.best_params_}")
    print(f"CV 5-fold best AUC: {gs.best_score_:.3f}")
    print(f"Test AUC: {auc_val:.3f} | Sensitivity: {sensitivity:.3f} | Specificity: {specificity:.3f} | Acc: {acc:.3f} | F1: {f1:.3f}")
    print(f"Confusion matrix (tn, fp, fn, tp): {tn}, {fp}, {fn}, {tp}")
    print(classification_report(y_test, y_pred, target_names=["Benign", "HCC"]))