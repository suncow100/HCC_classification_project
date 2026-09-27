import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, accuracy_score,
    f1_score, recall_score, classification_report
)

FEATURES_CSV = "/home/jovyan/gcubme/SW_BAEK/HCC/radiomics_features_portal.csv"

# ---------- 1. 데이터 로드 ----------
df = pd.read_csv(FEATURES_CSV)

meta_cols = ["patient_id", "class", "binary_label", "series_path"]
feature_cols = [c for c in df.columns if c not in meta_cols]

X = df[feature_cols].values
y = df["binary_label"].values

print(f"전체 샘플 수: {len(df)}, 피처 수: {len(feature_cols)}")
print(f"클래스 분포: {df['binary_label'].value_counts().to_dict()}  (1=HCC, 0=Benign)")

# ---------- 2. train/test 분할 (환자 단위 = 이미 1행/환자) ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
print(f"\ntrain: {len(X_train)}, test: {len(X_test)}")

# ---------- 3. 정규화 ----------
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------- 4. RFE로 피처 10개 선택 (LogisticRegression 기준) ----------
N_FEATURES = 10
rfe = RFE(
    estimator=LogisticRegression(max_iter=5000, random_state=42),
    n_features_to_select=N_FEATURES
)
rfe.fit(X_train_scaled, y_train)

selected_mask = rfe.support_
selected_features = [f for f, m in zip(feature_cols, selected_mask) if m]
print(f"\n선택된 피처 {N_FEATURES}개:")
for f in selected_features:
    print(" -", f)

X_train_sel = X_train_scaled[:, selected_mask]
X_test_sel = X_test_scaled[:, selected_mask]

# ---------- 5. 모델별 GridSearchCV ----------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

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

results = {}
for name, (estimator, grid) in param_grids.items():
    gs = GridSearchCV(estimator, grid, cv=cv, scoring="roc_auc", n_jobs=-1)
    gs.fit(X_train_sel, y_train)

    best_model = gs.best_estimator_
    y_pred = best_model.predict(X_test_sel)
    y_proba = best_model.predict_proba(X_test_sel)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    results[name] = {
        "best_params": gs.best_params_,
        "cv_best_auc": gs.best_score_,
        "test_auc": auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": acc,
        "f1": f1,
        "confusion_matrix": (tn, fp, fn, tp),
    }

    print(f"\n=== {name} ===")
    print(f"best params: {gs.best_params_}")
    print(f"CV 5-fold best AUC: {gs.best_score_:.3f}")
    print(f"Test AUC: {auc:.3f} | Sensitivity: {sensitivity:.3f} | Specificity: {specificity:.3f} | Acc: {acc:.3f} | F1: {f1:.3f}")
    print(f"Confusion matrix (tn, fp, fn, tp): {tn}, {fp}, {fn}, {tp}")
    print(classification_report(y_test, y_pred, target_names=["Benign", "HCC"]))