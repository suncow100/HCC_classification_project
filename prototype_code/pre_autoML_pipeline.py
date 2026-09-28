'''
HCC/pre_idea_train_pipeline.py 에 따라서 다중공선성 분석을 넣을까 했지만,
다중공선성 분석을(임계치 90 이상) 거치면서 제거된 피쳐가 생각보다 너무많고 유의미하다고 생각해서
다중공선성 필터링된 57개(또는 LASSO의 46개)
일단은 다시 107개 피쳐(파이라디오믹스 기본값(필터하나도없을때)) + 내가 의도적으로 추가한 3개피쳐
동맥기 문맥기 지연기 피쳐만 넣어서 학습시킨 결과로만
시각화 자료를 뽑았음

'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, f1_score
)
from autogluon.tabular import TabularPredictor

FEATURES_CSV = "/home/jovyan/gcubme/SW_BAEK/HCC/radiomics_features_multiphase.csv"
SAVE_PATH = "/home/jovyan/gcubme/SW_BAEK/HCC/AutogluonModels/hcc_multiphase"
OUT_DIR = "/home/jovyan/gcubme/SW_BAEK/HCC"

# ---------- 1. 데이터 로드 ----------
df = pd.read_csv(FEATURES_CSV)

meta_cols = ["patient_id", "category", "class", "binary_label", "phase", "series_path"]
radiomics_cols = [c for c in df.columns if c not in meta_cols]

phase_dummies = pd.get_dummies(df["phase"], prefix="phase")
df = pd.concat([df, phase_dummies], axis=1)
feature_cols = radiomics_cols + list(phase_dummies.columns)

print(f"전체 시리즈(행) 수: {len(df)}, 환자 수: {df['patient_id'].nunique()}, 피처 수: {len(feature_cols)}")

# ---------- 2. 환자 단위 train/test 분할 (기존 로직 그대로) ----------
patient_labels = df[["patient_id", "binary_label"]].drop_duplicates("patient_id")

label_consistency = df.groupby("patient_id")["binary_label"].nunique()
assert (label_consistency == 1).all(), "같은 환자인데 binary_label이 다른 시리즈가 있습니다"

train_patients, test_patients = train_test_split(
    patient_labels["patient_id"],
    test_size=0.25,
    stratify=patient_labels["binary_label"],
    random_state=42
)

train_df = df[df["patient_id"].isin(train_patients)].reset_index(drop=True)
test_df = df[df["patient_id"].isin(test_patients)].reset_index(drop=True)

print(f"train: 환자 {train_df['patient_id'].nunique()}명 / 시리즈 {len(train_df)}행")
print(f"test:  환자 {test_df['patient_id'].nunique()}명 / 시리즈 {len(test_df)}행")

# ---------- 3. AutoGluon 학습 ----------
ag_train_df = train_df[feature_cols + ["binary_label"]].rename(columns={"binary_label": "label"})
ag_test_df = test_df[feature_cols + ["binary_label"]].rename(columns={"binary_label": "label"})

predictor = TabularPredictor(
    label="label",
    eval_metric="roc_auc",
    problem_type="binary",
    path=SAVE_PATH,
).fit(
    ag_train_df,
    presets="best_quality",
    num_gpus=1,        # CUDA_VISIBLE_DEVICES=7 설정하고 실행할 것
    time_limit=1800,
)

leaderboard = predictor.leaderboard(ag_test_df, silent=True)
print("\n=== Leaderboard (test 기준) ===")
print(leaderboard.head(10))

perf = predictor.evaluate(ag_test_df, auxiliary_metrics=True)
print("\n=== Best model test 성능 ===")
print(perf)

y_true = ag_test_df["label"].values
y_proba = predictor.predict_proba(ag_test_df.drop(columns=["label"]))[1].values
y_pred = predictor.predict(ag_test_df.drop(columns=["label"])).values

# ---------- 4. ROC curve ----------
fpr, tpr, _ = roc_curve(y_true, y_proba)
roc_auc_val = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, linewidth=2, label=f"AutoGluon (AUC = {roc_auc_val:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.title("ROC Curve — HCC vs Benign (multiphase, patient-level split)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/roc_curve_multiphase.png", dpi=150)
plt.close()
print(f"-> roc_curve_multiphase.png 저장 완료")

# ---------- 5. Confusion Matrix ----------
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "HCC"])

fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax, cmap="Blues", values_format="d")
plt.title("Confusion Matrix — HCC vs Benign (test set)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confusion_matrix_multiphase.png", dpi=150)
plt.close()
print(f"-> confusion_matrix_multiphase.png 저장 완료")

acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
print(f"\nAccuracy: {acc:.3f} | F1: {f1:.3f}")

# ---------- 6. Permutation Feature Importance ----------
importance_df = predictor.feature_importance(
    ag_test_df, subsample_size=len(ag_test_df), num_shuffle_sets=10
)
importance_df.to_csv(f"{OUT_DIR}/feature_importance_multiphase.csv")
print("\n=== 피처 중요도 상위 20개 ===")
print(importance_df.head(20))

top_n = 20
top_importance = importance_df.head(top_n).sort_values("importance")

plt.figure(figsize=(9, 8))
plt.barh(top_importance.index, top_importance["importance"], xerr=top_importance["stddev"])
plt.xlabel("Permutation Importance (AUC drop)")
plt.title(f"Top {top_n} Feature Importance (multiphase, HCC vs Benign)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/feature_importance_multiphase.png", dpi=150)
plt.close()
print("-> feature_importance_multiphase.png 저장 완료")