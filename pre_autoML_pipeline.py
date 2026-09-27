import pandas as pd
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor
import matplotlib.pyplot as plt

FEATURES_CSV = "/home/jovyan/gcubme/SW_BAEK/HCC/radiomics_features_portal.csv"
SAVE_PATH = "/home/jovyan/gcubme/SW_BAEK/HCC/AutogluonModels/hcc_binary"

df = pd.read_csv(FEATURES_CSV)

meta_cols = ["patient_id", "class", "binary_label", "series_path"]
feature_cols = [c for c in df.columns if c not in meta_cols]
print(f"전체 피처 수: {len(feature_cols)}")

df_model = df[feature_cols + ["binary_label"]].rename(columns={"binary_label": "label"})

train_df, test_df = train_test_split(
    df_model, test_size=0.25, stratify=df_model["label"], random_state=42
)
print(f"train: {len(train_df)}, test: {len(test_df)}")

# ---------- AutoGluon 학습 ----------
predictor = TabularPredictor(
    label="label",
    eval_metric="roc_auc",
    problem_type="binary",
    path=SAVE_PATH,
).fit(
    train_df,
    presets="best_quality",
)

leaderboard = predictor.leaderboard(test_df, silent=True)
print("\n=== 모델별 leaderboard (test set 기준) ===")
print(leaderboard)

perf = predictor.evaluate(test_df, auxiliary_metrics=True)
print("\n=== best model test 성능 ===")
print(perf)

# ---------- Permutation feature importance ----------
importance_df = predictor.feature_importance(
    test_df, subsample_size=len(test_df), num_shuffle_sets=10
)
importance_df.to_csv("/home/jovyan/gcubme/SW_BAEK/HCC/feature_importance.csv")
print("\n=== 피처 중요도 상위 20개 ===")
print(importance_df.head(20))

# ---------- 시각화 ----------
top_n = 20
top_importance = importance_df.head(top_n).sort_values("importance")

plt.figure(figsize=(9, 8))
plt.barh(top_importance.index, top_importance["importance"], xerr=top_importance["stddev"])
plt.xlabel("Permutation Importance (AUC drop)")
plt.title(f"Top {top_n} Radiomics Feature Importance (HCC vs Benign)")
plt.tight_layout()
plt.savefig("/home/jovyan/gcubme/SW_BAEK/HCC/feature_importance_top20.png", dpi=150)
print("\n-> feature_importance_top20.png 저장 완료")