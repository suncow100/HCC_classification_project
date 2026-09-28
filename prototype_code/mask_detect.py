import pydicom
import pandas as pd
from pathlib import Path

IN_CSV = "/home/jovyan/gcubme/SW_BAEK/HCC/dataset_manifest_with_mask.csv"
OUT_CSV = "/home/jovyan/gcubme/SW_BAEK/HCC/dataset_manifest_portal_only.csv"

df = pd.read_csv(IN_CSV)

def get_series_description(series_path):
    first_dcm = next(Path(series_path).glob("*.dcm"), None)
    if first_dcm is None:
        return None
    ds = pydicom.dcmread(str(first_dcm), stop_before_pixels=True)
    return getattr(ds, "SeriesDescription", None)

df["series_description"] = df["series_path"].apply(get_series_description)
desc = df["series_description"].fillna("")

# portal 또는 venous(오타 VENOS 포함) 계열 -> 문맥기로 간주
# 동시에 arterial/artery/delay가 같이 들어간 경우는 제외 (혹시 모를 오탐 방지)
is_portal_like = (
    desc.str.contains("portal", case=False)
    | desc.str.contains("veno", case=False)
    | desc.str.contains(r"\bpvp\b", case=False, regex=True)
)
is_other_phase = desc.str.contains("arteri|artery|delay", case=False, regex=True)

portal_df = df[is_portal_like & ~is_other_phase].copy()
print(f"portal/venous로 잡힌 시리즈: {len(portal_df)}개")

dup_counts = portal_df.groupby("patient_id").size()
dups = dup_counts[dup_counts > 1]
print(f"환자당 2개 이상 잡힌 경우: {len(dups)}명")
if len(dups) > 0:
    print(portal_df[portal_df["patient_id"].isin(dups.index)]
          [["patient_id", "class", "series_description"]].to_string())

all_patients = set(df["patient_id"])
portal_patients = set(portal_df["patient_id"])
missing = all_patients - portal_patients
print(f"\n여전히 누락된 환자: {len(missing)}명")
if missing:
    # 이 환자들의 '모든' 시리즈 설명을 다 보여줌 (진짜 없는건지 확인)
    missing_all = df[df["patient_id"].isin(missing)][["patient_id", "class", "series_description"]]
    print(missing_all.to_string())

portal_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print(f"\n-> {OUT_CSV} 저장 ({len(portal_df)}행, {portal_df['patient_id'].nunique()}명)")