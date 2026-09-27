# 메타데이터에서 환자의 phase(동맥기 문맥기 지연기) 라벨정도가 정확히 있는지 확인하는 코드

import pydicom
import pandas as pd
import re
from pathlib import Path

IN_CSV = "/home/jovyan/gcubme/SW_BAEK/HCC/dataset_manifest_with_mask.csv"
OUT_CSV = "/home/jovyan/gcubme/SW_BAEK/HCC/phase_completeness.csv"

df = pd.read_csv(IN_CSV)

def get_series_description(series_path):
    first_dcm = next(Path(series_path).glob("*.dcm"), None)
    if first_dcm is None:
        return ""
    ds = pydicom.dcmread(str(first_dcm), stop_before_pixels=True)
    return getattr(ds, "SeriesDescription", "") or ""

df["series_description"] = df["series_path"].apply(get_series_description)
desc = df["series_description"]

# 이전에 검증한 키워드 규칙 그대로 사용
is_arterial = desc.str.contains(r"arteri|artery", case=False, regex=True)
is_portal_venous = (
    desc.str.contains("portal", case=False)
    | desc.str.contains("veno", case=False)
    | desc.str.contains(r"\bpvp\b", case=False, regex=True)
) & ~desc.str.contains(r"arteri|artery|delay", case=False, regex=True)
is_delay = desc.str.contains("delay", case=False) & ~desc.str.contains(r"arteri|artery", case=False, regex=True)

df["phase"] = "other"
df.loc[is_arterial, "phase"] = "arterial"
df.loc[is_portal_venous, "phase"] = "portal"
df.loc[is_delay, "phase"] = "delay"

# 환자별로 어떤 phase를 갖고 있는지 집계
phase_summary = df.groupby("patient_id")["phase"].apply(lambda x: set(x)).reset_index()
phase_summary["has_arterial"] = phase_summary["phase"].apply(lambda s: "arterial" in s)
phase_summary["has_portal"] = phase_summary["phase"].apply(lambda s: "portal" in s)
phase_summary["has_delay"] = phase_summary["phase"].apply(lambda s: "delay" in s)
phase_summary["n_phases"] = phase_summary[["has_arterial", "has_portal", "has_delay"]].sum(axis=1)

# class 정보도 붙이기
patient_class = df[["patient_id", "class", "category"]].drop_duplicates("patient_id")
phase_summary = phase_summary.merge(patient_class, on="patient_id")

print(f"전체 환자 수: {len(phase_summary)}")
print("\n=== phase 조합 개수 분포 ===")
print(phase_summary["n_phases"].value_counts().sort_index())

print("\n=== 3-phase(동맥+문맥+지연) 다 가진 환자 수 (class별) ===")
full3 = phase_summary[phase_summary["n_phases"] == 3]
print(full3.groupby("class").size())

print("\n=== 2-phase 이하인 환자 수 (class별) ===")
partial = phase_summary[phase_summary["n_phases"] < 3]
print(partial.groupby("class").size())

print("\n=== phase가 하나도 안 잡힌(=전부 other) 환자 목록 ===")
none_matched = phase_summary[phase_summary["n_phases"] == 0]
print(none_matched[["patient_id", "class"]].to_string())

phase_summary.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print(f"\n-> {OUT_CSV} 저장 완료")

# 참고: long format용 최종 학습 데이터는 df에서 phase="other"인 행만 제외하면 됨
usable_series = df[df["phase"] != "other"]
print(f"\n최종 학습에 쓸 수 있는 시리즈(phase 식별된 것) 수: {len(usable_series)} / 전체 {len(df)}")