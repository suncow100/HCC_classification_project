import os
import re
import csv
from pathlib import Path
from collections import Counter

ROOT = Path("/home/jovyan/gcubme/SW_BAEK/HCC/학생연구원_이윤석교수님/data")
OUT_ROOT = Path("/home/jovyan/gcubme/SW_BAEK/HCC")  # 결과물 저장 위치

# 폴더명 -> 표준 클래스 라벨 매핑
CLASS_LABEL_MAP = {
    "liver_간암 ct(nia)": "HCC",
    "cyst": "cyst",
    "hemangioma": "hemangioma",
    "dn": "DN",
    "fnh": "FNH",
    "adenoma": "adenoma",
}

def has_mask_subfolder(folder: Path):
    for sub in folder.iterdir():
        if sub.is_dir() and sub.name.lower() == "mask":
            return sub
    return None

def get_class_and_patient(series_folder: Path):
    parts = series_folder.parts
    category = raw_cls = patient_id = None
    for i, p in enumerate(parts):
        low = p.lower()
        if low in ("malignant", "benign"):
            category = low
            if i + 1 < len(parts):
                m = re.match(r"(\d+)\.(.+)", parts[i + 1])
                if m:
                    raw_cls = m.group(2).lower()
            if category == "benign":
                if i + 3 < len(parts):
                    patient_id = parts[i + 3]
            else:
                if i + 2 < len(parts):
                    patient_id = parts[i + 2]
            break
    return category, raw_cls, patient_id

rows = []
skipped_no_dcm = []  # 마스크는 있지만 dcm 없는 고아 마스크 기록용

for dirpath, dirnames, filenames in os.walk(ROOT):
    folder = Path(dirpath)
    mask_folder = has_mask_subfolder(folder)
    if mask_folder is None:
        continue

    has_dcm = any(f.lower().endswith(".dcm") for f in filenames)
    if not has_dcm:
        skipped_no_dcm.append(str(folder))
        continue

    category, raw_cls, patient_id = get_class_and_patient(folder)
    cls = CLASS_LABEL_MAP.get(raw_cls, raw_cls)

    rows.append({
        "patient_id": patient_id,
        "category": category,          # malignant / benign
        "class": cls,                  # HCC / cyst / hemangioma / DN / FNH / adenoma
        "binary_label": 1 if category == "malignant" else 0,
        "series_path": str(folder),
        "mask_path": str(mask_folder),
        "num_dicom": sum(1 for f in filenames if f.lower().endswith(".dcm")),
        "num_mask_files": sum(1 for _ in mask_folder.iterdir()),
    })

out_csv = OUT_ROOT / "dataset_manifest_with_mask.csv"
with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"학습용 manifest 생성 완료: {out_csv}")
print(f"총 시리즈(마스크+dcm 모두 있음): {len(rows)}개")
print(f"제외된 고아 마스크(dcm 없음): {len(skipped_no_dcm)}개 -> {skipped_no_dcm}")

print("\n=== 클래스별 (시리즈 수 / 환자 수) ===")
cls_counter = Counter(r["class"] for r in rows)
for c in cls_counter:
    n_patients = len({r["patient_id"] for r in rows if r["class"] == c})
    print(f"  {c}: {cls_counter[c]} series / {n_patients} patients")