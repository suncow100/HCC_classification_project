import json
from pathlib import Path
import pandas as pd

ROOT = Path("/home/jovyan/gcubme/SW_BAEK/HCC/학생연구원_이윤석교수님/data/Malignant")

def find_metadata_json(patient_folder: Path):
    meta_dir = patient_folder / "metadata"
    if not meta_dir.exists():
        return None
    jsons = list(meta_dir.glob("*.json"))
    return jsons[0] if jsons else None

def extract_by_text(components, target_text):
    for comp in components:
        if comp.get("code", {}).get("text") == target_text:
            return comp.get("valueString", "")
    return None

rows = []
for cls_folder in ROOT.iterdir():
    if not cls_folder.is_dir():
        continue
    for patient_folder in cls_folder.iterdir():
        if not patient_folder.is_dir():
            continue
        json_path = find_metadata_json(patient_folder)
        if json_path is None:
            continue
        data = json.loads(json_path.read_text(encoding="utf-8"))
        components = data.get("component", [])

        opbx = extract_by_text(components, "OPBx")
        pathology_text = extract_by_text(components, "PATHOLOGYRESULTTEXT") or ""

        rows.append({
            "patient_id": patient_folder.name,
            "opbx": opbx,
            "has_pathology_text": len(pathology_text) > 0,
        })

df = pd.DataFrame(rows)
print(f"전체 HCC 환자: {len(df)}")
print("\n=== OPBX 값별 분포 ===")
print(df["opbx"].value_counts(dropna=False))

print("\n=== OPBX 값 x 병리텍스트 존재 여부 교차표 ===")
print(pd.crosstab(df["opbx"], df["has_pathology_text"]))