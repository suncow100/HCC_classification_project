import json
import re
from pathlib import Path
import pandas as pd

BASE = Path("/home/jovyan/gcubme/SW_BAEK/HCC/학생연구원_이윤석교수님/data")

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

def extract_by_code(components, target_code):
    for comp in components:
        for c in comp.get("code", {}).get("coding", []):
            if c.get("code") == target_code:
                return comp.get("valueString", "")
    return None

POSITIVE_PATTERN = re.compile(r"\bvascular invasion\b", re.IGNORECASE)
LYMPHO_POSITIVE_PATTERN = re.compile(r"\blymphovascular invasion\b", re.IGNORECASE)
NEGATION_PATTERN = re.compile(
    r"(no|without|absen(t|ce)\s+of|negative\s+for)\s+(\w+\s+){0,3}\b(lympho)?vascular invasion\b",
    re.IGNORECASE
)

rows = []

# --- Malignant ---
for cls_folder in (BASE / "Malignant").iterdir():
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
        pathology_text = extract_by_text(components, "PATHOLOGYRESULTTEXT") or ""
        rows.append({
            "patient_id": patient_folder.name,
            "category": "malignant",
            "class": cls_folder.name,
            "pathology_text_len": len(pathology_text),
            "has_vascular_mention": bool(POSITIVE_PATTERN.search(pathology_text) or LYMPHO_POSITIVE_PATTERN.search(pathology_text)),
        })

# --- Benign (class/CT/환자ID 구조) ---
for cls_folder in (BASE / "Benign").iterdir():
    if not cls_folder.is_dir():
        continue
    ct_folder = cls_folder / "CT"
    if not ct_folder.exists():
        continue
    for patient_folder in ct_folder.iterdir():
        if not patient_folder.is_dir():
            continue
        json_path = find_metadata_json(patient_folder)
        if json_path is None:
            continue
        data = json.loads(json_path.read_text(encoding="utf-8"))
        components = data.get("component", [])
        pathology_text = extract_by_text(components, "PATHOLOGYRESULTTEXT") or ""
        rows.append({
            "patient_id": patient_folder.name,
            "category": "benign",
            "class": cls_folder.name,
            "pathology_text_len": len(pathology_text),
            "has_vascular_mention": bool(POSITIVE_PATTERN.search(pathology_text) or LYMPHO_POSITIVE_PATTERN.search(pathology_text)),
        })

df = pd.DataFrame(rows)

print("=== category별 전체 환자 수 ===")
print(df.groupby("category").size())

print("\n=== category별 병리 텍스트(PATHOLOGYRESULTTEXT) 있는 환자 수 ===")
print(df[df["pathology_text_len"] > 0].groupby("category").size())

print("\n=== class별 병리 텍스트 있는 환자 수 ===")
print(df[df["pathology_text_len"] > 0].groupby(["category", "class"]).size())

print("\n=== vascular/lymphovascular invasion 언급 있는 환자 수 (category별) ===")
print(df[df["has_vascular_mention"]].groupby("category").size())

df.to_csv("/home/jovyan/gcubme/SW_BAEK/HCC/pathology_text_scan_all.csv", index=False, encoding="utf-8-sig")
print("\n-> pathology_text_scan_all.csv 저장 완료")