import json
import re
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

def extract_by_code(components, target_code):
    for comp in components:
        for c in comp.get("code", {}).get("coding", []):
            if c.get("code") == target_code:
                return comp.get("valueString", "")
    return None

# "vascular invasion" 앞에 no/negative/absence of/without 등이 있으면 음성으로 판단
NEGATION_PATTERN = re.compile(
    r"(no|without|absen(t|ce)\s+of|negative\s+for)\s+(\w+\s+){0,3}vascular invasion",
    re.IGNORECASE
)
POSITIVE_PATTERN = re.compile(r"vascular invasion", re.IGNORECASE)

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

        pathology_text = extract_by_text(components, "PATHOLOGYRESULTTEXT") or ""
        macro = extract_by_code(components, "MACROVASCULARINVASION")

        has_mention = bool(POSITIVE_PATTERN.search(pathology_text))
        is_negated = bool(NEGATION_PATTERN.search(pathology_text))

        if not pathology_text:
            mvi_label = None  # 병리 텍스트 자체가 없음 (수술/조직검사 안 받은 환자)
        elif has_mention and not is_negated:
            mvi_label = 1
        elif has_mention and is_negated:
            mvi_label = 0
        else:
            mvi_label = 0  # "vascular invasion" 언급 자체가 없으면 없다고 간주 (주의: 확실친 않음)

        rows.append({
            "patient_id": patient_folder.name,
            "macro_invasion_field": macro,
            "pathology_text_len": len(pathology_text),
            "has_vascular_invasion_mention": has_mention,
            "is_negated": is_negated,
            "mvi_label": mvi_label,
            "pathology_text_snippet": pathology_text[:300],
        })

df = pd.DataFrame(rows)
print(f"전체 HCC 환자: {len(df)}")
print(f"병리 텍스트 있는 환자(=조직검사 받은 환자): {(df['pathology_text_len'] > 0).sum()}")
print(f"\nMVI 라벨 분포 (텍스트 있는 환자 대상):")
print(df[df["pathology_text_len"] > 0]["mvi_label"].value_counts())

df.to_csv("/home/jovyan/gcubme/SW_BAEK/HCC/mvi_label_candidate.csv", index=False, encoding="utf-8-sig")
print("\n-> mvi_label_candidate.csv 저장. pathology_text_snippet 전수 눈으로 검토 필수")