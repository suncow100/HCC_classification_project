import re
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from pathlib import Path
from PIL import Image
from radiomics import featureextractor

MANIFEST_CSV = "/home/jovyan/gcubme/SW_BAEK/HCC/dataset_manifest_portal_only.csv"
OUT_CSV = "/home/jovyan/gcubme/SW_BAEK/HCC/radiomics_features_portal.csv"

UID_PATTERN = re.compile(r"(\d+(?:\.\d+){3,})")

def extract_uid(filename: str):
    m = UID_PATTERN.search(filename)
    return m.group(1) if m else None

def load_series_with_mask(series_path: Path, mask_path: Path):
    dcm_files = sorted(series_path.glob("*.dcm"))
    if not dcm_files:
        return None, None

    slices = [(pydicom.dcmread(str(f)), extract_uid(f.name)) for f in dcm_files]

    def sort_key(item):
        ds = item[0]
        if hasattr(ds, "ImagePositionPatient"):
            return float(ds.ImagePositionPatient[2])
        return int(getattr(ds, "InstanceNumber", 0))
    slices.sort(key=sort_key)

    mask_uid_map = {extract_uid(f.name): f for f in mask_path.iterdir() if extract_uid(f.name)}

    img_slices, mask_slices = [], []
    for ds, uid in slices:
        if uid not in mask_uid_map:
            return None, None  # UID 매칭 실패 -> 이 시리즈는 스킵
        img_slices.append(ds.pixel_array.astype(np.float32))
        mask_img = np.array(Image.open(mask_uid_map[uid]))
        mask_slices.append((mask_img > 0).astype(np.uint8))

    image_np = np.stack(img_slices, axis=0)
    mask_np = np.stack(mask_slices, axis=0)

    first_ds = slices[0][0]
    px_spacing = [float(x) for x in first_ds.PixelSpacing]
    if len(slices) > 1 and hasattr(first_ds, "ImagePositionPatient"):
        z0 = float(slices[0][0].ImagePositionPatient[2])
        z1 = float(slices[1][0].ImagePositionPatient[2])
        slice_thickness = abs(z1 - z0) or 1.0
    else:
        slice_thickness = float(getattr(first_ds, "SliceThickness", 1.0))
    spacing = (px_spacing[1], px_spacing[0], slice_thickness)

    image_sitk = sitk.GetImageFromArray(image_np)
    mask_sitk = sitk.GetImageFromArray(mask_np)
    image_sitk.SetSpacing(spacing)
    mask_sitk.SetSpacing(spacing)
    return image_sitk, mask_sitk

def main():
    df = pd.read_csv(MANIFEST_CSV)
    extractor = featureextractor.RadiomicsFeatureExtractor()

    results = []
    for i, row in df.iterrows():
        try:
            image_sitk, mask_sitk = load_series_with_mask(Path(row["series_path"]), Path(row["mask_path"]))
            if image_sitk is None:
                print(f"[스킵-UID불일치] {row['patient_id']}")
                continue
            if sitk.GetArrayFromImage(mask_sitk).sum() == 0:
                print(f"[스킵-빈마스크] {row['patient_id']}")
                continue

            fv = extractor.execute(image_sitk, mask_sitk)
            feat = {k: v for k, v in fv.items() if not k.startswith("diagnostics_")}
            feat.update({
                "patient_id": row["patient_id"],
                "class": row["class"],
                "binary_label": row["binary_label"],
                "series_path": row["series_path"],
            })
            results.append(feat)
            print(f"[{i+1}/{len(df)}] {row['patient_id']} ({row['class']}) 완료")
        except Exception as e:
            print(f"[에러] {row['patient_id']}: {e}")

    pd.DataFrame(results).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n피처 추출 완료 -> {OUT_CSV}")

if __name__ == "__main__":
    main()