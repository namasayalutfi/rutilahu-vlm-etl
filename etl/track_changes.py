import json
from pathlib import Path

BASE_FILE = Path("metadata/mkn2_metadata_production_ready.json")
FINAL_FILE = Path("metadata/reconciled_metadata_mkn2.json")

OUTPUT_DETAILS_TXT = Path("metadata/changed_house_details.txt")
OUTPUT_IDS_TXT = Path("metadata/changed_house_ids.txt")


def load_json_list(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Isi file {path} harus berupa list JSON.")
    return data


def normalize_record(record):
    images = record.get("images", [])
    if not isinstance(images, list):
        images = []

    actual_label = record.get("actual_label", {})
    if not isinstance(actual_label, dict):
        actual_label = {}

    normalized_images = []
    for img in images:
        if not isinstance(img, dict):
            continue
        normalized_images.append({
            "image_id": str(img.get("image_id", "")).strip(),
            "view_type": str(img.get("view_type", "")).strip().lower(),
        })

    normalized_images.sort(key=lambda x: x["image_id"])

    return {
        "house_id": str(record.get("house_id", "")).strip(),
        "house_type": record.get("house_type"),
        "images": normalized_images,
        "actual_label": {
            "atap": actual_label.get("atap"),
            "dinding": actual_label.get("dinding"),
            "lantai": actual_label.get("lantai"),
        }
    }


def index_by_house_id(records):
    idx = {}
    for rec in records:
        house_id = rec["house_id"]
        if house_id:
            idx[house_id] = rec
    return idx


base_data = [normalize_record(r) for r in load_json_list(BASE_FILE)]
final_data = [normalize_record(r) for r in load_json_list(FINAL_FILE)]

base_idx = index_by_house_id(base_data)
final_idx = index_by_house_id(final_data)

all_house_ids = sorted(set(base_idx.keys()) | set(final_idx.keys()))

changed_details = []
changed_ids = []

for house_id in all_house_ids:
    base_rec = base_idx.get(house_id)
    final_rec = final_idx.get(house_id)

    if base_rec is None:
        changed_details.append(f"{house_id} | missing_in_base_file")
        changed_ids.append(house_id)
        continue

    if final_rec is None:
        changed_details.append(f"{house_id} | missing_in_final_file")
        changed_ids.append(house_id)
        continue

    changes = []

    # house_type
    if base_rec.get("house_type") != final_rec.get("house_type"):
        changes.append(
            f"house_type: {base_rec.get('house_type')} -> {final_rec.get('house_type')}"
        )

    # jumlah images
    base_images = base_rec.get("images", [])
    final_images = final_rec.get("images", [])
    if len(base_images) != len(final_images):
        changes.append(
            f"image_count: {len(base_images)} -> {len(final_images)}"
        )

    # view_type untuk image_id yang sama
    base_map = {img["image_id"]: img["view_type"] for img in base_images if img["image_id"]}
    final_map = {img["image_id"]: img["view_type"] for img in final_images if img["image_id"]}

    common_image_ids = sorted(set(base_map.keys()) & set(final_map.keys()))
    view_type_changes = []
    for image_id in common_image_ids:
        if base_map[image_id] != final_map[image_id]:
            view_type_changes.append(f"{image_id}: {base_map[image_id]} -> {final_map[image_id]}")

    if view_type_changes:
        changes.append("view_type_changed: " + "; ".join(view_type_changes))

    # actual_label
    if base_rec.get("actual_label") != final_rec.get("actual_label"):
        changes.append(
            f"actual_label: {base_rec.get('actual_label')} -> {final_rec.get('actual_label')}"
        )

    if changes:
        changed_details.append(f"{house_id} | " + " | ".join(changes))
        changed_ids.append(house_id)

# simpan txt detail
OUTPUT_DETAILS_TXT.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_DETAILS_TXT, "w", encoding="utf-8") as f:
    for line in changed_details:
        f.write(line + "\n")

# simpan txt house_id saja
with open(OUTPUT_IDS_TXT, "w", encoding="utf-8") as f:
    for house_id in changed_ids:
        f.write(house_id + "\n")

print(f"Total house_id dibandingkan : {len(all_house_ids)}")
print(f"Total house_id berubah      : {len(changed_ids)}")
print(f"Detail perubahan disimpan   : {OUTPUT_DETAILS_TXT}")
print(f"Daftar house_id disimpan    : {OUTPUT_IDS_TXT}")