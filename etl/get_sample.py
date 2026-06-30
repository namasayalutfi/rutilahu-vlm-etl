import json
from pathlib import Path

# =========================
# CONFIG
# =========================

INPUT_PATH = Path("metadata/reconciled_mkn2_metadata_final.json")
OUTPUT_PATH = Path("metadata/mkn2_test_metadata.json")

TARGET_SPLIT = "test"


# =========================
# LOAD JSON
# =========================

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

if not isinstance(data, list):
    raise ValueError("Input JSON harus berupa list/array.")


# =========================
# FILTER TEST SPLIT
# =========================

filtered_data = []

for record in data:
    split = record.get("split")

    if split == TARGET_SPLIT:
        filtered_data.append(record)


# =========================
# SAVE OUTPUT
# =========================

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(
        filtered_data,
        f,
        ensure_ascii=False,
        indent=2,
    )


# =========================
# SUMMARY
# =========================

print(f"Total original data : {len(data)}")
print(f"Total test data     : {len(filtered_data)}")
print(f"Saved to            : {OUTPUT_PATH}")