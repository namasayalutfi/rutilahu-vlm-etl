import json
from pathlib import Path

BASE_DIR = Path("data/sft_dataset")
CHANGED_DIR = Path("data/sft_dataset_changed_only")
OUTPUT_DIR = Path("data/sft_dataset_merged")

SPLITS = ["train", "val", "test"]


def load_jsonl(path: Path):
    records = []
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_house_id(record):
    """
    Skema:
    {
      "id": {"house_id": "H10177"},
      ...
    }
    """
    id_block = record.get("id", {})
    if isinstance(id_block, dict):
        house_id = id_block.get("house_id")
        if house_id is not None:
            return str(house_id).strip()
    return ""


def merge_split(split_name: str):
    base_path = BASE_DIR / f"{split_name}.jsonl"
    changed_path = CHANGED_DIR / f"{split_name}.jsonl"
    output_path = OUTPUT_DIR / f"{split_name}.jsonl"

    base_records = load_jsonl(base_path)
    changed_records = load_jsonl(changed_path)

    changed_index = {}
    for rec in changed_records:
        house_id = get_house_id(rec)
        if house_id:
            changed_index[house_id] = rec

    merged_records = []
    used_changed_ids = set()

    # Prioritas: data dari changed_only menggantikan base jika house_id sama
    for rec in base_records:
        house_id = get_house_id(rec)
        if house_id in changed_index:
            merged_records.append(changed_index[house_id])
            used_changed_ids.add(house_id)
        else:
            merged_records.append(rec)

    # Tambahkan record baru yang hanya ada di changed_only
    for rec in changed_records:
        house_id = get_house_id(rec)
        if house_id and house_id not in used_changed_ids and house_id not in {get_house_id(x) for x in base_records}:
            merged_records.append(rec)

    save_jsonl(merged_records, output_path)

    return {
        "split": split_name,
        "base_records": len(base_records),
        "changed_records": len(changed_records),
        "output_records": len(merged_records),
        "output_path": str(output_path),
    }


def main():
    summary = []
    for split in SPLITS:
        result = merge_split(split)
        summary.append(result)
        print(f"[OK] {split} selesai")
        print(f"  base_records   : {result['base_records']}")
        print(f"  changed_records: {result['changed_records']}")
        print(f"  output_records : {result['output_records']}")
        print(f"  output_path    : {result['output_path']}")
        print()

    print("[OK] Semua split selesai.")


if __name__ == "__main__":
    main()