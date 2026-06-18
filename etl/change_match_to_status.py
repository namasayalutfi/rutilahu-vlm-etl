import json
from pathlib import Path

file_path = Path("metadata/metadata.jsonl")

# baca jsonl
data = []
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            data.append(json.loads(line))

# modifikasi
for record in data:
    record.pop("match", None)

    record["status"] = {
        "atap": None,
        "dinding": None,
        "lantai": None,
    }

# tulis kembali sebagai jsonl
with open(file_path, "w", encoding="utf-8") as f:
    for record in data:
        f.write(json.dumps(record, ensure_ascii=False))
        f.write("\n")

print(f"Updated {len(data)} records")