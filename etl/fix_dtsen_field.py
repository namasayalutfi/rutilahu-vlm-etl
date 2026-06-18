import json
from pathlib import Path

file_path = Path("metadata/metadata.jsonl")
temp_path = file_path.with_suffix(".tmp")

count = 0

with open(file_path, "r", encoding="utf-8") as fin, \
     open(temp_path, "w", encoding="utf-8") as fout:

    for line in fin:
        line = line.strip()
        if not line:
            continue

        record = json.loads(line)

        record["dtsen"] = {
            "atap": None,
            "dinding": None,
            "lantai": None,
        }

        fout.write(json.dumps(record, ensure_ascii=False))
        fout.write("\n")

        count += 1

# replace file asli
temp_path.replace(file_path)

print(f"Updated {count} records")
print(f"Overwritten {file_path}")