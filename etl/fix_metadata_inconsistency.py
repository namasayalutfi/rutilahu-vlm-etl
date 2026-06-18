#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

VALID_SCHEMA = {
    "atap": [
        "Beton", "Genteng", "Seng", "Asbes", "Bambu", "Kayu/sirap",
        "Jerami/ijuk/daun-daunan/rumbia", "Lainnya", "Tidak terdeteksi",
    ],
    "dinding": [
        "Tembok", "Plesteran anyaman bambu/kawat", "Kayu/papan/gypsum/GRC/calciboard",
        "Anyaman bambu", "Batang kayu", "Bambu", "Lainnya", "Tidak terdeteksi",
    ],
    "lantai": [
        "Marmer/granit", "Keramik", "Parket/vinil/karpet", "Ubin/tegel/teraso",
        "Kayu/papan", "Semen/bata merah", "Bambu", "Tanah", "Lainnya", "Tidak terdeteksi",
    ],
}

CANONICAL_MAP = {
    "atap": {
        "beton": "Beton",
        "genteng": "Genteng",
        "seng": "Seng",
        "asbes": "Asbes",
        "bambu": "Bambu",
        "kayu/sirap": "Kayu/sirap",
        "kayu sirap": "Kayu/sirap",
        "jerami/ijuk/daun-daunan/rumbia": "Jerami/ijuk/daun-daunan/rumbia",
        "jerami/ijuk/daun_daunan/rumbia": "Jerami/ijuk/daun-daunan/rumbia",
        "lainnya": "Lainnya",
        "tidak terdeteksi": "Tidak terdeteksi",
        "tidak_terdeteksi": "Tidak terdeteksi",
    },
    "dinding": {
        "tembok": "Tembok",
        "plesteran anyaman bambu/kawat": "Plesteran anyaman bambu/kawat",
        "plesteran_anyaman_bambu/kawat": "Plesteran anyaman bambu/kawat",
        "kayu/papan/gypsum/grc/calciboard": "Kayu/papan/gypsum/GRC/calciboard",
        "kayu papan gypsum grc calciboard": "Kayu/papan/gypsum/GRC/calciboard",
        "anyaman bambu": "Anyaman bambu",
        "anyaman_bambu": "Anyaman bambu",
        "batang kayu": "Batang kayu",
        "batang_kayu": "Batang kayu",
        "bambu": "Bambu",
        "lainnya": "Lainnya",
        "tidak terdeteksi": "Tidak terdeteksi",
        "tidak_terdeteksi": "Tidak terdeteksi",
    },
    "lantai": {
        "marmer/granit": "Marmer/granit",
        "keramik": "Keramik",
        "parket/vinil/karpet": "Parket/vinil/karpet",
        "ubin/tegel/teraso": "Ubin/tegel/teraso",
        "kayu/papan": "Kayu/papan",
        "semen/bata merah": "Semen/bata merah",
        "semen/bata_merah": "Semen/bata merah",
        "bambu": "Bambu",
        "tanah": "Tanah",
        "lainnya": "Lainnya",
        "tidak terdeteksi": "Tidak terdeteksi",
        "tidak_terdeteksi": "Tidak terdeteksi",
    },
}

# sekarang langsung sesuai dengan struktur metadata Anda
FIELD_MAP = {
    "atap": "atap",
    "dinding": "dinding",
    "lantai": "lantai",
}

def normalize_key(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip().lower()
    s = s.replace("_", " ")
    s = " ".join(s.split())
    return s

def canonicalize(category: str, value: Any) -> Any:
    if value is None:
        return value
    raw = str(value).strip()
    if raw == "":
        return raw
    if raw.lower() == "unclassified":
        return "unclassified"

    norm = normalize_key(raw)
    mapping = CANONICAL_MAP[category]
    if norm in mapping:
        return mapping[norm]
    if raw.lower() in mapping:
        return mapping[raw.lower()]
    return raw

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at line {line_no} in {path}: {e}") from e
    return records

def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")

def fix_metadata(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Counter]]:
    out = []
    changes = defaultdict(Counter)

    for rec in records:
        new_rec = deepcopy(rec)
        actual = new_rec.get("actual_label", {})

        if isinstance(actual, dict):
            for category, field_name in FIELD_MAP.items():
                old = actual.get(field_name, None)
                new = canonicalize(category, old)
                if new != old:
                    changes[f"actual_label.{field_name}"][f"{old} -> {new}"] += 1
                    actual[field_name] = new

            new_rec["actual_label"] = actual

        out.append(new_rec)

    return out, changes

def main() -> int:
    parser = argparse.ArgumentParser(description="Fix label inconsistencies in metadata/metadata.jsonl")
    parser.add_argument("--metadata", type=Path, default=Path("metadata/metadata.jsonl"))
    parser.add_argument("--out-metadata", type=Path, default=Path("metadata/metadata_fixed.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("metadata/metadata_fix_report.json"))
    parser.add_argument("--inplace", action="store_true")
    args = parser.parse_args()

    metadata_path = args.metadata
    out_metadata_path = metadata_path if args.inplace else args.out_metadata
    report_path = args.report

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    metadata_records = load_jsonl(metadata_path)
    print(f"Loaded metadata records: {len(metadata_records)}")

    print("\nUnique values BEFORE fix:")
    for category, field in FIELD_MAP.items():
        counter = Counter()
        for rec in metadata_records:
            actual = rec.get("actual_label", {})
            if isinstance(actual, dict):
                val = actual.get(field, None)
                if val is not None:
                    counter[str(val)] += 1
        print(f"\n=== {category.upper()} ({field}) ===")
        print(sorted(counter.keys()))

    fixed_metadata, md_changes = fix_metadata(metadata_records)

    out_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    write_jsonl(out_metadata_path, fixed_metadata)

    report = {
        "inputs": {"metadata": str(metadata_path)},
        "outputs": {"metadata": str(out_metadata_path)},
        "changes": {"metadata": {k: dict(v) for k, v in md_changes.items()}},
        "valid_schema": VALID_SCHEMA,
        "notes": ["unclassified is preserved as-is", "Labels outside the canonical mapping are left unchanged"],
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nSaved cleaned metadata: {out_metadata_path}")
    print(f"Saved report: {report_path}")

    print("\nChanges summary:")
    if not md_changes:
        print("No changes were needed.")
    else:
        for field, counter in md_changes.items():
            print(f"\nMetadata changes for {field}:")
            for k, v in counter.most_common():
                print(f"  {k}: {v}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())