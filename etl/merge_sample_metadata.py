from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


@dataclass
class MergeSampleMetadataConfig:
    mkn2_metadata_path: Path = Path("metadata/mkn2_metadata.json")
    metadata_jsonl_path: Path = Path("metadata/metadata.jsonl")
    crawled_metadata_path: Path = Path("metadata/crawled_img_metadata.json")
    output_path: Path = Path("metadata/mkn2_metadata_merged.json")
    max_per_label: int = 1000


class SampleMetadataMerger:
    TARGET_ATAP = {
        "Asbes",
        "Seng",
        "Beton",
    }

    TARGET_DINDING = {
        "Kayu/papan/gypsum/GRC/calciboard",
    }

    TARGET_LANTAI = {
        "Tanah",
        "Ubin/tegel/teraso",
    }

    def __init__(self, config: MergeSampleMetadataConfig | None = None):
        self.config = config or MergeSampleMetadataConfig()

    @staticmethod
    def _read_json(path: Path) -> Any:
        if not path.exists():
            raise FileNotFoundError(f"File tidak ditemukan: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"File tidak ditemukan: {path}")

        records: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    @staticmethod
    def _write_json(data: Any, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _norm_text(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text if text else None

    @staticmethod
    def _norm_label(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        if text.lower() == "tidak terdeteksi":
            return "Tidak terdeteksi"
        return text

    def _matched_target_labels(self, record: Dict[str, Any]) -> Dict[str, str]:
        actual = record.get("actual_label", {})
        if not isinstance(actual, dict):
            return {}

        matched: Dict[str, str] = {}

        atap = self._norm_label(actual.get("atap"))
        dinding = self._norm_label(actual.get("dinding"))
        lantai = self._norm_label(actual.get("lantai"))

        if atap in self.TARGET_ATAP:
            matched["atap"] = atap
        if dinding in self.TARGET_DINDING:
            matched["dinding"] = dinding
        if lantai in self.TARGET_LANTAI:
            matched["lantai"] = lantai

        return matched

    @staticmethod
    def _extract_no_kk_set(records: List[Dict[str, Any]]) -> Set[str]:
        out: Set[str] = set()
        for rec in records:
            no_kk = rec.get("no_kk")
            if no_kk is None:
                continue
            no_kk_str = str(no_kk).strip()
            if no_kk_str:
                out.add(no_kk_str)
        return out

    @staticmethod
    def _normalize_metadata_record(record: Dict[str, Any]) -> Dict[str, Any]:
        rec = dict(record)

        if "house_id" not in rec:
            rec["house_id"] = None
        if "no_kk" not in rec:
            rec["no_kk"] = None
        if "house_type" not in rec:
            rec["house_type"] = None
        if "split" not in rec:
            rec["split"] = None
        if "images" not in rec or not isinstance(rec["images"], list):
            rec["images"] = []

        actual = rec.get("actual_label", {})
        if not isinstance(actual, dict):
            actual = {}
        rec["actual_label"] = {
            "atap": actual.get("atap"),
            "dinding": actual.get("dinding"),
            "lantai": actual.get("lantai"),
        }

        dtsen = rec.get("dtsen", {})
        if not isinstance(dtsen, dict):
            dtsen = {}
        rec["dtsen"] = {
            "atap": dtsen.get("atap"),
            "dinding": dtsen.get("dinding"),
            "lantai": dtsen.get("lantai"),
        }

        status = rec.get("status", {})
        if not isinstance(status, dict):
            status = {}
        rec["status"] = {
            "atap": status.get("atap"),
            "dinding": status.get("dinding"),
            "lantai": status.get("lantai"),
        }

        return rec

    def merge(self) -> Dict[str, Any]:
        # mkn2 hanya dipakai sebagai reference no_kk
        mkn2_records = self._read_json(self.config.mkn2_metadata_path)
        if not isinstance(mkn2_records, list):
            raise ValueError("mkn2_metadata.json harus berupa JSON array/list.")

        mkn2_records = [self._normalize_metadata_record(r) for r in mkn2_records]
        existing_no_kk = self._extract_no_kk_set(mkn2_records)

        # metadata.jsonl -> difilter dan dimasukkan
        source_records = self._read_jsonl(self.config.metadata_jsonl_path)

        filtered_new_records: List[Dict[str, Any]] = []
        skipped_duplicate_no_kk = 0
        skipped_not_target = 0
        skipped_label_limit = 0
        skipped_duplicate_within_jsonl = 0

        label_counts = {
            "atap": {label: 0 for label in self.TARGET_ATAP},
            "dinding": {label: 0 for label in self.TARGET_DINDING},
            "lantai": {label: 0 for label in self.TARGET_LANTAI},
        }

        seen_no_kk_in_new_data: Set[str] = set()

        for rec in source_records:
            rec = self._normalize_metadata_record(rec)

            no_kk = self._norm_text(rec.get("no_kk"))
            if no_kk:
                if no_kk in existing_no_kk or no_kk in seen_no_kk_in_new_data:
                    skipped_duplicate_no_kk += 1
                    skipped_duplicate_within_jsonl += 1 if no_kk in seen_no_kk_in_new_data else 0
                    continue

            matched = self._matched_target_labels(rec)
            if not matched:
                skipped_not_target += 1
                continue

            over_limit = False
            for comp, label in matched.items():
                if label_counts[comp][label] >= self.config.max_per_label:
                    over_limit = True
                    break

            if over_limit:
                skipped_label_limit += 1
                continue

            filtered_new_records.append(rec)

            if no_kk:
                existing_no_kk.add(no_kk)
                seen_no_kk_in_new_data.add(no_kk)

            for comp, label in matched.items():
                label_counts[comp][label] += 1

        # crawled metadata langsung ditambahkan
        crawled_records = self._read_json(self.config.crawled_metadata_path)
        if not isinstance(crawled_records, list):
            raise ValueError("crawled_img_metadata.json harus berupa JSON array/list.")

        crawled_records = [self._normalize_metadata_record(r) for r in crawled_records]

        merged_records = filtered_new_records + crawled_records

        self._write_json(merged_records, self.config.output_path)

        return {
            "mkn2_reference_records": len(mkn2_records),
            "filtered_from_jsonl": len(filtered_new_records),
            "crawled_records": len(crawled_records),
            "total_output_records": len(merged_records),
            "skipped_duplicate_no_kk": skipped_duplicate_no_kk,
            "skipped_duplicate_within_jsonl": skipped_duplicate_within_jsonl,
            "skipped_not_target": skipped_not_target,
            "skipped_label_limit": skipped_label_limit,
            "label_counts": label_counts,
            "max_per_label": self.config.max_per_label,
            "output_path": str(self.config.output_path),
        }