from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class MergeConfig:
    reconciled_metadata_path: Path = Path("metadata/reconciled_mkn2_metadata.json")
    mkn2_metadata_path: Path = Path("metadata/mkn2_metadata.json")
    output_path: Path = Path("metadata/mkn2_metadata_final.json")
    dedupe_by_house_id: bool = False


class MetadataMerger:
    def __init__(self, config: MergeConfig | None = None):
        self.config = config or MergeConfig()

    @staticmethod
    def _read_json(path: Path) -> Any:
        if not path.exists():
            raise FileNotFoundError(f"File tidak ditemukan: {path}")

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _write_json(data: Any, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _ensure_list(obj: Any) -> List[Dict[str, Any]]:
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict):
            return [obj]
        raise ValueError("Format JSON harus berupa list atau object.")

    @staticmethod
    def _get_house_id(record: Dict[str, Any]) -> str:
        value = record.get("house_id")
        return str(value).strip() if value is not None else ""

    @staticmethod
    def _normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
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
        reconciled_raw = self._read_json(self.config.reconciled_metadata_path)
        mkn2_raw = self._read_json(self.config.mkn2_metadata_path)

        reconciled_records = self._ensure_list(reconciled_raw)
        mkn2_records = self._ensure_list(mkn2_raw)

        reconciled_records = [self._normalize_record(r) for r in reconciled_records]
        mkn2_records = [self._normalize_record(r) for r in mkn2_records]

        merged_records: List[Dict[str, Any]] = []

        if self.config.dedupe_by_house_id:
            seen_house_ids = set()

            for rec in reconciled_records + mkn2_records:
                hid = self._get_house_id(rec)
                if hid and hid in seen_house_ids:
                    continue
                if hid:
                    seen_house_ids.add(hid)
                merged_records.append(rec)
        else:
            merged_records = reconciled_records + mkn2_records

        self._write_json(merged_records, self.config.output_path)

        return {
            "reconciled_records": len(reconciled_records),
            "mkn2_records": len(mkn2_records),
            "total_output_records": len(merged_records),
            "output_path": str(self.config.output_path),
        }