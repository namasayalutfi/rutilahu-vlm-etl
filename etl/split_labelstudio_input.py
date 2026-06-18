from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class LabelStudioSplitConfig:
    input_json: Path = Path("data/labelstudio_input.json")
    output_dir: Path = Path("data/labelstudio_input_split")
    num_splits: int = 8
    seed: int = 42


class LabelStudioInputSplitter:
    def __init__(self, config: LabelStudioSplitConfig | None = None):
        self.config = config or LabelStudioSplitConfig()
        self.rng = random.Random(self.config.seed)

    @staticmethod
    def _read_json(path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"File tidak ditemukan: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("labelstudio_input.json harus berupa JSON array/list.")

        return data

    @staticmethod
    def _write_json(data: Any, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _extract_house_type(item: Dict[str, Any]) -> str:
        data = item.get("data", {})
        if not isinstance(data, dict):
            return "unknown"
        house_type = data.get("house_type")
        if house_type is None:
            return "unknown"
        text = str(house_type).strip().lower()
        return text if text else "unknown"

    def _group_by_house_type(self, records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for item in records:
            ht = self._extract_house_type(item)
            groups[ht].append(item)
        return groups

    def _split_round_robin_stratified(self, records: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Split merata dengan stratifikasi sederhana berdasarkan house_type.
        Setiap grup house_type di-shuffle lalu dibagikan round-robin ke bucket.
        """
        groups = self._group_by_house_type(records)
        buckets: List[List[Dict[str, Any]]] = [[] for _ in range(self.config.num_splits)]

        group_names = sorted(groups.keys())
        for group_idx, group_name in enumerate(group_names):
            group_items = groups[group_name][:]
            self.rng.shuffle(group_items)

            start_offset = group_idx % self.config.num_splits
            for i, item in enumerate(group_items):
                bucket_idx = (start_offset + i) % self.config.num_splits
                buckets[bucket_idx].append(item)

        # shuffle kecil di dalam tiap bucket supaya urutannya tidak terlalu blok berdasarkan house_type
        for bucket in buckets:
            self.rng.shuffle(bucket)

        return buckets

    def run(self) -> Dict[str, Any]:
        records = self._read_json(self.config.input_json)
        buckets = self._split_round_robin_stratified(records)

        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        out_paths = []
        for idx, bucket in enumerate(buckets, start=1):
            out_path = self.config.output_dir / f"labelstudio_input_part_{idx:02d}.json"
            self._write_json(bucket, out_path)
            out_paths.append(str(out_path))

        summary = {
            "total_records": len(records),
            "num_splits": self.config.num_splits,
            "split_sizes": {f"part_{i+1:02d}": len(buckets[i]) for i in range(self.config.num_splits)},
            "output_dir": str(self.config.output_dir),
            "output_files": out_paths,
        }

        self._write_json(summary, self.config.output_dir / "split_manifest.json")
        return summary