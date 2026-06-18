from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class LabelStudioMergeConfig:
    input_dir: Path = Path("data/labelstudio_output_split")
    output_json: Path = Path("data/labelstudio_output_merged.json")
    recursive: bool = False


class LabelStudioOutputMerger:
    def __init__(self, config: LabelStudioMergeConfig | None = None):
        self.config = config or LabelStudioMergeConfig()

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

    def _list_json_files(self) -> List[Path]:
        if not self.config.input_dir.exists():
            raise FileNotFoundError(f"Folder tidak ditemukan: {self.config.input_dir}")

        if self.config.recursive:
            files = sorted(
                p for p in self.config.input_dir.rglob("*.json")
                if p.is_file()
            )
        else:
            files = sorted(
                p for p in self.config.input_dir.glob("*.json")
                if p.is_file()
            )

        if not files:
            raise FileNotFoundError(f"Tidak ada file .json di folder: {self.config.input_dir}")

        return files

    @staticmethod
    def _normalize_items(obj: Any) -> List[Dict[str, Any]]:
        """
        Label Studio output biasanya JSON array.
        Kalau suatu file berisi 1 object saja, tetap dikemas jadi list.
        """
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict):
            return [obj]
        return []

    def merge(self) -> Dict[str, Any]:
        files = self._list_json_files()

        merged: List[Dict[str, Any]] = []
        per_file_counts: Dict[str, int] = {}

        for path in files:
            obj = self._read_json(path)
            items = self._normalize_items(obj)
            merged.extend(items)
            per_file_counts[path.name] = len(items)

        self._write_json(merged, self.config.output_json)

        return {
            "input_dir": str(self.config.input_dir),
            "output_json": str(self.config.output_json),
            "total_files": len(files),
            "total_records": len(merged),
            "per_file_counts": per_file_counts,
        }