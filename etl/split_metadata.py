from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class SplitConfig:
    input_path: Path = Path("output/mkn2_cleaned_metadata.json")
    output_dir: Path = Path("splited_metadata")
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42

    # Disimpan untuk kompatibilitas dan laporan, tidak dipakai sebagai constraint utama.
    min_combo_support: int = 10

    # Jika True, pembagian kecil tetap diusahakan mengikuti proporsi.
    enforce_min_split_when_possible: bool = True


class HouseTypeAwareHierarchicalStratifiedSplitter:
    """
    Strategi split yang dipakai:

    1) Split dilakukan pada level house_id, jadi satu rumah tidak pernah pecah.
    2) Split dilakukan terpisah per house_type.
    3) Dalam setiap house_type, strata yang dipakai hanya label material yang relevan:
       - multi -> atap, dinding, lantai
       - single_exterior_only -> atap, dinding
       - single_interior_only -> lantai
    4) Combo tidak dipakai sebagai constraint split, hanya untuk laporan distribusi.
    """

    SPLITS = ("train", "val", "test")

    def __init__(self, config: SplitConfig | None = None):
        self.config = config or SplitConfig()
        self.rng = random.Random(self.config.seed)

    def load_records(self) -> List[Dict[str, Any]]:
        path = self.config.input_path
        if not path.exists():
            raise FileNotFoundError(f"Input file tidak ditemukan: {path}")

        if path.suffix.lower() == ".jsonl":
            records: List[Dict[str, Any]] = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            return records

        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        if not isinstance(obj, list):
            raise ValueError("Input JSON harus berupa list of records.")

        return obj

    def save_json(self, data: List[Dict[str, Any]], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _normalize_house_type(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip().lower()
        if not text:
            return None

        mapping = {
            "multi": "multi",
            "single_exterior_only": "single_exterior_only",
            "single_interior_only": "single_interior_only",
            "exterior_only": "single_exterior_only",
            "interior_only": "single_interior_only",
        }
        return mapping.get(text, text)

    @staticmethod
    def _normalize_label(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        if text.lower() == "tidak terdeteksi":
            return "Tidak terdeteksi"
        return text

    def _normalize_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        rec = dict(record)
        rec["house_type"] = self._normalize_house_type(rec.get("house_type"))

        actual = rec.get("actual_label", {})
        if not isinstance(actual, dict):
            actual = {}

        rec["actual_label"] = {
            "atap": self._normalize_label(actual.get("atap")),
            "dinding": self._normalize_label(actual.get("dinding")),
            "lantai": self._normalize_label(actual.get("lantai")),
        }
        return rec

    def _normalized_ratios(self) -> List[float]:
        ratios = [
            max(float(self.config.train_ratio), 0.0),
            max(float(self.config.val_ratio), 0.0),
            max(float(self.config.test_ratio), 0.0),
        ]
        total = sum(ratios)
        if total <= 0:
            return [0.8, 0.1, 0.1]
        return [r / total for r in ratios]

    def _relevant_components(self, house_type: str) -> Tuple[str, ...]:
        if house_type == "multi":
            return ("atap", "dinding", "lantai")
        if house_type == "single_exterior_only":
            return ("atap", "dinding")
        if house_type == "single_interior_only":
            return ("lantai",)
        return ()

    def _strata_key(self, record: Dict[str, Any]) -> str:
        """
        Key strata yang dipakai untuk balancing.
        """
        ht = self._normalize_house_type(record.get("house_type"))
        if ht is None:
            return "unknown"

        actual = record.get("actual_label", {})
        if not isinstance(actual, dict):
            actual = {}

        parts = [ht]
        for comp in self._relevant_components(ht):
            val = self._normalize_label(actual.get(comp)) or "None"
            parts.append(f"{comp}={val}")

        return "||".join(parts)

    def _combo_key(self, record: Dict[str, Any], relevant_components: Tuple[str, ...]) -> str:
        actual = record.get("actual_label", {})
        if not isinstance(actual, dict):
            actual = {}

        parts = []
        for comp in relevant_components:
            val = self._normalize_label(actual.get(comp)) or "None"
            parts.append(f"{comp}={val}")
        return "||".join(parts)

    def _desired_split_sizes(self, n: int) -> Dict[str, int]:
        """
        Hitung ukuran split yang proporsional secara saklek per kombinasi/strata lokal.
        """
        if n <= 0:
            return {"train": 0, "val": 0, "test": 0}

        # FIX: Aturan Saklek Kelompok Kecil per Kombinasi
        if n == 1:
            return {"train": 1, "val": 0, "test": 0}

        if n == 2:
            return {"train": 1, "val": 0, "test": 1}

        if n == 3:
            return {"train": 1, "val": 1, "test": 1}

        # Aturan untuk n > 3 (Mengikuti proporsi rasio global)
        ratios = self._normalized_ratios()
        raw = [n * r for r in ratios]
        base = [int(math.floor(x)) for x in raw]
        remainder = n - sum(base)

        frac_order = sorted(range(3), key=lambda i: (raw[i] - base[i]), reverse=True)
        for i in frac_order[:remainder]:
            base[i] += 1

        if self.config.enforce_min_split_when_possible:
            for i in range(3):
                if base[i] == 0 and n >= 3:
                    donor = max(range(3), key=lambda x: base[x])
                    if base[donor] > 1:
                        base[donor] -= 1
                        base[i] += 1

        while sum(base) < n:
            donor = max(range(3), key=lambda i: ratios[i])
            base[donor] += 1

        while sum(base) > n:
            donor = max(range(3), key=lambda i: base[i])
            if base[donor] > 1:
                base[donor] -= 1
            else:
                break

        return {"train": base[0], "val": base[1], "test": base[2]}

    def _build_units(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Bentuk unit split berdasarkan house_id.
        """
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for idx, record in enumerate(records):
            rec = self._normalize_record(record)
            house_id = str(rec.get("house_id") or f"__row_{idx}__")
            grouped[house_id].append(rec)

        units: List[Dict[str, Any]] = []
        for house_id, recs in grouped.items():
            house_type = None
            for rec in recs:
                ht = self._normalize_house_type(rec.get("house_type"))
                if ht in {"multi", "single_exterior_only", "single_interior_only"}:
                    house_type = ht
                    break

            if house_type is None:
                continue

            strata_key = None
            for rec in recs:
                strata_key = self._strata_key(rec)
                break

            if strata_key is None:
                continue

            units.append(
                {
                    "house_id": house_id,
                    "house_type": house_type,
                    "records": recs,
                    "size": len(recs),
                    "strata_key": strata_key,
                }
            )

        return units

    def _split_units_in_group(self, units: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        total_size = sum(unit["size"] for unit in units)
        if total_size <= 0:
            return {"train": [], "val": [], "test": []}

        buckets: Dict[str, List[Dict[str, Any]]] = {
            "train": [],
            "val": [],
            "test": [],
        }

        # Group berdasarkan kombinasi strata unik
        strata_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for unit in units:
            strata_groups[unit["strata_key"]].append(unit)

        # Urutkan dari strata dengan jumlah rumah terbanyak
        ordered_strata = sorted(
            strata_groups.items(),
            key=lambda kv: (-sum(u["size"] for u in kv[1]), kv[0]),
        )

        for _, strata_units in ordered_strata:
            self.rng.shuffle(strata_units)
            
            # FIX UTAMA: Hitung target split berdasarkan total rumah di kombinasi ini
            num_units = len(strata_units)
            alloc = self._desired_split_sizes(num_units)

            # Buat sequence distribusi split berdasarkan target alloc kombinasi ini
            split_sequence: List[str] = (
                ["train"] * alloc["train"] +
                ["val"] * alloc["val"] +
                ["test"] * alloc["test"]
            )

            # Acak urutan sequence split agar distribusi antar rumah bersifat random
            self.rng.shuffle(split_sequence)

            for unit, split in zip(strata_units, split_sequence):
                for rec in unit["records"]:
                    rec_to_store = dict(rec)
                    rec_to_store["split"] = split
                    buckets[split].append(rec_to_store)

        return buckets

    def split(self, records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        units = self._build_units(records)

        grouped_by_type: Dict[str, List[Dict[str, Any]]] = {
            "multi": [],
            "single_exterior_only": [],
            "single_interior_only": [],
        }

        for unit in units:
            ht = unit["house_type"]
            if ht in grouped_by_type:
                grouped_by_type[ht].append(unit)

        final_buckets: Dict[str, List[Dict[str, Any]]] = {
            "train": [],
            "val": [],
            "test": [],
        }

        for house_type, type_units in grouped_by_type.items():
            if not type_units:
                continue

            split_result = self._split_units_in_group(type_units)

            for split in self.SPLITS:
                final_buckets[split].extend(split_result[split])

        for split in self.SPLITS:
            self.rng.shuffle(final_buckets[split])

        return final_buckets

    def _counter_by_component(self, records: List[Dict[str, Any]], comp: str) -> Counter:
        counter = Counter()
        for rec in records:
            actual = rec.get("actual_label", {})
            if not isinstance(actual, dict):
                continue
            val = actual.get(comp)
            if val is not None:
                counter[str(val)] += 1
        return counter

    def _combo_counter(self, records: List[Dict[str, Any]], house_type: Optional[str] = None) -> Counter:
        counter = Counter()
        for rec in records:
            ht = self._normalize_house_type(rec.get("house_type"))
            if house_type is not None and ht != house_type:
                continue

            relevant_components = self._relevant_components(ht or "")
            if not relevant_components:
                continue

            counter[self._combo_key(rec, relevant_components)] += 1

        return counter

    def summarize(self, split_buckets: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        split_sizes = {s: len(split_buckets[s]) for s in self.SPLITS}

        house_type_distribution = {
            split: dict(Counter(rec.get("house_type", "unknown") for rec in split_buckets[split]))
            for split in self.SPLITS
        }

        label_distribution_global = {
            comp: {
                split: dict(self._counter_by_component(split_buckets[split], comp))
                for split in self.SPLITS
            }
            for comp in ("atap", "dinding", "lantai")
        }

        label_distribution_by_schema: Dict[str, Dict[str, Dict[str, int]]] = {}
        combo_distribution_by_schema: Dict[str, Dict[str, Dict[str, int]]] = {}

        for schema in ("multi", "single_exterior_only", "single_interior_only"):
            schema_label_dist: Dict[str, Dict[str, int]] = {}
            for comp in self._relevant_components(schema):
                schema_label_dist[comp] = {
                    split: dict(
                        Counter(
                            str(rec.get("actual_label", {}).get(comp))
                            for rec in split_buckets[split]
                            if self._normalize_house_type(rec.get("house_type")) == schema
                            and isinstance(rec.get("actual_label", {}), dict)
                            and rec.get("actual_label", {}).get(comp) is not None
                        )
                    )
                    for split in self.SPLITS
                }
            label_distribution_by_schema[schema] = schema_label_dist

            combo_distribution_by_schema[schema] = {
                split: dict(self._combo_counter(split_buckets[split], house_type=schema))
                for split in self.SPLITS
            }

        combo_distribution_global = {
            split: dict(self._combo_counter(split_buckets[split], house_type=None))
            for split in self.SPLITS
        }

        return {
            "split_sizes": split_sizes,
            "house_type_distribution": house_type_distribution,
            "label_distribution_global": label_distribution_global,
            "label_distribution_by_schema": label_distribution_by_schema,
            "combo_distribution_global": combo_distribution_global,
            "combo_distribution_by_schema": combo_distribution_by_schema,
        }

    def run(self) -> Dict[str, Any]:
        records = self.load_records()
        split_buckets = self.split(records)

        out_dir = self.config.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        train_path = out_dir / "train.json"
        val_path = out_dir / "val.json"
        test_path = out_dir / "test.json"
        all_path = out_dir / "all.json"

        self.save_json(split_buckets["train"], train_path)
        self.save_json(split_buckets["val"], val_path)
        self.save_json(split_buckets["test"], test_path)

        all_records = split_buckets["train"] + split_buckets["val"] + split_buckets["test"]
        self.save_json(all_records, all_path)

        summary = self.summarize(split_buckets)

        return {
            "total_records": len(records),
            "train_path": str(train_path),
            "val_path": str(val_path),
            "test_path": str(test_path),
            "all_path": str(all_path),
            **summary,
        }