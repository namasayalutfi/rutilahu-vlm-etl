import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any


@dataclass
class DTSENDummyConfig:
    input_path: Path = Path("metadata/splited_metadata/all.json")
    output_path: Path = Path("metadata/mkn2_metadata_production_ready.json")
    seed: int = 42
    same_probability: float = 0.7


class DTSENDummyGenerator:
    LABELS = {
        "atap": [
            "Beton",
            "Genteng",
            "Seng",
            "Asbes",
            "Bambu",
            "Kayu/sirap",
            "Jerami/ijuk/daun-daunan/rumbia",
            "Lainnya",
        ],
        "dinding": [
            "Tembok",
            "Plesteran anyaman bambu/kawat",
            "Kayu/papan/gypsum/GRC/calciboard",
            "Anyaman bambu",
            "Batang kayu",
            "Bambu",
            "Lainnya",
        ],
        "lantai": [
            "Marmer/granit",
            "Keramik",
            "Parket/vinil/karpet",
            "Ubin/tegel/teraso",
            "Kayu/papan",
            "Semen/bata merah",
            "Bambu",
            "Tanah",
            "Lainnya",
        ],
    }

    def __init__(self, config: DTSENDummyConfig):
        self.config = config
        self.rng = random.Random(config.seed)

    @staticmethod
    def load_json(path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"File tidak ditemukan: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Input JSON harus berupa list of records.")
        return data

    @staticmethod
    def save_json(data: List[Dict[str, Any]], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def generate_dtsen_label(self, actual: str, component: str) -> str:
        candidates = self.LABELS[component]

        # actual bisa "Tidak terdeteksi", tapi dtsen tidak boleh punya label itu
        if actual == "Tidak terdeteksi" or actual is None:
            return self.rng.choice(candidates)

        if self.rng.random() < self.config.same_probability:
            return actual

        other = [x for x in candidates if x != actual]
        return self.rng.choice(other)

    @staticmethod
    def generate_status(actual: str, dtsen: str) -> str:
        if actual == "Tidak terdeteksi":
            return "Tidak teridentifikasi"
        if actual == dtsen:
            return "Sesuai"
        return "Tidak sesuai"

    def process_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        actual = record.get("actual_label", {})
        if not isinstance(actual, dict):
            actual = {}

        dtsen: Dict[str, str] = {}
        status: Dict[str, str] = {}

        for component in ["atap", "dinding", "lantai"]:
            actual_value = actual.get(component, "Tidak terdeteksi")
            if actual_value is None:
                actual_value = "Tidak terdeteksi"

            dtsen_value = self.generate_dtsen_label(actual_value, component)
            dtsen[component] = dtsen_value
            status[component] = self.generate_status(actual_value, dtsen_value)

        record["dtsen"] = dtsen
        record["status"] = status
        return record

    def run(self) -> Dict[str, Any]:
        data = self.load_json(self.config.input_path)

        output = []
        summary = {
            "Sesuai": 0,
            "Tidak sesuai": 0,
            "Tidak teridentifikasi": 0,
        }

        for record in data:
            updated = self.process_record(record)
            output.append(updated)

            for value in updated["status"].values():
                if value in summary:
                    summary[value] += 1

        self.save_json(output, self.config.output_path)

        return {
            "total_house": len(output),
            "output_path": str(self.config.output_path),
            **summary,
        }