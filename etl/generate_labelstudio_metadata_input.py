from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class LabelStudioConfig:
    input_json: Path = Path("metadata/mkn2_test_metadata.json")
    output_json: Path = Path("data/mkn2_data_test_input.json")


class LabelStudioMetadataGenerator:
    def __init__(self, config: LabelStudioConfig | None = None):
        self.config = config or LabelStudioConfig()

    @staticmethod
    def _read_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"File tidak ditemukan: {path}")

        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        if not content:
            return []

        # JSON array / JSON object
        if content.startswith("[") or content.startswith("{"):
            obj = json.loads(content)
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict):
                return [obj]
            raise ValueError(f"Format JSON tidak dikenali: {path}")

        # Fallback JSONL
        records: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    @staticmethod
    def _normalize_house_type_for_ls(house_type: Optional[str]) -> str:
        mapping = {
            "multi": "multi",
            "single_exterior_only": "single_exterior_only",
            "single_interior_only": "single_interior_only",
            "single": "single_exterior_only",
        }

        if house_type is None:
            return "multi"

        text = str(house_type).strip().lower()
        if not text or text in {"none", "null"}:
            return "multi"

        return mapping.get(text, "multi")

    @staticmethod
    def _normalize_choice_text(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        if text == "" or text.lower() in {"none", "null"}:
            return None
        return text

    @staticmethod
    def _actual_label_to_choice(
        value: Any,
        field_name: str,
        house_type: str,
    ) -> str:
        normalized = LabelStudioMetadataGenerator._normalize_choice_text(value)
        if normalized is not None:
            return normalized
        return "Belum Diisi"

    @staticmethod
    def _get_images_by_view(record: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        images = record.get("images", [])
        if isinstance(images, str):
            try:
                images = json.loads(images)
            except Exception:
                images = []

        if not isinstance(images, list):
            images = []

        exterior_images = []
        interior_images = []

        for img in images:
            if not isinstance(img, dict):
                continue
            view_type = str(img.get("view_type", "")).strip().lower()
            if view_type == "exterior":
                exterior_images.append(img)
            elif view_type == "interior":
                interior_images.append(img)

        return exterior_images, interior_images

    @staticmethod
    def _infer_house_type_from_images(record: Dict[str, Any]) -> str:
        exterior_images, interior_images = LabelStudioMetadataGenerator._get_images_by_view(record)

        has_exterior = len(exterior_images) > 0
        has_interior = len(interior_images) > 0

        if has_exterior and has_interior:
            return "multi"
        if has_exterior:
            return "single_exterior_only"
        if has_interior:
            return "single_interior_only"

        return "multi"

    @staticmethod
    def _slot_image_data(
        images: List[Dict[str, Any]],
        idx: int,
    ) -> Tuple[str, str]:
        if idx >= len(images):
            return "", ""

        img = images[idx]
        image_url = str(img.get("image_db_url") or img.get("image_path") or "")
        image_id = str(img.get("image_id") or "")
        return image_url, image_id

    @staticmethod
    def _build_choice_result(from_name: str, to_name: str, choice_value: str) -> Dict[str, Any]:
        return {
            "from_name": from_name,
            "to_name": to_name,
            "type": "choices",
            "value": {
                "choices": [choice_value],
            },
        }

    def _resolve_house_type(self, record: Dict[str, Any]) -> str:
        raw_house_type = record.get("house_type")
        if raw_house_type is None or str(raw_house_type).strip().lower() in {"", "none", "null"}:
            return self._infer_house_type_from_images(record)
        return str(raw_house_type).strip().lower()

    def _build_data_block(self, record: Dict[str, Any]) -> Dict[str, Any]:
        house_id = record.get("house_id", "")
        house_type = self._resolve_house_type(record)

        exterior_images, interior_images = self._get_images_by_view(record)

        ext_img_1_db, ext_img_1_id = self._slot_image_data(exterior_images, 0)
        ext_img_2_db, ext_img_2_id = self._slot_image_data(exterior_images, 1)
        int_img_1_db, int_img_1_id = self._slot_image_data(interior_images, 0)
        int_img_2_db, int_img_2_id = self._slot_image_data(interior_images, 1)

        return {
            "house_id": house_id,
            "house_type": self._normalize_house_type_for_ls(house_type),

            "ext_img_1_exists": "yes" if len(exterior_images) >= 1 else "no",
            "ext_img_2_exists": "yes" if len(exterior_images) >= 2 else "no",
            "int_img_1_exists": "yes" if len(interior_images) >= 1 else "no",
            "int_img_2_exists": "yes" if len(interior_images) >= 2 else "no",

            "ext_img_1": ext_img_1_db,
            "ext_img_2": ext_img_2_db,
            "int_img_1": int_img_1_db,
            "int_img_2": int_img_2_db,

            "ext_img_1_id": ext_img_1_id,
            "ext_img_2_id": ext_img_2_id,
            "int_img_1_id": int_img_1_id,
            "int_img_2_id": int_img_2_id,
        }

    def _build_predictions(self, record: Dict[str, Any]) -> List[Dict[str, Any]]:
        house_type = self._resolve_house_type(record)

        actual_label = record.get("actual_label", {})
        if isinstance(actual_label, str):
            try:
                actual_label = json.loads(actual_label)
            except Exception:
                actual_label = {}

        if not isinstance(actual_label, dict):
            actual_label = {}

        exterior_images, interior_images = self._get_images_by_view(record)

        result: List[Dict[str, Any]] = []

        result.append(
            self._build_choice_result(
                from_name="house_type_valid",
                to_name="house_anchor",
                choice_value=self._normalize_house_type_for_ls(house_type),
            )
        )

        result.append(self._build_choice_result("ext_img_1_exists_ctrl", "house_anchor", "yes" if len(exterior_images) >= 1 else "no"))
        result.append(self._build_choice_result("ext_img_2_exists_ctrl", "house_anchor", "yes" if len(exterior_images) >= 2 else "no"))
        result.append(self._build_choice_result("int_img_1_exists_ctrl", "house_anchor", "yes" if len(interior_images) >= 1 else "no"))
        result.append(self._build_choice_result("int_img_2_exists_ctrl", "house_anchor", "yes" if len(interior_images) >= 2 else "no"))

        result.append(
            self._build_choice_result(
                "jenis_atap_terluas",
                "house_anchor",
                self._actual_label_to_choice(actual_label.get("atap"), "atap", house_type),
            )
        )
        result.append(
            self._build_choice_result(
                "jenis_dinding_terluas",
                "house_anchor",
                self._actual_label_to_choice(actual_label.get("dinding"), "dinding", house_type),
            )
        )
        result.append(
            self._build_choice_result(
                "jenis_lantai_terluas",
                "house_anchor",
                self._actual_label_to_choice(actual_label.get("lantai"), "lantai", house_type),
            )
        )

        if len(exterior_images) >= 1:
            result.append(self._build_choice_result("ext_img_1_status", "ext_img_1", "KEEP"))
        if len(exterior_images) >= 2:
            result.append(self._build_choice_result("ext_img_2_status", "ext_img_2", "KEEP"))
        if len(interior_images) >= 1:
            result.append(self._build_choice_result("int_img_1_status", "int_img_1", "KEEP"))
        if len(interior_images) >= 2:
            result.append(self._build_choice_result("int_img_2_status", "int_img_2", "KEEP"))

        return result

    def run(self) -> Path:
        records = self._read_json_or_jsonl(self.config.input_json)

        output_items: List[Dict[str, Any]] = []
        for record in records:
            item = {
                "data": self._build_data_block(record),
                "predictions": [
                    {
                        "result": self._build_predictions(record),
                    }
                ],
            }
            output_items.append(item)

        self.config.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config.output_json, "w", encoding="utf-8") as f:
            json.dump(output_items, f, ensure_ascii=False, indent=2)

        return self.config.output_json