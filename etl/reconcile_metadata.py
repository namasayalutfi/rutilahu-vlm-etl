from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set


ATAP_MAP = {
    "Beton": "Beton",
    "Genteng": "Genteng",
    "Seng": "Seng",
    "Asbes": "Asbes",
    "Bambu": "Bambu",
    "Kayu/sirap": "Kayu/sirap",
    "Jerami/ijuk/daun-daunan/rumbia": "Jerami/ijuk/daun-daunan/rumbia",
    "Lainnya": "Lainnya",
    "Tidak terdeteksi": "Tidak terdeteksi",
    "Tidak Terdeteksi": "Tidak terdeteksi",
    "Belum Diisi": "Belum Diisi",
}

DINDING_MAP = {
    "Tembok": "Tembok",
    "Plesteran anyaman bambu/kawat": "Plesteran anyaman bambu/kawat",
    "Kayu/papan/gypsum/GRC/calciboard": "Kayu/papan/gypsum/GRC/calciboard",
    "Anyaman bambu": "Anyaman bambu",
    "Batang kayu": "Batang kayu",
    "Bambu": "Bambu",
    "Lainnya": "Lainnya",
    "Tidak terdeteksi": "Tidak terdeteksi",
    "Tidak Terdeteksi": "Tidak terdeteksi",
    "Belum Diisi": "Belum Diisi",
}

LANTAI_MAP = {
    "Marmer/granit": "Marmer/granit",
    "Keramik": "Keramik",
    "Parket/vinil/karpet": "Parket/vinil/karpet",
    "Ubin/tegel/teraso": "Ubin/tegel/teraso",
    "Kayu/papan": "Kayu/papan",
    "Semen/bata merah": "Semen/bata merah",
    "Bambu": "Bambu",
    "Tanah": "Tanah",
    "Lainnya": "Lainnya",
    "Tidak terdeteksi": "Tidak terdeteksi",
    "Tidak Terdeteksi": "Tidak terdeteksi",
    "Belum Diisi": "Belum Diisi",
}


@dataclass
class ReconcileConfig:
    metadata_path: Path = Path("metadata/mkn2_metadata_merged.json")
    labelstudio_output_path: Path = Path("data/labelstudio_output_merged.json")
    output_json_path: Path = Path("metadata/reconciled_mkn2_metadata.json")


class LabelStudioMetadataReconciler:
    def __init__(self, config: ReconcileConfig | None = None):
        self.config = config or ReconcileConfig()

    @staticmethod
    def _read_json_records(path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"File tidak ditemukan: {path}")

        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        if not content:
            return []

        obj = json.loads(content)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict):
            return [obj]

        raise ValueError(f"Format file tidak dikenali: {path}")

    @staticmethod
    def _extract_house_id(record: Dict[str, Any]) -> str:
        if record.get("house_id"):
            return str(record["house_id"])

        data = record.get("data", {})
        if isinstance(data, dict) and data.get("house_id"):
            return str(data["house_id"])

        return ""

    @staticmethod
    def _get_anomaly_notes(record: Dict[str, Any]) -> str:
        notes = record.get("anomaly_notes")
        if notes is None:
            notes = record.get("anomaly_notes_text")

        if notes is None:
            data = record.get("data", {})
            if isinstance(data, dict):
                notes = data.get("anomaly_notes") or data.get("anomaly_notes_text")

        if notes is None:
            return ""

        if isinstance(notes, list):
            return " | ".join(str(x) for x in notes)

        return str(notes)

    @staticmethod
    def _parse_anomaly_flags(notes: str) -> Set[str]:
        text = (notes or "").lower()
        flags: Set[str] = set()

        patterns = {
            "anomali_1": [r"\banomali[_\s-]?1\b", r"\banomaly[_\s-]?1\b"],
            "anomali_2": [r"\banomali[_\s-]?2\b", r"\banomaly[_\s-]?2\b"],
            "anomali_3": [r"\banomali[_\s-]?3\b", r"\banomaly[_\s-]?3\b"],
            "anomali_4": [r"\banomali[_\s-]?4\b", r"\banomaly[_\s-]?4\b"],
        }

        for name, pats in patterns.items():
            for pat in pats:
                if re.search(pat, text):
                    flags.add(name)
                    break

        return flags

    @staticmethod
    def _normalize_choice(value: Any, field_name: str) -> Any:
        if value is None:
            return None

        text = str(value).strip()
        if text == "":
            return None

        if field_name == "atap":
            return ATAP_MAP.get(text, text)
        if field_name == "dinding":
            return DINDING_MAP.get(text, text)
        if field_name == "lantai":
            return LANTAI_MAP.get(text, text)

        return text

    @staticmethod
    def _normalize_meta_house_type(house_type: str) -> str:
        ht = str(house_type).strip().lower()
        mapping = {
            "multi": "multi",
            "exterior_only": "single_exterior_only",
            "interior_only": "single_interior_only",
            "single_exterior_only": "single_exterior_only",
            "single_interior_only": "single_interior_only",
        }
        return mapping.get(ht, ht)

    @staticmethod
    def _normalize_labelstudio_house_type(house_type: Any) -> str:
        ht = str(house_type).strip().lower()
        mapping = {
            "multi": "multi",
            "single": "multi",
            "exterior_only": "single_exterior_only",
            "interior_only": "single_interior_only",
            "single_exterior_only": "single_exterior_only",
            "single_interior_only": "single_interior_only",
        }
        return mapping.get(ht, "")

    @staticmethod
    def _load_sample_index(sample_records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        index: Dict[str, Dict[str, Any]] = {}
        for rec in sample_records:
            house_id = str(rec.get("house_id", "")).strip()
            if house_id:
                index[house_id] = rec
        return index

    @staticmethod
    def _clone_image(img: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "image_id": img.get("image_id"),
            "image_path": img.get("image_path"),
            "image_ori_url": img.get("image_ori_url", ""),
            "image_db_url": img.get("image_db_url", ""),
            "view_type": img.get("view_type", ""),
        }

    @staticmethod
    def _flip_view_type(view_type: str) -> str:
        vt = str(view_type).strip().lower()
        if vt == "exterior":
            return "interior"
        if vt == "interior":
            return "exterior"
        return view_type

    @staticmethod
    def _is_keep(status: Optional[str]) -> bool:
        if status is None:
            return True
        return str(status).strip().upper() == "KEEP"

    @staticmethod
    def _resolve_slot_fields(record: Dict[str, Any]) -> Dict[Tuple[str, int], Dict[str, Any]]:
        slot_pattern = re.compile(r"^(ext|int)_img_(\d+)_(db|id|exists|status)$", re.IGNORECASE)

        slot_buckets: Dict[Tuple[str, int], Dict[str, Any]] = {}
        for key, value in record.items():
            m = slot_pattern.match(str(key))
            if not m:
                continue

            prefix = m.group(1).lower()
            idx = int(m.group(2))
            suffix = m.group(3).lower()

            bucket = slot_buckets.setdefault((prefix, idx), {"prefix": prefix, "idx": idx})
            bucket[suffix] = value

        return slot_buckets

    def _build_source_lookup(
        self,
        source_images: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        src_by_db_url: Dict[str, Dict[str, Any]] = {}
        src_by_id: Dict[str, Dict[str, Any]] = {}

        for img in source_images:
            if not isinstance(img, dict):
                continue

            db_url = str(img.get("image_db_url", "")).strip()
            img_id = str(img.get("image_id", "")).strip()

            if db_url:
                src_by_db_url[db_url] = img
            if img_id:
                src_by_id[img_id] = img

        return src_by_db_url, src_by_id

    def _extract_house_type_from_ls(self, ls_record: Dict[str, Any]) -> str:
        candidates = [
            ls_record.get("house_type_valid"),
            ls_record.get("house_type"),
        ]

        data = ls_record.get("data", {})
        if isinstance(data, dict):
            candidates.append(data.get("house_type"))

        for c in candidates:
            normalized = self._normalize_labelstudio_house_type(c)
            if normalized:
                return normalized

        return ""

    @staticmethod
    def _infer_house_type_from_kept_images(images: List[Dict[str, Any]]) -> str:
        has_exterior = any(str(img.get("view_type", "")).strip().lower() == "exterior" for img in images)
        has_interior = any(str(img.get("view_type", "")).strip().lower() == "interior" for img in images)

        if has_exterior and has_interior:
            return "multi"
        if has_exterior:
            return "single_exterior_only"
        if has_interior:
            return "single_interior_only"
        return ""

    @staticmethod
    def _resolve_anomaly_mode(anomaly_flags: Set[str]) -> str:
        """
        anomaly_1: multi, ext/int slot tertukar -> swap slot prefix
        anomaly_2: single, slot/view terbalik -> flip view_type image yang keep
        """
        if "anomali_1" in anomaly_flags:
            return "swap_slots"
        if "anomali_2" in anomaly_flags:
            return "flip_kept_images"
        return "normal"

    def _collect_kept_images(
        self,
        ls_record: Dict[str, Any],
        source_record: Dict[str, Any],
        anomaly_mode: str,
    ) -> List[Dict[str, Any]]:
        source_images = source_record.get("images", [])
        if isinstance(source_images, str):
            try:
                source_images = json.loads(source_images)
            except Exception:
                source_images = []
        if not isinstance(source_images, list):
            source_images = []

        src_by_db_url, src_by_id = self._build_source_lookup(source_images)
        slot_buckets = self._resolve_slot_fields(ls_record)
        slots = sorted(slot_buckets.values(), key=lambda x: (0 if x["prefix"] == "ext" else 1, x["idx"]))

        final_images: List[Dict[str, Any]] = []

        for slot in slots:
            prefix = slot["prefix"]
            idx = slot["idx"]

            db_url = str(slot.get("db", "") or "").strip()
            img_id = str(slot.get("id", "") or "").strip()

            exists_raw = slot.get("exists")
            if exists_raw is None:
                exists = bool(db_url or img_id)
            else:
                exists = str(exists_raw).strip().lower() == "yes"

            if not exists:
                continue

            status_name = f"{prefix}_img_{idx}_status"
            status = ls_record.get(status_name)
            if not self._is_keep(status):
                continue

            matched_source = None
            if db_url and db_url in src_by_db_url:
                matched_source = src_by_db_url[db_url]
            elif img_id and img_id in src_by_id:
                matched_source = src_by_id[img_id]

            if matched_source is None:
                matched_source = {
                    "image_id": img_id or "",
                    "image_path": "",
                    "image_ori_url": "",
                    "image_db_url": db_url or "",
                    "view_type": "exterior" if prefix == "ext" else "interior",
                }

            cloned = self._clone_image(matched_source)

            if anomaly_mode == "swap_slots":
                # anomaly_1:
                # ext slot dianggap interior, int slot dianggap exterior
                cloned["view_type"] = self._flip_view_type("exterior" if prefix == "int" else "interior")
            elif anomaly_mode == "flip_kept_images":
                # anomaly_2:
                # image yang dipertahankan dibalik view_type-nya
                current_view = str(cloned.get("view_type") or ("exterior" if prefix == "ext" else "interior")).strip().lower()
                cloned["view_type"] = self._flip_view_type(current_view)
            else:
                cloned["view_type"] = "exterior" if prefix == "ext" else "interior"

            if not cloned.get("image_db_url"):
                cloned["image_db_url"] = db_url
            if not cloned.get("image_path"):
                cloned["image_path"] = matched_source.get("image_path", "")
            if not cloned.get("image_ori_url"):
                cloned["image_ori_url"] = matched_source.get("image_ori_url", "")

            final_images.append(cloned)

        return final_images

    def _reconcile_actual_label(
        self,
        ls_record: Dict[str, Any],
        final_house_type: str,
        source_record: Dict[str, Any],
    ) -> Dict[str, Any]:
        source_label = source_record.get("actual_label", {})
        if isinstance(source_label, str):
            try:
                source_label = json.loads(source_label)
            except Exception:
                source_label = {}

        if not isinstance(source_label, dict):
            source_label = {}

        def get_choice(field_name: str, ls_key: str) -> Any:
            raw = ls_record.get(ls_key)
            if raw is None:
                raw = source_label.get(field_name)
            return self._normalize_choice(raw, field_name)

        atap = get_choice("atap", "jenis_atap_terluas")
        dinding = get_choice("dinding", "jenis_dinding_terluas")
        lantai = get_choice("lantai", "jenis_lantai_terluas")

        if final_house_type == "single_exterior_only":
            lantai = "Tidak terdeteksi"
        elif final_house_type == "single_interior_only":
            atap = "Tidak terdeteksi"
            dinding = "Tidak terdeteksi"

        return {
            "atap": atap,
            "dinding": dinding,
            "lantai": lantai,
        }

    @staticmethod
    def _null_status_object() -> Dict[str, None]:
        return {"atap": None, "dinding": None, "lantai": None}

    def reconcile(self) -> Dict[str, Any]:
        sample_records = self._read_json_records(self.config.metadata_path)
        sample_index = self._load_sample_index(sample_records)

        labelstudio_items = self._read_json_records(self.config.labelstudio_output_path)
        labelstudio_index: Dict[str, Dict[str, Any]] = {}
        for item in labelstudio_items:
            if not isinstance(item, dict):
                continue
            house_id = self._extract_house_id(item)
            if house_id:
                labelstudio_index[house_id] = item

        reconciled_records: List[Dict[str, Any]] = []
        matched = 0
        skipped_no_labelstudio = 0

        for house_id, source_record in sample_index.items():
            ls_item = labelstudio_index.get(house_id)
            if ls_item is None:
                skipped_no_labelstudio += 1
                continue

            matched += 1

            anomaly_notes = self._get_anomaly_notes(ls_item)
            anomaly_flags = self._parse_anomaly_flags(anomaly_notes)
            anomaly_mode = self._resolve_anomaly_mode(anomaly_flags)

            kept_images = self._collect_kept_images(
                ls_record=ls_item,
                source_record=source_record,
                anomaly_mode=anomaly_mode,
            )

            if not kept_images:
                continue

            final_house_type = self._infer_house_type_from_kept_images(kept_images)
            final_house_type = self._normalize_meta_house_type(final_house_type)
            if not final_house_type:
                continue

            reconciled_records.append(
                {
                    "house_id": source_record.get("house_id", house_id),
                    "no_kk": source_record.get("no_kk"),
                    "house_type": final_house_type,
                    "split": source_record.get("split", None),
                    "images": kept_images,
                    "actual_label": self._reconcile_actual_label(
                        ls_record=ls_item,
                        final_house_type=final_house_type,
                        source_record=source_record,
                    ),
                    "dtsen": self._null_status_object(),
                    "status": self._null_status_object(),
                }
            )

        self.config.output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config.output_json_path, "w", encoding="utf-8") as f:
            json.dump(reconciled_records, f, ensure_ascii=False, indent=2)

        return {
            "sample_records": len(sample_records),
            "labelstudio_records": len(labelstudio_items),
            "matched_records": matched,
            "skipped_no_labelstudio": skipped_no_labelstudio,
            "final_records": len(reconciled_records),
            "output_path": str(self.config.output_json_path),
        }