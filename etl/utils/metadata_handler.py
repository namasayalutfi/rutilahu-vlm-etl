import json
import os
from typing import Any

from config import METADATA_PATH, OUTPUT_DIR


def load_metadata(path: str = METADATA_PATH) -> list[dict]:
    """Load metadata JSON utama."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Toleran terhadap dua kemungkinan struktur:
    #   1. Root langsung list  → [{"house_id": ...}, ...]
    #   2. Root adalah dict    → {"data": [...]}
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "houses", "records"):
            if key in data and isinstance(data[key], list):
                return data[key]
    raise ValueError(f"Format metadata tidak dikenali di {path}")


def save_json(obj: Any, path: str, indent: int = 2) -> None:
    """Simpan objek ke file JSON, buat direktori jika belum ada."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
    print(f"[metadata_handler] Tersimpan → {path}")


def extract_image_records(metadata: list[dict]) -> list[dict]:
    """
    Flatten metadata menjadi list image records yang siap di-embed.

    Setiap record berisi:
        image_id   : str   — ID unik gambar
        image_url  : str   — URL untuk download (image_db_url atau image_ori_url)
        view_type  : str   — "exterior" / "interior"
        house_id   : str   — parent house
        house_type : str   — "multi" / "single_interior_only" / "single_exterior_only"
        no_kk      : str | None
    """
    records = []
    seen_image_ids = set()

    for house in metadata:
        house_id   = house.get("house_id", "")
        house_type = house.get("house_type", "")
        no_kk      = house.get("no_kk")

        for img in house.get("images", []):
            image_id = img.get("image_id", "")

            # Skip jika image_id duplikat di level metadata (data kotor)
            if image_id in seen_image_ids:
                print(f"[metadata_handler] WARNING: image_id duplikat di metadata → {image_id}, skip.")
                continue
            seen_image_ids.add(image_id)

            # Pilih URL: utamakan image_db_url, fallback ke image_ori_url
            url = img.get("image_db_url") or img.get("image_ori_url")
            if not url:
                print(f"[metadata_handler] WARNING: image_id {image_id} tidak punya URL, skip.")
                continue

            records.append({
                "image_id"  : image_id,
                "image_url" : url,
                "view_type" : img.get("view_type", "unknown"),
                "house_id"  : house_id,
                "house_type": house_type,
                "no_kk"     : no_kk,
            })

    print(f"[metadata_handler] Total image records diekstrak: {len(records)}")
    return records