import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import requests
from PIL import Image

from config import (
    DOWNLOAD_TIMEOUT,
    DOWNLOAD_WORKERS,
    MAX_RETRIES,
    MINIO_ACCESS_KEY,
    MINIO_BUCKET,
    MINIO_ENDPOINT,
    MINIO_SECRET_KEY,
    MINIO_SECURE,
    USE_MINIO_SDK,
)

if USE_MINIO_SDK:
    try:
        from minio import Minio

        _minio_client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_SECURE,
        )
        print("[image_loader] MinIO SDK client aktif.")
    except ImportError:
        print("[image_loader] WARNING: minio package tidak ditemukan, fallback ke HTTP.")
        USE_MINIO_SDK = False



def _download_via_http(url: str) -> Optional[Image.Image]:
    """Download gambar via HTTP GET dengan retry."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=DOWNLOAD_TIMEOUT, stream=True)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            return img
        except Exception as e:
            if attempt == MAX_RETRIES:
                print(f"[image_loader] GAGAL download {url}: {e}")
                return None
            time.sleep(1.5 * attempt)  # backoff sederhana


def _download_via_minio(object_name: str) -> Optional[Image.Image]:
    """Download gambar via MinIO SDK."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = _minio_client.get_object(MINIO_BUCKET, object_name)
            img = Image.open(io.BytesIO(response.read())).convert("RGB")
            response.close()
            response.release_conn()
            return img
        except Exception as e:
            if attempt == MAX_RETRIES:
                print(f"[image_loader] GAGAL MinIO download {object_name}: {e}")
                return None
            time.sleep(1.5 * attempt)


def download_image(record: dict) -> tuple[str, Optional[Image.Image]]:
    """
    Download satu gambar berdasarkan record.
    Return: (image_id, PIL.Image atau None jika gagal)
    """
    image_id = record["image_id"]

    if USE_MINIO_SDK:
        # object_name diambil dari image_path (relatif terhadap bucket)
        object_name = record.get("image_path", "")
        img = _download_via_minio(object_name)
    else:
        img = _download_via_http(record["image_url"])

    return image_id, img


# ── Batch download dengan thread pool ────────────────────────────────────────

def download_images_batch(
    records: list[dict],
    workers: int = DOWNLOAD_WORKERS,
) -> dict[str, Image.Image]:
    """
    Download semua gambar secara paralel.

    Return: dict { image_id → PIL.Image }
    Gambar yang gagal didownload tidak dimasukkan ke dict.
    """
    results: dict[str, Image.Image] = {}
    failed: list[str] = []

    print(f"[image_loader] Mulai download {len(records)} gambar ({workers} workers)...")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_image, rec): rec for rec in records}
        done = 0
        for future in as_completed(futures):
            image_id, img = future.result()
            done += 1
            if img is not None:
                results[image_id] = img
            else:
                failed.append(image_id)

            if done % 500 == 0 or done == len(records):
                print(f"[image_loader] Progress: {done}/{len(records)} "
                      f"(berhasil: {len(results)}, gagal: {len(failed)})")

    if failed:
        print(f"[image_loader] {len(failed)} gambar gagal didownload: {failed[:10]}{'...' if len(failed) > 10 else ''}")

    return results