import gc
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterator, Optional

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

# ── Opsional: MinIO SDK ──────────────────────────────────────────────────────
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


# ── Download single image ────────────────────────────────────────────────────

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


# ── STREAMING: download per-chunk (RECOMMENDED untuk dataset besar) ─────────

def chunk_records(records: list[dict], chunk_size: int) -> Iterator[list[dict]]:
    """Pecah list records jadi beberapa chunk kecil."""
    for i in range(0, len(records), chunk_size):
        yield records[i: i + chunk_size]


def download_images_streaming(
    records: list[dict],
    chunk_size: int,
    workers: int = DOWNLOAD_WORKERS,
) -> Iterator[tuple[dict[str, Image.Image], list[str]]]:
    """
    Download gambar secara streaming, per-chunk kecil.

    Generator ini yield (chunk_results, chunk_failed) setiap kali satu chunk
    selesai didownload. Memory hanya menahan 1 chunk pada satu waktu — chunk
    sebelumnya otomatis bisa di-garbage-collect oleh caller setelah dipakai
    (misal setelah di-encode oleh CLIP).

    Args:
        records    : seluruh image records yang mau didownload
        chunk_size : jumlah gambar per chunk (mis. 200). Sesuaikan dengan RAM.
        workers    : jumlah thread paralel PER CHUNK

    Yield:
        (results, failed) untuk setiap chunk:
            results : dict { image_id → PIL.Image }
            failed  : list image_id yang gagal didownload
    """
    total_chunks = (len(records) + chunk_size - 1) // chunk_size
    total_done   = 0
    total_failed = 0

    print(f"[image_loader] Streaming download: {len(records)} gambar, "
          f"{total_chunks} chunk (chunk_size={chunk_size}, workers={workers})")

    for chunk_idx, chunk in enumerate(chunk_records(records, chunk_size), start=1):
        results: dict[str, Image.Image] = {}
        failed: list[str] = []

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(download_image, rec): rec for rec in chunk}
            for future in as_completed(futures):
                image_id, img = future.result()
                if img is not None:
                    results[image_id] = img
                else:
                    failed.append(image_id)

        total_done   += len(results)
        total_failed += len(failed)

        print(f"[image_loader] Chunk {chunk_idx}/{total_chunks} selesai "
              f"({len(results)} ok, {len(failed)} gagal) | "
              f"Total progress: {total_done + total_failed}/{len(records)} "
              f"(berhasil: {total_done}, gagal: {total_failed})")

        yield results, failed

        # Bersihkan referensi chunk ini sebelum lanjut ke chunk berikutnya.
        # Caller (embedder) seharusnya sudah selesai pakai `results` di titik ini.
        del results, failed
        gc.collect()


# ── LEGACY: download semua sekaligus (HINDARI untuk dataset besar) ──────────

def download_images_batch(
    records: list[dict],
    workers: int = DOWNLOAD_WORKERS,
) -> dict[str, Image.Image]:
    """
    [LEGACY] Download semua gambar secara paralel, ditahan di RAM sekaligus.

    !! PERINGATAN: untuk dataset > beberapa ribu gambar, fungsi ini bisa
    menghabiskan puluhan GB RAM dan menyebabkan crash. Gunakan
    `download_images_streaming()` untuk dataset besar.

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