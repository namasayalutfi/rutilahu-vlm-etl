import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    CLEANED_METADATA,
    DOWNLOAD_CHUNK_SIZE,
    DUPLICATE_REPORT,
    EMBEDDING_CACHE,
    OUTPUT_DIR,
    SIMILARITY_THRESHOLD,
)
from utils.dedup_engine import QdrantDedupEngine
from utils.embedder import CLIPEmbedder, load_embedding_cache
from utils.image_loader import download_images_streaming
from utils.metadata_handler import (
    extract_image_records,
    load_metadata,
    save_json,
)


# ── CLI args ──────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="CLIP + Qdrant Duplicate Detection")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download gambar, langsung pakai embedding cache jika ada.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Hanya deteksi duplikat, jangan tulis cleaned metadata.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=SIMILARITY_THRESHOLD,
        help=f"Cosine similarity threshold (default: {SIMILARITY_THRESHOLD})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Jumlah nearest neighbor yang di-query per gambar (default: 5)",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Jangan hapus Qdrant collection setelah selesai (berguna untuk inspeksi).",
    )
    parser.add_argument(
        "--recreate-collection",
        action="store_true",
        default=True,
        help="Hapus dan buat ulang Qdrant collection (default: True)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DOWNLOAD_CHUNK_SIZE,
        help=f"Jumlah gambar per chunk untuk streaming download+encode "
             f"(default: {DOWNLOAD_CHUNK_SIZE}). Turunkan jika RAM terbatas.",
    )
    return parser.parse_args()


# ── Apply dedup ke metadata ───────────────────────────────────────────────────

def apply_deduplication(
    metadata: list[dict],
    images_to_remove: set[str],
) -> tuple[list[dict], dict]:
    """
    Hapus gambar duplikat dari metadata, dan hapus house yang tidak punya gambar lagi.

    Return:
        cleaned_metadata : metadata bersih
        stats            : ringkasan statistik
    """
    original_house_count = len(metadata)
    original_image_count = sum(len(h.get("images", [])) for h in metadata)

    cleaned_metadata = []
    removed_houses   = []

    for house in metadata:
        original_images = house.get("images", [])
        kept_images = [
            img for img in original_images
            if img.get("image_id") not in images_to_remove
        ]

        if not kept_images:
            # Seluruh gambar di house ini dihapus sebagai duplikat
            removed_houses.append(house["house_id"])
            continue

        # Buat salinan house dengan gambar yang sudah difilter
        cleaned_house = {**house, "images": kept_images}

        # Update house_type jika jumlah gambar berubah
        if len(kept_images) == 1 and house.get("house_type") == "multi":
            remaining_view = kept_images[0].get("view_type", "")
            if remaining_view == "exterior":
                cleaned_house["house_type"] = "single_exterior_only"
            elif remaining_view == "interior":
                cleaned_house["house_type"] = "single_interior_only"

        cleaned_metadata.append(cleaned_house)

    final_image_count = sum(len(h.get("images", [])) for h in cleaned_metadata)

    stats = {
        "original_houses"        : original_house_count,
        "original_images"        : original_image_count,
        "images_removed"         : len(images_to_remove),
        "houses_removed"         : len(removed_houses),
        "removed_house_ids"      : removed_houses,
        "final_houses"           : len(cleaned_metadata),
        "final_images"           : final_image_count,
        "dedup_rate_images_pct"  : round(len(images_to_remove) / original_image_count * 100, 2)
                                   if original_image_count else 0,
    }
    return cleaned_metadata, stats


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print(" CLIP + Qdrant Duplicate Detection Pipeline")
    print("=" * 60)
    print(f" Threshold    : {args.threshold}")
    print(f" Top-K        : {args.top_k}")
    print(f" Dry-run      : {args.dry_run}")
    print(f" Skip download: {args.skip_download}")
    print("=" * 60)

    # ── STEP 1: Load metadata ─────────────────────────────────────────────────
    print("\n[STEP 1] Load metadata...")
    metadata = load_metadata()
    records  = extract_image_records(metadata)
    print(f"  → {len(metadata)} houses, {len(records)} image records")

    # ── STEP 2 & 3: Streaming download + CLIP encoding ────────────────────────
    # Kedua step ini digabung jadi satu loop: download 1 chunk kecil → encode
    # chunk itu → flush ke cache di disk → buang dari RAM → lanjut chunk
    # berikutnya. Ini mencegah RAM penuh saat dataset besar (20rb+ gambar).
    print("\n[STEP 2+3] Streaming download + CLIP encoding...")

    embedder = CLIPEmbedder()

    if args.skip_download:
        print("  → --skip-download aktif, langsung pakai embedding cache.")
        embeddings = load_embedding_cache(EMBEDDING_CACHE)
    else:
        # RESUME LOGIC: skip image_id yang sudah ada di cache (misal dari
        # run sebelumnya yang crash di tengah jalan), supaya tidak download
        # ulang gambar yang sudah berhasil di-encode.
        existing_cache = load_embedding_cache(EMBEDDING_CACHE)
        records_to_process = [
            r for r in records if r["image_id"] not in existing_cache
        ]
        skipped = len(records) - len(records_to_process)
        if skipped > 0:
            print(f"  → {skipped} gambar sudah ada di cache (dari run sebelumnya), "
                  f"akan diskip. Sisa: {len(records_to_process)} gambar.")

        if records_to_process:
            chunk_iterator = download_images_streaming(
                records=records_to_process,
                chunk_size=args.chunk_size,
            )
            embeddings = embedder.encode_streaming(
                chunk_iterator=chunk_iterator,
                cache_path=EMBEDDING_CACHE,
                flush_every_chunk=True,  # crash-safe: simpan progress tiap chunk
            )
        else:
            print("  → Semua gambar sudah ada di cache, tidak ada yang perlu didownload.")
            embeddings = existing_cache

    if not embeddings:
        print("[STEP 2+3] ERROR: Tidak ada embedding yang dihasilkan. Pipeline berhenti.")
        return

    print(f"  → {len(embeddings)} embedding siap")

    # ── STEP 4: Setup Qdrant ──────────────────────────────────────────────────
    print("\n[STEP 4] Setup Qdrant collection...")
    engine = QdrantDedupEngine(threshold=args.threshold)
    engine.setup_collection(recreate=args.recreate_collection)

    # ── STEP 5: Upsert ke Qdrant ──────────────────────────────────────────────
    print("\n[STEP 5] Upsert embeddings ke Qdrant...")
    engine.upsert_embeddings(embeddings=embeddings, records=records)

    # ── STEP 6: Cari duplikat ─────────────────────────────────────────────────
    print("\n[STEP 6] Mencari pasangan duplikat...")
    duplicate_pairs = engine.find_duplicate_pairs(top_k=args.top_k)

    # ── STEP 7: Resolve cluster & tentukan winner ─────────────────────────────
    print("\n[STEP 7] Resolving duplicate clusters...")
    images_to_remove, duplicate_report = engine.resolve_duplicates(
        duplicate_pairs=duplicate_pairs,
        records=records,
    )

    # ── STEP 8: Apply ke metadata & simpan output ─────────────────────────────
    print("\n[STEP 8] Menyimpan hasil...")

    # Simpan duplicate report selalu (bukan dry-run)
    full_report = {
        "summary": {
            "total_duplicate_pairs"   : len(duplicate_pairs),
            "total_duplicate_clusters": len(duplicate_report),
            "total_images_to_remove"  : len(images_to_remove),
            "threshold_used"          : args.threshold,
        },
        "clusters": duplicate_report,
    }
    save_json(full_report, DUPLICATE_REPORT)
    print(f"  → Duplicate report: {DUPLICATE_REPORT}")

    if not args.dry_run:
        cleaned_metadata, stats = apply_deduplication(metadata, images_to_remove)
        save_json(cleaned_metadata, CLEANED_METADATA)
        print(f"  → Cleaned metadata: {CLEANED_METADATA}")

        print("\n" + "=" * 60)
        print(" RINGKASAN HASIL DEDUPLICATION")
        print("=" * 60)
        print(f"  Houses (sebelum)  : {stats['original_houses']}")
        print(f"  Images (sebelum)  : {stats['original_images']}")
        print(f"  Images dihapus    : {stats['images_removed']} ({stats['dedup_rate_images_pct']}%)")
        print(f"  Houses dihapus    : {stats['houses_removed']}")
        print(f"  Houses (sesudah)  : {stats['final_houses']}")
        print(f"  Images (sesudah)  : {stats['final_images']}")
        print("=" * 60)
    else:
        print(f"\n[DRY-RUN] {len(images_to_remove)} gambar akan dihapus jika bukan dry-run.")
        print(f"  Cleaned metadata TIDAK disimpan (--dry-run aktif).")

    # ── STEP 9: Cleanup Qdrant ────────────────────────────────────────────────
    print("\n[STEP 9] Cleanup...")
    engine.cleanup(delete_collection=not args.no_cleanup)

    print("\n✓ Pipeline selesai.")


if __name__ == "__main__":
    main()