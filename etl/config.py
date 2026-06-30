# =============================================================================
# config.py — Konfigurasi pipeline duplicate detection
# Disesuaikan untuk: NVIDIA RTX 3050 Laptop (4GB VRAM), CUDA 13.0, Windows
# =============================================================================

import os

# --- Path ---
# Windows-safe: os.path.join otomatis pakai backslash di Windows
BASE_DIR         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METADATA_PATH    = os.path.join(BASE_DIR, "metadata", "mkn2_final_metadata.json")
OUTPUT_DIR       = os.path.join(BASE_DIR, "output")
DUPLICATE_REPORT = os.path.join(OUTPUT_DIR, "duplicate_report.json")
CLEANED_METADATA = os.path.join(OUTPUT_DIR, "mkn2_cleaned_metadata.json")
EMBEDDING_CACHE  = os.path.join(OUTPUT_DIR, "embeddings_cache.npz")  # cache embedding agar tidak re-encode jika crash

# --- MinIO / Image Source ---
# Ganti sesuai kredensial MinIO Anda
MINIO_ENDPOINT   = "76.13.194.250:9010"
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_BUCKET     = "mkn2"
MINIO_SECURE     = False  # True jika pakai HTTPS

# Alternatif: gunakan image_db_url langsung via HTTP (tanpa MinIO SDK)
# Set USE_MINIO_SDK = False jika image_db_url sudah accessible via HTTP
USE_MINIO_SDK    = False

# --- CLIP Model ---
# !! PENTING: ViT-L/14 butuh ~4.5GB VRAM → OOM di RTX 3050 Laptop 4GB
#
# Pilihan untuk 4GB VRAM:
#   "ViT-B/32" → ~1.5GB VRAM, dim=512, paling aman, encoding paling cepat
#   "ViT-B/16" → ~2.5GB VRAM, dim=512, akurasi lebih baik, masih aman
#
# Rekomendasi: mulai dengan ViT-B/16. Jika OOM, turun ke ViT-B/32.
CLIP_MODEL_NAME  = "ViT-B/16"

# "cuda" untuk pakai RTX 3050, "cpu" sebagai fallback jika ada masalah CUDA
CLIP_DEVICE      = "cuda"

# Batch size untuk VRAM 4GB:
#   ViT-B/16 → max aman ~24, pakai 16 untuk ada headroom
#   ViT-B/32 → max aman ~48, pakai 32 untuk ada headroom
# Turunkan ke 8 jika masih OOM (VRAM sudah terpakai browser/OS)
CLIP_BATCH_SIZE  = 16

# --- Qdrant ---
QDRANT_HOST       = "localhost"
QDRANT_PORT       = 6333
QDRANT_COLLECTION = "house_images_dedup"
# Dimensi embedding sesuai model:
#   ViT-B/32  → 512
#   ViT-B/16  → 512  ← sesuaikan jika ganti model
#   ViT-L/14  → 768
QDRANT_VECTOR_DIM = 512

# --- Duplicate Detection ---
# Cosine similarity threshold (0.0 - 1.0)
# ≥ 0.98 → hampir pasti duplikat (exact / re-compressed)
# ≥ 0.95 → near-duplicate (beda resolusi, crop ringan)
# ≥ 0.90 → sangat mirip (bisa false positive untuk foto rumah sejenis)
SIMILARITY_THRESHOLD = 0.95

# --- Prioritas retain saat duplikat ditemukan ---
# Index lebih kecil = prioritas lebih tinggi (dipertahankan)
HOUSE_TYPE_PRIORITY = {
    "multi": 0,
    "single_interior_only": 1,
    "single_exterior_only": 2,
}
# Data dengan no_kk (data DTSEN asli) lebih diprioritaskan dari crawled
# Ini dihandle otomatis: no_kk=null → crawled → prioritas lebih rendah

# --- Download ---
DOWNLOAD_TIMEOUT  = 30   # detik per gambar
# Windows lebih konservatif dengan thread — 8 sudah cukup agresif
# Turunkan ke 4 jika sering timeout atau koneksi tidak stabil
DOWNLOAD_WORKERS  = 8
MAX_RETRIES       = 3    # retry jika download gagal

# --- Streaming Pipeline (PENTING untuk RAM terbatas) ---
# Jumlah gambar yang ditahan di RAM sekaligus sebelum di-encode lalu dibuang.
# Dataset 22rb gambar JANGAN didownload semua dulu baru encode — itu yang
# menyebabkan RAM penuh dan laptop crash sebelumnya.
#
# 200 gambar per chunk biasanya aman untuk laptop 8-16GB RAM.
# Turunkan ke 100 jika laptop masih terasa berat / RAM kecil (8GB).
DOWNLOAD_CHUNK_SIZE = 1103