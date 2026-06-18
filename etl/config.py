import os

# --- Path ---
BASE_DIR         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METADATA_PATH    = os.path.join(BASE_DIR, "metadata", "mkn2_final_metadata.json")
OUTPUT_DIR       = os.path.join(BASE_DIR, "output")
DUPLICATE_REPORT = os.path.join(OUTPUT_DIR, "duplicate_report.json")
CLEANED_METADATA = os.path.join(OUTPUT_DIR, "mkn2_cleaned_metadata.json")
EMBEDDING_CACHE  = os.path.join(OUTPUT_DIR, "embeddings_cache.npz")  # cache embedding agar tidak re-encode jika crash

# --- MinIO / Image Source ---
MINIO_ENDPOINT   = "76.13.194.250:9010"
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_BUCKET     = "mkn2"
MINIO_SECURE     = False  # True jika pakai HTTPS

# Alternatif: gunakan image_db_url langsung via HTTP (tanpa MinIO SDK)
# Set USE_MINIO_SDK = False jika image_db_url sudah accessible via HTTP
USE_MINIO_SDK    = False

# --- CLIP Model ---
CLIP_MODEL_NAME  = "ViT-L/14"   # opsi: "ViT-B/32", "ViT-B/16", "ViT-L/14"
CLIP_DEVICE      = "cuda"        
CLIP_BATCH_SIZE  = 256           # batch size untuk encoding; turunkan jika OOM

# --- Qdrant ---
QDRANT_HOST       = "localhost"
QDRANT_PORT       = 6333
QDRANT_COLLECTION = "house_images_dedup"
# Dimensi embedding sesuai model:
#   ViT-B/32  → 512
#   ViT-B/16  → 512
#   ViT-L/14  → 768
QDRANT_VECTOR_DIM = 768

# --- Duplicate Detection ---
# Cosine similarity threshold (0.0 - 1.0)
# ≥ 0.98 → hampir pasti duplikat (exact / re-compressed)
# ≥ 0.95 → near-duplicate (beda resolusi, crop ringan)
# ≥ 0.90 → sangat mirip (bisa false positive untuk foto rumah sejenis)
SIMILARITY_THRESHOLD = 0.97

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
DOWNLOAD_TIMEOUT  = 30    # detik per gambar
DOWNLOAD_WORKERS  = 16    # thread paralel untuk download gambar
MAX_RETRIES       = 3     # retry jika download gagal