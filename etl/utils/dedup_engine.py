from typing import Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from tqdm import tqdm

from config import (
    HOUSE_TYPE_PRIORITY,
    QDRANT_COLLECTION,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_VECTOR_DIM,
    SIMILARITY_THRESHOLD,
)


# ── Union-Find untuk clustering duplikat ────────────────────────────────────

class UnionFind:
    """
    Struktur data untuk mengelompokkan duplikat ke dalam cluster.
    Setiap elemen yang terhubung (A duplikat B, B duplikat C) → satu cluster.
    """

    def __init__(self):
        self.parent: dict = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # path compression
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[ry] = rx

    def get_clusters(self) -> dict[str, list[str]]:
        """Return dict { root → [member, ...] } untuk cluster dengan > 1 anggota."""
        groups: dict[str, list] = {}
        for x in self.parent:
            root = self.find(x)
            groups.setdefault(root, []).append(x)
        return {root: members for root, members in groups.items() if len(members) > 1}


# ── Prioritas retain ─────────────────────────────────────────────────────────

def _retain_score(payload: dict) -> tuple:
    """
    Hitung skor prioritas untuk menentukan gambar mana yang dipertahankan.
    Tuple lebih kecil = prioritas lebih tinggi.

    Urutan prioritas:
    1. house_type (multi > single_interior_only > single_exterior_only)
    2. no_kk tidak null (data DTSEN asli > data crawled)
    3. image_id (tie-breaker deterministik)
    """
    house_type_score = HOUSE_TYPE_PRIORITY.get(payload.get("house_type", ""), 99)
    is_crawled       = 1 if payload.get("no_kk") is None else 0
    return (house_type_score, is_crawled, payload.get("image_id", ""))


# ── Qdrant Dedup Engine ──────────────────────────────────────────────────────

class QdrantDedupEngine:

    def __init__(
        self,
        host: str      = QDRANT_HOST,
        port: int       = QDRANT_PORT,
        collection: str = QDRANT_COLLECTION,
        vector_dim: int = QDRANT_VECTOR_DIM,
        threshold: float = SIMILARITY_THRESHOLD,
    ):
        print(f"[dedup_engine] Koneksi ke Qdrant {host}:{port}...")
        self.client     = QdrantClient(host=host, port=port)
        self.collection = collection
        self.vector_dim = vector_dim
        self.threshold  = threshold

        # Mapping image_id → int (Qdrant butuh integer atau UUID sebagai point id)
        self._id_to_int:  dict[str, int] = {}
        self._int_to_id:  dict[int, str] = {}

    # ── Setup collection ─────────────────────────────────────────────────────

    def setup_collection(self, recreate: bool = True) -> None:
        """
        Buat Qdrant collection untuk menyimpan embedding gambar.

        Args:
            recreate: Jika True, hapus collection lama dulu.
                      Set False jika ingin resume upsert yang terputus.
        """
        existing = [c.name for c in self.client.get_collections().collections]

        if recreate and self.collection in existing:
            print(f"[dedup_engine] Menghapus collection lama '{self.collection}'...")
            self.client.delete_collection(self.collection)

        if self.collection not in existing or recreate:
            print(f"[dedup_engine] Membuat collection '{self.collection}' "
                  f"(dim={self.vector_dim}, distance=Cosine)...")
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.vector_dim,
                    distance=Distance.COSINE,
                    # on_disk=True  # aktifkan jika RAM terbatas dan ingin simpan vector ke disk
                ),
            )
            print(f"[dedup_engine] Collection '{self.collection}' siap.")
        else:
            print(f"[dedup_engine] Collection '{self.collection}' sudah ada, pakai yang existing.")

    # ── Upsert embeddings ────────────────────────────────────────────────────

    def upsert_embeddings(
        self,
        embeddings: dict[str, np.ndarray],
        records: list[dict],
        batch_size: int = 512,
    ) -> None:
        """
        Masukkan semua embedding + metadata ke Qdrant collection.

        Args:
            embeddings : { image_id → np.ndarray }
            records    : list image records dari metadata_handler.extract_image_records()
            batch_size : jumlah point per upsert call
        """
        # Buat lookup record berdasarkan image_id
        record_lookup = {r["image_id"]: r for r in records}

        # Assign integer ID (Qdrant pakai int/UUID, bukan string)
        for idx, image_id in enumerate(embeddings.keys()):
            self._id_to_int[image_id] = idx
            self._int_to_id[idx]      = image_id

        print(f"[dedup_engine] Upsert {len(embeddings)} points ke Qdrant "
              f"(batch_size={batch_size})...")

        items      = list(embeddings.items())
        total_done = 0

        for batch_start in range(0, len(items), batch_size):
            batch = items[batch_start: batch_start + batch_size]
            points = []

            for image_id, vector in batch:
                rec = record_lookup.get(image_id, {})
                points.append(
                    PointStruct(
                        id=self._id_to_int[image_id],
                        vector=vector.tolist(),
                        payload={
                            "image_id"  : image_id,
                            "house_id"  : rec.get("house_id", ""),
                            "house_type": rec.get("house_type", ""),
                            "view_type" : rec.get("view_type", ""),
                            "no_kk"     : rec.get("no_kk"),
                            "image_url" : rec.get("image_url", ""),
                        },
                    )
                )

            self.client.upsert(collection_name=self.collection, points=points)
            total_done += len(batch)
            print(f"[dedup_engine] Upsert progress: {total_done}/{len(items)}")

        print(f"[dedup_engine] Semua {len(embeddings)} points berhasil di-upsert.")

    # ── Search duplikat ──────────────────────────────────────────────────────

    def find_duplicate_pairs(
        self,
        top_k: int = 5,
    ) -> list[tuple[str, str, float]]:
        """
        Query setiap point ke Qdrant untuk mencari nearest neighbors.
        Return list of (image_id_A, image_id_B, similarity_score).

        Args:
            top_k: jumlah kandidat nearest neighbor per query.
                   Nilai kecil (3-5) cukup untuk near-duplicate detection.
                   Naikkan jika satu gambar bisa punya banyak duplikat.
        """
        total_points = self.client.count(self.collection).count
        print(f"[dedup_engine] Mencari duplikat dari {total_points} points "
              f"(threshold={self.threshold}, top_k={top_k})...")

        duplicate_pairs: list[tuple[str, str, float]] = []
        seen_pairs: set[frozenset] = set()

        # Scroll semua point untuk ambil ID-nya
        # (kita query satu per satu; untuk 20K gambar ini tetap cepat karena
        #  Qdrant melakukan ANN search bukan brute-force)
        all_ids = list(self._int_to_id.keys())

        for int_id in tqdm(all_ids, desc="[dedup_engine] Searching"):
            image_id_a = self._int_to_id[int_id]

            results = self.client.query_points(
                collection_name=self.collection,
                query=int_id,                # query by existing point ID
                limit=top_k + 1,            # +1 karena hasil selalu include dirinya sendiri
                with_payload=True,
                score_threshold=self.threshold,
            ).points

            for hit in results:
                image_id_b = hit.payload.get("image_id", "")
                score      = hit.score

                # Skip dirinya sendiri
                if image_id_b == image_id_a:
                    continue

                pair = frozenset([image_id_a, image_id_b])
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                duplicate_pairs.append((image_id_a, image_id_b, float(score)))

        print(f"[dedup_engine] Ditemukan {len(duplicate_pairs)} pasang duplikat kandidat.")
        return duplicate_pairs

    # ── Clustering & keputusan retain ────────────────────────────────────────

    def resolve_duplicates(
        self,
        duplicate_pairs: list[tuple[str, str, float]],
        records: list[dict],
    ) -> tuple[set[str], list[dict]]:
        """
        Cluster semua pasangan duplikat, lalu pilih 1 winner per cluster.

        Return:
            images_to_remove : set image_id yang harus dihapus dari metadata
            duplicate_report : list detail cluster untuk logging/audit
        """
        record_lookup = {r["image_id"]: r for r in records}
        uf = UnionFind()

        for id_a, id_b, _ in duplicate_pairs:
            uf.union(id_a, id_b)

        clusters = uf.get_clusters()
        print(f"[dedup_engine] {len(clusters)} cluster duplikat ditemukan.")

        images_to_remove: set[str] = set()
        duplicate_report: list[dict] = []

        for root, members in clusters.items():
            # Ambil payload dari Qdrant untuk masing-masing member
            payloads = {}
            for img_id in members:
                int_id = self._id_to_int.get(img_id)
                if int_id is None:
                    continue
                results = self.client.retrieve(
                    collection_name=self.collection,
                    ids=[int_id],
                    with_payload=True,
                )
                if results:
                    payloads[img_id] = results[0].payload

            if not payloads:
                continue

            # Pilih winner berdasarkan prioritas
            winner = min(payloads.keys(), key=lambda x: _retain_score(payloads[x]))
            losers = [img_id for img_id in members if img_id != winner]

            images_to_remove.update(losers)

            # Pasangan similarity untuk report
            pair_scores = [
                {"image_id_a": a, "image_id_b": b, "similarity": round(s, 6)}
                for a, b, s in duplicate_pairs
                if a in members and b in members
            ]

            duplicate_report.append({
                "cluster_size": len(members),
                "winner"      : winner,
                "winner_meta" : payloads.get(winner, {}),
                "losers"      : [
                    {"image_id": lid, "meta": payloads.get(lid, {})}
                    for lid in losers
                ],
                "pairs"       : pair_scores,
            })

        print(f"[dedup_engine] {len(images_to_remove)} gambar akan dihapus dari metadata.")
        return images_to_remove, duplicate_report

    # ── Cleanup ──────────────────────────────────────────────────────────────

    def cleanup(self, delete_collection: bool = True) -> None:
        """Hapus collection setelah selesai (opsional)."""
        if delete_collection:
            self.client.delete_collection(self.collection)
            print(f"[dedup_engine] Collection '{self.collection}' dihapus.")
        self.client.close()
        print("[dedup_engine] Koneksi Qdrant ditutup.")