import os
from typing import Iterator, Optional

import clip
import numpy as np
import torch
from PIL import Image

from config import (
    CLIP_BATCH_SIZE,
    CLIP_DEVICE,
    CLIP_MODEL_NAME,
    EMBEDDING_CACHE,
)


def load_embedding_cache(cache_path: str = EMBEDDING_CACHE) -> dict[str, np.ndarray]:
    """Load embedding cache dari disk. Return dict kosong jika belum ada."""
    if cache_path and os.path.exists(cache_path):
        cache = np.load(cache_path, allow_pickle=True)
        return {k: cache[k] for k in cache.files}
    return {}


def save_embedding_cache(
    embeddings: dict[str, np.ndarray],
    cache_path: str = EMBEDDING_CACHE,
) -> None:
    """Overwrite cache di disk dengan dict embedding lengkap (lama + baru)."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, **embeddings)


class CLIPEmbedder:
    """
    Wrapper CLIP untuk batch encoding gambar ke normalized float32 vectors.

    Fitur:
    - GPU batching dengan batch size yang bisa dikonfigurasi
    - Auto-normalize embeddings (siap untuk cosine similarity)
    - Cache embedding ke disk (.npz) agar pipeline bisa resume jika crash
    - Mode streaming: encode per-chunk dan flush ke disk setiap chunk,
      sehingga gambar mentah (PIL.Image) tidak perlu menumpuk di RAM
    """

    def __init__(
        self,
        model_name: str = CLIP_MODEL_NAME,
        device: str = CLIP_DEVICE,
        batch_size: int = CLIP_BATCH_SIZE,
    ):
        self.device     = device if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size

        print(f"[embedder] Loading CLIP model '{model_name}' ke {self.device}...")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

        # Dimensi embedding berdasarkan model
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224).to(self.device)
            self.embedding_dim = self.model.encode_image(dummy).shape[-1]

        print(f"[embedder] Model siap. Embedding dim: {self.embedding_dim}, device: {self.device}")

    def _encode_dict(self, image_dict: dict[str, Image.Image]) -> dict[str, np.ndarray]:
        """Encode satu dict gambar (di GPU, dalam mini-batch internal) ke embedding."""
        new_embeddings: dict[str, np.ndarray] = {}
        if not image_dict:
            return new_embeddings

        image_ids = list(image_dict.keys())
        images    = list(image_dict.values())
        total     = len(image_ids)

        for batch_start in range(0, total, self.batch_size):
            batch_ids  = image_ids[batch_start: batch_start + self.batch_size]
            batch_imgs = images[batch_start: batch_start + self.batch_size]

            tensors = torch.stack(
                [self.preprocess(img) for img in batch_imgs]
            ).to(self.device)

            with torch.no_grad():
                features = self.model.encode_image(tensors)
                features = features / features.norm(dim=-1, keepdim=True)
                features_np = features.cpu().float().numpy()

            for img_id, vec in zip(batch_ids, features_np):
                new_embeddings[img_id] = vec

            # Bebaskan VRAM secepatnya — penting untuk GPU 4GB
            del tensors, features
            if self.device == "cuda":
                torch.cuda.empty_cache()

        return new_embeddings

    def encode(
        self,
        image_dict: dict[str, Image.Image],
        cache_path: Optional[str] = EMBEDDING_CACHE,
    ) -> dict[str, np.ndarray]:
        """
        [MODE NON-STREAMING] Encode semua gambar yang SUDAH ada di RAM sekaligus.

        Cocok untuk dataset kecil. Untuk dataset besar (ribuan gambar),
        gunakan `encode_streaming()` agar gambar tidak perlu ditahan
        semua di RAM sebelum encoding dimulai.

        Args:
            image_dict : { image_id → PIL.Image }
            cache_path : Path file .npz untuk simpan/load cache.

        Return:
            { image_id → np.ndarray shape (embedding_dim,) float32 }
        """
        cached_embeddings = load_embedding_cache(cache_path) if cache_path else {}

        to_encode = {
            img_id: img
            for img_id, img in image_dict.items()
            if img_id not in cached_embeddings
        }
        print(f"[embedder] {len(to_encode)} gambar perlu di-encode "
              f"({len(cached_embeddings)} sudah dari cache).")

        new_embeddings = self._encode_dict(to_encode)

        if new_embeddings and cache_path:
            merged = {**cached_embeddings, **new_embeddings}
            save_embedding_cache(merged, cache_path)
            print(f"[embedder] Cache diperbarui → {cache_path} ({len(merged)} total)")

        all_embeddings = {**cached_embeddings, **new_embeddings}
        print(f"[embedder] Total embedding siap: {len(all_embeddings)}")
        return all_embeddings

    def encode_streaming(
        self,
        chunk_iterator: Iterator[tuple[dict[str, Image.Image], list[str]]],
        cache_path: str = EMBEDDING_CACHE,
        flush_every_chunk: bool = True,
    ) -> dict[str, np.ndarray]:
        """
        [MODE STREAMING — RECOMMENDED untuk dataset besar / RAM terbatas]

        Terima generator chunk dari `image_loader.download_images_streaming()`,
        encode tiap chunk begitu chunk itu selesai didownload, lalu langsung
        flush hasilnya ke disk. Gambar mentah (PIL.Image) di tiap chunk otomatis
        dibuang dari RAM setelah chunk itu selesai diproses.

        Jika proses crash di tengah jalan, cukup jalankan ulang — chunk yang
        sudah di-flush ke cache tidak akan di-download/encode ulang karena
        `image_id` yang sudah ada di cache akan diskip oleh caller
        (lihat `clip_dedup.py` yang memfilter records sebelum streaming).

        Args:
            chunk_iterator    : generator yang yield (image_dict_chunk, failed_ids)
            cache_path        : path .npz untuk cache, WAJIB diisi (bukan None)
                                 karena inti dari mode ini adalah flush per-chunk
            flush_every_chunk : True = save ke disk setelah SETIAP chunk (paling aman,
                                 sedikit overhead I/O). False = save setelah semua
                                 chunk selesai (lebih cepat tapi tidak crash-safe).

        Return:
            { image_id → np.ndarray } — seluruh embedding (cache lama + baru)
        """
        embeddings = load_embedding_cache(cache_path)
        print(f"[embedder] Streaming encode mulai. {len(embeddings)} embedding "
              f"sudah ada di cache sebelumnya.")

        all_failed: list[str] = []
        chunk_num = 0

        for image_dict_chunk, failed_ids in chunk_iterator:
            chunk_num += 1
            all_failed.extend(failed_ids)

            if image_dict_chunk:
                new_embeddings = self._encode_dict(image_dict_chunk)
                embeddings.update(new_embeddings)
                print(f"[embedder] Chunk {chunk_num} di-encode: "
                      f"{len(new_embeddings)} embedding baru "
                      f"(total cache: {len(embeddings)})")

                if flush_every_chunk:
                    save_embedding_cache(embeddings, cache_path)

            # `image_dict_chunk` keluar dari scope di iterasi berikutnya →
            # PIL.Image di chunk ini bisa di-garbage-collect, RAM tidak menumpuk.

        # Flush terakhir untuk memastikan semua tersimpan
        if not flush_every_chunk:
            save_embedding_cache(embeddings, cache_path)

        print(f"[embedder] Streaming encode selesai. Total embedding: {len(embeddings)}, "
              f"total gagal download: {len(all_failed)}")
        if all_failed:
            print(f"[embedder] Daftar gagal (max 10 ditampilkan): "
                  f"{all_failed[:10]}{'...' if len(all_failed) > 10 else ''}")

        return embeddings