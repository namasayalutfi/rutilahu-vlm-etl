import os
from typing import Optional

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


class CLIPEmbedder:
    """
    Wrapper CLIP untuk batch encoding gambar ke normalized float32 vectors.

    Fitur:
    - GPU batching dengan batch size yang bisa dikonfigurasi
    - Auto-normalize embeddings (siap untuk cosine similarity)
    - Cache embedding ke disk (.npz) agar pipeline bisa resume jika crash
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

    def encode(
        self,
        image_dict: dict[str, Image.Image],
        cache_path: Optional[str] = EMBEDDING_CACHE,
    ) -> dict[str, np.ndarray]:
        """
        Encode semua gambar ke embedding vector.

        Args:
            image_dict : { image_id → PIL.Image }
            cache_path : Path file .npz untuk simpan/load cache.
                         Jika None, cache dinonaktifkan.

        Return:
            { image_id → np.ndarray shape (embedding_dim,) float32 }
        """
        # ── Load cache jika ada ───────────────────────────────────────────────
        cached_embeddings: dict[str, np.ndarray] = {}
        if cache_path and os.path.exists(cache_path):
            print(f"[embedder] Ditemukan cache embedding → {cache_path}")
            cache = np.load(cache_path, allow_pickle=True)
            cached_embeddings = {k: cache[k] for k in cache.files}
            print(f"[embedder] {len(cached_embeddings)} embedding di-load dari cache.")

        # ── Tentukan gambar yang belum di-encode ─────────────────────────────
        to_encode = {
            img_id: img
            for img_id, img in image_dict.items()
            if img_id not in cached_embeddings
        }
        print(f"[embedder] {len(to_encode)} gambar perlu di-encode "
              f"({len(cached_embeddings)} sudah dari cache).")

        new_embeddings: dict[str, np.ndarray] = {}

        if to_encode:
            image_ids = list(to_encode.keys())
            images    = list(to_encode.values())
            total     = len(image_ids)

            for batch_start in range(0, total, self.batch_size):
                batch_ids = image_ids[batch_start: batch_start + self.batch_size]
                batch_imgs = images[batch_start: batch_start + self.batch_size]

                # Preprocess & stack ke tensor
                tensors = torch.stack(
                    [self.preprocess(img) for img in batch_imgs]
                ).to(self.device)

                with torch.no_grad():
                    features = self.model.encode_image(tensors)
                    # L2-normalize agar dot product == cosine similarity
                    features = features / features.norm(dim=-1, keepdim=True)
                    features_np = features.cpu().float().numpy()

                for img_id, vec in zip(batch_ids, features_np):
                    new_embeddings[img_id] = vec

                done = min(batch_start + self.batch_size, total)
                print(f"[embedder] Encoded: {done}/{total}")

            # ── Simpan cache (gabung lama + baru) ────────────────────────────
            if cache_path:
                merged = {**cached_embeddings, **new_embeddings}
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                np.savez_compressed(cache_path, **merged)
                print(f"[embedder] Cache diperbarui → {cache_path} ({len(merged)} total)")

        all_embeddings = {**cached_embeddings, **new_embeddings}
        print(f"[embedder] Total embedding siap: {len(all_embeddings)}")
        return all_embeddings