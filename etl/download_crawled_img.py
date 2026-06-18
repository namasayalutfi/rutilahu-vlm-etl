from __future__ import annotations

import json
import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from minio import Minio
from minio.error import S3Error

try:
    from PIL import Image
except Exception:
    Image = None

load_dotenv()


@dataclass
class CrawledImageMinioMetadataConfig:
    crawler_output_dir: Path = Path("data/crawler_outputs")
    output_metadata_path: Path = Path("metadata/crawled_img_metadata.json")

    # MinIO config from .env
    minio_endpoint: Optional[str] = None
    minio_access_key: Optional[str] = None
    minio_secret_key: Optional[str] = None
    minio_secure: bool = False
    minio_bucket_name: Optional[str] = None
    minio_public_base_url: Optional[str] = None

    # Output object naming
    minio_folder: str = "crawled_image"
    image_file_prefix: str = "mkn2_crawled_img_"
    image_file_start_index: int = 1

    # Metadata id naming
    house_id_prefix: str = "CRAWLED_H"
    house_id_start_index: int = 1
    image_id_prefix: str = "CRAWLED_IMG_"
    image_id_start_index: int = 0

    # Workers
    workers: int = 8

    # Network
    request_timeout: int = 30
    request_user_agent: str = "Mozilla/5.0"

    def __post_init__(self):
        if self.minio_endpoint is None:
            self.minio_endpoint = os.getenv("MINIO_ENDPOINT")
        if self.minio_access_key is None:
            self.minio_access_key = os.getenv("MINIO_ACCESS_KEY")
        if self.minio_secret_key is None:
            self.minio_secret_key = os.getenv("MINIO_SECRET_KEY")
        if self.minio_bucket_name is None:
            self.minio_bucket_name = os.getenv("MINIO_BUCKET_NAME")
        if self.minio_public_base_url is None:
            self.minio_public_base_url = os.getenv("MINIO_PUBLIC_BASE_URL")

        secure_raw = os.getenv("MINIO_SECURE", "False").strip().lower()
        self.minio_secure = secure_raw in {"true", "1", "yes"}

        required = {
            "MINIO_ENDPOINT": self.minio_endpoint,
            "MINIO_ACCESS_KEY": self.minio_access_key,
            "MINIO_SECRET_KEY": self.minio_secret_key,
            "MINIO_BUCKET_NAME": self.minio_bucket_name,
            "MINIO_PUBLIC_BASE_URL": self.minio_public_base_url,
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise ValueError(f"Environment variable required belum lengkap: {missing}")


@dataclass
class CrawledImageTask:
    source_file: Path
    source_url: str
    category_key: str
    actual_label_field: str
    actual_label_value: str
    view_type: str
    house_id: str
    image_id: str
    image_filename: str
    object_name: str


class CrawledImageMinioMetadataPipeline:
    """
    Pipeline:
    1) baca URL dari file txt di data/crawler_outputs
    2) download image
    3) konversi JPG
    4) upload ke MinIO
    5) buat metadata JSON bersamaan
    """

    LABEL_RULES: Dict[str, Dict[str, str]] = {
        "url_atap_jerami_ijuk_daun_rumbia": {
            "actual_label_field": "atap",
            "actual_label_value": "Jerami/ijuk/daun-daunan/rumbia",
            "view_type": "exterior",
        },
        "url_atap_kayu_sirap": {
            "actual_label_field": "atap",
            "actual_label_value": "Kayu/sirap",
            "view_type": "exterior",
        },
        "url_dinding_anyaman_bambu": {
            "actual_label_field": "dinding",
            "actual_label_value": "Anyaman bambu",
            "view_type": "exterior",
        },
        "url_dinding_bambu": {
            "actual_label_field": "dinding",
            "actual_label_value": "Bambu",
            "view_type": "exterior",
        },
        "url_dinding_batang_kayu": {
            "actual_label_field": "dinding",
            "actual_label_value": "Batang kayu",
            "view_type": "exterior",
        },
        "url_lantai_kayu_papan": {
            "actual_label_field": "lantai",
            "actual_label_value": "Kayu/papan",
            "view_type": "interior",
        },
        "url_lantai_marmer_granit": {
            "actual_label_field": "lantai",
            "actual_label_value": "Marmer/granit",
            "view_type": "interior",
        },
        "url_lantai_parket_vinil_karpet": {
            "actual_label_field": "lantai",
            "actual_label_value": "Parket/vinil/karpet",
            "view_type": "interior",
        },
    }

    def __init__(self, config: CrawledImageMinioMetadataConfig):
        self.config = config
        self.config.output_metadata_path.parent.mkdir(parents=True, exist_ok=True)

        self.client = Minio(
            endpoint=self.config.minio_endpoint,
            access_key=self.config.minio_access_key,
            secret_key=self.config.minio_secret_key,
            secure=self.config.minio_secure,
        )
        self._ensure_bucket()

        self.next_object_idx = self._get_next_object_index()
        self.next_house_idx = self._get_next_house_index()
        self.next_image_idx = self._get_next_image_index()

    def _ensure_bucket(self) -> None:
        if not self.client.bucket_exists(self.config.minio_bucket_name):
            self.client.make_bucket(self.config.minio_bucket_name)

    def _get_next_object_index(self) -> int:
        """
        Scan object existing di MinIO agar penamaan lanjut dari nomor terbesar.
        """
        prefix = f"{self.config.minio_folder}/"
        pattern = f"{self.config.minio_folder}/{self.config.image_file_prefix}"

        max_idx = 0
        for obj in self.client.list_objects(
            self.config.minio_bucket_name,
            prefix=prefix,
            recursive=True,
        ):
            name = obj.object_name
            if not name.startswith(pattern):
                continue
            stem = Path(name).stem
            suffix = stem.replace(self.config.image_file_prefix, "")
            if suffix.isdigit():
                max_idx = max(max_idx, int(suffix))

        return max_idx + 1

    def _get_next_house_index(self) -> int:
        """
        Jika output metadata sudah ada, lanjutkan house_id dari nomor terbesar.
        """
        if not self.config.output_metadata_path.exists():
            return self.config.house_id_start_index

        try:
            with open(self.config.output_metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return self.config.house_id_start_index

        max_idx = 0
        if isinstance(data, list):
            for rec in data:
                hid = str(rec.get("house_id", ""))
                digits = "".join(ch for ch in hid if ch.isdigit())
                if digits:
                    max_idx = max(max_idx, int(digits))

        return max_idx + 1 if max_idx > 0 else self.config.house_id_start_index

    def _get_next_image_index(self) -> int:
        """
        Jika output metadata sudah ada, lanjutkan image_id dari nomor terbesar.
        """
        if not self.config.output_metadata_path.exists():
            return self.config.image_id_start_index

        try:
            with open(self.config.output_metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return self.config.image_id_start_index

        max_idx = -1
        if isinstance(data, list):
            for rec in data:
                images = rec.get("images", [])
                if not isinstance(images, list):
                    continue
                for img in images:
                    image_id = str(img.get("image_id", ""))
                    digits = "".join(ch for ch in image_id if ch.isdigit())
                    if digits:
                        max_idx = max(max_idx, int(digits))

        return max_idx + 1 if max_idx >= 0 else self.config.image_id_start_index

    def _read_urls_from_txt(self, path: Path) -> List[str]:
        urls: List[str] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                url = line.strip()
                if not url:
                    continue
                if url.startswith("#"):
                    continue
                urls.append(url)
        return urls

    def _load_tasks(self) -> List[CrawledImageTask]:
        tasks: List[CrawledImageTask] = []
        txt_files = sorted(self.config.crawler_output_dir.rglob("*.txt"))

        for txt_file in txt_files:
            key = txt_file.stem
            if key not in self.LABEL_RULES:
                continue

            rules = self.LABEL_RULES[key]
            urls = self._read_urls_from_txt(txt_file)

            for url in urls:
                house_id = f"{self.config.house_id_prefix}{self.next_house_idx:05d}"
                image_id = f"{self.config.image_id_prefix}{self.next_image_idx:05d}"
                image_filename = f"{self.config.image_file_prefix}{self.next_object_idx:05d}.jpg"
                object_name = f"{self.config.minio_folder}/{image_filename}"

                tasks.append(
                    CrawledImageTask(
                        source_file=txt_file,
                        source_url=url,
                        category_key=key,
                        actual_label_field=rules["actual_label_field"],
                        actual_label_value=rules["actual_label_value"],
                        view_type=rules["view_type"],
                        house_id=house_id,
                        image_id=image_id,
                        image_filename=image_filename,
                        object_name=object_name,
                    )
                )

                self.next_house_idx += 1
                self.next_image_idx += 1
                self.next_object_idx += 1

        return tasks

    def _download_image_bytes(self, url: str) -> bytes:
        headers = {"User-Agent": self.config.request_user_agent}
        response = requests.get(url, headers=headers, timeout=self.config.request_timeout)
        response.raise_for_status()
        return response.content

    def _convert_to_jpg(self, raw_bytes: bytes) -> bytes:
        """
        Konversi bytes image ke JPEG agar file final konsisten .jpg.
        Jika Pillow tidak tersedia atau gagal decode, fallback ke raw bytes.
        """
        if Image is None:
            return raw_bytes

        try:
            with Image.open(BytesIO(raw_bytes)) as img:
                if img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")
                elif img.mode == "L":
                    img = img.convert("RGB")

                buf = BytesIO()
                img.save(buf, format="JPEG", quality=95, optimize=True)
                return buf.getvalue()
        except Exception:
            return raw_bytes

    def _upload_to_minio(self, image_bytes: bytes, object_name: str) -> None:
        data = BytesIO(image_bytes)
        data.seek(0)
        self.client.put_object(
            bucket_name=self.config.minio_bucket_name,
            object_name=object_name,
            data=data,
            length=len(image_bytes),
            content_type="image/jpeg",
        )

    def _build_record(self, task: CrawledImageTask) -> Dict[str, Any]:
        actual_label = {
            "atap": None,
            "dinding": None,
            "lantai": None,
        }
        actual_label[task.actual_label_field] = task.actual_label_value

        image_db_url = (
            f"{self.config.minio_public_base_url.rstrip('/')}/"
            f"{self.config.minio_bucket_name}/"
            f"{task.object_name}"
        )

        return {
            "house_id": task.house_id,
            "no_kk": None,
            "house_type": None,
            "split": None,
            "images": [
                {
                    "image_id": task.image_id,
                    "image_path": f"{self.config.minio_folder}/{task.image_filename}",
                    "image_ori_url": None,
                    "image_db_url": image_db_url,
                    "view_type": task.view_type,
                }
            ],
            "actual_label": actual_label,
            "dtsen": {
                "atap": None,
                "dinding": None,
                "lantai": None,
            },
            "status": {
                "atap": None,
                "dinding": None,
                "lantai": None,
            },
        }

    def _process_task(self, task: CrawledImageTask) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        try:
            raw_bytes = self._download_image_bytes(task.source_url)
            jpg_bytes = self._convert_to_jpg(raw_bytes)
            self._upload_to_minio(jpg_bytes, task.object_name)
            record = self._build_record(task)
            return True, record, None
        except Exception as e:
            err = f"{task.source_file.name} | {task.source_url} | {str(e)}"
            return False, None, err

    def run(self) -> Dict[str, Any]:
        tasks = self._load_tasks()

        records: List[Dict[str, Any]] = []
        errors: List[str] = []

        if not tasks:
            self.config.output_metadata_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config.output_metadata_path, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)

            return {
                "total_tasks": 0,
                "successful": 0,
                "failed": 0,
                "output_path": str(self.config.output_metadata_path),
                "minio_bucket": self.config.minio_bucket_name,
            }

        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=self.config.workers) as executor:
            futures = [executor.submit(self._process_task, task) for task in tasks]

            for future in as_completed(futures):
                success, record, error = future.result()
                if success and record is not None:
                    records.append(record)
                elif error:
                    errors.append(error)

        # sort agar konsisten berdasarkan house_id numerik
        def _house_sort_key(rec: Dict[str, Any]) -> int:
            hid = str(rec.get("house_id", ""))
            digits = "".join(ch for ch in hid if ch.isdigit())
            return int(digits) if digits else 0

        records.sort(key=_house_sort_key)

        self.config.output_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config.output_metadata_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        error_path = self.config.output_metadata_path.parent / "crawled_img_metadata_errors.json"
        with open(error_path, "w", encoding="utf-8") as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)

        return {
            "total_tasks": len(tasks),
            "successful": len(records),
            "failed": len(errors),
            "output_path": str(self.config.output_metadata_path),
            "error_path": str(error_path),
            "minio_bucket": self.config.minio_bucket_name,
        }