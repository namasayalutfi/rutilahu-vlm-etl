from __future__ import annotations

import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from etl.extract_dinsos_house_images_data import DinsosHouseImagesExtractor
from etl.download_images_and_metadata import DinsosHouseDownloadMetadataPipeline
from etl.sample_metadata_from_multi import DinsosHouseMetadataSampler
from etl.augment_sample_metadata_from_crawling import AugmentConfig, SampleMetadataAugmentor
from etl.generate_labelstudio_metadata_input import LabelStudioConfig, LabelStudioMetadataGenerator
from etl.reconcile_metadata_v2 import ReconcileConfig, LabelStudioMetadataReconciler
from etl.generate_dtsen_status_dummy_v2 import DTSENDummyConfig, DTSENDummyGenerator
from etl.split_metadata import (
    SplitConfig,
    HouseTypeAwareHierarchicalStratifiedSplitter,
)
from etl.crawl_google_images import GoogleImageCrawlerConfig, GoogleImageCrawler
from etl.download_crawled_img import CrawledImageMinioMetadataConfig, CrawledImageMinioMetadataPipeline
from etl.merge_sample_metadata import MergeSampleMetadataConfig, SampleMetadataMerger
from etl.split_labelstudio_input import LabelStudioInputSplitter, LabelStudioSplitConfig
from etl.merge_labelstudio_outputs import LabelStudioMergeConfig, LabelStudioOutputMerger
from etl.merge_two_metadata import MergeConfig, MetadataMerger

class RutilahuETLPipeline:

    def __init__(self):
        self.extractor = DinsosHouseImagesExtractor()
        self.downloader = DinsosHouseDownloadMetadataPipeline()
        self.augmentor = SampleMetadataAugmentor(AugmentConfig())
        self.labelstudio_generator = LabelStudioMetadataGenerator(LabelStudioConfig())
        self.reconciler = LabelStudioMetadataReconciler(ReconcileConfig())
        self.dtsen_generator = DTSENDummyGenerator(DTSENDummyConfig())

        self.splitter = HouseTypeAwareHierarchicalStratifiedSplitter(
            SplitConfig(
                input_path=Path("output/mkn2_cleaned_metadata.json"),
                output_dir=Path("metadata/splited_metadata"),
                train_ratio=0.8,
                val_ratio=0.1,
                test_ratio=0.1,
                seed=42,
            )
        )

        self.image_crawler = GoogleImageCrawler(
            GoogleImageCrawlerConfig(
                keyword_dir=PROJECT_ROOT / "data" / "keywords_test",
                output_dir=PROJECT_ROOT / "data" / "crawled_urls_mkn2",
                max_urls_per_keyword=200,
                headless=True,
                file_workers=2,
                keyword_workers=1,
                stagger_delay_min=0.5,
                stagger_delay_max=2.0,
                thumbnails_to_click = 50,
                click_delay_min = 0.1,
                click_delay_max = 0.3,
            )
        )
        self.crawled_image_pipeline = CrawledImageMinioMetadataPipeline(
            CrawledImageMinioMetadataConfig(
                crawler_output_dir=PROJECT_ROOT / "data" / "crawler_outputs",
                output_metadata_path=PROJECT_ROOT / "metadata" / "crawled_img_metadata.json",
                workers=8,
            )
        )
        self.sample_metadata_merger = SampleMetadataMerger(
            MergeSampleMetadataConfig(
                metadata_jsonl_path=PROJECT_ROOT / "metadata" / "metadata.jsonl",
                crawled_metadata_path=PROJECT_ROOT / "metadata" / "crawled_img_metadata.json",
                output_path=PROJECT_ROOT / "metadata" / "mkn2_metadata_merged.json",
            )
        )
        self.labelstudio_splitter = LabelStudioInputSplitter(
            LabelStudioSplitConfig(
                input_json=PROJECT_ROOT / "data" / "labelstudio_input.json",
                output_dir=PROJECT_ROOT / "data" / "labelstudio_input_split",
                num_splits=8,
                seed=42,
            )
        )
        self.labelstudio_output_merger = LabelStudioOutputMerger(
            LabelStudioMergeConfig(
                input_dir=PROJECT_ROOT / "data" / "labelstudio_output_split",
                output_json=PROJECT_ROOT / "data" / "labelstudio_output_merged.json",
                recursive=False,
            )
        )
        self.two_metadata_merger = MetadataMerger(
            MergeConfig(
                reconciled_metadata_path=PROJECT_ROOT / "metadata" / "reconciled_mkn2_metadata.json",
                mkn2_metadata_path=PROJECT_ROOT / "metadata" / "mkn2_metadata.json",
                output_path=PROJECT_ROOT / "metadata" / "mkn2_metadata_final.json",
                dedupe_by_house_id=False,
            )
        )
        self._sampler = None

    @property
    def sampler(self):
        if self._sampler is None:
            self._sampler = DinsosHouseMetadataSampler()
        return self._sampler

    def run_extract(self) -> None:
        df = self.extractor.run()
        print(f"[OK] Extract selesai. Total rows: {len(df):,}")

    def run_download_and_metadata(self) -> None:
        outputs = self.downloader.run()
        print("[OK] Download + metadata selesai.")
        for k, v in outputs.items():
            print(f"{k}: {v}")

    def run_sample_metadata(self) -> None:
        outputs = self.sampler.run()
        print("[OK] Sampling metadata selesai.")
        for k, v in outputs.items():
            print(f"{k}: {v}")

    def run_augment_sample_metadata(self) -> None:
        result = self.augmentor.run()
        print("[OK] Augment sample metadata selesai.")
        for k, v in result.items():
            print(f"{k}: {v}")

    def run_build_labelstudio_input(self) -> None:
        out_path = self.labelstudio_generator.run()
        print(f"[OK] Label Studio metadata generated: {out_path}")

    def run_reconcile_metadata(self) -> None:
        result = self.reconciler.reconcile()
        print("[OK] Reconcile metadata selesai.")
        for k, v in result.items():
            print(f"{k}: {v}")

    def run_generate_dtsen_dummy(self) -> None:
        result = self.dtsen_generator.run()
        print("[OK] Generate DTSEN dummy selesai.")
        for k, v in result.items():
            print(f"{k}: {v}")

    def run_split_metadata(self) -> None:
        result = self.splitter.run()
        print("[OK] Split metadata selesai.")
        print(f"total_records: {result['total_records']}")
        print(f"all_path: {result['all_path']}")
        print()
        print("split_sizes:")
        print(f"  train: {result['split_sizes']['train']}")
        print(f"  val: {result['split_sizes']['val']}")
        print(f"  test: {result['split_sizes']['test']}")
        print()
        print("house_type_distribution:")
        for split, dist in result["house_type_distribution"].items():
            print(f"  {split}: {dist}")
        print()
        print("label_distribution_global:")
        for comp, comp_dist in result["label_distribution_global"].items():
            print(f"  {comp}: {comp_dist}")
        print()
        print("label_distribution_by_schema:")
        for schema, schema_dist in result["label_distribution_by_schema"].items():
            print(f"  {schema}: {schema_dist}")
        print()
        print("combo_distribution_global:")
        for split, dist in result["combo_distribution_global"].items():
            print(f"  {split}: {dist}")
        print()
        print("combo_distribution_by_schema:")
        for schema, schema_dist in result["combo_distribution_by_schema"].items():
            print(f"  {schema}: {schema_dist}")


    def run_crawl_images(self) -> None:
        """Crawl image URL dari Google Images berdasarkan semua file keyword."""
        results = self.image_crawler.run()
        print("[OK] Crawling Google Images selesai.")
        for kw_file, out_path in results.items():
            print(f"  {kw_file} → {out_path}")

    def run_build_crawled_img_metadata(self) -> None:
        result = self.crawled_image_pipeline.run()
        print("[OK] Crawled image download + MinIO upload + metadata selesai.")
        for k, v in result.items():
            print(f"{k}: {v}")

    def run_merge_sample_metadata(self) -> None:
        result = self.sample_metadata_merger.merge()
        print("[OK] Merge sample metadata selesai.")
        for k, v in result.items():
            print(f"{k}: {v}")

    def run_split_labelstudio_input(self) -> None:
        result = self.labelstudio_splitter.run()
        print("[OK] Split labelstudio_input selesai.")
        print(f"total_records: {result['total_records']}")
        print(f"output_dir: {result['output_dir']}")
        print("split_sizes:")
        for k, v in result["split_sizes"].items():
            print(f"  {k}: {v}")

    def run_merge_labelstudio_outputs(self) -> None:
        result = self.labelstudio_output_merger.merge()
        print("[OK] Merge labelstudio outputs selesai.")
        print(f"input_dir: {result['input_dir']}")
        print(f"output_json: {result['output_json']}")
        print(f"total_files: {result['total_files']}")
        print(f"total_records: {result['total_records']}")
        print("per_file_counts:")
        for k, v in result["per_file_counts"].items():
            print(f"  {k}: {v}")

    def run_merge_two_metadata(self) -> None:
        result = self.two_metadata_merger.merge()
        print("[OK] Merge dua metadata selesai.")
        for k, v in result.items():
            print(f"{k}: {v}")

    def run_all(self) -> None:
        self.run_download_and_metadata()
        self.run_sample_metadata()
        self.run_augment_sample_metadata()
        self.run_build_labelstudio_input()
        self.run_reconcile_metadata()
        self.run_generate_dtsen_dummy()
        self.run_split_metadata()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rutilahu ETL Pipeline")
    parser.add_argument("--extract_data", action="store_true", help="Jalankan extract data dari excel lokal")
    parser.add_argument(
        "--download_metadata",
        action="store_true",
        help="Jalankan download image ke MinIO + pembuatan metadata",
    )
    parser.add_argument(
        "--sample_metadata",
        action="store_true",
        help="Jalankan sampling metadata multi menjadi multi + single split",
    )
    parser.add_argument(
        "--augment_sample_metadata",
        action="store_true",
        help="Tambah metadata crawling ke sample_metadata.jsonl",
    )
    parser.add_argument(
        "--build_labelstudio_input",
        action="store_true",
        help="Generate labelstudio_metadata_input.json from sample_metadata_augmented.jsonl",
    )
    parser.add_argument(
        "--reconcile_metadata",
        action="store_true",
        help="Repair sample_metadata_augmented.jsonl using labelstudio_output.json",
    )
    parser.add_argument(
        "--generate_dtsen_dummy",
        action="store_true",
        help="Generate dummy dtsen labels and status in reconciled_sample_metadata.json",
    )
    parser.add_argument(
        "--split_metadata",
        action="store_true",
        help="Split metadata menjadi train/val/test dengan house_type-aware iterative stratification",
    )
    parser.add_argument(
        "--crawl_images",
        action="store_true",
        help="Crawl image URL dari Google Images berdasarkan file keyword di data/",
    )
    parser.add_argument(
        "--build_crawled_img_metadata",
        action="store_true",
        help="Download image hasil crawl, upload ke MinIO, dan buat metadata",
    )
    parser.add_argument(
        "--merge_metadata",
        action="store_true",
        help="Gabungkan mkn2_metadata.json + filtered metadata.jsonl + crawled_img_metadata.json",
    )
    parser.add_argument(
        "--split_labelstudio_input",
        action="store_true",
        help="Pecah labelstudio_input.json menjadi 8 file JSON merata",
    )
    parser.add_argument(
        "--merge_labelstudio_outputs",
        action="store_true",
        help="Merge semua file JSON output Label Studio dari folder split",
    )
    parser.add_argument(
        "--merge_two_metadata",
        action="store_true",
        help="Merge reconciled_mkn2_metadata.json dan mkn2_metadata.json",
    )
    parser.add_argument("--all", action="store_true", help="Jalankan keseluruhan pipeline ETL")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    pipeline = RutilahuETLPipeline()

    if args.all:
        pipeline.run_all()
        return

    did_run = False

    if args.extract_data:
        pipeline.run_extract()
        did_run = True

    if args.download_metadata:
        pipeline.run_download_and_metadata()
        did_run = True

    if args.sample_metadata:
        pipeline.run_sample_metadata()
        did_run = True

    if args.augment_sample_metadata:
        pipeline.run_augment_sample_metadata()
        did_run = True

    if args.build_labelstudio_input:
        pipeline.run_build_labelstudio_input()
        did_run = True

    if args.reconcile_metadata:
        pipeline.run_reconcile_metadata()
        did_run = True

    if args.generate_dtsen_dummy:
        pipeline.run_generate_dtsen_dummy()
        did_run = True

    if args.split_metadata:
        pipeline.run_split_metadata()
        did_run = True

    if args.crawl_images:
        pipeline.run_crawl_images()
        did_run = True

    if args.build_crawled_img_metadata:
        pipeline.run_build_crawled_img_metadata()
        did_run = True

    if args.merge_metadata:
        pipeline.run_merge_sample_metadata()
        did_run = True
    
    if args.split_labelstudio_input:
        pipeline.run_split_labelstudio_input()
        did_run = True

    if args.merge_labelstudio_outputs:
        pipeline.run_merge_labelstudio_outputs()
        did_run = True  
    
    if args.merge_two_metadata:
        pipeline.run_merge_two_metadata()
        did_run = True

    if not did_run:
        parser.print_help()


if __name__ == "__main__":
    main()