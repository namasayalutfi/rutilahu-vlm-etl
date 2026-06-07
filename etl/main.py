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
from etl.reconcile_metadata import ReconcileConfig, LabelStudioMetadataReconciler
from etl.generate_dtsen_dummy import DTSENDummyConfig, DTSENDummyGenerator
from etl.split_metadata import (
    SplitConfig,
    HouseTypeAwareHierarchicalStratifiedSplitter,
)

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
                input_path=Path("metadata_sample/reconciled_sample_metadata_with_dtsen.json"),
                output_dir=Path("metadata_sample/splits_house_type_aware"),
                train_ratio=0.8,
                val_ratio=0.1,
                test_ratio=0.1,
                seed=42,
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

    def run_all(self) -> None:
        # extract dijalankan di local,
        # sedangkan download + metadata + sampling dijalankan di server.
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

    if not did_run:
        parser.print_help()


if __name__ == "__main__":
    main()