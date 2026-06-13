import json
from pathlib import Path
from collections import defaultdict

def extract_balanced_samples(input_file: Path, output_file: Path, n_per_split: int = 5):
    if not input_file.exists():
        print(f"File tidak ditemukan: {input_file}")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Buffer untuk menampung sampel
    samples = defaultdict(list)
    counts = defaultdict(int)

    # Iterasi data untuk mengambil sampel
    for record in data:
        split = record.get("split", "unknown")
        if counts[split] < n_per_split:
            samples[split].append(record)
            counts[split] += 1
        
        # Stop jika semua split sudah memenuhi kuota
        if all(counts[s] >= n_per_split for s in ["train", "val", "test"]):
            break

    # Gabungkan semua sampel
    final_samples = []
    for split_list in samples.values():
        final_samples.extend(split_list)

    # Simpan ke file baru
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_samples, f, ensure_ascii=False, indent=2)

    print(f"Berhasil mengambil {len(final_samples)} sampel.")
    for split, count in counts.items():
        print(f"- {split}: {count} sampel")

# Konfigurasi Path
input_path = Path(r"metadata_sample\splits_house_type_aware\mkn2_metadata.json")
output_path = Path(r"metadata_sample\mkn2_metadata_sample.json")

extract_balanced_samples(input_path, output_path)