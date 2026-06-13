import urllib.parse
import time
from pathlib import Path
from scrapling import Fetcher

class GoogleImageScraperConfig:
    def __init__(
        self, 
        data_dir: Path, 
        output_dir: Path, 
        max_scrolls: int = 3
    ):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.max_scrolls = max_scrolls
        # List file keyword yang diminta
        self.target_files = [
            "keyword_atap_jerami_ijuk_daun_rumbia.txt",
            "keyword_atap_kayu_sirap.txt",
            "keyword_dinding_anyaman_bambu.txt",
            "keyword_dinding_bambu.txt",
            "keyword_dinding_batang_kayu.txt",
            "keyword_lantai_marmer_granit.txt",
            "keyword_lantai_parket_vinil_karpet.txt"
        ]

class GoogleImageUrlsExtractor:
    def __init__(self, config: GoogleImageScraperConfig):
        self.config = config
        # Memastikan folder output tersedia
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def _scrape_keyword(self, fetcher, keyword: str) -> list[str]:
        urls = set()
        query = urllib.parse.quote_plus(keyword)
        # Parameter tbm=isch digunakan untuk mengakses Google Images
        search_url = f"https://www.google.com/search?tbm=isch&q={query}"
        
        # Buka halaman pencarian
        page = fetcher.get(search_url)
        
        # Scroll perlahan ke bawah untuk men-trigger lazy-load gambar
        for _ in range(self.config.max_scrolls):
            try:
                page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
                time.sleep(1.5)
            except Exception as e:
                print(f"[WARN] Gagal scroll: {e}")
                break
            
        # Ekstrak elemen img
        img_elements = page.css("img")
        for img in img_elements:
            # Di Google Images, gambar asli kadang disimpan di data-src sebelum di-render
            src = img.attrib.get("src")
            data_src = img.attrib.get("data-src")
            
            url_to_save = data_src if data_src else src
            # Hanya simpan valid HTTP URL (mencegah base64 image yang terlalu panjang/rusak)
            if url_to_save and url_to_save.startswith("http"):
                urls.add(url_to_save)
                
        return list(urls)

    def run(self) -> dict:
        results = {}
        
        # Jalankan Scrapling Fetcher dengan mode headless
        with Fetcher(headless=True) as fetcher:
            for filename in self.config.target_files:
                input_path = self.config.data_dir / filename
                if not input_path.exists():
                    print(f"[WARN] File keyword tidak ditemukan: {input_path}")
                    continue
                    
                # Ubah format penamaan output file, misal dari keyword_xxx.txt menjadi url_xxx.txt
                output_filename = filename.replace("keyword_", "url_")
                output_path = self.config.output_dir / output_filename
                
                with open(input_path, "r", encoding="utf-8") as f:
                    keywords = [line.strip() for line in f if line.strip()]
                    
                all_urls = set()
                for keyword in keywords:
                    print(f"[*] Scraping images for keyword: '{keyword}' from {filename}")
                    scraped_urls = self._scrape_keyword(fetcher, keyword)
                    all_urls.update(scraped_urls)
                    
                # Simpan hasil scraping dari file keyword yang bersangkutan ke txt output
                with open(output_path, "w", encoding="utf-8") as f:
                    for url in all_urls:
                        f.write(f"{url}\n")
                        
                results[filename] = len(all_urls)
                print(f"[OK] Disimpan {len(all_urls)} URL untuk file {output_filename}")
                
        return results