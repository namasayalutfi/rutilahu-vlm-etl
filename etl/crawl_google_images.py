from __future__ import annotations

import logging
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import quote_plus, urlparse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GoogleImageCrawlerConfig:
    keyword_dir: Path = Path("../data/keywords")
    output_dir: Path = Path("../data/crawled_urls")

    max_urls_per_keyword: int = 200

    # Scroll: dikurangi dari 15 → 8, delay diperkecil
    max_scrolls: int = 8
    scroll_delay_min: float = 0.8
    scroll_delay_max: float = 1.5

    # Timeout diperkecil: page 30s (dari 60s), element 5s (dari 8s)
    page_timeout: int = 30_000
    element_wait_timeout: int = 5_000

    headless: bool = True

    # Parallelism — KUNCI UTAMA penghematan RAM:
    # file_workers=2 artinya 2 file diproses bersamaan (OK)
    # keyword_workers=1 artinya 1 keyword per file pada satu waktu (hemat RAM)
    # Total Chromium aktif = file_workers × keyword_workers = 2×1 = 2 (vs 4 sebelumnya)
    file_workers: int = 2
    keyword_workers: int = 1

    stagger_delay_min: float = 0.5
    stagger_delay_max: float = 2.0

    # Klik thumbnail: fallback saja, dikurangi drastis 200 → 30
    thumbnails_to_click: int = 30
    click_delay_min: float = 0.1
    click_delay_max: float = 0.3

    # Reuse browser antar keyword dalam satu file (hemat waktu init ~2s/keyword)
    reuse_browser: bool = True

    keyword_to_output_map: Optional[dict[str, str]] = None
    keyword_files: Optional[list[str]] = None


# ---------------------------------------------------------------------------
# URL Validator
# ---------------------------------------------------------------------------

class _ImageUrlValidator:
    _IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff", ".avif", ".jfif"}

    _BLOCKED_DOMAINS = {
        "www.google.com", "google.com", "google.co.id", "google.co.uk",
        "encrypted-tbn0.gstatic.com", "encrypted-tbn1.gstatic.com",
        "encrypted-tbn2.gstatic.com", "encrypted-tbn3.gstatic.com",
        "ssl.gstatic.com", "www.gstatic.com", "fonts.gstatic.com",
        "googleusercontent.com",
        "www.googleadservices.com", "googleadservices.com",
        "pagead2.googlesyndication.com", "tpc.googlesyndication.com",
        "www.google-analytics.com", "analytics.google.com",
        "doubleclick.net", "googlesyndication.com",
        "facebook.com", "fbcdn.net", "instagram.com",
        "schema.org", "ogp.me", "opengraph.io",
        "fonts.googleapis.com", "ajax.googleapis.com",
        "cdn.jsdelivr.net", "cdnjs.cloudflare.com",
    }

    _BLOCKED_PATH_PATTERNS = [
        "/pagead/", "/aclk", "/intl/", "/policies/", "/preferences",
        "/search?", "/maps/", "/translate", "/accounts/",
        "/blogs/read/", "/read/", "/artikel/", "/news/", "/berita/",
        "/product/", "/shop/", "/category/", "/tag/", "/author/",
        "/page/", "/post/", "/2023/", "/2024/", "/2025/",
    ]

    _BLOCKED_EXTENSIONS = {
        ".html", ".htm", ".php", ".asp", ".aspx", ".jsp",
        ".js", ".css", ".json", ".xml", ".svg", ".ico",
        ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".zip",
        ".mp4", ".mp3", ".avi", ".mov", ".wmv", ".flv",
    }

    def is_valid(self, url: str) -> bool:
        if not url or not url.startswith("http"):
            return False
        if url.startswith("data:"):
            return False
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower().lstrip("www.")
            path = parsed.path.lower()

            if self._is_blocked_domain(domain):
                return False
            if not path or path == "/":
                return False
            for pattern in self._BLOCKED_PATH_PATTERNS:
                if pattern in path:
                    return False
            for ext in self._BLOCKED_EXTENSIONS:
                if path.endswith(ext):
                    return False

            for ext in self._IMAGE_EXTS:
                if path.endswith(ext):
                    return True
            path_without_qs = path.split("?")[0]
            for ext in self._IMAGE_EXTS:
                if path_without_qs.endswith(ext):
                    return True

            query = (parsed.query or "").lower()
            if re.search(r'format=(jpg|jpeg|png|webp|gif|avif)', query):
                return True
            if re.search(r'\.(jpg|jpeg|png|webp|gif|avif)(\b|&|$)', query):
                return True

            return False
        except Exception:
            return False

    def _is_blocked_domain(self, domain: str) -> bool:
        if domain in self._BLOCKED_DOMAINS:
            return True
        for blocked in self._BLOCKED_DOMAINS:
            if domain.endswith("." + blocked):
                return True
        return False


_VALIDATOR = _ImageUrlValidator()


# ---------------------------------------------------------------------------
# _BrowserSession — reuse browser antar keyword dalam satu file
# ---------------------------------------------------------------------------

class _BrowserSession:
    """
    Satu instance Playwright + browser yang di-reuse antar keyword.
    Dibuat sekali per file worker, bukan per keyword.
    Menghemat ~2 detik init Chromium × jumlah keyword.
    """

    _USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    ]

    _STEALTH_JS = """
        () => {
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            Object.defineProperty(navigator, 'plugins',   { get: () => [1, 2, 3, 4, 5] });
            Object.defineProperty(navigator, 'languages', { get: () => ['id-ID', 'id', 'en-US', 'en'] });
            window.chrome = { runtime: {} };
        }
    """

    def __init__(self, pw, config: GoogleImageCrawlerConfig):
        self.config = config
        user_agent = random.choice(self._USER_AGENTS)
        viewport = {"width": random.randint(1366, 1920), "height": random.randint(768, 1080)}

        self.browser = pw.chromium.launch(
            headless=config.headless,
            args=[
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--disable-infobars",
                "--disable-extensions",
                "--disable-gpu",
                # Tambahan: matikan fitur berat yang tidak dibutuhkan
                "--disable-background-networking",
                "--disable-sync",
                "--disable-translate",
                "--disable-notifications",
                "--blink-settings=imagesEnabled=false",  # Nonaktifkan load gambar di halaman (hemat bandwidth)
            ],
        )
        self.context = self.browser.new_context(
            user_agent=user_agent,
            viewport=viewport,
            locale="id-ID",
            timezone_id="Asia/Jakarta",
            java_script_enabled=True,
            permissions=[],
            # Blokir resource tidak perlu: font, media, stylesheet
            # (gambar diblokir di level args; network intercept tetap aktif untuk URL)
        )
        self.context.add_init_script(self._STEALTH_JS)
        self._consent_done = False

    def close(self):
        try:
            self.context.close()
        except Exception:
            pass
        try:
            self.browser.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# _KeywordWorker
# ---------------------------------------------------------------------------

class _KeywordWorker:
    """
    Strategi ekstraksi (urutan prioritas):
    1. [BARU] Intercept network requests: tangkap URL gambar dari traffic browser
       → tidak perlu klik thumbnail sama sekali, jauh lebih cepat
    2. Klik thumbnail (fallback, max 30 klik) jika intercept kurang
    3. Fallback DOM: ambil img[src] yang ter-render

    Browser di-reuse antar keyword (via _BrowserSession) jika reuse_browser=True.
    """

    _GOOGLE_IMAGES_URL = "https://www.google.com/search?tbm=isch&q={query}&safe=off"

    _THUMBNAIL_SELECTORS = [
        "div[jsname='dTDiAc'] img",
        "div[data-id] img",
        "g-img > img",
        "div[jsaction*='mousedown'] img",
        "img.YQ4gaf",
        "img.rg_i",
    ]

    _DETAIL_PANEL_JS = """
        () => {
            const selectors = [
                'img.sFlh5c', 'img.iPVvYb', 'img[jsname="kn3ccd"]',
                'img.r48jcc', 'div.tvh9oe img', 'c-wiz img',
            ];
            for (const sel of selectors) {
                const imgs = document.querySelectorAll(sel);
                for (const img of imgs) {
                    const src = img.src || img.getAttribute('data-src') || '';
                    if (src.startsWith('http') && !src.includes('encrypted-tbn')
                        && !src.includes('gstatic.com') && img.naturalWidth > 100) {
                        return src;
                    }
                }
            }
            return null;
        }
    """

    _ALL_IMG_JS = """
        () => {
            const results = [];
            document.querySelectorAll('img').forEach(img => {
                const src = img.src || img.getAttribute('data-src') || '';
                if (src.startsWith('http') && img.naturalWidth > 100) results.push(src);
            });
            return [...new Set(results)];
        }
    """

    # Pola URL gambar full-res yang muncul di network requests
    _FULLRES_URL_PATTERN = re.compile(
        r'https?://(?!(?:www\.google|encrypted-tbn|gstatic|googleadservices|doubleclick)[./])'
        r'[^\s"\'<>]+?\.(?:jpg|jpeg|png|webp|gif|avif)(?:\?[^\s"\'<>]*)?',
        re.IGNORECASE
    )

    def __init__(self, config: GoogleImageCrawlerConfig):
        self.config = config

    def crawl_with_session(self, keyword: str, session: _BrowserSession) -> list[str]:
        """Crawl satu keyword menggunakan session yang sudah ada (reuse browser)."""
        return self._run_keyword(keyword, session)

    def crawl_fresh(self, keyword: str) -> list[str]:
        """Crawl satu keyword dengan browser baru (digunakan jika reuse=False)."""
        from playwright.sync_api import sync_playwright
        time.sleep(random.uniform(self.config.stagger_delay_min, self.config.stagger_delay_max))
        with sync_playwright() as pw:
            session = _BrowserSession(pw, self.config)
            try:
                return self._run_keyword(keyword, session)
            finally:
                session.close()

    def _run_keyword(self, keyword: str, session: _BrowserSession) -> list[str]:
        """Inti crawl: buka halaman, intercept + scroll + (opsional) klik."""
        intercepted_urls: list[str] = []
        intercept_lock = threading.Lock()

        page = session.context.new_page()
        page.set_default_timeout(self.config.page_timeout)

        # --- Block resource berat yang tidak dibutuhkan ---
        # Kita butuh JS (untuk scroll dan DOM) tapi tidak butuh gambar/font/media
        def handle_route(route):
            req = route.request
            resource_type = req.resource_type
            if resource_type in ("image", "font", "media", "stylesheet"):
                # Kecuali: jangan blokir jika URL adalah gambar target (dari network intercept)
                route.abort()
            else:
                route.continue_()

        page.route("**/*", handle_route)

        # --- Network intercept: tangkap URL gambar full-res dari traffic ---
        def on_response(response):
            url = response.url
            content_type = response.headers.get("content-type", "")
            # Terima jika content-type gambar ATAU URL punya ekstensi gambar
            if (
                "image/" in content_type
                and "image/svg" not in content_type
                and "gstatic.com" not in url
                and "encrypted-tbn" not in url
                and _VALIDATOR.is_valid(url)
            ):
                with intercept_lock:
                    intercepted_urls.append(url)

        page.on("response", on_response)

        try:
            search_url = self._GOOGLE_IMAGES_URL.format(query=quote_plus(keyword))
            page.goto(search_url, wait_until="load", timeout=self.config.page_timeout)

            # Karena gambar diblokir di level route, intercept dari response
            # akan menangkap URL dari XHR/fetch yang Google lakukan saat buka panel detail.
            # Tapi kita tetap perlu dismiss consent dulu.
            if not session._consent_done:
                self._dismiss_consent(page)
                session._consent_done = True

            thumbnail_appeared = self._wait_for_thumbnails(page, keyword)
            if not thumbnail_appeared:
                logger.warning("[%s] Thumbnail tidak muncul, skip", keyword)
                page.close()
                return []

            # Scroll untuk trigger lazy-load (Google fetch URL gambar via XHR saat thumbnail muncul)
            self._scroll_to_load(page, keyword)

            # Kumpulkan dari intercept sejauh ini
            with intercept_lock:
                urls = list(dict.fromkeys(intercepted_urls))

            logger.info("[%s] Intercept: %d URL", keyword, len(urls))

            # Jika intercept kurang, fallback klik thumbnail (lebih sedikit dari sebelumnya)
            if len(urls) < self.config.max_urls_per_keyword:
                needed = self.config.max_urls_per_keyword - len(urls)
                click_limit = min(self.config.thumbnails_to_click, needed)
                if click_limit > 0:
                    click_urls = self._extract_via_clicks(page, keyword, click_limit)
                    existing = set(urls)
                    for u in click_urls:
                        if u not in existing:
                            urls.append(u)

            # Jika masih kurang, fallback DOM
            if len(urls) < 20:
                dom_urls = self._extract_from_dom(page)
                existing = set(urls)
                for u in dom_urls:
                    if u not in existing:
                        urls.append(u)

            urls = list(dict.fromkeys(urls))[: self.config.max_urls_per_keyword]

        except Exception as e:
            logger.error("[%s] Error saat crawl: %s", keyword, e)
            urls = []
        finally:
            try:
                page.close()
            except Exception:
                pass

        return urls

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _wait_for_thumbnails(self, page, keyword: str) -> bool:
        for sel in self._THUMBNAIL_SELECTORS:
            try:
                page.wait_for_selector(sel, timeout=self.config.element_wait_timeout, state="visible")
                return True
            except Exception:
                continue
        return False

    def _scroll_to_load(self, page, keyword: str) -> None:
        """Scroll bertahap. Delay lebih singkat dari v1."""
        for i in range(self.config.max_scrolls):
            prev_height = page.evaluate("document.body.scrollHeight")
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(random.uniform(self.config.scroll_delay_min, self.config.scroll_delay_max))
            self._click_show_more(page)
            new_height = page.evaluate("document.body.scrollHeight")
            if new_height == prev_height and i > 2:
                logger.debug("[%s] Halaman berhenti tumbuh pada scroll %d", keyword, i + 1)
                break

    def _extract_via_clicks(self, page, keyword: str, limit: int) -> list[str]:
        """Klik thumbnail sebagai fallback. Limit jauh lebih kecil dari v1."""
        urls: list[str] = []
        thumbnails = self._get_thumbnails(page)
        to_click = min(len(thumbnails), limit)

        for idx in range(to_click):
            try:
                thumb = thumbnails[idx]
                thumb.scroll_into_view_if_needed()
                time.sleep(0.05)
                thumb.click(timeout=3_000)
                src = self._read_detail_panel(page)
                if src and _VALIDATOR.is_valid(src):
                    urls.append(src)
            except Exception as e:
                logger.debug("[%s] Klik %d gagal: %s", keyword, idx + 1, e)
            time.sleep(random.uniform(self.config.click_delay_min, self.config.click_delay_max))

        return list(dict.fromkeys(urls))

    def _get_thumbnails(self, page) -> list:
        for sel in self._THUMBNAIL_SELECTORS:
            elements = page.query_selector_all(sel)
            if len(elements) > 5:
                return elements
        return page.query_selector_all("div[role='listitem'] img, div[tabindex='0'] img") or []

    def _read_detail_panel(self, page, retries: int = 2, wait_ms: int = 300) -> Optional[str]:
        for attempt in range(retries):
            try:
                src = page.evaluate(self._DETAIL_PANEL_JS)
                if src:
                    return src
                time.sleep(wait_ms / 1000)
            except Exception:
                time.sleep(wait_ms / 1000)
        return None

    def _extract_from_dom(self, page) -> list[str]:
        urls: list[str] = []
        try:
            all_srcs: list[str] = page.evaluate(self._ALL_IMG_JS)
            for src in all_srcs:
                if _VALIDATOR.is_valid(src):
                    urls.append(src)
        except Exception as e:
            logger.debug("DOM fallback error: %s", e)
        return list(dict.fromkeys(urls))

    def _dismiss_consent(self, page) -> None:
        for sel in [
            "button:has-text('Terima semua')",
            "button:has-text('Accept all')",
            "button:has-text('I agree')",
            "#L2AGLb",
            "form button:last-child",
        ]:
            try:
                btn = page.query_selector(sel)
                if btn and btn.is_visible():
                    btn.click()
                    time.sleep(1.0)
                    return
            except Exception:
                continue

    def _click_show_more(self, page) -> None:
        for sel in [
            "input[value='Show more results']",
            "a:has-text('Show more')",
            "div[jsname='xPjF4b']",
        ]:
            try:
                btn = page.query_selector(sel)
                if btn and btn.is_visible():
                    btn.click()
                    time.sleep(1.0)
                    return
            except Exception:
                continue


# ---------------------------------------------------------------------------
# GoogleImageCrawler — public API
# ---------------------------------------------------------------------------

class GoogleImageCrawler:
    """
    Scraper image URL dari Google Images.

    Perubahan performa vs v1:
    - Strategi intercept network (bukan klik 200x) → hemat 2–4 menit/keyword
    - keyword_workers=1 default → 2 Chromium aktif (bukan 4) → hemat ~800MB RAM
    - Browser di-reuse antar keyword dalam satu file → hemat ~2s init/keyword
    - Resource blocking (gambar, font, stylesheet) → page load lebih cepat
    - Timeout diperkecil → gagal cepat, tidak nunggu 60s
    """

    def __init__(self, config: Optional[GoogleImageCrawlerConfig] = None):
        self.config = config or GoogleImageCrawlerConfig()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()
        self._setup_logging()

    def run(self) -> dict[str, Path]:
        self._check_playwright()
        keyword_files = self._resolve_keyword_files()
        if not keyword_files:
            logger.warning("Tidak ada file keyword di %s", self.config.keyword_dir)
            return {}

        logger.info(
            "Memulai: %d file, file_workers=%d, keyword_workers=%d, reuse_browser=%s",
            len(keyword_files), self.config.file_workers,
            self.config.keyword_workers, self.config.reuse_browser,
        )

        results: dict[str, Path] = {}
        with ThreadPoolExecutor(
            max_workers=self.config.file_workers,
            thread_name_prefix="file-worker",
        ) as ex:
            future_to_file = {ex.submit(self._process_keyword_file, f): f for f in keyword_files}
            for future in as_completed(future_to_file):
                kw_file = future_to_file[future]
                try:
                    out_path = future.result()
                    if out_path:
                        results[kw_file.name] = out_path
                except Exception as exc:
                    logger.error("[%s] Gagal: %s", kw_file.name, exc)

        logger.info("Selesai. %d/%d file berhasil.", len(results), len(keyword_files))
        return results

    def _process_keyword_file(self, kw_file: Path) -> Optional[Path]:
        from playwright.sync_api import sync_playwright

        keywords = self._read_keywords(kw_file)
        if not keywords:
            logger.warning("[%s] Kosong, dilewati.", kw_file.name)
            return None

        output_path = self._resolve_output_path(kw_file)
        logger.info("[%s] %d keyword → %s", kw_file.name, len(keywords), output_path)

        worker = _KeywordWorker(self.config)
        collected: dict[str, list[str]] = {}

        if self.config.reuse_browser and self.config.keyword_workers == 1:
            # Mode hemat RAM: 1 browser per file, keyword diproses berurutan
            # Chromium hanya di-init sekali per file, bukan per keyword
            with sync_playwright() as pw:
                session = _BrowserSession(pw, self.config)
                try:
                    for kw in keywords:
                        time.sleep(random.uniform(
                            self.config.stagger_delay_min,
                            self.config.stagger_delay_max,
                        ))
                        try:
                            urls = worker.crawl_with_session(kw, session)
                            collected[kw] = urls
                            logger.info("[%s] ✓ '%s' → %d URL", kw_file.name, kw, len(urls))
                        except Exception as exc:
                            logger.error("[%s] ✗ '%s': %s", kw_file.name, kw, exc)
                            collected[kw] = []
                finally:
                    session.close()
        else:
            # Mode paralel: keyword_workers browser sekaligus (lebih cepat, lebih boros RAM)
            lock = threading.Lock()
            with ThreadPoolExecutor(
                max_workers=self.config.keyword_workers,
                thread_name_prefix=f"kw-{kw_file.stem[:12]}",
            ) as ex:
                future_to_kw = {ex.submit(worker.crawl_fresh, kw): kw for kw in keywords}
                for future in as_completed(future_to_kw):
                    kw = future_to_kw[future]
                    try:
                        urls = future.result()
                        with lock:
                            collected[kw] = urls
                        logger.info("[%s] ✓ '%s' → %d URL", kw_file.name, kw, len(urls))
                    except Exception as exc:
                        logger.error("[%s] ✗ '%s': %s", kw_file.name, kw, exc)
                        with lock:
                            collected[kw] = []

        all_urls: list[str] = []
        for kw in keywords:
            all_urls.extend(collected.get(kw, []))
        unique_urls = list(dict.fromkeys(all_urls))

        with self._write_lock:
            self._save_urls(unique_urls, output_path)

        logger.info("[%s] %d URL unik disimpan.", kw_file.name, len(unique_urls))
        return output_path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_keyword_files(self) -> list[Path]:
        kw_dir = self.config.keyword_dir
        if self.config.keyword_files:
            return [kw_dir / f for f in self.config.keyword_files if (kw_dir / f).exists()]
        return sorted(kw_dir.glob("keyword_*.txt"))

    def _read_keywords(self, path: Path) -> list[str]:
        lines = path.read_text(encoding="utf-8").splitlines()
        return [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]

    def _resolve_output_path(self, kw_file: Path) -> Path:
        if self.config.keyword_to_output_map and kw_file.name in self.config.keyword_to_output_map:
            return self.config.output_dir / self.config.keyword_to_output_map[kw_file.name]
        return self.config.output_dir / kw_file.name.replace("keyword_", "urls_", 1)

    def _save_urls(self, urls: list[str], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(urls), encoding="utf-8")
        logger.info("Saved %d URLs → %s", len(urls), output_path)

    @staticmethod
    def _check_playwright() -> None:
        try:
            from playwright.sync_api import sync_playwright  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Playwright belum terinstall:\n"
                "  pip install playwright\n"
                "  playwright install chromium"
            ) from exc

    @staticmethod
    def _setup_logging() -> None:
        if not logging.root.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(levelname)s] %(threadName)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )