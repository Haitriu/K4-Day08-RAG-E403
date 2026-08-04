"""
Task 2 — Crawl bài viết/hướng dẫn về luật lao động.

Chủ đề dữ liệu: luật lao động Việt Nam (Bộ luật Lao động 2019, BHXH, BHTN,
hợp đồng lao động, tiền lương, an toàn lao động, sa thải/kỷ luật, nghỉ phép,
thai sản, lao động nước ngoài...).

Hướng dẫn gốc đề nghị dùng Crawl4AI, nhưng thư viện này phụ thuộc `litellm`
cần Rust/Cargo để build trên môi trường này. Do đó dùng `requests` +
`BeautifulSoup` (đã có sẵn, không cần compiler) để crawl thật nội dung.

Cài đặt:
    pip install requests beautifulsoup4 lxml

Output: mỗi bài viết lưu 1 file JSON trong data/landing/news/ với format:
    {
        "url": str,
        "title": str,
        "published_date": str | None,
        "crawl_date": str (ISO date),
        "content": str
    }
"""

import json
import re
from datetime import date
from pathlib import Path

import requests
from bs4 import BeautifulSoup

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

# 7 bài viết hướng dẫn về luật lao động Việt Nam (chủ đề của K4 variant)
ARTICLES = [
    {
        "slug": "che-do-nghi-phep-2026",
        "url": "https://giadinh.suckhoedoisong.vn/quy-dinh-ve-che-do-nghi-phep-2026-theo-luat-lao-dong-moi-nhat-nguoi-lao-dong-can-biet-172260316153824736.htm",
    },
    {
        "slug": "cach-tinh-luong-lam-them-gio",
        "url": "https://aztax.com.vn/cach-tinh-tien-luong-lam-them-gio-moi-nhat/",
    },
    {
        "slug": "quy-dinh-thoi-gian-thu-viec",
        "url": "https://fastwork.vn/quy-dinh-ve-thoi-gian-thu-viec/",
    },
    {
        "slug": "che-do-thai-san-2026",
        "url": "https://ebh.vn/nghiep-vu-tong-hop/muc-huong-che-do-thai-san-nam-2026",
    },
    {
        "slug": "sa-thai-trai-phap-luat-quyen-loi",
        "url": "https://hanoiluat.vn/lam-gi-khi-nguoi-lao-dong-bi-sa-thai-trai-phap-luat",
    },
    {
        "slug": "mau-hop-dong-lao-dong-2026",
        "url": "https://amis.misa.vn/35431/mau-hop-dong-lao-dong/",
    },
    {
        "slug": "thu-tuc-bao-hiem-that-nghiep",
        "url": "https://ebh.vn/bao-hiem-that-nghiep/infographic-5-buoc-nhan-tro-cap-that-nghiep-online",
    },
]


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _extract_title(soup: BeautifulSoup) -> str:
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)
    return "Untitled"


def _extract_main_text(soup: BeautifulSoup) -> str:
    """Lấy nội dung chính: ưu tiên <article>, fallback sang toàn bộ <p>."""
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()

    container = soup.find("article") or soup.find(class_=re.compile(r"(content|article|post)", re.I))
    paragraphs = (container or soup).find_all("p")

    text = "\n".join(
        p.get_text(" ", strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 30
    )
    return text.strip()


def crawl_article(url: str) -> dict:
    """Crawl một bài viết bằng requests + BeautifulSoup, trả về dict metadata + content."""
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")
    title = _extract_title(soup)
    content = _extract_main_text(soup)

    if len(content) < 100:
        raise ValueError(f"Nội dung crawl quá ngắn ({len(content)} ký tự) — có thể trang dùng JS render")

    return {
        "url": url,
        "title": title,
        "published_date": None,
        "crawl_date": date.today().isoformat(),
        "content": content,
    }


def crawl_all():
    """Crawl toàn bộ bài viết trong ARTICLES và lưu JSON vào DATA_DIR."""
    setup_directory()
    ok, failed = 0, []

    for i, item in enumerate(ARTICLES, 1):
        slug, url = item["slug"], item["url"]
        print(f"[{i}/{len(ARTICLES)}] Crawling: {url}")
        try:
            article = crawl_article(url)
        except Exception as e:
            print(f"  ✗ Lỗi: {e}")
            failed.append(url)
            continue

        filepath = DATA_DIR / f"{slug}.json"
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  ✓ Saved: {filepath.name} ({len(article['content'])} ký tự)")
        ok += 1

    print(f"\nHoàn tất: {ok}/{len(ARTICLES)} bài crawl thành công.")
    if failed:
        print("Thất bại:", *failed, sep="\n  - ")


if __name__ == "__main__":
    crawl_all()
