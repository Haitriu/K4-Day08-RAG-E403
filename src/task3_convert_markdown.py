"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install "markitdown[pdf]"
    # Lưu ý: cần extra [pdf] để convert được file PDF. Chỉ "pip install markitdown"
    # (không có extra) sẽ báo MissingDependencyException khi convert PDF, dù JSON/DOCX
    # vẫn convert bình thường.

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import json
import re
from pathlib import Path

from markitdown import MarkItDown

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def _extract_legacy_doc_text(filepath: Path) -> str:
    """Fallback cho file .doc nhị phân cũ (không phải .docx) mà MarkItDown không hỗ trợ.

    File .doc (OLE Compound Document) lưu văn bản dạng UTF-16LE xen kẽ control bytes.
    Decode toàn bộ file như UTF-16LE rồi lọc các đoạn ký tự in được liên tục
    (bao gồm dấu tiếng Việt) — kỹ thuật "strings" cổ điển, không cần MS Word/LibreOffice.
    """
    data = filepath.read_bytes()
    text = data.decode("utf-16-le", errors="ignore")
    runs = re.findall(r"[ -~ -ỹ]{4,}", text)
    return "\n".join(runs)


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    md = MarkItDown()

    for filepath in legal_dir.iterdir():
        if filepath.suffix.lower() in (".pdf", ".docx", ".doc"):
            print(f"Converting: {filepath.name}")
            try:
                text_content = md.convert(str(filepath)).text_content
            except Exception as e:
                if filepath.suffix.lower() == ".doc":
                    print(f"  ⚠ MarkItDown không hỗ trợ .doc cũ ({e.__class__.__name__}), dùng fallback extractor")
                    text_content = _extract_legacy_doc_text(filepath)
                else:
                    print(f"  ✗ Bỏ qua ({e.__class__.__name__}): {e}")
                    continue
            output_path = output_dir / f"{filepath.stem}.md"
            output_path.write_text(text_content, encoding="utf-8")
            print(f"  ✓ Saved: {output_path}")


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filepath in news_dir.iterdir():
        if filepath.suffix.lower() == ".json":
            print(f"Converting: {filepath.name}")
            data = json.loads(filepath.read_text(encoding="utf-8"))
            output_path = output_dir / f"{filepath.stem}.md"

            # Thêm metadata header
            header = f"# {data.get('title', 'Unknown')}\n\n"
            header += f"**Source:** {data.get('url', 'N/A')}\n"
            header += f"**Crawled:** {data.get('crawl_date', 'N/A')}\n\n---\n\n"

            content = header + data.get("content", "")
            output_path.write_text(content, encoding="utf-8")
            print(f"  ✓ Saved: {output_path}")


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    print("\n✓ Done! Output tại:", OUTPUT_DIR)


if __name__ == "__main__":
    convert_all()
