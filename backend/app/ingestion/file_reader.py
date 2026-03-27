"""Universal file reader — extracts text content from any file type."""

from __future__ import annotations

import base64
import csv
import io
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Extensions that can be sent as images to GPT-4o vision
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"}

# Extensions treated as plain text
TEXT_EXTENSIONS = {
    ".txt", ".md", ".rst", ".log", ".ini", ".cfg", ".conf",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".swift",
    ".kt", ".scala", ".r", ".sql", ".sh", ".bash", ".zsh",
    ".yaml", ".yml", ".toml", ".xml", ".html", ".css", ".scss",
    ".env", ".gitignore", ".dockerfile",
}


def read_file(path: Path) -> dict:
    """Read a file and return its content in a structured format.

    Returns:
        {
            "filename": str,
            "type": "text" | "image",
            "content": str,          # for text files
            "image_base64": str,     # for image files
            "mime_type": str,        # for image files
        }
    """
    ext = path.suffix.lower()
    filename = path.name

    try:
        if ext == ".pdf":
            try:
                return {"filename": filename, "type": "text", "content": _read_pdf(path)}
            except Exception:
                return {"filename": filename, "type": "text", "content": _read_text(path)}

        if ext in (".docx", ".doc"):
            try:
                return {"filename": filename, "type": "text", "content": _read_docx(path)}
            except Exception:
                # Fallback: file might be plain text with .docx extension
                return {"filename": filename, "type": "text", "content": _read_text(path)}

        if ext in (".xlsx", ".xls"):
            try:
                return {"filename": filename, "type": "text", "content": _read_excel(path)}
            except Exception:
                return {"filename": filename, "type": "text", "content": _read_text(path)}

        if ext in (".pptx", ".ppt"):
            try:
                return {"filename": filename, "type": "text", "content": _read_pptx(path)}
            except Exception:
                return {"filename": filename, "type": "text", "content": _read_text(path)}

        if ext == ".csv":
            return {"filename": filename, "type": "text", "content": _read_csv(path)}

        if ext == ".json":
            return {"filename": filename, "type": "text", "content": _read_json(path)}

        if ext in IMAGE_EXTENSIONS:
            return _read_image(path, filename)

        if ext in TEXT_EXTENSIONS or _is_likely_text(path):
            return {"filename": filename, "type": "text", "content": _read_text(path)}

        # Fallback: try reading as text
        return {"filename": filename, "type": "text", "content": _read_text(path)}

    except Exception as e:
        logger.error("Failed to read %s: %s", path, e)
        return {"filename": filename, "type": "text", "content": f"[Error reading file: {e}]"}


def _read_text(path: Path) -> str:
    """Read as plain text with encoding fallback."""
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return path.read_text(encoding=encoding)
        except (UnicodeDecodeError, ValueError):
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def _read_pdf(path: Path) -> str:
    from PyPDF2 import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"--- Page {i + 1} ---\n{text}")
    return "\n\n".join(pages) if pages else "[Empty PDF]"


def _read_docx(path: Path) -> str:
    from docx import Document

    doc = Document(str(path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    # Also read tables
    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                paragraphs.append(" | ".join(cells))
    return "\n\n".join(paragraphs) if paragraphs else "[Empty document]"


def _read_excel(path: Path) -> str:
    from openpyxl import load_workbook

    wb = load_workbook(str(path), read_only=True, data_only=True)
    sheets = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        rows = []
        for row in ws.iter_rows(values_only=True):
            values = [str(v) if v is not None else "" for v in row]
            if any(v.strip() for v in values):
                rows.append(" | ".join(values))
        if rows:
            sheets.append(f"--- Sheet: {sheet_name} ---\n" + "\n".join(rows))
    wb.close()
    return "\n\n".join(sheets) if sheets else "[Empty spreadsheet]"


def _read_pptx(path: Path) -> str:
    from pptx import Presentation

    prs = Presentation(str(path))
    slides = []
    for i, slide in enumerate(prs.slides):
        texts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for paragraph in shape.text_frame.paragraphs:
                    text = paragraph.text.strip()
                    if text:
                        texts.append(text)
        if texts:
            slides.append(f"--- Slide {i + 1} ---\n" + "\n".join(texts))
    return "\n\n".join(slides) if slides else "[Empty presentation]"


def _read_csv(path: Path) -> str:
    text = _read_text(path)
    reader = csv.reader(io.StringIO(text))
    rows = []
    for row in reader:
        rows.append(" | ".join(row))
    return "\n".join(rows[:500])  # Cap at 500 rows


def _read_json(path: Path) -> str:
    text = _read_text(path)
    try:
        data = json.loads(text)
        return json.dumps(data, indent=2, ensure_ascii=False)[:10000]  # Cap size
    except json.JSONDecodeError:
        return text[:10000]


def _read_image(path: Path, filename: str) -> dict:
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("utf-8")
    ext = path.suffix.lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".svg": "image/svg+xml",
    }
    return {
        "filename": filename,
        "type": "image",
        "image_base64": b64,
        "mime_type": mime_map.get(ext, "image/png"),
        "content": "",
    }


def _is_likely_text(path: Path) -> bool:
    """Heuristic: check if file looks like text by reading first bytes."""
    try:
        chunk = path.read_bytes()[:1024]
        # If most bytes are printable ASCII or common UTF-8, it's text
        text_chars = sum(1 for b in chunk if b < 128 or b > 191)
        return text_chars / max(len(chunk), 1) > 0.85
    except Exception:
        return False
