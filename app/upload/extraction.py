"""
Text extraction for uploaded receipts — PDF embedded-text layer first, OCR
fallback for scanned pages and plain images. Everything runs locally:
PyMuPDF for PDF parsing + page rasterization, Tesseract (via pytesseract)
for OCR. No paid/cloud OCR service.

Wrapped in a hard wall-clock timeout per file (ProcessingTimeout) so one
malformed or huge file can't hang a request indefinitely.
"""

import concurrent.futures
import io

import pymupdf as fitz  # PyMuPDF — "pymupdf" is the current recommended import name
from PIL import Image


class ExtractionError(RuntimeError):
    """Unreadable/unsupported file content — a clean, user-facing message,
    not a raw library exception."""


class ProcessingTimeout(RuntimeError):
    pass


class OCRUnavailable(RuntimeError):
    """Tesseract isn't installed on this machine. Distinct from
    ExtractionError so the API can give a specific, actionable message
    ("ask your administrator to install tesseract-ocr") instead of implying
    the file itself is bad."""


def _ocr_image(image: Image.Image) -> str:
    import pytesseract

    try:
        return pytesseract.image_to_string(image).strip()
    except pytesseract.TesseractNotFoundError as exc:
        raise OCRUnavailable(
            "OCR engine (tesseract) is not installed on the server. "
            "PDF pages with an embedded text layer still work; scanned "
            "pages and image files need tesseract-ocr installed."
        ) from exc


def _extract_pdf(file_bytes: bytes, max_pages: int) -> list[dict]:
    """Returns one dict per page: {page_number, text, used_ocr}."""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:  # noqa: BLE001 — PyMuPDF raises its own exception types
        raise ExtractionError(f"Could not open this file as a PDF: {exc}") from exc

    if doc.page_count == 0:
        raise ExtractionError("This PDF has no pages.")
    if doc.page_count > max_pages:
        raise ExtractionError(
            f"This PDF has {doc.page_count} pages, which exceeds the {max_pages}-page limit."
        )

    pages = []
    ocr_unavailable_seen = False
    for i, page in enumerate(doc):
        embedded_text = page.get_text().strip()
        if embedded_text:
            pages.append({"page_number": i + 1, "text": embedded_text, "used_ocr": False})
            continue

        # No embedded text layer on this page — it's a scanned image, fall
        # back to OCR by rasterizing the page.
        if ocr_unavailable_seen:
            pages.append({"page_number": i + 1, "text": "", "used_ocr": False,
                          "warning": "No embedded text and OCR is unavailable."})
            continue
        try:
            pix = page.get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            ocr_text = _ocr_image(img)
            pages.append({"page_number": i + 1, "text": ocr_text, "used_ocr": True})
        except OCRUnavailable:
            ocr_unavailable_seen = True
            pages.append({"page_number": i + 1, "text": "", "used_ocr": False,
                          "warning": "No embedded text and OCR is unavailable."})

    return pages


def _extract_image(file_bytes: bytes) -> list[dict]:
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.load()  # force-decode now, so a corrupt image fails here, not later
    except Exception as exc:  # noqa: BLE001 — Pillow raises various exception types for bad images
        raise ExtractionError(f"Could not read this image file: {exc}") from exc

    text = _ocr_image(img)
    return [{"page_number": 1, "text": text, "used_ocr": True}]


def extract_pages(file_bytes: bytes, content_type: str, max_pages: int, timeout_s: int) -> list[dict]:
    """content_type: 'application/pdf', 'image/jpeg', or 'image/png'.
    Raises ExtractionError / OCRUnavailable / ProcessingTimeout."""

    def _run():
        if content_type == "application/pdf":
            return _extract_pdf(file_bytes, max_pages)
        return _extract_image(file_bytes)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run)
        try:
            return future.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError as exc:
            raise ProcessingTimeout(
                f"Processing took longer than {timeout_s}s and was stopped."
            ) from exc
