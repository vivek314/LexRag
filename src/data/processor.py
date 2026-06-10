# processor.py — Extract and clean text from PDFs, preserving page boundaries.
from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

try:
    import fitz  # PyMuPDF
    _FITZ_AVAILABLE = True
except (ImportError, OSError):
    _FITZ_AVAILABLE = False
import yaml

logger = logging.getLogger(__name__)

@dataclass
class PageContent:
    page_number: int    # 1-indexed — fitz is 0-indexed, we'll convert
    text: str
    char_count: int
    has_table: bool

@dataclass
class Document:
    doc_id: str
    title: str
    source: str
    domain: str
    date: str
    local_path: str
    num_pages: int
    pages: list[PageContent]
    metadata: dict = field(default_factory=dict)

def extract_page(page: fitz.Page) -> PageContent:
    # Get page text as a dict — gives us blocks with bounding boxes
    page_dict = page.get_text("dict")
    page_height = page.rect.height

    # Header zone = top 8%, footer zone = bottom 8%
    header_cutoff = page_height * 0.08
    footer_cutoff = page_height * 0.92

    lines_out = []
    has_table = False
    tab_line_count = 0

    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:   # 0 = text, 1 = image — skip images
            continue

        block_y0 = block["bbox"][1]  # top y-coordinate
        block_y1 = block["bbox"][3]  # bottom y-coordinate

        # Skip if block is entirely in header or footer zone
        if block_y1 <= header_cutoff or block_y0 >= footer_cutoff:
            continue

        for line in block.get("lines", []):
            line_text = " ".join(
                span["text"] for span in line.get("spans", [])
            ).strip()

            if not line_text:
                continue

            # Detect table-like lines (lots of numbers with big gaps)
            if re.search(r"\d+\s{3,}\d+", line_text):
                tab_line_count += 1

            lines_out.append(line_text)

    if tab_line_count >= 3:
        has_table = True

    text = "\n".join(lines_out).strip()

    return PageContent(
        page_number=page.number + 1,  # convert 0-indexed → 1-indexed
        text=text,
        char_count=len(text),
        has_table=has_table,
    )

def process_pdf(local_path: str, meta: dict, cfg: dict) -> Optional[Document]:
    if not _FITZ_AVAILABLE:
        raise RuntimeError("PyMuPDF is not available in this environment. PDF ingestion is disabled.")
    path = Path(local_path)
    if not path.exists():
        logger.warning(f"File not found: {local_path}")
        return None

    try:
        pdf = fitz.open(str(path))
    except Exception as e:
        logger.error(f"Cannot open {path.name}: {e}")
        return None

    num_pages = len(pdf)

    # Guard: skip too-short or too-long PDFs
    if num_pages < cfg["data"]["download"]["min_pages"]:
        logger.info(f"Skipping (too short): {path.name}")
        pdf.close()
        return None

    pages = []
    for i in range(min(num_pages, cfg["data"]["download"]["max_pages"])):
        page_content = extract_page(pdf[i])
        if page_content.char_count < 50:   # skip blank pages
            continue
        pages.append(page_content)

    pdf.close()

    if not pages:
        logger.warning(f"No readable pages: {path.name}")
        return None

    logger.info(f"{path.name}: {len(pages)} pages extracted")

    return Document(
        doc_id=meta.get("doc_id", path.stem),
        title=meta.get("title", path.stem),
        source=meta.get("source", ""),
        domain=meta.get("domain", "unknown"),
        date=meta.get("date", "unknown"),
        local_path=str(path),
        num_pages=len(pages),
        pages=pages,
        metadata=meta,
    )


def process_corpus(raw_dir: str, output_dir: str, cfg: dict) -> list[Document]:
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load manifest for metadata (title, date, source per PDF)
    manifest_path = Path(cfg["data"]["manifest_file"])
    manifest = []
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

    # Build a lookup: file path → metadata dict
    meta_by_path = {m["local_path"].replace("\\", "/"): m for m in manifest}

    pdf_files = list(raw_path.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDFs in {raw_dir}")

    documents = []
    for pdf_path in pdf_files:
        meta = meta_by_path.get(str(pdf_path).replace("\\", "/"), {
            "doc_id": pdf_path.stem,
            "title": pdf_path.stem,
            "source": "",
            "domain": "unknown",
            "date": "unknown",
        })
        out_file = out_path / f"{pdf_path.stem}.json"

        # Incremental: skip if already processed
        if out_file.exists():
            logger.info(f"Already processed: {pdf_path.name}")
            with open(out_file, encoding="utf-8") as f:
                raw = json.load(f)
            pages = [PageContent(**p) for p in raw.get("pages", [])]
            raw["pages"] = pages
            documents.append(Document(**raw))
            continue

        doc = process_pdf(str(pdf_path), meta, cfg)
        if doc is None:
            continue

        # Save to JSON
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(asdict(doc), f, indent=2, ensure_ascii=False)

        documents.append(doc)

    logger.info(f"Done: {len(documents)} documents ready")
    return documents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw")
    parser.add_argument("--output", default="data/processed")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    docs = process_corpus(args.input, args.output, cfg)
    print(f"\nProcessed {len(docs)} documents")
    for doc in docs:
        print(f"  {doc.doc_id} | {doc.num_pages} pages")

if __name__ == "__main__":
    main()