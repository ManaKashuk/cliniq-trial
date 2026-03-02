#!/usr/bin/env python3
"""
Simple ingestion utility:
- Copies or saves given text/PDFs into data/sops
- (Optional) Converts PDFs to text for quick inspection
This MVP relies on on-the-fly TF‑IDF indexing in app.py, so no pre-index file is produced.
"""
import shutil
from pathlib import Path
import argparse
from pypdf import PdfReader

DATA_DIR = Path(__file__).parent / "data" / "sops"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def add_file(path: Path):
    dest = DATA_DIR / path.name
    shutil.copy2(path, dest)
    print(f"Added {dest}")

def pdf_preview(path: Path):
    try:
        reader = PdfReader(str(path))
        text = "\n".join([(p.extract_text() or "") for p in reader.pages[:2]])
        print("--- PDF preview (first ~2 pages) ---")
        print(text[:2000])
    except Exception as e:
        print(f"(Preview failed) {e}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="Paths to .txt or .pdf files to add")
    ap.add_argument("--preview", action="store_true", help="Print first pages of PDFs")
    args = ap.parse_args()

    for raw in args.inputs:
        p = Path(raw)
        if not p.exists():
            print(f"Skip missing: {p}")
            continue
        add_file(p)
        if args.preview and p.suffix.lower() == ".pdf":
            pdf_preview(p)
