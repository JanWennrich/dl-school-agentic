#!/usr/bin/env python3
"""
Chunk the PDFs in pdfs/ into a corpus in the handbook-corpus.json format.

Each chunk carries heading (section), source file, chunk number and page.
Headings are detected by font size relative to the body text.

Usage:
  uv run --with pymupdf python3 data/pdfs-to-corpus.py
  uv run --with pymupdf python3 data/pdfs-to-corpus.py --pdf-dir pdfs --out data/my-corpus.json
"""

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent

HEADING_FACTOR = 1.3  # line counts as heading if its font is this much larger than body text
MIN_HEADING_LETTERS = 4
MAX_HEADING_CHARS = 80
TARGET_CHARS = 800
MIN_CHARS = 150

LIGATURE = re.compile(r"([ﬀ-ﬆ])\s+")
PRINT_MARKS = re.compile(r"\.indd\b|\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}")


def clean(text):
    text = LIGATURE.sub(r"\1", text)  # "Konﬂ ikt" -> "Konﬂikt"
    text = re.sub(r"\u00ad\s*$", "-", text)  # soft hyphen at line end -> rejoined by join_lines
    text = re.sub(r"\u00ad\s*", "", text)  # soft hyphen mid-line: "Geschwin­ digkeit"
    text = unicodedata.normalize("NFKC", text)  # "ﬂ" -> "fl", NBSP -> space
    text = re.sub(r"[\x00-\x08\x0b-\x1f]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def doc_slug(path):
    """heat-824968091319-regel.pdf -> heat"""
    return re.split(r"-\d{6,}", path.stem)[0]


def line_text(line):
    """Join spans of a line, dropping fake-bold duplicates (same text, same position)."""
    parts, seen = [], set()
    for span in line["spans"]:
        key = (span["text"], round(span["origin"][0]), round(span["origin"][1]))
        if key in seen:
            continue
        seen.add(key)
        parts.append(span["text"])
    return "".join(parts)


def line_size(line):
    return max((s["size"] for s in line["spans"] if s["text"].strip()), default=0)


def body_size(doc):
    sizes = Counter()
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    sizes[round(span["size"])] += len(span["text"].strip())
    return sizes.most_common(1)[0][0]


def is_heading(text, size, body):
    letters = sum(c.isalpha() for c in text)
    return (
        size >= body * HEADING_FACTOR
        and letters >= MIN_HEADING_LETTERS
        and len(text) <= MAX_HEADING_CHARS
    )


def reading_order(page):
    """Blocks sorted left half before right half, then top to bottom.

    The content stream order is unreliable: some pages store headings after their body text.
    """
    middle = page.rect.width / 2
    blocks = page.get_text("dict")["blocks"]
    return sorted(blocks, key=lambda b: (b["bbox"][0] >= middle, b["bbox"][1]))


def extract_elements(doc):
    """Yield ("heading" | "para", text, page) in reading order."""
    body = body_size(doc)
    for page in doc:
        page_no = page.number + 1
        for block in reading_order(page):
            heading, para, prev = [], [], None
            for line in block.get("lines", []):
                raw = line_text(line)
                text = clean(raw)
                if not text or text == prev or PRINT_MARKS.search(text):
                    continue
                if re.fullmatch(r"[\d\W]+", text):  # page numbers, step markers
                    continue
                prev = text
                if is_heading(text, line_size(line), body):
                    if para:
                        yield "para", join_lines(para), page_no
                        para = []
                    heading.append(text)
                else:
                    if heading:
                        yield "heading", " ".join(heading), page_no
                        heading = []
                    para.append(text)
            if heading:
                yield "heading", " ".join(heading), page_no
            if para:
                yield "para", join_lines(para), page_no


def join_lines(lines):
    """Join lines of a paragraph, undoing end-of-line hyphenation."""
    out = ""
    for line in lines:
        if out.endswith("-") and line[:1].islower():
            out = out[:-1] + line
        else:
            out = f"{out} {line}" if out else line
    return out


def split_long(text):
    """Split a paragraph longer than TARGET_CHARS at sentence boundaries."""
    if len(text) <= TARGET_CHARS:
        return [text]
    pieces, current = [], ""
    for sentence in re.split(r"(?<=[.!?:])\s+", text):
        if current and len(current) + len(sentence) + 1 > TARGET_CHARS:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def stack_heading(previous, text):
    """Combine headings that follow each other without body text in between."""
    if text in previous.split(" – "):
        return previous  # fake-bold duplicate in a separate block
    last = previous.split(" – ")[-1]
    if last.isupper() and text.isupper() and (last.endswith(",") or len(text.split()) <= 2 and len(last.split()) <= 2):
        combined = f"{previous} {text}"  # one title broken into several blocks
    else:
        combined = f"{previous} – {text}"
    return combined if len(combined) <= MAX_HEADING_CHARS else text


def build_sections(elements, title):
    """Group paragraphs under their most recent heading."""
    sections = [{"heading": title, "page": 1, "paras": []}]
    for kind, text, page in elements:
        if kind == "heading":
            last = sections[-1]
            if not last["paras"] and last["heading"] != title:
                last["heading"] = stack_heading(last["heading"], text)
            else:
                sections.append({"heading": text, "page": page, "paras": []})
        else:
            sections[-1]["paras"].extend((p, page) for p in split_long(text))
    return [s for s in sections if s["paras"]]


def chunk_pdf(path):
    doc = pymupdf.open(path)
    slug = doc_slug(path)
    sections = build_sections(extract_elements(doc), title=slug)

    chunks = []
    for section_no, section in enumerate(sections, start=1):
        groups, current = [], []
        for para, page in section["paras"]:
            size = sum(len(p) for p, _ in current)
            if current and size + len(para) > TARGET_CHARS:
                groups.append(current)
                current = []
            current.append((para, page))
        if current:
            # merge a tiny tail into the previous chunk of the same section
            if groups and sum(len(p) for p, _ in current) < MIN_CHARS:
                groups[-1].extend(current)
            else:
                groups.append(current)

        for group in groups:
            index = len(chunks) + 1
            heading = section["heading"]
            chunks.append(
                {
                    "id": f"{slug}-c{index:03d}",
                    "page": group[0][1],
                    "section": heading,
                    "section_id": f"{slug}-s{section_no:02d}",
                    "source_file": path.name,
                    "chunk_index": index,
                    "chunk_text": f"{heading} — " + " ".join(p for p, _ in group),
                }
            )
    return chunks


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pdf-dir", default=str(ROOT / "pdfs"))
    parser.add_argument("--out", default=str(ROOT / "data/my-corpus.json"))
    args = parser.parse_args()

    pdfs = sorted(Path(args.pdf_dir).glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {args.pdf_dir}")

    chunks = []
    for pdf in pdfs:
        pdf_chunks = chunk_pdf(pdf)
        print(f"{pdf.name}: {len(pdf_chunks)} chunks")
        chunks.extend(pdf_chunks)

    corpus = {
        "doc": ", ".join(p.name for p in pdfs),
        "version": "1",
        "chunks": chunks,
    }
    Path(args.out).write_text(json.dumps(corpus, ensure_ascii=False, indent=2) + "\n")
    print(f"✓ Wrote {len(chunks)} chunks to {args.out}")


if __name__ == "__main__":
    main()
