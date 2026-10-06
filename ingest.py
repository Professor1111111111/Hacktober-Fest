from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from pypdf import PdfReader


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_name: str
    page: int
    text: str
    char_start: int
    char_end: int


def _doc_id(path: Path) -> str:
    return hashlib.sha1(
        str(path.resolve()).encode()
    ).hexdigest()[:12]


def _read_pdf(path: Path) -> Iterator[tuple[int, str]]:
    reader = PdfReader(str(path))

    for i, page in enumerate(reader.pages, start=1):
        try:
            yield i, page.extract_text() or ""
        except Exception:
            yield i, ""


def _read_docx(path: Path) -> Iterator[tuple[int, str]]:
    from docx import Document

    doc = Document(str(path))

    yield 1, "\n".join(
        paragraph.text
        for paragraph in doc.paragraphs
    )


def _read_text(path: Path) -> Iterator[tuple[int, str]]:
    yield 1, path.read_text(
        encoding="utf-8",
        errors="ignore"
    )


READERS = {
    ".pdf": _read_pdf,
    ".docx": _read_docx,
    ".txt": _read_text,
    ".md": _read_text,
}


def _chunk_page(
    text: str,
    target_chars: int = 700
) -> list[tuple[int, int, str]]:

    text = re.sub(r"[ \t]+", " ", text)

    sentences = []

    for match in re.finditer(
        r"[^.!?\n]+[.!?]?",
        text
    ):
        start, end = match.start(), match.end()

        if text[start:end].strip():
            sentences.append((start, end))

    chunks = []
    current_start = None
    current_end = None
    current_length = 0

    for start, end in sentences:

        sentence_length = end - start

        if current_start is None:
            current_start = start
            current_end = end
            current_length = sentence_length

        elif current_length + sentence_length <= target_chars:
            current_end = end
            current_length += sentence_length

        else:
            chunks.append(
                (
                    current_start,
                    current_end,
                    text[current_start:current_end].strip()
                )
            )

            current_start = start
            current_end = end
            current_length = sentence_length

    if current_start is not None:
        chunks.append(
            (
                current_start,
                current_end,
                text[current_start:current_end].strip()
            )
        )

    return chunks


def ingest_paths(paths: list[Path]) -> list[Chunk]:

    chunks = []

    for path in paths:

        reader = READERS.get(
            path.suffix.lower()
        )

        if not reader:
            continue

        doc_id = _doc_id(path)

        for page_number, page_text in reader(path):

            if not page_text.strip():
                continue

            page_chunks = _chunk_page(page_text)

            for char_start, char_end, chunk_text in page_chunks:

                chunk_id = hashlib.sha1(
                    f"{doc_id}:{page_number}:{char_start}".encode()
                ).hexdigest()[:16]

                chunks.append(
                    Chunk(
                        chunk_id,
                        doc_id,
                        path.name,
                        page_number,
                        chunk_text,
                        char_start,
                        char_end,
                    )
                )

    return chunks


def discover(root: Path) -> list[Path]:

    extensions = set(READERS.keys())

    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in extensions
    )