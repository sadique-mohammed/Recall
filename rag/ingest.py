"""PDF ingestion: load documents, split into overlapping chunks, assign stable IDs."""

import hashlib
from collections import Counter
from pathlib import Path

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path("data")
CHUNK_SIZE = 2000      # characters, ~500 tokens
CHUNK_OVERLAP = 200    # ~10% overlap so context isn't lost at chunk boundaries


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def make_chunk_id(source: str, index: int) -> str:
    # Deterministic so chunk IDs stay stable across runs and eval labels remain valid.
    return hashlib.sha1(f"{source}:{index}".encode()).hexdigest()[:12]


def load_chunks() -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks: list[dict] = []
    for path in sorted(DATA_DIR.glob("*.pdf")):
        for i, piece in enumerate(splitter.split_text(read_pdf(path))):
            chunks.append(
                {"chunk_id": make_chunk_id(path.name, i), "source": path.name, "text": piece}
            )
    return chunks


def main() -> None:
    chunks = load_chunks()
    if not chunks:
        print(f"No PDFs found in '{DATA_DIR}/'.")
        return

    for source, n in Counter(c["source"] for c in chunks).items():
        print(f"  {source}: {n} chunks")
    print(f"\n{len(chunks)} chunks from {len({c['source'] for c in chunks})} document(s).")


if __name__ == "__main__":
    main()
