"""
ingest.py — Step 1 of the project.

What this does, in plain words:
  1. Reads every PDF in the `data/` folder.
  2. Cuts the text into small, slightly-overlapping pieces ("chunks").
  3. Gives each piece a short, stable ID.
  4. Prints how many pieces it made, and shows one as an example.

That is ALL it does for today. Storing the pieces and searching them
comes in the next step — don't worry about that yet.

How to run it (from the project's main folder):
    python rag/ingest.py
"""

import hashlib
from pathlib import Path

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# The folder where your PDFs live.
DATA_DIR = Path("data")

# How big each piece is (counted in characters), and how much
# neighbouring pieces overlap. ~2000 characters is roughly 500 words —
# a sensible starting size. The overlap stops us cutting a sentence in
# half and losing the meaning.
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200


def read_pdf(path: Path) -> str:
    """Open one PDF and return all of its text as a single long string."""
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def make_chunk_id(source: str, index: int) -> str:
    """Make a short, stable ID for a piece.

    'Stable' means: the same file and the same position will always get
    the same ID, every time you run this. We need that later so our
    test answers don't break when we re-run things.
    """
    raw = f"{source}:{index}"
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def main() -> None:
    pdf_paths = sorted(DATA_DIR.glob("*.pdf"))

    if not pdf_paths:
        print(f"No PDFs found in '{DATA_DIR}/'.")
        print("Create a 'data' folder, put 2-3 PDF files in it, and run again.")
        return

    # This is the tool that cuts the text into pieces at natural
    # break points (paragraphs, then lines, then spaces).
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    all_chunks = []
    for path in pdf_paths:
        text = read_pdf(path)
        pieces = splitter.split_text(text)
        for i, piece in enumerate(pieces):
            all_chunks.append(
                {
                    "chunk_id": make_chunk_id(path.name, i),
                    "source": path.name,
                    "text": piece,
                }
            )
        print(f"  {path.name}: {len(pieces)} pieces")

    print(f"\nLoaded {len(pdf_paths)} PDF(s)  ->  {len(all_chunks)} pieces total.\n")

    # Show one piece so you can see what a "chunk" actually looks like.
    if all_chunks:
        example = all_chunks[0]
        print("Here is one example piece:")
        print(f"  id     : {example['chunk_id']}")
        print(f"  source : {example['source']}")
        print(f"  text   : {example['text'][:200]} ...")


if __name__ == "__main__":
    main()