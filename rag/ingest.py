"""PDF ingestion: load documents from S3 (or a local folder), split into overlapping chunks, assign stable IDs."""

import hashlib
import os
from collections import Counter
from io import BytesIO
from pathlib import Path

import boto3
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path("data")
S3_BUCKET = os.environ.get("RAG_S3_BUCKET")  # if set, read PDFs from S3; otherwise from data/
S3_REGION = "eu-central-1"

CHUNK_SIZE = 2000      # characters, ~500 tokens
CHUNK_OVERLAP = 200    # ~10% overlap so context isn't lost at chunk boundaries


def _read_pdf(source) -> str:
    reader = PdfReader(source)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _load_documents() -> list[tuple[str, str]]:
    """Return (filename, text) per PDF, from S3 if RAG_S3_BUCKET is set, else the local folder."""
    if S3_BUCKET:
        s3 = boto3.client("s3", region_name=S3_REGION)
        objects = s3.list_objects_v2(Bucket=S3_BUCKET).get("Contents", [])
        documents = []
        for obj in objects:
            key = obj["Key"]
            if key.lower().endswith(".pdf"):
                body = s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
                documents.append((Path(key).name, _read_pdf(BytesIO(body))))
        return documents
    return [(p.name, _read_pdf(str(p))) for p in sorted(DATA_DIR.glob("*.pdf"))]


def make_chunk_id(source: str, index: int) -> str:
    # Deterministic so chunk IDs stay stable across runs and eval labels remain valid.
    return hashlib.sha1(f"{source}:{index}".encode()).hexdigest()[:12]


def load_chunks() -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks: list[dict] = []
    for source, text in _load_documents():
        for i, piece in enumerate(splitter.split_text(text)):
            chunks.append(
                {"chunk_id": make_chunk_id(source, i), "source": source, "text": piece}
            )
    return chunks


def main() -> None:
    chunks = load_chunks()
    if not chunks:
        print("No PDFs found. Set RAG_S3_BUCKET, or add PDFs to the data/ folder.")
        return

    for source, count in Counter(c["source"] for c in chunks).items():
        print(f"  {source}: {count} chunks")
    origin = f"S3 bucket '{S3_BUCKET}'" if S3_BUCKET else f"local folder '{DATA_DIR}/'"
    print(f"\n{len(chunks)} chunks from {len({c['source'] for c in chunks})} document(s) [{origin}].")


if __name__ == "__main__":
    main()
