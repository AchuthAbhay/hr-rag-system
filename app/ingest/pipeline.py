from pathlib import Path
import uuid
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings

from qdrant_client import QdrantClient, models

from app.db.mongo import store_doc_metadata


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "hr_knowledge_base"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100


# =========================================================
# EMBEDDINGS (Lazy Singleton - Render Safe)
# =========================================================

_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        print("🔹 Connecting to HuggingFace Inference API...")
        _embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            huggingfacehub_api_token=os.getenv("HF_API_KEY"),
        )
    return _embeddings



# =========================================================
# QDRANT CLIENT
# =========================================================

def get_qdrant_client():
    """
    Create Qdrant client using cloud credentials.
    """
    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )


# =========================================================
# LOAD FILE
# =========================================================

def load_file(path: Path):

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))

    elif suffix in [".txt", ".md", ".markdown"]:  # ✅ merged into one
        loader = TextLoader(str(path), encoding="utf-8")

    else:
        raise ValueError(f"Unsupported file type: {path.name}")

    docs = loader.load()

    for d in docs:
        d.metadata["source_file"] = path.name
        d.metadata["file_type"] = suffix.replace(".", "")

    return docs

# =========================================================
# SPLIT DOCUMENTS
# =========================================================

def split_docs(docs):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )

    chunks = splitter.split_documents(docs)

    return chunks


# =========================================================
# CREATE COLLECTION IF NOT EXISTS
# =========================================================

def ensure_collection(client, vector_size):

    if client.collection_exists(COLLECTION_NAME):
        return

    print("🗄 Creating new collection...")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )


# =========================================================
# EMBED AND STORE
# =========================================================

def embed_and_store(chunks):

    embeddings = get_embeddings()
    client = get_qdrant_client()

    print("📐 Creating embeddings...")
    texts = [c.page_content for c in chunks]

    if not texts:
        raise ValueError("No text chunks to embed")

    vectors = embeddings.embed_documents(texts)

    if not vectors:
        raise ValueError("Embedding API returned empty vectors")

    ensure_collection(client, len(vectors[0]))

    print("⬆ Uploading vectors in batches...")

    points = []

    for i, (vec, doc) in enumerate(zip(vectors, chunks)):

        doc_id = str(uuid.uuid4())

        payload = {
            "page_content": doc.page_content,
            "metadata": {
                "source_file": doc.metadata.get("source_file", "unknown"),
                "file_type": doc.metadata.get("file_type", "unknown"),
                "chunk_id": i,
            },
        }

        points.append(
            models.PointStruct(
                id=doc_id,
                vector=vec,
                payload=payload,
            )
        )

        store_doc_metadata(
            doc_id=doc_id,
            source_file=payload["metadata"]["source_file"],
            chunk_id=i,
            metadata=payload["metadata"],
        )

    # ✅ Upload in batches of 50 instead of all at once
    BATCH_SIZE = 50
    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i:i + BATCH_SIZE]
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch,
        )
        print(f"✅ Uploaded batch {i//BATCH_SIZE + 1} ({len(batch)} points)")

    print("✅ All vectors stored in Qdrant")
    return len(points)



# =========================================================
# MAIN INGEST FUNCTION
# =========================================================

def ingest_file(path: Path):

    print(f"\n📥 Ingesting file: {path.name}")

    docs = load_file(path)

    chunks = split_docs(docs)

    count = embed_and_store(chunks)

    print("🎉 Ingestion complete\n")

    return {
        "pages": len(docs),
        "chunks": len(chunks),
        "vectors": count,
    }