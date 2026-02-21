from pathlib import Path
import uuid

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from qdrant_client import QdrantClient, models

from app.db.mongo import store_doc_metadata
from dotenv import load_dotenv
import os 
load_dotenv()


# =========================================================
# CONFIG
# =========================================================

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "hr_knowledge_base"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100


# =========================================================
# LOAD FILE
# =========================================================

def load_file(path: Path):

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))

    elif suffix == ".txt":
        loader = TextLoader(str(path), encoding="utf-8")

    elif suffix in [".md", ".markdown"]:
        loader = UnstructuredMarkdownLoader(str(path))

    else:
        raise ValueError(f"Unsupported file type: {path.name}")

    docs = loader.load()

    # attach metadata correctly
    for d in docs:
        d.metadata["source_file"] = path.name
        d.metadata["file_type"] = suffix.replace(".", "")

    return docs


# =========================================================
# SPLIT DOCS
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
# STORE IN QDRANT (CRITICAL FIX HERE)
# =========================================================

def embed_and_store(chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

    print("📐 Creating embeddings...")

    texts = [c.page_content for c in chunks]
    vectors = embeddings.embed_documents(texts)

    # recreate collection cleanly
    if client.collection_exists(COLLECTION_NAME):
        print("🗑 Collection exists. Deleting...")
        client.delete_collection(COLLECTION_NAME)

    print("🗄 Creating collection...")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=len(vectors[0]),
            distance=models.Distance.COSINE,
        ),
    )

    print("⬆ Uploading vectors...")

    points = []

    for i, (vec, doc) in enumerate(zip(vectors, chunks)):

        doc_id = str(uuid.uuid4())

        # ✅ NESTED PAYLOAD (LangChain-compatible)
        payload = {
    "page_content": doc.page_content,
    "metadata": {
        "source_file": doc.metadata.get("source_file", "unknown"),
        "file_type": doc.metadata.get("file_type", "unknown"),
        "chunk_id": i
    }
}

        points.append(
            models.PointStruct(
                id=doc_id,
                vector=vec,
                payload=payload
            )
        )

        # store metadata in Mongo
        store_doc_metadata(
            doc_id=doc_id,
            source_file=doc.metadata.get("source_file", "unknown"),
            chunk_id=i,
            metadata=payload["metadata"]  # ✅ pass the metadata dict
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    print("✅ Stored in Qdrant successfully")

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
        "vectors": count
    }
