from pathlib import Path
from typing import List
import uuid

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

from qdrant_client import QdrantClient, models

from app.db.mongo import store_doc_metadata


# -------------------------
# CONFIG
# -------------------------

DATA_DIR = Path("data/hr_docs")
COLLECTION_NAME = "hr_knowledge_base"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100


# -------------------------
# LOADERS
# -------------------------

def load_file(path: Path) -> List[Document]:
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))

    elif suffix == ".txt":
        loader = TextLoader(str(path), encoding="utf-8")

    elif suffix in [".md", ".markdown"]:
        loader = UnstructuredMarkdownLoader(str(path))

    else:
        print(f"⚠️ Skipping unsupported file: {path.name}")
        return []

    docs = loader.load()

    # attach metadata
    for d in docs:
        d.metadata["source_file"] = path.name
        d.metadata["file_type"] = suffix.replace(".", "")

    return docs


# -------------------------
# LOAD ALL DOCUMENTS
# -------------------------

def load_all_documents() -> List[Document]:
    all_docs = []

    for path in DATA_DIR.glob("**/*"):
        if path.is_file():
            all_docs.extend(load_file(path))

    print(f"✅ Loaded {len(all_docs)} raw documents/pages")
    return all_docs


# -------------------------
# CHUNKING
# -------------------------

def split_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
        ],
    )

    chunks = splitter.split_documents(docs)

    print(f"✅ Created {len(chunks)} chunks")
    return chunks


# -------------------------
# QDRANT STORE + MONGO META
# -------------------------

def store_in_qdrant(chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    client = QdrantClient("http://localhost:6333")

    print("📐 Creating embeddings...")

    vectors = embeddings.embed_documents(
        [doc.page_content for doc in chunks]
    )

    print("🗄 Recreating collection...")

    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=len(vectors[0]),
            distance=models.Distance.COSINE,
        ),
    )

    print("⬆️ Uploading vectors...")

    points = []

    for i, (vec, doc) in enumerate(zip(vectors, chunks)):

        doc_id = str(uuid.uuid4())

        # ---- FLAT PAYLOAD (IMPORTANT FIX) ----
        payload = {
            "page_content": doc.page_content
        }

        # flatten metadata into payload
        for k, v in doc.metadata.items():
            payload[k] = v

        # ---- Qdrant point ----
        points.append(
            models.PointStruct(
                id=doc_id,
                vector=vec,
                payload=payload
            )
        )

        # ---- Mongo metadata ----
        store_doc_metadata(
            doc_id=doc_id,
            source_file=doc.metadata.get("source_file"),
            chunk_id=i,
            metadata=doc.metadata
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    print("✅ Stored embeddings in Qdrant + Mongo")


# -------------------------
# MAIN
# -------------------------

if __name__ == "__main__":
    print("\n📥 HR Document Ingestion Started...\n")

    docs = load_all_documents()
    chunks = split_documents(docs)
    store_in_qdrant(chunks)

    print("\n🎉 Ingestion complete.\n")

