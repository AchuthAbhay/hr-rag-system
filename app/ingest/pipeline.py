from pathlib import Path
import uuid

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from qdrant_client import QdrantClient, models

from app.db.mongo import store_doc_metadata

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "hr_knowledge_base"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100


# -------------------------
# LOAD SINGLE FILE
# -------------------------

def load_file(path: Path):

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    elif suffix == ".txt":
        loader = TextLoader(str(path), encoding="utf-8")
    elif suffix in [".md", ".markdown"]:
        loader = UnstructuredMarkdownLoader(str(path))
    else:
        raise ValueError("Unsupported file type")

    docs = loader.load()

    for d in docs:
        d.metadata["source_file"] = path.name
        d.metadata["file_type"] = suffix.replace(".", "")

    return docs


# -------------------------
# SPLIT
# -------------------------

def split_docs(docs):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )

    return splitter.split_documents(docs)


# -------------------------
# STORE
# -------------------------

def embed_and_store(chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    client = QdrantClient(QDRANT_URL)

    texts = [d.page_content for d in chunks]
    vectors = embeddings.embed_documents(texts)

    # create collection if missing
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=len(vectors[0]),
                distance=models.Distance.COSINE,
            ),
        )

    points = []

    for i, (vec, doc) in enumerate(zip(vectors, chunks)):

        doc_id = str(uuid.uuid4())

        payload = {
            "page_content": doc.page_content,
            **doc.metadata
        }

        points.append(
            models.PointStruct(
                id=doc_id,
                vector=vec,
                payload=payload
            )
        )

        # Mongo metadata
        store_doc_metadata(
            doc_id=doc_id,
            source_file=doc.metadata.get("source_file"),
            chunk_id=i,
            metadata=doc.metadata
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    return len(points)


# -------------------------
# FULL PIPELINE
# -------------------------

def ingest_file(path: Path):

    docs = load_file(path)
    chunks = split_docs(docs)
    count = embed_and_store(chunks)

    return {
        "pages": len(docs),
        "chunks": len(chunks),
        "vectors": count
    }
