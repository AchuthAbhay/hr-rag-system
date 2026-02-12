from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from pathlib import Path
import shutil

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from app.ingest.pipeline import ingest_file


# -------------------------
# CONFIG
# -------------------------

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "hr_knowledge_base"

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------
# FASTAPI INIT
# -------------------------

app = FastAPI(
    title="HR RAG API",
    description="HR Policy Retrieval & QA System",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------
# REQUEST MODELS
# -------------------------

class QuestionRequest(BaseModel):
    question: str
    k: int = 4


class SearchRequest(BaseModel):
    query: str
    k: int = 4


# -------------------------
# GLOBAL OBJECTS
# -------------------------

print("🔧 Loading embeddings + vector store...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

client = QdrantClient(QDRANT_URL)

vectorstore = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)

llm = ChatOllama(
    model="mistral",
    temperature=0
)

prompt = ChatPromptTemplate.from_messages([
("system",
"You are a friendly and professional HR policy assistant.\n"
"If the user greets, reply briefly and politely.\n\n"

"STRICT RULES:\n"
"- Answer ONLY what is asked.\n"
"- Use ONLY sentences found in the context.\n"
"- Do NOT add related policy sections.\n"
"- Do NOT give extra rules or recommendations.\n"
"- Do NOT explain beyond the question.\n"
"- Do NOT merge multiple policy sections unless the question asks for all.\n"
"- If the exact answer is not clearly present, say exactly: Not specified in policy.\n"
),
("human",
"Context:\n{context}\n\nQuestion: {question}")
])

chain = prompt | llm | StrOutputParser()

print("✅ RAG components loaded")


# -------------------------
# HELPERS
# -------------------------

def extract_source(d):
    if not d.metadata:
        return None

    if "source_file" in d.metadata:
        return d.metadata["source_file"]

    if "metadata" in d.metadata:
        return d.metadata["metadata"].get("source_file")

    return None


def format_docs(docs):
    texts = []

    for d in docs:
        src = extract_source(d) or "unknown"
        texts.append(f"[Source: {src}]\n{d.page_content}")

    return "\n\n".join(texts)


# -------------------------
# ROUTES
# -------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


# ---------- SEARCH ONLY ----------

@app.post("/search")
def semantic_search(req: SearchRequest):

    docs = vectorstore.similarity_search(req.query, k=req.k)

    return {
        "results": [
            {
                "text": d.page_content,
                "source": extract_source(d)
            }
            for d in docs
        ]
    }


# ---------- CLEAN ASK ----------

@app.post("/ask")
def ask_question(req: QuestionRequest):

    docs = vectorstore.similarity_search(
        req.question,
        k=req.k
    )

    context = format_docs(docs)

    print("\n===== CONTEXT =====")
    print(context[:1200])
    print("===================\n")

    answer = chain.invoke({
        "context": context,
        "question": req.question
    })

    sources = list({
        extract_source(d)
        for d in docs
        if extract_source(d)
    })

    return {
        "answer": answer,
        "sources": sources
    }


# ---------- UPLOAD ----------

@app.post("/upload-doc")
def upload_document(file: UploadFile = File(...)):

    suffix = Path(file.filename).suffix.lower()

    if suffix not in [".pdf", ".txt", ".md"]:
        return {"error": "Unsupported file type"}

    save_path = UPLOAD_DIR / file.filename

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    stats = ingest_file(save_path)

    return {
        "status": "ingested",
        "file": file.filename,
        "stats": stats
    }