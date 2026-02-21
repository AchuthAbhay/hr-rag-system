from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import shutil
import re
import os

from dotenv import load_dotenv
load_dotenv()

# ============================
# LangChain / AI imports
# ============================

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# ============================
# Internal imports
# ============================

from app.ingest.pipeline import ingest_file
from app.db.mongo import log_query, get_query_analytics
from app.rag.confidence import compute_confidence


# =========================================================
# CONFIG
# =========================================================

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

COLLECTION_NAME = "hr_knowledge_base"

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# FASTAPI INIT
# =========================================================

app = FastAPI(
    title="HR RAG API",
    description="HR Policy Retrieval & Question Answering System",
    version="2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# LAZY LOAD GLOBALS (CRITICAL FOR RENDER MEMORY)
# =========================================================

_embeddings = None
_client = None


def get_embeddings():
    global _embeddings

    if _embeddings is None:
        print("🔹 Loading embedding model...")
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    return _embeddings


def get_qdrant_client():
    global _client

    if _client is None:
        print("🔹 Connecting to Qdrant Cloud...")
        _client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY
        )

    return _client


# =========================================================
# LOAD GROQ LLM
# =========================================================

print("🔹 Loading Groq LLM...")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=GROQ_API_KEY
)

print("✅ LLM ready.")


# =========================================================
# PROMPTS
# =========================================================

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a strict HR Policy Assistant for the company.\n\n"

        "CRITICAL RULES (MUST FOLLOW):\n"

        "1. Answer ONLY using the provided context.\n"
        "2. Do NOT use outside knowledge.\n"
        "3. Do NOT infer, assume, or guess anything.\n"
        "4. Do NOT add explanations beyond what is explicitly stated.\n"
        "5. If the answer is not explicitly present in the context, respond EXACTLY with:\n"
        "   Not specified in policy.\n"
        "6. If the user greets (hi, hello, hey), respond EXACTLY with:\n"
        "   Hello! How can I assist you with HR policies today?\n"
        "7. Do NOT mention sources, metadata, or context in your answer.\n"
        "8. Do NOT add extra HR knowledge.\n"
        "9. Keep answers concise and factual.\n"
        "10. Your job is ONLY to extract and return information from context.\n"
    ),
    (
        "human",
        "Context:\n{context}\n\n"
        "Question: {question}"
    )
])

chain = prompt | llm | StrOutputParser()
email_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a professional HR assistant.\n\n"

        "Your job is to write a professional email based on the user's request.\n\n"

        "IMPORTANT RULES:\n"
        "- You MAY write general professional emails even if exact scenario is not in policy.\n"
        "- Use the provided context if relevant.\n"
        "- Do NOT invent company policies, benefits, approvals, or rules.\n"
        "- Do NOT claim policy support unless explicitly present in context.\n"
        "- Do NOT invent attachments, medical certificates, or approvals.\n"
        "- Keep the email professional, clear, and concise.\n"
        "- Use proper email format with subject, greeting, body, and closing.\n"
        "- Do NOT mention the context or policy in the email.\n"
        "- Do NOT hallucinate company-specific rules.\n"
    ),
    (
        "human",
        "Context:\n{context}\n\n"
        "User Request:\n{request}"
    )
])
email_chain = email_prompt | llm | StrOutputParser()




# =========================================================
# REQUEST MODELS
# =========================================================

class QuestionRequest(BaseModel):
    question: str
    k: int = 8


class SearchRequest(BaseModel):
    query: str
    k: int = 8


class EmailRequest(BaseModel):
    request: str
    k: int = 8


# =========================================================
# VECTORSTORE
# =========================================================

def get_vectorstore():

    client = get_qdrant_client()

    if not client.collection_exists(COLLECTION_NAME):
        return None

    embeddings = get_embeddings()

    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
        content_payload_key="page_content",
        metadata_payload_key="metadata"
    )


# =========================================================
# HELPERS
# =========================================================

def extract_source(doc):
    meta = doc.metadata or {}

    if "source_file" in meta:
        return meta["source_file"]

    return "Unknown Source"


def format_docs(docs):

    formatted = []

    for d in docs:
        source = extract_source(d)
        formatted.append(f"[Source: {source}]\n{d.page_content}")

    return "\n\n".join(formatted)


def extract_keywords(text):

    words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())

    stopwords = {
        "what", "when", "where", "which",
        "their", "there", "about",
        "policy", "please", "tell"
    }

    return [w for w in words if w not in stopwords]


def keyword_coverage(question, docs):

    keywords = extract_keywords(question)

    if not keywords:
        return 0, 0

    combined = " ".join(d.page_content.lower() for d in docs)

    hits = sum(1 for k in keywords if k in combined)

    return hits, len(keywords)


# =========================================================
# ROUTES
# =========================================================

@app.get("/health")
def health():
    return {"status": "ok"}


# =========================================================
# SEARCH
# =========================================================

@app.post("/search")
def search(req: SearchRequest):

    vectorstore = get_vectorstore()

    if vectorstore is None:
        return {"results": []}

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


# =========================================================
# ASK
# =========================================================

@app.post("/ask")
def ask(req: QuestionRequest):

    vectorstore = get_vectorstore()

    if vectorstore is None:

        answer = "Knowledge base is empty."

        log_query(req.question, answer, [], 0)

        return {
            "answer": answer,
            "sources": [],
            "confidence": 0
        }

    results = vectorstore.similarity_search_with_score(
        req.question,
        k=req.k
    )

    docs = []
    scores = []

    for doc, score in results:
        docs.append(doc)
        scores.append(1 - score)

    avg_score = sum(scores) / len(scores)

    keyword_hits, expected_keywords = keyword_coverage(
        req.question,
        docs
    )

    confidence = compute_confidence(
        avg_score,
        keyword_hits,
        expected_keywords,
        len(docs),
        req.k
    )

    context = format_docs(docs)

    answer = chain.invoke({
        "context": context,
        "question": req.question
    }).strip()

    sources = list(set(extract_source(d) for d in docs))

    log_query(
        req.question,
        answer,
        sources,
        confidence
    )

    return {
        "answer": answer,
        "sources": sources,
        "confidence": confidence
    }


# =========================================================
# EMAIL
# =========================================================

@app.post("/generate-email")
def generate_email(req: EmailRequest):

    vectorstore = get_vectorstore()

    if vectorstore is None:
        return {"email": "Knowledge base empty"}

    docs = vectorstore.similarity_search(
        req.request,
        k=req.k
    )

    context = format_docs(docs)

    email = email_chain.invoke({
        "context": context,
        "request": req.request
    })

    return {"email": email.strip()}


# =========================================================
# UPLOAD
# =========================================================

@app.post("/upload-doc")
def upload_doc(file: UploadFile = File(...)):

    path = UPLOAD_DIR / file.filename

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    stats = ingest_file(path)

    return {
        "status": "success",
        "stats": stats
    }


# =========================================================
# ANALYTICS
# =========================================================

@app.get("/analytics")
def analytics():

    stats = get_query_analytics()

    return {
        "status": "success",
        "analytics": stats
    }


# =========================================================
# DEBUG
# =========================================================

@app.get("/debug-qdrant")
def debug_qdrant():

    client = get_qdrant_client()

    result = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=5,
        with_payload=True,
        with_vectors=False
    )

    return result[0]