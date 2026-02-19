from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import shutil

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from app.rag.confidence import compute_confidence
import re

from app.ingest.pipeline import ingest_file
from app.db.mongo import log_query
from app.db.mongo import get_query_analytics

# =========================================================
# CONFIG
# =========================================================

QDRANT_URL = "http://localhost:6333"
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
# REQUEST MODELS
# =========================================================

class QuestionRequest(BaseModel):
    question: str
    k: int = 6


class SearchRequest(BaseModel):
    query: str
    k: int = 6


class EmailRequest(BaseModel):
    request: str
    k: int = 6


# =========================================================
# GLOBAL COMPONENTS
# =========================================================

print("Loading embeddings and LLM...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

client = QdrantClient(QDRANT_URL)

llm = ChatOllama(
    model="mistral",
    temperature=0
)

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
        "You are a strict HR assistant.\n\n"

        "Write a professional HR email using ONLY the provided context.\n\n"

        "STRICT RULES:\n"
        "- Do NOT invent attachments.\n"
        "- Do NOT invent actions not mentioned.\n"
        "- Do NOT assume medical certificates, forms, or approvals.\n"
        "- Only write what is requested.\n"
        "- Use professional format.\n"
        "- If policy details missing, write generic email without adding policy claims.\n"
        "- Do NOT mention context.\n"
        "- Do NOT hallucinate.\n"
    ),
    (
        "human",
        "Context:\n{context}\n\n"
        "User Request:\n{request}"
    )
])

email_chain = email_prompt | llm | StrOutputParser()

print("System ready.")


# =========================================================
# VECTORSTORE LOADER (CRITICAL FIX)
# =========================================================

def get_vectorstore():

    if not client.collection_exists(COLLECTION_NAME):
        return None

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
    """
    Extract source from LangChain document.
    """
    meta = doc.metadata or {}
    
    # After fix, source_file will be directly in metadata
    if "source_file" in meta:
        return meta["source_file"]
        
    return "Unknown Source"

def format_docs(docs):
    """
    Format documents into context string.
    """

    formatted = []

    for d in docs:

        source = extract_source(d) or "unknown"

        formatted.append(
            f"[Source: {source}]\n{d.page_content}"
        )

    return "\n\n".join(formatted)


def extract_keywords(text: str):
    """
    Extract important keywords from question.
    Simple production-safe keyword extraction.
    """
    words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())

    stopwords = {
        "what", "when", "where", "which", "their", "there",
        "about", "policy", "please", "tell", "does", "have",
        "this", "that", "with", "from"
    }

    return [w for w in words if w not in stopwords]


def keyword_coverage(question: str, docs):
    """
    Measure keyword coverage inside retrieved chunks.
    """
    keywords = extract_keywords(question)

    if not keywords:
        return 0, 0

    combined = " ".join([d.page_content.lower() for d in docs])

    hits = sum(1 for k in keywords if k in combined)

    return hits, len(keywords)


# =========================================================
# ROUTES
# =========================================================

@app.get("/health")
def health():
    return {"status": "ok"}


# =========================================================
# SEARCH ENDPOINT
# =========================================================

@app.post("/search")
def search(req: SearchRequest):

    vectorstore = get_vectorstore()

    if vectorstore is None:
        return {"results": []}

    docs = vectorstore.similarity_search(
        req.query,
        k=req.k
    )

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
# ASK ENDPOINT (MAIN RAG)
# =========================================================

@app.post("/ask")
def ask(req: QuestionRequest):

    vectorstore = get_vectorstore()

    if vectorstore is None:

        final_answer = "Knowledge base is empty. Please upload documents first."

        log_query(
            question=req.question,
            answer=final_answer,
            sources=[],
            confidence=0.0
        )

        return {
            "answer": final_answer,
            "sources": [],
            "confidence": 0.0
        }

    # similarity search with score
    results = vectorstore.similarity_search_with_score(
        req.question,
        k=req.k
    )

    if not results:

        final_answer = "Not specified in policy."

        log_query(
            question=req.question,
            answer=final_answer,
            sources=[],
            confidence=0.0
        )

        return {
            "answer": final_answer,
            "sources": [],
            "confidence": 0.0
        }

    docs = []
    scores = []

    for doc, score in results:

        docs.append(doc)

        # convert Qdrant distance → similarity
        similarity = 1 - score
        scores.append(similarity)

    avg_score = sum(scores) / len(scores)

    # keyword coverage
    keyword_hits, expected_keywords = keyword_coverage(
        req.question,
        docs
    )

    # confidence calculation
    confidence = compute_confidence(
        avg_rerank_score=avg_score,
        keyword_hits=keyword_hits,
        expected_keywords=expected_keywords,
        retrieved_chunks=len(docs),
        k=req.k
    )

    # build context
    context = format_docs(docs)

    answer = chain.invoke({
        "context": context,
        "question": req.question
    })

    final_answer = answer.strip()

    # extract sources
    sources = list(set([
        extract_source(d)
        for d in docs
    ]))

    # ✅ QUERY LOGGING (CRITICAL ADDITION)
    log_query(
        question=req.question,
        answer=final_answer,
        sources=sources,
        confidence=confidence
    )

    return {
        "answer": final_answer,
        "sources": sources,
        "confidence": confidence
    }

# =========================================================
# DOCUMENT UPLOAD ENDPOINT
# =========================================================

@app.post("/upload-doc")
def upload_doc(file: UploadFile = File(...)):

    suffix = Path(file.filename).suffix.lower()

    if suffix not in [".pdf", ".txt", ".md"]:

        return {
            "error": "Unsupported file type"
        }

    save_path = UPLOAD_DIR / file.filename

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    stats = ingest_file(save_path)

    return {
        "status": "success",
        "file": file.filename,
        "stats": stats
    }

@app.get("/debug-qdrant")
def debug_qdrant():

    result = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=5,
        with_payload=True,
        with_vectors=False
    )

    points = result[0]

    output = []

    for p in points:
        output.append({
            "id": str(p.id),
            "payload": p.payload
        })

    return output


@app.get("/debug/raw")
def debug_raw():

    result = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=3,
        with_payload=True,
        with_vectors=False
    )

    points = result[0]

    output = []

    for p in points:

        output.append({
            "id": p.id,
            "payload": p.payload
        })

    return output


@app.post("/generate-email")
def generate_email(req: EmailRequest):

    vectorstore = get_vectorstore()

    if vectorstore is None:
        return {
            "email": "Knowledge base empty.",
            "sources": [],
            "confidence": 0.0
        }

    docs = vectorstore.similarity_search(
        req.request,
        k=req.k
    )

    if not docs:
        return {
            "email": "Unable to generate email.",
            "sources": [],
            "confidence": 0.0
        }

    context = format_docs(docs)

    email = email_chain.invoke({
        "context": context,
        "request": req.request
    })

    # Extract sources
    sources = list(set([
        extract_source(d)
        for d in docs
        if extract_source(d)
    ]))

    # Simple confidence calculation
    avg_rerank_score = 0.8
    keyword_hits = len(req.request.split())
    expected_keywords = len(req.request.split())
    retrieved_chunks = len(docs)

    confidence = compute_confidence(
        avg_rerank_score,
        keyword_hits,
        expected_keywords,
        retrieved_chunks,
        req.k
    )

    return {
        "email": email.strip(),
        "sources": sources,
        "confidence": confidence
    }


@app.get("/analytics")
def analytics():

    stats = get_query_analytics()

    return {
        "status": "success",
        "analytics": stats
    }