from pymongo import MongoClient
from datetime import datetime
from collections import Counter
from dotenv import load_dotenv
import os

# Load .env
load_dotenv()

# Use Atlas connection
MONGO_URL = os.getenv("MONGO_URL")

# Create client
client = MongoClient(MONGO_URL)

# Database config
DB_NAME = "hr_rag"
COLLECTION = "documents"

db = client[DB_NAME]
docs_col = db[COLLECTION]
query_logs_col = db["query_logs"]


# =====================================================
# STORE DOCUMENT METADATA
# =====================================================

def store_doc_metadata(doc_id, source_file, chunk_id, metadata):

    record = {
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "source_file": source_file,
        "metadata": metadata,
        "ingested_at": datetime.utcnow()
    }

    docs_col.insert_one(record)


# =====================================================
# GET DOC CHUNKS
# =====================================================

def get_doc_chunks(source_file):
    return list(docs_col.find({"source_file": source_file}))


# =====================================================
# LOG QUERY
# =====================================================

def log_query(question, answer, sources, confidence):

    record = {
        "question": question,
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
        "timestamp": datetime.utcnow()
    }

    query_logs_col.insert_one(record)


# =====================================================
# ANALYTICS
# =====================================================

def get_query_analytics():

    total_queries = query_logs_col.count_documents({})

    if total_queries == 0:
        return {
            "total_queries": 0,
            "avg_confidence": 0,
            "top_questions": [],
            "top_sources": []
        }

    # Average confidence
    pipeline = [
        {
            "$group": {
                "_id": None,
                "avg_conf": {"$avg": "$confidence"}
            }
        }
    ]

    avg_result = list(query_logs_col.aggregate(pipeline))
    avg_confidence = round(avg_result[0]["avg_conf"], 3)

    # Top questions
    questions = [
        q["question"]
        for q in query_logs_col.find({}, {"question": 1})
    ]

    top_questions = Counter(questions).most_common(5)

    # Top sources
    sources = []

    for doc in query_logs_col.find({}, {"sources": 1}):
        sources.extend(doc.get("sources", []))

    top_sources = Counter(sources).most_common(5)

    return {
        "total_queries": total_queries,
        "avg_confidence": avg_confidence,
        "top_questions": top_questions,
        "top_sources": top_sources
    }