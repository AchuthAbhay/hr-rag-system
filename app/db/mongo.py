from pymongo import MongoClient
from datetime import datetime

MONGO_URL = "mongodb://localhost:27017"
DB_NAME = "hr_rag"
COLLECTION = "documents"


client = MongoClient(MONGO_URL)
db = client[DB_NAME]
docs_col = db[COLLECTION]


def store_doc_metadata(doc_id, source_file, chunk_id, metadata):

    record = {
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "source_file": source_file,
        "metadata": metadata,
        "ingested_at": datetime.utcnow()
    }

    docs_col.insert_one(record)


def get_doc_chunks(source_file):
    return list(docs_col.find({"source_file": source_file}))
