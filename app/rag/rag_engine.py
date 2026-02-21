from langchain_groq import ChatGroq
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()


# -------------------------
# CONFIG
# -------------------------

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "hr_knowledge_base"


# -------------------------
# LAZY LOAD EMBEDDINGS (IMPORTANT FIX)
# -------------------------

_embeddings = None

def get_embeddings():
    global _embeddings

    if _embeddings is None:
        print("Loading embedding model (rag_engine)...")
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    return _embeddings


# -------------------------
# CONNECT VECTOR STORE
# -------------------------

def get_vectorstore():

    embeddings = get_embeddings()

    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    return vectorstore


# -------------------------
# BUILD RAG CHAIN
# -------------------------

def build_chain():

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an HR policy assistant. "
            "Answer ONLY using the provided context. "
            "If unsure, say you don't know."
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion: {question}"
        )
    ])

    chain = prompt | llm | StrOutputParser()

    return chain


# -------------------------
# FORMAT DOCS
# -------------------------

def format_docs(docs):

    texts = []

    for d in docs:
        src = d.metadata.get("source_file", "unknown")
        texts.append(f"[Source: {src}]\n{d.page_content}")

    return "\n\n".join(texts)


# -------------------------
# MAIN LOOP (LOCAL TESTING ONLY)
# -------------------------

def main():

    print("\n🔎 Connecting to Qdrant...")

    vectorstore = get_vectorstore()

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 8}
    )

    chain = build_chain()

    print("✅ HR RAG Engine Ready")
    print("Type 'exit' to quit\n")

    while True:

        question = input("❓ ")

        if question.lower() in ["exit", "quit"]:
            break

        docs = retriever.invoke(question)

        context = format_docs(docs)

        answer = chain.invoke({
            "context": context,
            "question": question
        })

        print("\n💡 Answer:\n", answer)

        print("\n📚 Sources:")

        for d in docs:
            print("-", d.metadata.get("source_file"))

        print("\n" + "-"*50 + "\n")


if __name__ == "__main__":
    main()