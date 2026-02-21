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

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "hr_knowledge_base"


# -------------------------
# CONNECT VECTOR STORE
# -------------------------

def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    client = QdrantClient(QDRANT_URL)

    vectorstore = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)


    return vectorstore


# -------------------------
# BUILD RAG CHAIN
# -------------------------

def build_chain(retriever):

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
        "You are an HR policy assistant. "
        "Answer ONLY using the provided context. "
        "Do NOT perform calculations unless explicitly stated in the policy. "
        "Do NOT combine categories into totals unless policy explicitly defines a total. "
        "If unsure, say you don't know."),

        ("human",
         "Context:\n{context}\n\nQuestion: {question}")
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
# MAIN LOOP
# -------------------------

def main():

    print("\n🔎 Connecting to Qdrant...")
    vectorstore = get_vectorstore()

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 8}
    )

    chain = build_chain(retriever)

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
