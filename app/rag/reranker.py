from sentence_transformers import CrossEncoder

print("🔁 Loading reranker model...")

reranker_model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def rerank(query, docs, top_n=3):
    if not docs:
        return docs

    pairs = [(query, d.page_content) for d in docs]

    scores = reranker_model.predict(pairs)

    scored = list(zip(scores, docs))
    scored.sort(key=lambda x: x[0], reverse=True)

    return [doc for _, doc in scored[:top_n]]
