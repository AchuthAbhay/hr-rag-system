from app.rag.reranker import reranker_model
import math


def score_chunks(query, docs):

    if not docs:
        return []

    pairs = [(query, d.page_content) for d in docs]
    raw_scores = reranker_model.predict(pairs)

    # sigmoid normalize
    scores = [
        1 / (1 + math.exp(-s))
        for s in raw_scores
    ]

    return scores


def keyword_hit_check(docs, keywords):

    text = " ".join(d.page_content.lower() for d in docs)

    hits = sum(
        1 for kw in keywords
        if kw.lower() in text
    )

    return hits
