def compute_confidence(
    avg_rerank_score: float,
    keyword_hits: int,
    expected_keywords: int,
    retrieved_chunks: int,
    k: int
):

    # keyword coverage ratio
    if expected_keywords == 0:
        keyword_ratio = 0
    else:
        keyword_ratio = keyword_hits / expected_keywords

    # retrieval fullness
    retrieval_ratio = min(retrieved_chunks / k, 1)

    score = (
        0.5 * avg_rerank_score +
        0.3 * keyword_ratio +
        0.2 * retrieval_ratio
    )

    return round(float(score), 3)
