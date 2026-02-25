import os
import numpy as np
import requests
from collections import defaultdict


# ============================================================
# 1. Embedding Client Setup
# ============================================================

class EmbeddingClient:
    def __init__(
        self,
        embedding_model_name="@cf/baai/bge-small-en-v1.5",
    ):
        self.embedding_model_name = embedding_model_name
        self.account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
        self.auth_token = os.getenv("EMBEDDING_API_KEY")
        if not self.account_id:
            raise ValueError("CLOUDFLARE_ACCOUNT_ID environment variable not set")
        if not self.auth_token:
            raise ValueError("EMBEDDING_API_KEY environment variable not set")
        self.base_url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run"

    def embed_texts(self, texts):
        """
        texts: list[str]
        Returns:
            np.ndarray shape (N, dim), L2-normalized
        """
        response = requests.post(
            f"{self.base_url}/{self.embedding_model_name}",
            headers={"Authorization": f"Bearer {self.auth_token}"},
            json={"text": texts}
        )
        response.raise_for_status()
        data = response.json()

        vectors = np.array(data["result"]["data"])

        # L2 normalize so cosine similarity = dot product
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.clip(norms, 1e-10, None)

        return vectors


# ============================================================
# 2. Chunk Scoring
# ============================================================

def chunk_scores_from_qemb(q_emb, chunk_texts, embedder):
    """
    q_emb: np.ndarray shape (1, dim)
    chunk_texts: list[str]
    embedder: EmbeddingClient instance

    Returns:
        list[float]
    """
    if not chunk_texts:
        return []

    c_embs = embedder.embed_texts(chunk_texts)  # (N, dim)

    sims = np.dot(c_embs, q_emb.T).squeeze(1)  # cosine similarity

    return sims.tolist()


# ============================================================
# 3. Doc Score Aggregation
# ============================================================

def doc_score_from_chunks(chunk_scores_list):
    """
    DocScore = (1/sqrt(N+1)) * sum(chunk_scores)
    """
    N = len(chunk_scores_list)
    if N == 0:
        return float("-inf")
    return (1.0 / ((N + 1) ** 0.5)) * float(np.sum(chunk_scores_list))


# ============================================================
# 4. Document Scoring
# ============================================================

def bert_doc_score(query_text, chunk_texts, embedder, q_emb=None):
    """
    query_text: str
    chunk_texts: list[str]
    embedder: EmbeddingClient
    q_emb: optional precomputed query embedding

    Returns:
        float
    """
    if q_emb is None:
        q_emb = embedder.embed_texts([query_text])  # (1, dim)

    chunk_sims = chunk_scores_from_qemb(q_emb, chunk_texts, embedder)

    return doc_score_from_chunks(chunk_sims)


# ============================================================
# 5. Rank Sources in HuggingFace Dataset
# ============================================================

def top_k_sources_for_question(hf_dataset, query_text, embedder, k=3):
    """
    hf_dataset: HuggingFace Dataset with:
        - source
        - text

    Returns:
        list[str] top-k file names only
    """

    grouped = defaultdict(list)

    for src, txt in zip(hf_dataset["source"], hf_dataset["text"]):
        if txt:
            grouped[str(src)].append(str(txt))

    # Embed query once
    q_emb = embedder.embed_texts([query_text])  # (1, dim)

    scored = []
    for src, chunk_texts in grouped.items():
        score = bert_doc_score(query_text, chunk_texts, embedder, q_emb=q_emb)
        scored.append((src, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    return [src for src, _ in scored[:k]]
