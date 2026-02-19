import tensorflow as tf
import numpy as np
from transformers import BertTokenizer, TFBertModel
from collections import defaultdict


# ============================================================
# 1. Load HuggingFace BERT Model + Tokenizer
# ============================================================

MODEL_NAME = "bert-base-uncased"

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
bert_model = TFBertModel.from_pretrained(MODEL_NAME)


# ============================================================
# 2. Embedding Function
# ============================================================

def embed_texts(texts, max_length=256):
    """
    Convert list[str] -> normalized BERT embeddings (CLS pooled).

    Returns:
        tf.Tensor shape (N, hidden_dim)
    """

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="tf"
    )

    outputs = bert_model(**encoded)

    # CLS token embedding: (N, hidden_dim)
    cls_embeddings = outputs.last_hidden_state[:, 0, :]

    # Normalize so cosine similarity = dot product
    normed = tf.math.l2_normalize(cls_embeddings, axis=1)

    return normed


# ============================================================
# 3. Chunk Scoring
# ============================================================

def chunk_scores_from_qemb(q_emb, chunk_texts):
    """
    Compute cosine similarity between query embedding and each chunk embedding.

    q_emb: shape (1, hidden)
    chunk_texts: list[str]

    Returns:
        list[float] similarity scores
    """

    if not chunk_texts:
        return []

    c_embs = embed_texts(chunk_texts)

    sims = tf.squeeze(tf.matmul(c_embs, q_emb, transpose_b=True), axis=1)

    return sims.numpy().tolist()


# ============================================================
# 4. Doc Score Aggregation
# ============================================================

def doc_score_from_chunks(chunk_scores_list):
    """
    DocScore = (1/sqrt(N+1)) * sum(chunk_scores)

    N = number of chunks
    """

    N = len(chunk_scores_list)
    if N == 0:
        return float("-inf")

    return (1.0 / ((N + 1) ** 0.5)) * sum(chunk_scores_list)


# ============================================================
# 5. Document Scoring
# ============================================================

def bert_doc_score(query_text, chunk_texts, q_emb=None):
    """
    Compute document score for one doc.

    query_text: str
    chunk_texts: list[str]
    q_emb: optional precomputed query embedding

    Returns:
        float doc score
    """

    if q_emb is None:
        q_emb = embed_texts([query_text])

    chunk_sims = chunk_scores_from_qemb(q_emb, chunk_texts)

    return doc_score_from_chunks(chunk_sims)


# ============================================================
# 6. Rank Sources in HuggingFace Dataset
# ============================================================

def top_k_sources_for_question(hf_dataset, query_text, k=3):
    """
    hf_dataset: HuggingFace Dataset with columns:
        - source (file name)
        - text   (chunk)

    Returns:
        list[str] top-k file names only
    """

    # --- Group chunks by source ---
    grouped = defaultdict(list)

    for src, txt in zip(hf_dataset["source"], hf_dataset["text"]):
        if txt:
            grouped[str(src)].append(str(txt))

    # --- Embed query once ---
    q_emb = embed_texts([query_text])

    # --- Score each document ---
    scored = []
    for src, chunk_texts in grouped.items():
        score = bert_doc_score(query_text, chunk_texts, q_emb=q_emb)
        scored.append((src, score))

    # --- Sort descending ---
    scored.sort(key=lambda x: x[1], reverse=True)

    # --- Return only names ---
    return [src for src, _ in scored[:k]]
