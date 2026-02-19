import tensorflow as tf
from transformers import BertTokenizer, TFBertModel

# 1) Load model + tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = TFBertModel.from_pretrained("bert-base-uncased")

def mean_pooling(last_hidden_state, attention_mask):
    """
    last_hidden_state: (batch, seq_len, hidden)
    attention_mask:    (batch, seq_len)
    """
    mask = tf.cast(tf.expand_dims(attention_mask, axis=-1), tf.float32)  # (batch, seq_len, 1)
    summed = tf.reduce_sum(last_hidden_state * mask, axis=1)             # (batch, hidden)
    counts = tf.reduce_sum(mask, axis=1)                                 # (batch, 1)
    return summed / tf.maximum(counts, 1e-9)

def embed_texts(texts, max_length=256):
    """
    Returns L2-normalized embeddings: (batch, hidden)
    """
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="tf"
    )
    outputs = bert(**inputs)
    pooled = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
    # L2 normalize so cosine similarity becomes dot product
    pooled = tf.math.l2_normalize(pooled, axis=1)
    return pooled
    import re
from typing import Dict, List, Tuple, Any

def _split_with_overlap_words(text: str, chunk_size_words: int, overlap_words: int) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    words = text.split()
    out = []
    start = 0
    step = max(1, chunk_size_words - overlap_words)
    while start < len(words):
        end = start + chunk_size_words
        out.append(" ".join(words[start:end]))
        start += step
    return out


def chunks_from_pageindex_tree(
    tree: Any,
    *,
    min_words: int = 120,
    max_words: int = 450,
    split_overlap_words: int = 80,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Build embedding chunks from PageIndex tree nodes.

    Parameters
    ----------
    tree : Any
      The tree result from pi_client.get_tree(... )['result'].

    min_words : int
      Nodes smaller than this may be merged with nearby nodes (same-ish page order).

    max_words : int
      Nodes larger than this are split into multiple overlapping chunks.

    split_overlap_words : int
      Overlap used when splitting large nodes.

    Returns
    -------
    chunk_texts : list[str]
    chunk_meta  : list[dict]  (same length as chunk_texts)
      Each meta includes: node_id, title, page, and optionally merged_from / part_index.
    """
    node_map = utils.create_node_mapping(tree)

    # Create a linear reading order. If PageIndex provides a better traversal utility, use it.
    # We'll sort nodes primarily by page_index, then node_id as a stable tie-breaker.
    nodes = list(node_map.values())
    nodes.sort(key=lambda n: (n.get("page_index") is None, n.get("page_index", 10**9), str(n.get("node_id", ""))))

    chunk_texts: List[str] = []
    chunk_meta: List[Dict[str, Any]] = []

    def add_chunk(text: str, meta: Dict[str, Any]):
        text = re.sub(r"\s+", " ", (text or "")).strip()
        if not text:
            return
        chunk_texts.append(text)
        chunk_meta.append(meta)

    buffer_texts = []
    buffer_meta = []

    def flush_buffer():
        if not buffer_texts:
            return
        merged_text = "\n\n".join(buffer_texts).strip()
        merged_from = [
            {
                "node_id": m["node_id"],
                "title": m["title"],
                "page": m["page"],
            }
            for m in buffer_meta
        ]
        # Use the first node as the "primary" location, but keep merged_from for traceability
        primary = buffer_meta[0]
        words = merged_text.split()

        if len(words) > max_words:
            parts = _split_with_overlap_words(merged_text, max_words, split_overlap_words)
            for i, part in enumerate(parts):
                add_chunk(part, {
                    "node_id": primary["node_id"],
                    "title": primary["title"],
                    "page": primary["page"],
                    "part_index": i,
                    "merged_from": merged_from,
                })
        else:
            add_chunk(merged_text, {
                "node_id": primary["node_id"],
                "title": primary["title"],
                "page": primary["page"],
                "merged_from": merged_from if len(merged_from) > 1 else None,
            })

        buffer_texts.clear()
        buffer_meta.clear()

    for node in nodes:
        text = node.get("text", "")
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue

        meta = {
            "node_id": node.get("node_id"),
            "title": node.get("title"),
            "page": node.get("page_index"),
        }

        n_words = len(text.split())

        # If node is huge, flush buffer then split this node alone
        if n_words > max_words:
            flush_buffer()
            parts = _split_with_overlap_words(text, max_words, split_overlap_words)
            for i, part in enumerate(parts):
                add_chunk(part, {**meta, "part_index": i})
            continue

        # If node is tiny, accumulate to buffer for merging
        if n_words < min_words:
            buffer_texts.append(text)
            buffer_meta.append(meta)
            # Flush if buffer is getting too big
            if len(" ".join(buffer_texts).split()) >= max_words:
                flush_buffer()
            continue

        # Otherwise medium sized: flush any pending buffer then add node as its own chunk
        flush_buffer()
        add_chunk(text, meta)

    flush_buffer()

    # Remove merged_from=None for cleanliness
    for m in chunk_meta:
        if m.get("merged_from") is None:
            m.pop("merged_from", None)

    return chunk_texts, chunk_meta
