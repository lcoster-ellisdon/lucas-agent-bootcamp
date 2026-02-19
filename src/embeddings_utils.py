import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Try to import embeddings for query embedding generation
try:
    from poc.embeddings import embed_texts
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logging.warning("embeddings module not available - query embedding will be unavailable")

# Try to import docScore for score computation
try:
    from poc.docScore import chunk_scores, doc_score_from_chunks
    DOCSCORE_AVAILABLE = True
except ImportError:
    DOCSCORE_AVAILABLE = False
    logging.warning("docScore module not available - doc_score computation will be unavailable")

logger = logging.getLogger(__name__)


class EmbeddingsDataFrame:
    """Wrapper class for efficient embedding operations on a DataFrame."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with a DataFrame containing embeddings.
        
        Args:
            df: DataFrame with columns: chunk_id, text, embedding, source, page_index, etc.
        """
        self.df = df
        
        # Validate required columns
        required_cols = ["chunk_id", "text", "embedding"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"DataFrame missing required columns: {missing_cols}")
        
        # Convert embeddings to numpy array matrix for efficient similarity computation
        logger.info("Converting embeddings to numpy matrix...")
        embeddings_list = []
        for emb in df["embedding"]:
            if isinstance(emb, list):
                embeddings_list.append(np.array(emb, dtype=np.float32))
            elif isinstance(emb, np.ndarray):
                embeddings_list.append(emb.astype(np.float32))
            else:
                raise ValueError(f"Unsupported embedding type: {type(emb)}")
        
        self.embeddings_matrix = np.vstack(embeddings_list)
        logger.info(f"Embeddings matrix shape: {self.embeddings_matrix.shape}")
        
        # Create index for fast lookup by chunk_id
        self.chunk_id_to_idx = {cid: idx for idx, cid in enumerate(df["chunk_id"])}
        logger.info(f"Created index with {len(self.chunk_id_to_idx)} chunk IDs")
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[dict]:
        """
        Retrieve a single chunk by its chunk_id.
        
        Args:
            chunk_id: The unique chunk identifier
        
        Returns:
            Dictionary with chunk data (text, embedding, metadata) or None if not found
        """
        idx = self.chunk_id_to_idx.get(chunk_id)
        if idx is None:
            return None
        
        row = self.df.iloc[idx]
        return {
            "chunk_id": row["chunk_id"],
            "text": row["text"],
            "embedding": self.embeddings_matrix[idx],
            "source": row["source"],
            "page_index": row["page_index"],
            "section_title": row.get("section_title"),
            "section_level": row.get("section_level"),
            "segment_index": row.get("segment_index"),
            "chunk_index": row.get("chunk_index"),
        }
    
    def search_by_embedding(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        similarity_threshold: float = 0.0,
        source_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Find top-k most similar chunks using cosine similarity.
        
        Args:
            query_embedding: numpy array of shape (768,) or (1, 768)
            k: Number of results to return
            similarity_threshold: Minimum cosine similarity (0.0 to 1.0)
            source_filter: Optional PDF source filename to filter results
        
        Returns:
            List of dicts with chunk_id, text, similarity_score, metadata
        """
        # Ensure query embedding is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Convert to float32 for consistency
        query_embedding = query_embedding.astype(np.float32)
        
        # Compute cosine similarities
        similarities = cosine_similarity(query_embedding, self.embeddings_matrix)[0]
        
        # Filter by threshold
        valid_indices = np.where(similarities >= similarity_threshold)[0]
        
        # If source filter is specified, further filter
        if source_filter:
            valid_indices = [
                idx for idx in valid_indices
                if self.df.iloc[idx]["source"] == source_filter
            ]
        
        # Sort by similarity (descending) and get top-k
        top_indices = sorted(
            valid_indices,
            key=lambda idx: similarities[idx],
            reverse=True
        )[:k]
        
        # Build result list
        results = []
        for idx in top_indices:
            row = self.df.iloc[idx]
            results.append({
                "chunk_id": row["chunk_id"],
                "text": row["text"],
                "similarity_score": float(similarities[idx]),
                "source": row["source"],
                "page_index": row["page_index"],
                "section_title": row.get("section_title"),
                "section_level": row.get("section_level"),
            })
        
        return results
    
    def search_by_text(
        self,
        query_text: str,
        k: int = 5,
        similarity_threshold: float = 0.0,
        source_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Find top-k most similar chunks by embedding a query text.
        
        Args:
            query_text: Text to search for
            k: Number of results to return
            similarity_threshold: Minimum cosine similarity (0.0 to 1.0)
            source_filter: Optional PDF source filename to filter results
        
        Returns:
            List of dicts with chunk_id, text, similarity_score, metadata
        """
        if not EMBEDDINGS_AVAILABLE:
            raise RuntimeError("embeddings module not available for query generation")
        
        # Generate embedding for query text
        query_embedding = embed_texts([query_text], max_length=256)
        query_embedding = query_embedding.numpy().astype(np.float32)
        
        # Use existing search method
        return self.search_by_embedding(
            query_embedding,
            k=k,
            similarity_threshold=similarity_threshold,
            source_filter=source_filter,
        )
    
    def compute_doc_score(
        self,
        query_text: str,
        chunk_ids: list[str],
    ) -> dict:
        """
        Compute document score for a set of chunks given a query.
        
        Uses the DocScore metric: (1/sqrt(N+1)) * sum(ChunkScore)
        where ChunkScore is cosine similarity between query and chunk.
        
        Args:
            query_text: Text to score against
            chunk_ids: List of chunk IDs to use for scoring
        
        Returns:
            Dictionary with:
                - doc_score: float (aggregate score)
                - chunk_scores: list of float (individual scores)
                - num_chunks: int
        """
        if not DOCSCORE_AVAILABLE:
            # Fallback: compute simple average similarity
            logger.warning("docScore module not available, using average similarity")
            return self._compute_simple_score(query_text, chunk_ids)
        
        # Get chunk texts
        chunk_texts = []
        valid_chunks = []
        for chunk_id in chunk_ids:
            chunk = self.get_chunk_by_id(chunk_id)
            if chunk:
                chunk_texts.append(chunk["text"])
                valid_chunks.append(chunk_id)
        
        if not chunk_texts:
            return {
                "doc_score": 0.0,
                "chunk_scores": [],
                "num_chunks": 0,
            }
        
        # Compute scores using docScore module
        try:
            scores = chunk_scores(query_text, chunk_texts)
            doc_score = doc_score_from_chunks(scores)
        except Exception as e:
            logger.error(f"Error computing doc_score: {e}")
            return self._compute_simple_score(query_text, valid_chunks)
        
        return {
            "doc_score": float(doc_score),
            "chunk_scores": [float(s) for s in scores],
            "num_chunks": len(chunk_texts),
        }
    
    def _compute_simple_score(
        self,
        query_text: str,
        chunk_ids: list[str],
    ) -> dict:
        """Fallback: compute average cosine similarity score."""
        if not EMBEDDINGS_AVAILABLE:
            logger.warning("Cannot compute scores without embeddings module")
            return {
                "doc_score": 0.0,
                "chunk_scores": [],
                "num_chunks": len(chunk_ids),
            }
        
        # Embed query
        query_emb = embed_texts([query_text], max_length=256).numpy()
        
        # Get similarities for each chunk
        scores = []
        for chunk_id in chunk_ids:
            chunk = self.get_chunk_by_id(chunk_id)
            if chunk:
                chunk_emb = np.array(chunk["embedding"], dtype=np.float32).reshape(1, -1)
                sim = cosine_similarity(query_emb.astype(np.float32), chunk_emb)[0, 0]
                scores.append(float(sim))
        
        avg_score = np.mean(scores) if scores else 0.0
        return {
            "doc_score": float(avg_score),
            "chunk_scores": scores,
            "num_chunks": len(scores),
        }
    
    def filter_by_source(self, source: str) -> "EmbeddingsDataFrame":
        """Return a new EmbeddingsDataFrame filtered to a specific source PDF."""
        filtered_df = self.df[self.df["source"] == source].reset_index(drop=True)
        if len(filtered_df) == 0:
            raise ValueError(f"No chunks found for source: {source}")
        return EmbeddingsDataFrame(filtered_df)
    
    def get_sources(self) -> list[str]:
        """Get list of unique PDF sources."""
        return sorted(self.df["source"].unique().tolist())
    
    def get_stats(self) -> dict:
        """Get summary statistics."""
        return {
            "total_chunks": len(self.df),
            "unique_sources": self.df["source"].nunique(),
            "embedding_dimension": self.embeddings_matrix.shape[1],
            "sources": self.get_sources(),
            "chunks_per_source": self.df.groupby("source").size().to_dict(),
        }


def load_embeddings_df(parquet_path: Path) -> EmbeddingsDataFrame:
    """
    Load embeddings DataFrame from Parquet file.
    
    Args:
        parquet_path: Path to the .parquet file
    
    Returns:
        EmbeddingsDataFrame instance for efficient searching
    """
    logger.info(f"Loading embeddings from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded {len(df)} chunks")
    return EmbeddingsDataFrame(df)


def batch_embed(texts: list[str], batch_size: int = 32, max_length: int = 256) -> np.ndarray:
    """
    Generate embeddings for a list of texts in batches.
    
    Args:
        texts: List of text strings to embed
        batch_size: Number of texts to process in each batch
        max_length: Maximum token length for BERT
    
    Returns:
        numpy array of shape (len(texts), 768) with L2-normalized BERT embeddings
    """
    if not EMBEDDINGS_AVAILABLE:
        raise RuntimeError("embeddings module not available")
    
    embeddings_list = []
    total_texts = len(texts)
    
    logger.info(f"Embedding {total_texts} texts in batches of {batch_size}...")
    
    for i in range(0, total_texts, batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_texts + batch_size - 1) // batch_size
        
        try:
            batch_embeddings = embed_texts(batch_texts, max_length=max_length)
            batch_embeddings_np = batch_embeddings.numpy().astype(np.float32)
            embeddings_list.append(batch_embeddings_np)
            logger.info(f"  Batch {batch_num}/{total_batches}")
        except Exception as e:
            logger.error(f"  Batch {batch_num}/{total_batches} - Error: {e}")
            embeddings_list.append(np.zeros((len(batch_texts), 768), dtype=np.float32))
    
    return np.vstack(embeddings_list)
