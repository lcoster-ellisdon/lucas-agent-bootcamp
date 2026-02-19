
import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

# Add parent directory to path for imports
_src_path = Path(__file__).parent.parent
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

# Import from existing codebase
from poc.embeddings import embed_texts
from utils.data.pdf_to_hf_dataset import (
    _compile_skip_patterns,
    _collect_records,
    _resolve_pdf_paths,
    _load_pymupdf,
    _resolve_openai_api_key,
    _get_openai_client,
    DEFAULT_SKIP_PATTERNS,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def batch_embed_texts(texts: list[str], batch_size: int = 32, max_length: int = 256) -> np.ndarray:
    """
    Generate embeddings for a list of texts in batches.
    
    Args:
        texts: List of text strings to embed
        batch_size: Number of texts to process in each batch
        max_length: Maximum token length for BERT
    
    Returns:
        numpy array of shape (len(texts), 768) with L2-normalized BERT embeddings
    """
    embeddings_list = []
    total_texts = len(texts)
    
    logger.info(f"Embedding {total_texts} texts in batches of {batch_size}...")
    
    for i in range(0, total_texts, batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_texts + batch_size - 1) // batch_size
        
        try:
            # embed_texts returns TensorFlow tensor, convert to numpy
            batch_embeddings = embed_texts(batch_texts, max_length=max_length)
            batch_embeddings_np = batch_embeddings.numpy()
            embeddings_list.append(batch_embeddings_np)
            
            logger.info(f"  Batch {batch_num}/{total_batches} - Embedded {len(batch_texts)} texts")
        except Exception as e:
            logger.error(f"  Batch {batch_num}/{total_batches} - Error embedding batch: {e}")
            # Add zeros for failed batch to keep alignment with texts
            embeddings_list.append(np.zeros((len(batch_texts), 768)))
    
    # Concatenate all batch embeddings
    all_embeddings = np.vstack(embeddings_list)
    
    # Verify L2 norm (should be ~1.0 for normalized embeddings)
    norms = np.linalg.norm(all_embeddings, axis=1)
    logger.info(f"Embedding norms - Min: {norms.min():.4f}, Max: {norms.max():.4f}, Mean: {norms.mean():.4f}")
    
    return all_embeddings


def process_pdfs(
    pdf_dir: Path,
    output_path: Path,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    model: str = "gemini-2.5-flash",
    max_pages_per_doc: Optional[int] = None,
    batch_embed_size: int = 32,
) -> pd.DataFrame:
    """
    Process all PDFs in a directory and create embeddings DataFrame.
    
    Args:
        pdf_dir: Directory containing PDF files
        output_path: Path where to save the Parquet file
        chunk_size: Max tokens per chunk
        chunk_overlap: Token overlap between chunks
        model: OCR model to use (OpenAI-compatible)
        max_pages_per_doc: Max pages to process per PDF (None = all)
        batch_embed_size: Batch size for embedding generation
    
    Returns:
        pandas DataFrame with embeddings and metadata
    """
    
    # Step 1: Find and validate PDFs
    logger.info(f"Scanning {pdf_dir} for PDF files...")
    pdf_paths = _resolve_pdf_paths(pdf_dir, recursive=False)
    logger.info(f"Found {len(pdf_paths)} PDFs to process")
    
    if not pdf_paths:
        logger.error("No PDFs found in the specified directory")
        return pd.DataFrame()
    
    # Step 2: Setup OCR infrastructure
    logger.info("Loading PyMuPDF library...")
    pymupdf = _load_pymupdf()
    
    logger.info("Setting up OpenAI-compatible client for OCR...")
    try:
        api_key = _resolve_openai_api_key()
        client = _get_openai_client(api_key, base_url=None)
    except ValueError as e:
        logger.error(f"Failed to setup API client: {e}")
        logger.info("Will attempt to use cached chunks if available")
        client = None
    
    # Step 3: Load tokenizer for chunking
    logger.info("Loading tokenizer (BAAI/bge-m3)...")
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    
    # Step 4: Compile skip patterns for filtering pages
    skip_patterns = _compile_skip_patterns((), use_default_skip_patterns=True)
    
    # Step 5: Process PDFs into chunks with metadata
    logger.info("Processing PDFs and extracting chunks...")
    start_time = time.time()
    
    records = _collect_records(
        pdf_paths,
        pymupdf,
        client,
        tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        model=model,
        prompt="Transcribe all readable text from this page in natural reading order. "
               "Return plain text only. Do not summarize or add commentary.",
        temperature=0.0,
        max_output_tokens=4096,
        seed=None,
        max_retries=3,
        retry_base_seconds=2.0,
        dpi=300,
        max_pages_per_doc=max_pages_per_doc,
        skip_front_pages=0,
        skip_back_pages=0,
        min_page_characters=200,
        min_page_words=0,
        skip_patterns=skip_patterns,
        skip_toc_detection=True,
        show_progress=True,
        structured_ocr=False,
        source_root=pdf_dir,
    )
    
    elapsed = time.time() - start_time
    logger.info(f"Extracted {len(records)} chunks in {elapsed:.1f}s")
    
    if not records:
        logger.error("No chunks extracted from PDFs")
        return pd.DataFrame()
    
    # Step 6: Create DataFrame from records
    logger.info("Creating DataFrame from chunks...")
    df = pd.DataFrame(records)
    
    # Step 7: Generate unique chunk IDs
    logger.info("Generating chunk IDs...")
    chunk_ids = []
    for idx, row in df.iterrows():
        # Format: PDF_filename_page_segment_chunk
        source = Path(row["source"]).stem
        chunk_id = f"{source}_{row['page_index']}_{row['segment_index']}_{row['chunk_index']}"
        chunk_ids.append(chunk_id)
    df.insert(0, "chunk_id", chunk_ids)
    
    # Step 8: Generate embeddings
    logger.info("Generating BERT embeddings...")
    start_time = time.time()
    embeddings = batch_embed_texts(
        df["text"].tolist(),
        batch_size=batch_embed_size,
        max_length=256,
    )
    elapsed = time.time() - start_time
    logger.info(f"Generated embeddings in {elapsed:.1f}s")
    
    # Step 9: Add embeddings to DataFrame
    logger.info("Adding embeddings to DataFrame...")
    df["embedding"] = [embeddings[i].tolist() for i in range(len(embeddings))]
    
    # Step 10: Save to Parquet
    logger.info(f"Saving DataFrame to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, compression="snappy", index=False)
    logger.info(f"Successfully saved {len(df)} chunks with embeddings")
    
    # Step 11: Generate and save report
    report_path = output_path.parent / (output_path.stem + "_report.json")
    report = {
        "total_chunks": len(df),
        "total_pdfs": len(pdf_paths),
        "embedding_dimension": 768,
        "embedding_model": "bert-base-uncased",
        "chunk_size_tokens": chunk_size,
        "chunk_overlap_tokens": chunk_overlap,
        "chunks_per_pdf": df.groupby("source").size().to_dict(),
        "unique_sources": df["source"].nunique(),
        "parquet_path": str(output_path),
        "processing_time_seconds": elapsed,
    }
    
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved processing report to {report_path}")
    
    # Step 12: Print summary statistics
    logger.info("\n" + "="*60)
    logger.info("EMBEDDING PIPELINE SUMMARY")
    logger.info("="*60)
    logger.info(f"Total Chunks: {len(df)}")
    logger.info(f"Total PDFs: {len(pdf_paths)}")
    logger.info(f"Unique Sources: {df['source'].nunique()}")
    logger.info(f"Avg Chunks per PDF: {len(df) / len(pdf_paths):.1f}")
    logger.info(f"Embedding Dimension: 768 (BERT)")
    logger.info(f"Embedding Model: bert-base-uncased")
    logger.info(f"Output File: {output_path}")
    logger.info(f"Output File Size: {output_path.stat().st_size / (1024*1024):.1f} MB")
    logger.info("="*60)
    
    return df


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Batch process PDFs to create embeddings and save to Parquet"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("datasets"),
        help="Directory containing PDF files (default: datasets/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("embeddings.parquet"),
        help="Output Parquet file path (default: embeddings.parquet)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Max tokens per chunk (default: 512)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=64,
        help="Token overlap between chunks (default: 64)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help="OCR model (default: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--max-pages-per-doc",
        type=int,
        default=None,
        help="Max pages per PDF (default: all)",
    )
    parser.add_argument(
        "--batch-embed-size",
        type=int,
        default=32,
        help="Batch size for embedding generation (default: 32)",
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not args.input_dir.exists():
        logger.error(f"Input directory not found: {args.input_dir}")
        sys.exit(1)
    
    if not args.input_dir.is_dir():
        logger.error(f"Input path is not a directory: {args.input_dir}")
        sys.exit(1)
    
    # Process PDFs
    try:
        df = process_pdfs(
            pdf_dir=args.input_dir,
            output_path=args.output,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            model=args.model,
            max_pages_per_doc=args.max_pages_per_doc,
            batch_embed_size=args.batch_embed_size,
        )
        logger.info("✓ Batch embedding pipeline completed successfully!")
    except Exception as e:
        logger.error(f"✗ Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
