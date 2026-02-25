import asyncio
from typing import Any, AsyncGenerator

import agents
import gradio as gr
from dotenv import load_dotenv
from gradio.components.chatbot import ChatMessage
from langfuse import propagate_attributes
import os
import pandas as pd
from openai import AsyncOpenAI

# Load environment variables
load_dotenv()

from pageindex import PageIndexClient
import pageindex.utils as utils

pi_client = PageIndexClient(api_key=os.getenv("PAGEINDEX_API_KEY"))


def query_pageindex(query: str, doc_id: str = None):
    """
    Submit a query to PageIndex and stream the response.
    
    Parameters:
    -----------
    query : str
        The query to submit to PageIndex
    doc_id : str, optional
        Document ID to search within. If None, searches all documents.
    """
    try:
        for chunk in pi_client.chat_completions(
            messages=[{"role": "user", "content": query}],
            doc_id=doc_id,
            stream=True
        ):
            print(chunk, end='', flush=True)
        print()  # New line after streaming completes
    except Exception as e:
        print(f"Error querying PageIndex: {e}")


import json
import os
import pandas as pd
from docScore import EmbeddingClient, top_k_sources_for_question

async def interactive_query(
    questions,
    top_k: int = 1,
    answers_per_doc: int | None = 1,
    datasets_root: str = "/datasets",
    ensure_pdf_ext: bool = True,
    pageindex_api_key: str | None = os.getenv("PAGEINDEX_API_KEY"),
    embedding_model_name: str = "@cf/baai/bge-small-en-v1.5",
):
    hf_dataset = pd.read_parquet("hf://datasets/Vaibhav42/ellisdonone/data/train-00000-of-00001.parquet")
    filterOn = pd.read_csv("uploaded_documents.csv")
    hf_dataset = hf_dataset[hf_dataset["source"].isin(filterOn["file_name"])]


    if not isinstance(questions, (list, tuple)):
        raise TypeError("questions must be a list[str]")

    # Debug: Print environment variables
    print("\n=== DEBUG: Environment Variables ===")
    print(f"CLOUDFLARE_ACCOUNT_ID: {os.getenv('CLOUDFLARE_ACCOUNT_ID')}")
    print(f"EMBEDDING_API_KEY loaded: {'Yes' if os.getenv('EMBEDDING_API_KEY') else 'No'}")
    print(f"PAGEINDEX_API_KEY loaded: {'Yes' if os.getenv('PAGEINDEX_API_KEY') else 'No'}")
    print("===================================\n")

    # ---- initialize embedder once ----
    embedder = EmbeddingClient(embedding_model_name=embedding_model_name)

    # ---- PageIndex client ----
    pi_client = PageIndexClient(api_key=pageindex_api_key or os.getenv("PAGEINDEX_API_KEY"))

    def _pdf_path(file_name: str) -> str:
        file_name = str(file_name)
        if ensure_pdf_ext and not file_name.lower().endswith(".pdf"):
            file_name += ".pdf"
        return f"{datasets_root.rstrip('/')}/{file_name}"

    async def _pageindex_retrieve_sources(pi_doc_id: str, query_text: str):
        if not pi_client.is_retrieval_ready(pi_doc_id):
            return [], "Document not ready for retrieval"

        try:
            # Use PageIndex native retrieval API instead of manual tree traversal + LLM
            results = pi_client.chat_completions(
                messages=[{"role": "user", "content":query_text}],
                doc_id=pi_doc_id
            )
           
            sources = []
            for node in results.get("result", []):
                sources.append({
                    "page": node.get("page_index"),
                    "node_id": node.get("node_id"),
                    "title": node.get("title"),
                    "text": node.get("text", "")
                })

            sources.sort(key=lambda s: (s["page"] is None, s["page"]))
            print(results["choices"][0]["message"]["content"])
            return sources, None

        except Exception as e:
            return [], str(e)

    async def _answer_from_sources(query_text: str, sources: list[dict]) -> str:
        context = "\n\n".join(s["text"] for s in sources if s.get("text"))
        answer_prompt = f"""
Answer the question based ONLY on the context:

Question: {query_text}

Context:
{context}

Provide a clear, concise answer based only on the context provided.
"""
        return await call_llm(answer_prompt)

    async def _answer_per_source_node(query_text: str, sources: list[dict], limit: int) -> list[dict]:
        out = []
        for s in sources[:limit]:
            node_context = s.get("text", "")
            answer_prompt = f"""
Answer the question based ONLY on the context:

Question: {query_text}

Context:
{node_context}

Provide a clear, concise answer based only on the context provided.
"""
            ans = await call_llm(answer_prompt)
            out.append({"answer": ans, "sources": [s]})
        return out

    all_out = []
    linking_dataset = pd.read_csv("uploaded_documents.csv")

    filename_to_docid = dict(
        zip(linking_dataset["file_name"], linking_dataset["doc_id"])
    )

    for q in questions:
        # 1) Rank top_k files using embedder-backed scoring
        top_files = top_k_sources_for_question(hf_dataset, q, embedder, k=top_k)

        # 2) PageIndex each top file
        responses = []
        for file_name in top_files:
            doc_id = filename_to_docid.get(file_name)

            if doc_id is None:
                responses.append({
                    "file": file_name,
                    "doc_id": None,
                    "answer": None,
                    "sources": [],
                    "error": f"doc_id not found for filename: {file_name}"
                })
                continue

            sources, err = await _pageindex_retrieve_sources(doc_id, q)

            if err:
                responses.append({
                    "file": file_name,
                    "doc_id": doc_id,
                    "answer": None,
                    "sources": [],
                    "error": err
                })
                continue

            if answers_per_doc is None:
                answer = await _answer_from_sources(q, sources)
                responses.append({
                    "file": file_name,
                    "doc_id": doc_id,
                    "answer": answer,
                    "sources": sources
                })
            else:
                node_answers = await _answer_per_source_node(q, sources, answers_per_doc)
                for item in node_answers:
                    responses.append({
                        "file": file_name,
                        "doc_id": doc_id,
                        "answer": item["answer"],
                        "sources": item["sources"]
                    })

        all_out.append({
            "question": q,
            "top_files": top_files,
            "responses": responses
        })

    return all_out



if __name__ == "__main__":
    # Mock questions for testing
    mock_questions = [
        "How many patients in the Lomustine treatment arm experienced periodontal disease?",
        "What is the standard dosage and administration for alembic-telmisartan"
    ]
    
    # Run the interactive query with mock questions
    results = asyncio.run(interactive_query(mock_questions))
    
    # Print results
    print("\n" + "="*80)
    print("INTERACTIVE QUERY RESULTS")
    print("="*80)
    for result in results:
        print(f"\nQuestion: {result['question']}")
        print(f"Top Files: {result['top_files']}")
        print(f"Number of Responses: {len(result['responses'])}")
        for i, response in enumerate(result['responses'], 1):
            print(f"\n  Response {i}:")
            print(f"    File: {response['file']}")
            print(f"    Doc ID: {response['doc_id']}")
            if response.get('error'):
                print(f"    Error: {response['error']}")
            else:
                print(f"    Answer: {response['answer']}")
                print(f"    Sources: {len(response['sources'])} source(s)")
    
    print("\n" + "="*80)



GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")

# Initialize OpenAI-compatible client for Gemini
client = AsyncOpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)

async def call_llm(prompt, model="gemini-2.0-flash", temperature=0):
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return response.choices[0].message.content
