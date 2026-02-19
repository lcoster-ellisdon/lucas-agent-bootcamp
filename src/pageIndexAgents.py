import asyncio
from typing import Any, AsyncGenerator

import agents
import gradio as gr
from dotenv import load_dotenv
from gradio.components.chatbot import ChatMessage
from langfuse import propagate_attributes
import os
import pandas as pd
import openai

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
from docScore import EmbeddingClient, top_k_sources_for_question  # adjust import path as needed

async def interactive_query(
    questions,
    
    top_k: int = 3,
    answers_per_doc: int | None = 1,
    datasets_root: str = "/datasets",
    ensure_pdf_ext: bool = True,
    pageindex_api_key: str | None = os.getenv("PAGEINDEX_API_KEY"),
    # --- embedder config ---
    embedding_model_name: str = "bge-small-en-v1.5",
    embedding_api_key: str | None = os.getenv("EMBEDDING_API_KEY"),
    embedding_base_url: str | None = os.getenv("EMBEDDING_BASE_URL"),
):
    hf_dataset = pd.read_parquet("hf://datasets/Vaibhav42/ellisdonone/data/train-00000-of-00001.parquet")

    """
    For each question:
      1) Use HF dataset chunks + OpenAI embeddings (via EmbeddingClient) to pick top_k files
      2) Call PageIndex on each top file
      3) Return answers + page sources

    Returns:
      [
        {
          "question": "...",
          "top_files": ["a.pdf","b.pdf","c.pdf"],
          "responses": [
            {
              "file": "a.pdf",
              "pdf_path": "/datasets/a.pdf",
              "answer": "...",
              "sources": [{"page": 5, "node_id": "...", "title": "...", "text": "..."}],
              "error": "..." (optional)
            }, ...
          ]
        }, ...
      ]
    """

    if not isinstance(questions, (list, tuple)):
        raise TypeError("questions must be a list[str]")

    # ---- initialize embedder once ----
    embedder = EmbeddingClient(
    
    )

    # ---- PageIndex client ----
    pi_client = PageIndexClient(api_key=pageindex_api_key or PAGEINDEX_API_KEY)

    def _pdf_path(file_name: str) -> str:
        file_name = str(file_name)
        if ensure_pdf_ext and not file_name.lower().endswith(".pdf"):
            file_name += ".pdf"
        return f"{datasets_root.rstrip('/')}/{file_name}"

    async def _pageindex_retrieve_sources(pi_doc_id: str, query_text: str):

        if not pi_client.is_retrieval_ready(pi_doc_id):
            return [], "Document not ready for retrieval"

        tree = pi_client.get_tree(pi_doc_id, node_summary=True)["result"]
        node_map = utils.create_node_mapping(tree)
        tree_without_text = utils.remove_fields(tree.copy(), fields=["text"])

        search_prompt = f"""
You are given a question and a tree structure of a document.
Each node contains a node id, node title, and a corresponding summary.
Your task is to find all nodes that are likely to contain the answer to the question.

Question: {query_text}

Document tree structure:
{json.dumps(tree_without_text, indent=2)}

Please reply in the following JSON format:
{{
  "node_list": ["node_id_1", "node_id_2", "..."]
}}
Directly return the final JSON structure. Do not output anything else.
"""
        tree_search_result = await call_llm(search_prompt)
        node_list = json.loads(tree_search_result).get("node_list", [])

        sources = []
        for node_id in node_list:
            node = node_map.get(node_id)
            if not node:
                continue
            sources.append({
                "page": node.get("page_index"),
                "node_id": node.get("node_id"),
                "title": node.get("title"),
                "text": node.get("text", "")
            })

        sources.sort(key=lambda s: (s["page"] is None, s["page"]))
        return sources, None

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
            out.append({"answer": ans, "sources": [s]})  # single page source
        return out

    all_out = []
    linking_dataset = pd.read_csv("uploaded_documents.csv")  # expects columns: filename, doc_id

    # Create a fast lookup dictionary: filename -> doc_id
    filename_to_docid = dict(
        zip(linking_dataset["file_name"], linking_dataset["doc_id"])
    )

    for q in questions:
        # 1) Rank top_k files using embedder-backed scoring
        top_files = top_k_sources_for_question(hf_dataset, q, embedder, k=top_k)

        # 2) PageIndex each top file
        responses = []
        for file_name in top_files:
            # 🔹 Get doc_id from linking dataset
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

            # 🔹 Call PageIndex using doc_id
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
                        "sources": item["sources"]  # single-page citation
                    })

        all_out.append({
            "question": q,
            "top_files": top_files,
            "responses": responses
        })

    return all_out



if __name__ == "__main__":
    interactive_query()



OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

async def call_llm(prompt, model="gpt-4.1", temperature=0):
    client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return response.choices[0].message.content.strip()
