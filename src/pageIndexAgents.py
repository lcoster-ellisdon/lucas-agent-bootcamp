import asyncio
from typing import Any, AsyncGenerator

import agents
import gradio as gr
from dotenv import load_dotenv
from gradio.components.chatbot import ChatMessage
from langfuse import propagate_attributes
import os
import pandas as pd

from src.utils import (
    oai_agent_stream_to_gradio_messages,
    set_up_logging,
    setup_langfuse_tracer,
)
from src.utils.agent_session import get_or_create_session
from src.utils.client_manager import AsyncClientManager
from src.utils.gradio import COMMON_GRADIO_CONFIG
from src.utils.langfuse.shared_client import langfuse_client
from src.utils.tools.gemini_grounding import (
    GeminiGroundingWithGoogleSearch,
    ModelSettings,
)
from docScore import bert_doc_score
from pageindex import PageIndexClient

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

async def interactive_query(
    questions,
    hf_dataset,
    top_k: int = 3,
    answers_per_doc: int | None = None,
    datasets_root: str = "/datasets",
    ensure_pdf_ext: bool = True,
    pageindex_api_key: str | None = None,
):
    """
    For each question:
      1) Rank sources (filenames) using HF dataset chunks (top_k)
      2) Call PageIndex on each top file
      3) Return relevant answers + page sources

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
            },
            ...
          ]
        },
        ...
      ]
    """

    if not isinstance(questions, (list, tuple)):
        raise TypeError("questions must be a list[str]")

    pi_client = PageIndexClient(api_key=pageindex_api_key or PAGEINDEX_API_KEY)

    def _pdf_path(file_name: str) -> str:
        file_name = str(file_name)
        if ensure_pdf_ext and not file_name.lower().endswith(".pdf"):
            file_name += ".pdf"
        return f"{datasets_root.rstrip('/')}/{file_name}"

    async def _pageindex_retrieve_sources(pdf_path: str, query_text: str):
        """
        Returns list of sources with page/node/title/text for nodes relevant to query_text.
        """
        submitted = pi_client.submit_document(pdf_path)
        pi_doc_id = submitted["doc_id"]

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
        """
        Multiple answers per doc: one per node (each answer has a single page source).
        """
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

    for q in questions:
        # 1) Rank top_k files using your dataset scorer (returns ONLY names)
        top_files = top_k_sources_for_question(hf_dataset, q, k=top_k)

        # 2) PageIndex each top file
        responses = []
        for file_name in top_files:
            pdf_path = _pdf_path(file_name)

            sources, err = await _pageindex_retrieve_sources(pdf_path, q)
            if err:
                responses.append({
                    "file": file_name,
                    "pdf_path": pdf_path,
                    "answer": None,
                    "sources": [],
                    "error": err
                })
                continue

            if answers_per_doc is None:
                answer = await _answer_from_sources(q, sources)
                responses.append({
                    "file": file_name,
                    "pdf_path": pdf_path,
                    "answer": answer,
                    "sources": sources
                })
            else:
                node_answers = await _answer_per_source_node(q, sources, answers_per_doc)
                for item in node_answers:
                    responses.append({
                        "file": file_name,
                        "pdf_path": pdf_path,
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

