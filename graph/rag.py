"""
RAG 向量检索 — 替代手写 rag.py

基于 FAISS + HuggingFace Embeddings 的文档检索。
复用已有 rag_index.json 或从文件目录重新索引。
"""

import json
import os
import glob
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool

_EMBEDDINGS = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
_RAG_INDEX_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rag_index.json")

_store = None


def _get_store():
    global _store
    if _store is not None:
        return _store

    if os.path.exists(_RAG_INDEX_PATH):
        with open(_RAG_INDEX_PATH, encoding="utf-8") as f:
            data = json.load(f)
        chunks = data.get("chunks", [])
        if chunks:
            _store = FAISS.from_texts(chunks, _EMBEDDINGS)
            print(f"[RAG] 已加载 {len(chunks)} 个文档片段")
            return _store

    _store = FAISS.from_texts(["[占位]"], _EMBEDDINGS)
    return _store


def ingest(file_path: str) -> bool:
    """加载单个文件到向量库"""
    if not os.path.exists(file_path):
        return False
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    if not text.strip():
        return False

    store = _get_store()
    store.add_texts([text])
    return True


def ingest_directory(dir_path: str) -> int:
    """加载目录下所有支持的文档"""
    total = 0
    for ext in ["*.md", "*.py", "*.txt", "*.yaml", "*.yml"]:
        for f in sorted(glob.glob(os.path.join(dir_path, ext))):
            if ingest(f):
                total += 1
    print(f"[RAG] 目录加载完成，共 {total} 个文件")
    return total


@tool
def rag_query(query: str, top_k: int = 3) -> str:
    """
    从本地文档库中检索与问题相关的知识。
    当用户问到产品文档、API 文档、项目知识库内容时使用。

    参数:
        query: 搜索关键词或问题
        top_k: 返回结果数量（默认3）
    """
    store = _get_store()
    docs = store.similarity_search(query, k=top_k)
    if not docs or (len(docs) == 1 and docs[0].page_content == "[占位]"):
        return "未在本地文档中找到相关信息"
    parts = ["以下是从文档中检索到的相关信息：\n"]
    for i, d in enumerate(docs, 1):
        parts.append(f"[{i}] {d.page_content[:500]}")
    return "\n\n".join(parts)
