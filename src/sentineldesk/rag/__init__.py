"""Retrieval / resolution layer. Currently: minimal Graph RAG (symptom -> cause -> resolution)."""

from .graph_rag import GraphResult, KnowledgeGraph

__all__ = ["KnowledgeGraph", "GraphResult"]
