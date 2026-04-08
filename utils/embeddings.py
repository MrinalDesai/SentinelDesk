import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from loguru import logger
from config.settings import settings
import uuid

client = QdrantClient(
    host=settings.QDRANT_HOST,
    port=settings.QDRANT_PORT
)

def setup_collection():
    collections = client.get_collections().collections
    names = [c.name for c in collections]
    
    if settings.COLLECTION_NAME not in names:
        client.create_collection(
            collection_name=settings.COLLECTION_NAME,
            vectors_config=VectorParams(
                size=1024,
                distance=Distance.COSINE
            )
        )
        logger.success(f"Collection '{settings.COLLECTION_NAME}' created!")
    else:
        logger.info(f"Collection '{settings.COLLECTION_NAME}' already exists")

def get_embedding(text: str) -> list:
    response = ollama.embeddings(
        model=settings.EMBEDDING_MODEL,
        prompt=text
    )
    return response['embedding']

def store_ticket(ticket_id: int, text: str, metadata: dict):
    embedding = get_embedding(text)
    
    client.upsert(
        collection_name=settings.COLLECTION_NAME,
        points=[
            PointStruct(
                id=ticket_id,
                vector=embedding,
                payload=metadata
            )
        ]
    )
    logger.info(f"Stored ticket {ticket_id} in Qdrant")

def search_similar(query: str, top_k: int = 5) -> list:
    query_embedding = get_embedding(query)
    
    results = client.query_points(
        collection_name=settings.COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
        with_payload=True
    ).points
    
    return results

if __name__ == "__main__":
    setup_collection()
    
    # Test embedding
    test_text = "VPN not connecting after password reset"
    embedding = get_embedding(test_text)
    logger.success(f"Embedding generated — dimensions: {len(embedding)}")