import os
import logging
from typing import List, Dict, Any

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

# --- 설정 (indexer.py와 반드시 동일해야 합니다) ---
load_dotenv()
CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "school_announcements"
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
collection = client.get_or_create_collection()

def search(query, n_results):
    query = query.replace("\n", " ")
    response = client.embeddings.create(input=[query], model=EMBED_MODEL)
    query_embedding = response.data[0].embedding
    
    # 1 유사도 검색
    similarity_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]    
    )
    # 2 키워드 검색
    
    # 3 RRF 알고리즘 사용해서 결과 통합