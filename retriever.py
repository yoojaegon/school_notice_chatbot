import os
import logging
import pickle
from typing import List, Dict, Any
import numpy as np

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi
from konlpy.tag import Okt

# --- 설정 (indexer.py와 반드시 동일해야 합니다) ---
load_dotenv()
CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "school_announcements"
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BM25_INDEX_PATH = os.path.join(CHROMA_DB_DIR, "bm25_index.pkl")

# --- 로깅 설정 ---
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 클라이언트 및 인덱스 로드 ---
try:
    # ChromaDB 클라이언트
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    
    # OpenAI 클라이언트
    oai_client = OpenAI(api_key=OPENAI_API_KEY)

    # BM25 인덱스 로드
    if os.path.exists(BM25_INDEX_PATH):
        with open(BM25_INDEX_PATH, 'rb') as f:
            bm25_data = pickle.load(f)
            bm25_index: BM25Okapi = bm25_data['bm25']
            bm25_doc_ids: List[str] = bm25_data['doc_ids']
        log.info(f"BM25 인덱스를 로드했습니다. ({len(bm25_doc_ids)}개 문서)")
    else:
        bm25_index = None
        bm25_doc_ids = None
        log.warning(f"BM25 인덱스 파일('{BM25_INDEX_PATH}')을 찾을 수 없습니다. 키워드 검색을 비활성화합니다.")

    # Okt 형태소 분석기 초기화
    okt = Okt()
    log.info("KoNLPy Okt 형태소 분석기를 로드했습니다.")

except Exception as e:
    log.critical(f"초기화 중 오류 발생: {e}", exc_info=True)
    chroma_client = None
    collection = None
    oai_client = None
    bm25_index = None
    okt = None


def hybrid_search(query: str, n_results: int = 10, rrf_k: int = 60) -> List[Dict[str, Any]]:
    """
    유사도 검색(Vector)과 키워드 검색(BM25)을 결합한 하이브리드 검색을 수행합니다.
    결과는 RRF(Reciprocal Rank Fusion) 알고리즘으로 재정렬됩니다.
    """
    if not collection or not oai_client:
        log.error("클라이언트가 초기화되지 않았습니다. 검색을 수행할 수 없습니다.")
        return []

    query_norm = query.replace("\n", " ").strip()
    
    # --- 1. 유사도 검색 (Vector Search) ---
    log.info(f"유사도 검색 수행: '{query_norm}'")
    try:
        response = oai_client.embeddings.create(input=[query_norm], model=EMBED_MODEL)
        query_embedding = response.data[0].embedding
        
        sim_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["metadatas"] # RRF에서는 순위만 필요하므로 documents는 나중에 한번에 가져옵니다.
        )
        sim_ids = sim_results['ids'][0]
        log.info(f"유사도 검색 결과: {len(sim_ids)}개")
    except Exception as e:
        log.error(f"유사도 검색 중 오류 발생: {e}", exc_info=True)
        sim_ids = []

    # --- 2. 키워드 검색 (BM25) ---
    bm25_ids = []
    if bm25_index and bm25_doc_ids and okt:
        log.info(f"BM25 키워드 검색 수행: '{query_norm}'")
        try:
            # indexer와 동일한 토크나이저(Okt)와 전처리 로직을 사용합니다.
            tokenized_query = okt.morphs(query_norm, stem=True)
            # 불용어 처리
            tokenized_query = [token for token in tokenized_query if len(token) > 1]
            log.debug(f"BM25 토큰화된 쿼리: {tokenized_query}")

            doc_scores = bm25_index.get_scores(tokenized_query)

            sorted_bm25_indices = np.argsort(doc_scores)[::-1]
            bm25_ids = [bm25_doc_ids[i] for i in sorted_bm25_indices[:n_results]]
            log.info(f"BM25 검색 결과: {len(bm25_ids)}개")
        except Exception as e:
            log.error(f"BM25 검색 중 오류 발생: {e}", exc_info=True)
            bm25_ids = []
    else:
        log.warning("BM25 인덱스 또는 Okt 분석기가 없어 키워드 검색을 건너뜁니다.")

    # --- 3. RRF(Reciprocal Rank Fusion)로 결과 통합 ---
    log.info("RRF를 사용하여 결과 랭킹 재조정")
    rank_scores = {}
    
    for rank, doc_id in enumerate(sim_ids):
        rank_scores[doc_id] = rank_scores.get(doc_id, 0) + 1 / (rrf_k + rank + 1)

    for rank, doc_id in enumerate(bm25_ids):
        rank_scores[doc_id] = rank_scores.get(doc_id, 0) + 1 / (rrf_k + rank + 1)
        
    if not rank_scores:
        log.warning("검색 결과가 없습니다.")
        return []

    sorted_fused_ids = sorted(rank_scores.keys(), key=lambda id: rank_scores[id], reverse=True)
    final_ids = sorted_fused_ids[:n_results]
    log.info(f"최종 융합 결과: {len(final_ids)}개 문서 ID")
    
    # ChromaDB에서 최종 문서 정보 조회
    final_results = collection.get(ids=final_ids, include=["documents", "metadatas"])
    
    # RRF 점수 순서대로 결과 재정렬
    id_map = {
        doc_id: {
            "id": doc_id,
            "document": final_results['documents'][i],
            "metadata": final_results['metadatas'][i],
            "score": rank_scores[doc_id]
        } for i, doc_id in enumerate(final_results['ids'])
    }
    
    ordered_results = [id_map[doc_id] for doc_id in final_ids if doc_id in id_map]

    return ordered_results


if __name__ == '__main__':
    # 테스트용 예시
    test_query = "장학금 신청 방법"
    log.info(f"\n--- 테스트 검색 시작: '{test_query}' ---")
    results = hybrid_search(test_query, n_results=5)
    
    if results:
        print(f"\n--- 총 {len(results)}개의 검색 결과 ---")
        for i, res in enumerate(results):
            print(f"\n[결과 {i+1}] (ID: {res['id']}, Score: {res['score']:.4f})")
            print(f"  - 출처: {res['metadata'].get('announcement_title', 'N/A')}")
            print(f"  - 내용: {res['document'][:150].replace(chr(10), ' ')}...")
    else:
        print("검색 결과가 없습니다.")