# indexer.py

import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

from preprocessing.handler import process_file

# --- 환경 변수 및 상수 설정 ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CRAWLED_DATA_DIR = "./crawled_data"  # 크롤링된 파일이 저장된 디렉토리
CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "school_announcements"

def main():
    """
    crawled_data 디렉토리의 모든 파일을 처리하여 ChromaDB에 색인합니다.
    """
    # 1. ChromaDB 클라이언트 및 임베딩 함수 설정
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name="text-embedding-3-small"
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=openai_ef,
        metadata={"hnsw:space": "cosine"} # 코사인 유사도 사용
    )

    # 2. 처리할 파일 목록 가져오기
    if not os.path.exists(CRAWLED_DATA_DIR):
        print(f"'{CRAWLED_DATA_DIR}' 디렉토리가 없습니다. 먼저 데이터를 크롤링해주세요.")
        return
        
    files_to_process = []
    for root, _, filenames in os.walk(CRAWLED_DATA_DIR):
        for filename in filenames:
            files_to_process.append(os.path.join(root, filename))

    # 3. 각 파일 처리 및 DB에 추가
    for file_path in files_to_process:
        print(f"\n--- 파일 처리 시작: {os.path.basename(file_path)} ---")
        
        structured_chunks = process_file(file_path)
        
        if not structured_chunks:
            print("처리할 청크가 없습니다. 다음 파일로 넘어갑니다.")
            continue
            
        # ChromaDB에 추가할 데이터 준비
        ids = []
        documents = []
        metadatas = []
        
        for i, chunk_data in enumerate(structured_chunks):
            # 각 청크에 대한 고유 ID 생성
            base_filename = os.path.basename(file_path)
            chunk_id = f"{base_filename}_{chunk_data['metadata']['type']}_{chunk_data['metadata']['chunk_index']}"
            
            ids.append(chunk_id)
            documents.append(chunk_data['text'])
            metadatas.append(chunk_data['metadata'])

        # DB에 데이터 추가
        try:
            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            print(f"✅ 성공: {len(ids)}개의 청크를 DB에 저장했습니다.")
        except Exception as e:
            print(f"❌ 실패: DB 저장 중 오류 발생 - {e}")

if __name__ == "__main__":
    main()