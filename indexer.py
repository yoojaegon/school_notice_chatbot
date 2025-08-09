# indexer.py

import os
import logging
import logging.handlers
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from tqdm import tqdm

# 전처리 패키지의 메인 컨트롤러 함수를 임포트합니다.
from preprocessing.handler import process_file

# --- 로깅 설정 ---
def setup_logging():
    """로그 설정: 콘솔과 파일에 다른 레벨로 로그를 남깁니다."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "indexer.log")

    logger = logging.getLogger()
    # 핸들러가 이미 설정된 경우 중복 추가를 방지합니다.
    if logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 콘솔 핸들러 (INFO 레벨 이상)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 파일 핸들러 (DEBUG 레벨 이상, 파일 로테이션)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file_path, maxBytes=5*1024*1024, backupCount=5, encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

# --- 환경 변수 및 상수 설정 ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CRAWLED_DATA_DIR = "./crawled_data"
CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "school_announcements"

log = logging.getLogger(__name__)

def get_post_id_from_path(file_path: str) -> str:
    """
    파일 경로에서 게시물 ID(폴더명의 첫 부분)를 추출합니다.
    예: '.../crawled_data/일반공지/1065997_제목/content.txt' -> '1065997'
    """
    try:
        # Path: .../crawled_data/일반공지/1065997_제목
        post_directory_path = os.path.dirname(file_path)
        # Name: 1065997_제목
        post_folder_name = os.path.basename(post_directory_path)
        # ID: 1065997
        post_id = post_folder_name.split('_', 1)[0]
        if post_id.isdigit():
            return post_id
        else:
            # 게시판 폴더 바로 아래에 있는 파일의 경우 (예: 일반공지/temp.txt)
            log.debug(f"경로에서 숫자 형식의 post_id를 찾지 못했습니다: {file_path}")
            return "unknown"
    except IndexError:
        log.warning(f"경로 구조가 예상과 달라 post_id를 추출할 수 없습니다: {file_path}")
        return "unknown"

def main():
    """
    crawled_data 디렉토리의 모든 파일을 처리하여 ChromaDB에 색인합니다.
    """
    log.info("="*50)
    log.info("데이터 색인 작업을 시작합니다.")
    log.info("="*50)

    # 1. ChromaDB 클라이언트 및 임베딩 함수 설정
    log.info("ChromaDB 클라이언트 및 임베딩 함수를 설정합니다.")
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name="text-embedding-3-small"
        )
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=openai_ef,
            metadata={"hnsw:space": "cosine"}
        )
        log.info(f"ChromaDB 컬렉션 '{COLLECTION_NAME}'을(를) 성공적으로 준비했습니다.")
    except Exception as e:
        log.critical("ChromaDB 설정 중 심각한 오류가 발생했습니다. API 키 등을 확인해주세요.", exc_info=True)
        return

    # 2. 처리할 파일 목록 가져오기
    if not os.path.exists(CRAWLED_DATA_DIR):
        log.error(f"'{CRAWLED_DATA_DIR}' 디렉토리가 없습니다. 먼저 'crawler.py'를 실행해주세요.")
        return
        
    files_to_process = []
    for root, _, filenames in os.walk(CRAWLED_DATA_DIR):
        for filename in filenames:
            if not filename.startswith('.'):
                files_to_process.append(os.path.join(root, filename))
    
    if not files_to_process:
        log.warning(f"'{CRAWLED_DATA_DIR}' 디렉토리에 처리할 파일이 없습니다.")
        return

    log.info(f"총 {len(files_to_process)}개의 파일을 처리 대상으로 찾았습니다.")

    # 3. 각 파일 처리 및 DB에 추가
    for file_path in tqdm(files_to_process, desc="파일 색인 중"):
        log.debug(f"--- 파일 처리 시작: {file_path} ---")
        
        post_id = get_post_id_from_path(file_path)
        if post_id == "unknown":
            log.warning(f"게시물 ID를 알 수 없는 파일은 건너뜁니다: {file_path}")
            continue

        # handler.py를 통해 전처리된 청크 리스트를 받아옵니다.
        structured_chunks = process_file(file_path)
        
        if not structured_chunks:
            log.warning(f"처리할 청크가 생성되지 않았습니다: {os.path.basename(file_path)}")
            continue
            
        ids, documents, metadatas = [], [], []
        
        for i, chunk_data in enumerate(structured_chunks):
            chunk_metadata = chunk_data['metadata']
            chunk_metadata['post_id'] = post_id
            
            base_filename = os.path.basename(file_path)
            chunk_id = f"{post_id}_{base_filename}_{chunk_metadata.get('type', 'chunk')}_{i}"
            
            ids.append(chunk_id)
            documents.append(chunk_data['page_content']) # 'page_content' 키 사용
            metadatas.append(chunk_metadata)

        # DB에 데이터 추가 (upsert는 ID가 같으면 덮어쓰므로 재실행에 안전)
        try:
            collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            log.info(f"✅ 성공: {len(ids)}개의 청크(post_id: {post_id})를 DB에 저장했습니다.")
        except Exception as e:
            log.error(f"❌ 실패: '{file_path}' 파일의 청크를 DB에 저장하는 중 오류 발생", exc_info=True)

    log.info("="*50)
    log.info("모든 파일의 색인 작업이 완료되었습니다.")
    log.info(f"최종적으로 데이터는 '{CHROMA_DB_DIR}' 폴더에 저장되었습니다.")
    log.info("="*50)

if __name__ == "__main__":
    setup_logging()
    main()