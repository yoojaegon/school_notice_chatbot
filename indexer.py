# indexer.py

import os
import logging
import logging.handlers
import datetime
from typing import List, Dict, Any, Sequence
import time

import chromadb
from dotenv import load_dotenv
from tqdm import tqdm

from preprocessing.handler import process_file

# ---- 새로 추가: OpenAI 임베딩 직접 호출 + 캐시 ----
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "128"))
EMBED_MAX_RETRIES = int(os.getenv("EMBED_MAX_RETRIES", "5"))
EMBED_BACKOFF_BASE = float(os.getenv("EMBED_BACKOFF_BASE", "1.5"))

# OpenAI SDK v1 (권장)
try:
    from openai import OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    _oai = OpenAI(api_key=OPENAI_API_KEY)
    def _embed_batch(model: str, texts: Sequence[str]) -> list[list[float]]:
        resp = _oai.embeddings.create(model=model, input=list(texts))
        return [d.embedding for d in resp.data]
except Exception:
    # v0.x 호환(가능하면 v1로 업그레이드 권장)
    import openai
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    openai.api_key = OPENAI_API_KEY
    def _embed_batch(model: str, texts: Sequence[str]) -> list[list[float]]:
        resp = openai.Embedding.create(model=model, input=list(texts))
        return [d["embedding"] for d in resp["data"]]

from embed_cache import get_cached, set_cached


# --- 로깅 설정 ---
def setup_logging():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "indexer.log")

    logger = logging.getLogger()
    if logger.handlers:
        for h in logger.handlers[:]:
            logger.removeHandler(h)

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.handlers.RotatingFileHandler(
        log_file_path, maxBytes=5 * 1024 * 1024, backupCount=5, encoding='utf-8'
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

load_dotenv()

CRAWLED_DATA_DIR = "./crawled_data"
CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "school_announcements"

EXTRACTED_SAVE_DIR = os.getenv("EXTRACTED_SAVE_DIR", "./extracted_data")
SAVE_EXTRACTED_TEXT = os.getenv("SAVE_EXTRACTED_TEXT", "true").lower() in ("1", "true", "yes", "y")

log = logging.getLogger(__name__)


def _sanitize_filename(name: str) -> str:
    cleaned = "".join(c for c in name if c not in '\\/*?:"<>|').strip()
    return cleaned or "untitled"

def get_post_id_from_path(file_path: str) -> str:
    try:
        post_directory_path = os.path.dirname(file_path)
        post_folder_name = os.path.basename(post_directory_path)
        post_id = post_folder_name.split('_', 1)[0]
        if post_id.isdigit():
            return post_id
        else:
            log.debug(f"경로에서 숫자 형식의 post_id를 찾지 못했습니다: {file_path}")
            return "unknown"
    except IndexError:
        log.warning(f"경로 구조가 예상과 달라 post_id를 추출할 수 없습니다: {file_path}")
        return "unknown"

def _normalize_category(raw: str) -> str:
    if not raw:
        return "chunk"
    raw_lower = str(raw).lower()
    if raw_lower in ("post_content", "content", "post", "post_body"):
        return "post_content"
    if raw_lower in ("attachment_text", "attachment", "file", "image_attachment_ocr", "ocr_image_attachment", "attachment_body"):
        return "attachment_text"
    if raw_lower in ("embedded_image_text", "embedded_image", "image_in_document", "image_embedded_ocr", "embedded_image_ocr"):
        return "embedded_image_text"
    return _sanitize_filename(raw_lower)

def _save_chunks_to_disk(post_id: str, file_path: str, structured_chunks: List[Dict[str, Any]]) -> None:
    if not SAVE_EXTRACTED_TEXT:
        return
    try:
        base_filename = os.path.basename(file_path)
        for i, chunk in enumerate(structured_chunks):
            meta = dict(chunk.get("metadata", {}))
            category = _normalize_category(meta.get("category") or meta.get("type") or "chunk")
            out_dir = os.path.join(EXTRACTED_SAVE_DIR, post_id, category)
            os.makedirs(out_dir, exist_ok=True)
            out_name = f"chunk_{i:03d}__{_sanitize_filename(base_filename)}.txt"
            out_path = os.path.join(out_dir, out_name)
            header_lines = [
                f"# saved_at: {datetime.datetime.now().isoformat(timespec='seconds')}",
                f"# post_id: {post_id}",
                f"# category: {category}",
                f"# source_file: {file_path}",
                f"# title: {meta.get('announcement_title', '')}",
                f"# source: {meta.get('source', '')}",
                f"# extra: board={meta.get('board_name', '')}, post_url={meta.get('post_url', '')}",
                "#" + "-" * 78,
                "",
            ]
            body = chunk.get("page_content", "")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(header_lines))
                f.write(body)
        log.info(f"📝 추출 텍스트 {len(structured_chunks)}개를 디스크에 저장했습니다: {os.path.join(EXTRACTED_SAVE_DIR, post_id)}")
    except Exception:
        log.error("추출 텍스트 저장 중 오류 발생", exc_info=True)

# ---- 임베딩 배치 + 캐시 ----
def embed_in_batches(texts: Sequence[str],
                     model: str = EMBED_MODEL,
                     batch_size: int = EMBED_BATCH_SIZE,
                     max_retries: int = EMBED_MAX_RETRIES,
                     backoff_base: float = EMBED_BACKOFF_BASE) -> list[list[float]]:
    vectors: list[list[float]] = []
    n = len(texts)
    for start in range(0, n, batch_size):
        chunk = texts[start:start+batch_size]

        # 캐시 확인
        cached_vecs: list[list[float]] = []
        to_call_idx, to_call_texts = [], []
        for i, t in enumerate(chunk):
            c = get_cached(t, model)
            if c is not None:
                cached_vecs.append(c)
            else:
                cached_vecs.append(None)
                to_call_idx.append(i)
                to_call_texts.append(t)

        # API 호출(필요한 것만)
        if to_call_texts:
            for attempt in range(max_retries):
                try:
                    api_vecs = _embed_batch(model, to_call_texts)
                    # 결과 매핑
                    for offset, v in zip(to_call_idx, api_vecs):
                        cached_vecs[offset] = v
                        set_cached(chunk[offset], model, v)
                    break
                except Exception as e:
                    wait = (backoff_base ** attempt) + 0.1 * attempt
                    log.warning(f"임베딩 배치 실패({attempt+1}/{max_retries}): {e} -> {wait:.1f}s 대기 후 재시도")
                    time.sleep(wait)
            else:
                raise RuntimeError("임베딩 배치 계산에 반복적으로 실패했습니다.")

        # 배치 결과 합치기
        vectors.extend(cached_vecs)
        log.info(f"임베딩 처리: {start} ~ {start+len(chunk)-1} (캐시 {len(chunk)-len(to_call_texts)}/{len(chunk)})")

    return vectors


def main():
    log.info("=" * 50)
    log.info("데이터 색인 작업을 시작합니다.")
    log.info("=" * 50)

    # 1) ChromaDB 클라이언트 (⚠ 임베딩 함수 제거)
    log.info("ChromaDB 클라이언트 설정...")
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        log.info(f"ChromaDB 컬렉션 '{COLLECTION_NAME}' 준비 완료.")
    except Exception:
        log.critical("ChromaDB 설정 중 오류. 경로/권한 확인 요망.", exc_info=True)
        return

    # 2) 처리 대상 파일 수집
    if not os.path.exists(CRAWLED_DATA_DIR):
        log.error(f"'{CRAWLED_DATA_DIR}' 디렉토리가 없습니다. 먼저 'crawler.py' 실행 필요.")
        return

    files_to_process: List[str] = []
    for root, _, filenames in os.walk(CRAWLED_DATA_DIR):
        for filename in filenames:
            if not filename.startswith('.'):
                files_to_process.append(os.path.join(root, filename))

    if not files_to_process:
        log.warning(f"'{CRAWLED_DATA_DIR}' 디렉토리에 처리할 파일이 없습니다.")
        return

    log.info(f"총 {len(files_to_process)}개의 파일을 처리 대상으로 찾았습니다.")

    # 3) 파일 처리
    for file_path in tqdm(files_to_process, desc="파일 색인 중"):
        log.debug(f"--- 파일 처리 시작: {file_path} ---")

        post_id = get_post_id_from_path(file_path)
        if post_id == "unknown":
            log.warning(f"게시물 ID를 알 수 없는 파일은 건너뜁니다: {file_path}")
            continue

        structured_chunks = process_file(file_path)
        if not structured_chunks:
            log.warning(f"처리할 청크가 생성되지 않았습니다: {os.path.basename(file_path)}")
            continue

        ids, documents, metadatas = [], [], []
        base_filename = os.path.basename(file_path)

        for i, chunk_data in enumerate(structured_chunks):
            meta = dict(chunk_data.get("metadata", {}))
            meta["post_id"] = post_id
            chunk_id = f"{post_id}_{base_filename}_{meta.get('category', meta.get('type', 'chunk'))}_{i}"

            ids.append(chunk_id)
            documents.append(chunk_data.get("page_content", ""))
            metadatas.append(meta)

        # ---- 여기서부터 핵심: 임베딩을 직접 배치 계산하고 바로 넣기 ----
        try:
            log.info(f"임베딩 계산 시작 (문서 수: {len(documents)})")
            embeddings = embed_in_batches(documents, model=EMBED_MODEL, batch_size=EMBED_BATCH_SIZE)
            # upsert: 같은 ID면 덮어쓰기 (재실행 안전)
            collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
            log.info(f"✅ 성공: {len(ids)}개의 청크(post_id: {post_id})를 DB에 저장했습니다.")
        except Exception:
            log.error(f"❌ 실패: '{file_path}' 임베딩/DB 저장 중 오류", exc_info=True)
            continue

        # (선택) 디스크 저장
        try:
            _save_chunks_to_disk(post_id, file_path, structured_chunks)
        except Exception:
            log.error("디스크 저장 호출 중 오류", exc_info=True)

    log.info("=" * 50)
    log.info("모든 파일의 색인 작업이 완료되었습니다.")
    log.info(f"최종 데이터는 '{CHROMA_DB_DIR}' 폴더에 저장되었습니다.")
    if SAVE_EXTRACTED_TEXT:
        log.info(f"추출 텍스트는 '{EXTRACTED_SAVE_DIR}' 폴더에 저장되었습니다.")
    log.info("=" * 50)

if __name__ == "__main__":
    setup_logging()
    main()