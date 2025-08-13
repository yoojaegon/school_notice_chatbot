# preprocessing/handler.py

import os
import io
import logging
from typing import List, Dict
from . import ocr
from .document_parser import parse_document  # 기존 파서 사용 (HWP/PDF/DOCX 등)

log = logging.getLogger(__name__)

DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".hwp", ".pptx", ".xlsx", ".txt"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
ARCHIVE_EXTENSIONS = {".zip"}  # zip은 indexer에서 풀어 처리

def _simple_chunks(text: str, max_len: int = 1000) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_len, n)
        chunks.append(text[start:end])
        start = end
    return chunks

def _base_metadata(file_path: str) -> Dict:
    return {
        "source": os.path.basename(file_path),
        "category": None,
    }

def process_file(file_path: str) -> List[Dict]:
    try:
        ext = os.path.splitext(file_path)[1].lower()
        filename = os.path.basename(file_path)
        parent_dir = os.path.basename(os.path.dirname(file_path))
        is_content_txt = (filename == "content.txt")

        results: List[Dict] = []

        # 1) 게시글 본문
        if is_content_txt:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    body = f.read()
            except UnicodeDecodeError:
                with open(file_path, "r", encoding="cp949") as f:
                    body = f.read()

            for i, ch in enumerate(_simple_chunks(body)):
                md = _base_metadata(file_path)
                md.update({
                    "category": "post_body",
                    "attachment_filename": None,
                    "attachment_ext": None,
                    "chunk_index": i,
                })
                results.append({"page_content": ch, "metadata": md})
            return results

        # 2) 첨부가 이미지 자체인 경우 → OCR
        if ext in IMAGE_EXTENSIONS:
            with open(file_path, "rb") as f:
                img_bytes = f.read()
            text = ocr.get_text_from_clova_ocr(img_bytes) or ""
            for i, ch in enumerate(_simple_chunks(text)):
                md = _base_metadata(file_path)
                md.update({
                    "category": "attachment_body",
                    "attachment_filename": filename,
                    "attachment_ext": ext,
                    "image_index": 0,
                    "ocr_model": "clova-ocr",
                    "chunk_index": i,
                })
                results.append({"page_content": ch, "metadata": md})
            return results

        # 3) 문서형 첨부 (HWP/PDF/DOCX/TXT 등)
        if ext in DOCUMENT_EXTENSIONS:
            if ext == ".txt":
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        body = f.read()
                except UnicodeDecodeError:
                    with open(file_path, "r", encoding="cp949") as f:
                        body = f.read()

                for i, ch in enumerate(_simple_chunks(body)):
                    md = _base_metadata(file_path)
                    md.update({
                        "category": "attachment_body",
                        "attachment_filename": filename,
                        "attachment_ext": ext,
                        "chunk_index": i,
                    })
                    results.append({"page_content": ch, "metadata": md})
                return results

            parsed = parse_document(file_path) or {}
            body_text = (parsed.get("text") or "").strip()
            embedded_images = parsed.get("images") or []  # [(index, bytes)] 또는 [bytes]

            # 3-1) 첨부파일 본문
            if body_text:
                for i, ch in enumerate(_simple_chunks(body_text)):
                    md = _base_metadata(file_path)
                    md.update({
                        "category": "attachment_body",
                        "attachment_filename": filename,
                        "attachment_ext": ext,
                        "chunk_index": i,
                    })
                    results.append({"page_content": ch, "metadata": md})

            # 3-2) 문서에 포함된 이미지 OCR (tuple 안전 언패킹)
            for img_idx, img in enumerate(embedded_images):
                # img가 (index, bytes) 이거나 bytes일 수 있음
                if isinstance(img, tuple):
                    # (idx, bytes) 형태 보장
                    _, img_bytes = img if len(img) >= 2 else (None, None)
                else:
                    img_bytes = img

                if not img_bytes:
                    continue

                text = ocr.get_text_from_clova_ocr(img_bytes) or ""
                if not text.strip():
                    continue

                for i, ch in enumerate(_simple_chunks(text)):
                    md = _base_metadata(file_path)
                    md.update({
                        "category": "embedded_image_ocr",
                        "attachment_filename": filename,
                        "attachment_ext": ext,
                        "image_index": img_idx,
                        "ocr_model": "clova-ocr",
                        "chunk_index": i,
                    })
                    results.append({"page_content": ch, "metadata": md})

            return results

        # 4) 기타 확장자
        log.warning(f"지원하지 않는 파일 형식, 건너뜁니다: {filename}")
        return []

    except Exception:
        log.error(f"파일 처리 중 오류: {file_path}", exc_info=True)
        return []