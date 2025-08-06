# preprocessing/handler.py

import os
import logging
from typing import List, Dict, Any
from pathlib import Path
import zipfile
import shutil

from . import document_parser, ocr, chunker

log = logging.getLogger(__name__)

# 지원하는 파일 형식 정의
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".hwp", ".pptx", ".xlsx"}
ARCHIVE_EXTENSIONS = {".zip"}

def _handle_zip_file(file_path: Path) -> List[Dict[str, Any]]:
    """
    ZIP 파일의 압축을 풀고 내부의 각 파일을 개별적으로 처리하여 청크 리스트를 반환합니다.
    """
    log.info(f"ZIP 아카이브 처리 시작: {file_path.name}")
    all_chunks = []
    
    extract_dir = file_path.parent / f".{file_path.stem}_unzipped_temp"
    
    try:
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(exist_ok=True)

        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        log.info(f"'{extract_dir.name}'에 압축 해제 완료. 내부 파일들을 처리합니다.")
        
        for root, _, files in os.walk(extract_dir):
            for name in files:
                extracted_file_path = os.path.join(root, name)
                chunks_from_file = process_file(extracted_file_path)
                if chunks_from_file:
                    all_chunks.extend(chunks_from_file)

    except zipfile.BadZipFile:
        log.error(f"손상되었거나 유효하지 않은 ZIP 파일입니다: {file_path.name}")
    except Exception as e:
        log.error(f"ZIP 파일 처리 중 오류 발생: {file_path.name}", exc_info=True)
    finally:
        # 임시 디렉토리 정리
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
            log.info(f"임시 압축해제 폴더를 삭제했습니다: {extract_dir.name}")
            
    return all_chunks

def process_file(file_path: str) -> List[Dict[str, Any]]:
    """
    단일 파일을 처리하여 메타데이터가 포함된 텍스트 청크 리스트를 생성합니다.
    """
    p_file_path = Path(file_path)
    if not p_file_path.is_file():
        return []

    file_name = p_file_path.name
    extension = p_file_path.suffix.lower()
    all_chunks = []
    log.info(f"파일 처리 시작: {file_name}")

    if extension in ARCHIVE_EXTENSIONS:
        return _handle_zip_file(p_file_path)

    elif extension in IMAGE_EXTENSIONS:
        try:
            with open(file_path, 'rb') as f:
                image_bytes = f.read()
            extracted_text = ocr.get_text_from_donut(image_bytes)
            text_chunks = chunker.chunk_text(extracted_text)
            
            for i, chunk in enumerate(text_chunks):
                metadata = {"source": file_name, "type": "image_content", "chunk_index": i + 1}
                all_chunks.append({"page_content": chunk, "metadata": metadata})
        except Exception as e:
            log.error(f"이미지 파일 처리 중 오류 발생: {file_name}", exc_info=True)

    elif extension in DOCUMENT_EXTENSIONS:
        parsed_data = document_parser.parse_document(file_path)
        
        body_text = parsed_data.get("text", "")
        if body_text:
            body_chunks = chunker.chunk_text(body_text)
            for i, chunk in enumerate(body_chunks):
                metadata = {"source": file_name, "type": "document_body", "chunk_index": i + 1}
                all_chunks.append({"page_content": chunk, "metadata": metadata})
            
        embedded_images = parsed_data.get("images", [])
        if embedded_images:
            log.info(f"{file_name}에서 {len(embedded_images)}개의 내장 이미지를 처리합니다.")
            for img_index, image_bytes in embedded_images:
                img_text = ocr.get_text_from_donut(image_bytes)
                if img_text:
                    img_chunks = chunker.chunk_text(img_text)
                    for i, chunk in enumerate(img_chunks):
                        metadata = {"source": file_name, "type": "embedded_image", "image_index": img_index, "chunk_index": i + 1}
                        all_chunks.append({"page_content": chunk, "metadata": metadata})
    else:
        # 임시폴더 내 시스템 파일(예: .DS_Store) 등을 무시하기 위해 debug 레벨로 변경
        if not file_name.startswith('.'):
            log.warning(f"지원하지 않는 파일 형식, 건너뜁니다: {file_name}")

    log.info(f"파일 처리 완료: {file_name}, 총 {len(all_chunks)}개의 청크 생성.")
    return all_chunks