# preprocessing/handler.py

import os
from typing import List, Dict, Any

from . import document_parser, ocr, chunker

# 지원하는 파일 형식 정의
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".hwp"}

def process_file(file_path: str) -> List[Dict[str, Any]]:
    """
    단일 파일을 처리하여 메타데이터가 포함된 텍스트 청크 리스트를 생성합니다.

    Args:
        file_path (str): 처리할 파일의 전체 경로.

    Returns:
        List[Dict[str, Any]]: 각 청크의 텍스트와 메타데이터를 담은 딕셔너리 리스트.
    """
    _, extension = os.path.splitext(file_path)
    extension = extension.lower()
    
    all_chunks = []

    if extension in IMAGE_EXTENSIONS:
        print(f"이미지 파일 처리 중: {file_path}")
        with open(file_path, 'rb') as f:
            image_bytes = f.read()
        
        # 이미지 파일 자체를 OCR 처리
        extracted_text = ocr.get_text_from_image(image_bytes, extension.replace('.', ''))
        text_chunks = chunker.chunk_text(extracted_text)
        
        for i, chunk in enumerate(text_chunks):
            metadata = {
                "source": os.path.basename(file_path),
                "type": "image_file",
                "chunk_index": i + 1
            }
            all_chunks.append({"text": chunk, "metadata": metadata})

    elif extension in DOCUMENT_EXTENSIONS:
        print(f"문서 파일 처리 중: {file_path}")
        parsed_data = document_parser.parse_document(file_path)
        
        # 1. 문서 본문 텍스트 처리
        body_text = parsed_data.get("text", "")
        body_chunks = chunker.chunk_text(body_text)
        for i, chunk in enumerate(body_chunks):
            metadata = {
                "source": os.path.basename(file_path),
                "type": "document_body",
                "chunk_index": i + 1
            }
            all_chunks.append({"text": chunk, "metadata": metadata})
            
        # 2. 문서 내 포함된 이미지 처리
        embedded_images = parsed_data.get("images", [])
        for img_index, image_bytes in embedded_images:
            img_text = ocr.get_text_from_image(image_bytes)
            img_chunks = chunker.chunk_text(img_text)
            
            for i, chunk in enumerate(img_chunks):
                metadata = {
                    "source": os.path.basename(file_path),
                    "type": f"embedded_image_{img_index}",
                    "chunk_index": i + 1
                }
                all_chunks.append({"text": chunk, "metadata": metadata})
    else:
        print(f"지원하지 않는 파일 형식 건너뛰기: {file_path}")

    return all_chunks