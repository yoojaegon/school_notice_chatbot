# preprocessing/document_parser.py

import os
import logging
from typing import Dict, List, Union, Tuple
import hwp5
import fitz  # PyMuPDF
import docx
import olefile
import pptx
import pandas as pd
from PIL import Image
import io

log = logging.getLogger(__name__)

def parse_document(file_path: str) -> Dict[str, Union[str, List[Tuple[int, bytes]]]]:
    """
    파일 경로를 받아 확장자에 맞는 파서를 호출하고, 텍스트와 이미지를 추출합니다.
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == ".pdf":
        return _parse_pdf(file_path)
    elif ext == ".docx":
        return _parse_docx(file_path)
    elif ext == ".hwp":
        return _parse_hwp(file_path)
    elif ext == ".pptx":
        return _parse_pptx(file_path)
    elif ext == ".xlsx":
        return _parse_xlsx(file_path)
    else:
        log.warning(f"지원하지 않는 파일 형식입니다: {file_path}")
        return {"text": "", "images": []}

def _parse_pdf(file_path: str):
    # ... (이전과 동일, 생략)
    pass

def _parse_docx(file_path: str):
    # ... (이전과 동일, 생략)
    pass

def _parse_hwp(file_path: str):
    # ... (이전과 동일, 생략)
    pass


def _parse_pptx(file_path: str) -> Dict[str, Union[str, List[Tuple[int, bytes]]]]:
    """PowerPoint(.pptx) 파일에서 텍스트와 이미지를 추출합니다."""
    log.info(f"PPTX 파일 파싱 시작: {os.path.basename(file_path)}")
    try:
        presentation = pptx.Presentation(file_path)
        full_text_list = []
        images = []
        
        for slide in presentation.slides:
            for shape in slide.shapes:
                if shape.has_text_frame:
                    full_text_list.append(shape.text_frame.text)
                if isinstance(shape, pptx.shapes.picture.Picture):
                    image_bytes = shape.image.blob
                    images.append((len(images) + 1, image_bytes))
                    
    except Exception as e:
        log.error(f"PPTX 파일 처리 중 오류 발생: {file_path}", exc_info=True)
        return {"text": "", "images": []}
        
    return {"text": "\n\n".join(full_text_list), "images": images}

def _parse_xlsx(file_path: str) -> Dict[str, Union[str, List[Tuple[int, bytes]]]]:
    """Excel(.xlsx) 파일에서 텍스트와 이미지를 추출합니다."""
    log.info(f"XLSX 파일 파싱 시작: {os.path.basename(file_path)}")
    full_text_list = []
    images = []
    
    try:
        # 1. Pandas를 사용하여 모든 시트의 텍스트 데이터 추출
        xls = pd.ExcelFile(file_path)
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
            if not df.empty:
                full_text_list.append(f"--- 시트: {sheet_name} ---\n{df.to_string(index=False)}")

        # 2. openpyxl을 사용하여 이미지 추출 (예외처리 강화)
        try:
            from openpyxl_image_loader import SheetImageLoader
            image_loader = SheetImageLoader(file_path)
            for sheet_name in image_loader.sheet_names:
                for image_name in image_loader.get_images_in_sheet(sheet_name):
                    image = image_loader.get_image(image_name)
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format='PNG')
                    images.append((len(images) + 1, img_byte_arr.getvalue()))
        except ImportError:
            log.warning("엑셀 이미지 추출을 위해 'pip install openpyxl-image-loader'를 설치해주세요. 이미지 추출을 건너뜁니다.")
        except Exception as img_e:
            log.error(f"XLSX에서 이미지 추출 중 오류 발생: {img_e}")

    except Exception as e:
        log.error(f"XLSX 파일 처리 중 오류 발생: {file_path}", exc_info=True)
        return {"text": "", "images": []}
        
    return {"text": "\n\n".join(full_text_list), "images": images}