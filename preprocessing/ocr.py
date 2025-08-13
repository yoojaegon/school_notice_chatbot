# preprocessing/ocr.py

import os
import json
import logging
import time
import uuid
import base64
import requests
from dotenv import load_dotenv

log = logging.getLogger(__name__)

# .env 파일에서 환경 변수 로드
load_dotenv()

# Naver CLOVA OCR API 정보
API_URL = os.getenv("NCP_CLOVA_OCR_API_URL")
SECRET_KEY = os.getenv("NCP_CLOVA_OCR_SECRET_KEY")

def get_text_from_clova_ocr(image_bytes: bytes) -> str:
    """
    Naver CLOVA OCR API를 사용하여 이미지에서 텍스트를 추출합니다.
    Args:
        image_bytes (bytes): 이미지 파일의 바이트 데이터
    Returns:
        str: 추출된 텍스트, 실패 시 빈 문자열
    """
    if not all([API_URL, SECRET_KEY]):
        log.error("Naver CLOVA OCR API 설정(.env의 NCP_CLOVA_OCR_API_URL, NCP_CLOVA_OCR_SECRET_KEY)이 필요합니다.")
        return ""

    if not image_bytes:
        log.warning("이미지 데이터가 비어있어 OCR을 건너뜁니다.")
        return ""

    request_json = {
        'images': [
            {
                'format': 'png',  # API는 포맷 값을 요구하며, 대부분의 이미지를 처리할 수 있는 png로 지정
                'name': 'ocr_image',
                'data': base64.b64encode(image_bytes).decode('utf-8')
            }
        ],
        'lang': 'ko',
        'requestId': str(uuid.uuid4()),
        'version': 'V2',
        'timestamp': int(time.time() * 1000)
    }

    headers = {
        'X-OCR-SECRET': SECRET_KEY,
        'Content-Type': 'application/json'
    }

    try:
        response = requests.post(API_URL, headers=headers, json=request_json, timeout=30)
        response.raise_for_status()
        result = response.json()

        # 인식된 모든 텍스트 필드를 공백으로 연결하여 반환
        text_parts = [field.get('inferText', '') for image in result.get('images', []) for field in image.get('fields', [])]
        full_text = ' '.join(filter(None, text_parts))

        log.info(f"CLOVA OCR로 텍스트 추출 성공 (길이: {len(full_text)})")
        return full_text

    except requests.exceptions.RequestException as e:
        log.error(f"CLOVA OCR API 요청 실패: {e}", exc_info=True)
        return ""
    except Exception as e:
        log.error(f"CLOVA OCR 결과 처리 중 오류 발생: {e}", exc_info=True)
        return ""