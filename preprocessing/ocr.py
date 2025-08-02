# preprocessing/ocr.py

import os
import json
import time
import uuid
import requests
from dotenv import load_dotenv

load_dotenv()

# --- 환경 변수 로드 ---
CLOVA_OCR_API_URL = os.getenv("CLOVA_OCR_API_URL")
CLOVA_OCR_SECRET_KEY = os.getenv("CLOVA_OCR_SECRET_KEY")

def get_text_from_image(image_bytes: bytes, file_format: str = 'png') -> str:
    """
    네이버 클로바 OCR API를 사용하여 이미지 바이트에서 텍스트를 추출합니다.

    Args:
        image_bytes (bytes): 텍스트를 추출할 이미지의 바이트 데이터.
        file_format (str): 이미지 파일 형식 (예: 'png', 'jpeg').

    Returns:
        str: 추출된 텍스트. API 호출 실패 시 빈 문자열을 반환합니다.
    """
    if not all([CLOVA_OCR_API_URL, CLOVA_OCR_SECRET_KEY]):
        print("OCR API 환경 변수가 설정되지 않았습니다.")
        return ""

    request_json = {
        'images': [
            {
                'format': file_format,
                'name': 'demo'
            }
        ],
        'requestId': str(uuid.uuid4()),
        'version': 'V2',
        'timestamp': int(round(time.time() * 1000))
    }

    payload = {'message': json.dumps(request_json).encode('UTF-8')}
    files = [
        ('file', image_bytes)
    ]
    headers = {
        'X-OCR-SECRET': CLOVA_OCR_SECRET_KEY
    }

    try:
        response = requests.post(CLOVA_OCR_API_URL, headers=headers, data=payload, files=files, timeout=30)
        response.raise_for_status()  # 200 이외의 응답 코드에 대해 예외 발생
        
        result = response.json()
        
        # 추출된 텍스트들을 하나의 문자열로 결합
        full_text = []
        for field in result['images'][0]['fields']:
            full_text.append(field['inferText'])
        
        return " ".join(full_text)

    except requests.exceptions.RequestException as e:
        print(f"OCR API 요청 실패: {e}")
        return ""
    except Exception as e:
        print(f"OCR 처리 중 에러 발생: {e}")
        return ""