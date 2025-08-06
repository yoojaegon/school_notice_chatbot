# preprocessing/ocr.py

import torch
import logging
import re
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
import io

log = logging.getLogger(__name__)

# --- Donut 모델 전역 변수 (효율적인 로딩을 위해) ---
DONUT_PROCESSOR = None
DONUT_MODEL = None
DEVICE = None

def _initialize_donut():
    """
    Donut 모델과 프로세서를 초기화하고 전역 변수에 할당합니다.
    이미 로드된 경우 이 과정을 건너뜁니다.
    """
    global DONUT_PROCESSOR, DONUT_MODEL, DEVICE
    
    if DONUT_MODEL is not None:
        return

    try:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = "naver-clova-ix/donut-base"
        
        log.info(f"Donut 모델을 처음으로 로드합니다. 장치: {DEVICE}")
        
        DONUT_PROCESSOR = DonutProcessor.from_pretrained(model_name)
        DONUT_MODEL = VisionEncoderDecoderModel.from_pretrained(model_name)
        
        DONUT_MODEL.to(DEVICE)
        DONUT_MODEL.eval()
        log.info("Donut 모델 로드 및 설정 완료.")
    
    except Exception as e:
        log.error("Donut 모델 로딩에 실패했습니다. PyTorch 및 의존성 라이브러리 설치를 확인하세요.", exc_info=True)
        DONUT_PROCESSOR, DONUT_MODEL = None, None

def get_text_from_donut(image_bytes: bytes) -> str:
    """
    Donut 모델을 사용하여 이미지 바이트에서 텍스트를 추출합니다.
    """
    _initialize_donut()

    if not all([DONUT_PROCESSOR, DONUT_MODEL]):
        log.warning("Donut 모델이 사용 불가능한 상태입니다. 텍스트 추출을 건너뜁니다.")
        return ""

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        pixel_values = DONUT_PROCESSOR(image, return_tensors="pt").pixel_values
        task_prompt = "<s_iitcdip>"
        decoder_input_ids = DONUT_PROCESSOR.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

        with torch.no_grad():
            outputs = DONUT_MODEL.generate(
                pixel_values.to(DEVICE),
                decoder_input_ids=decoder_input_ids.to(DEVICE),
                max_length=DONUT_MODEL.decoder.config.max_position_embeddings,
                pad_token_id=DONUT_PROCESSOR.tokenizer.pad_token_id,
                eos_token_id=DONUT_PROCESSOR.tokenizer.eos_token_id,
                use_cache=True,
                bad_words_ids=[[DONUT_PROCESSOR.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )

        sequence = DONUT_PROCESSOR.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(DONUT_PROCESSOR.tokenizer.eos_token, "").replace(DONUT_PROCESSOR.tokenizer.pad_token, "")
        result_text = re.sub(r"<.*?>", "", sequence).strip()

        log.info(f"Donut으로 텍스트 추출 성공. (길이: {len(result_text)})")
        return result_text

    except Exception as e:
        log.error("Donut 추론 중 오류 발생", exc_info=True)
        return ""