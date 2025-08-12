# preprocessing/ocr.py

import io
import logging
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch

log = logging.getLogger(__name__)

# Donut 모델과 프로세서 로드
DONUT_MODEL_NAME = "naver-clova-ix/donut-base-finetuned-cord-v2"
DONUT_PROCESSOR = DonutProcessor.from_pretrained(DONUT_MODEL_NAME)
DONUT_MODEL = VisionEncoderDecoderModel.from_pretrained(DONUT_MODEL_NAME)
DONUT_MODEL.eval()

def get_text_from_donut(image_bytes: bytes) -> str:
    """
    Donut 모델을 사용하여 이미지에서 텍스트를 추출합니다.
    Args:
        image_bytes (bytes): 이미지 파일의 바이트 데이터
    Returns:
        str: 추출된 텍스트
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 이미지 크기가 0이 아닌지 확인
        if image.width == 0 or image.height == 0:
            log.warning("이미지 크기가 0입니다. OCR을 건너뜁니다.")
            return ""

        # 채널 형식 명확히 지정
        pixel_values = DONUT_PROCESSOR(
            image,
            return_tensors="pt",
            input_data_format="channels_last"
        ).pixel_values

        decoder_input_ids = DONUT_PROCESSOR.tokenizer(
            "<s_cord-v2>",
            add_special_tokens=False,
            return_tensors="pt"
        ).input_ids

        with torch.no_grad():
            outputs = DONUT_MODEL.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=512,
                pad_token_id=DONUT_PROCESSOR.tokenizer.pad_token_id,
                eos_token_id=DONUT_PROCESSOR.tokenizer.eos_token_id
            )

        sequence = DONUT_PROCESSOR.batch_decode(outputs, skip_special_tokens=True)[0]
        text = sequence.replace(DONUT_PROCESSOR.tokenizer.eos_token, "").strip()

        log.info(f"Donut으로 텍스트 추출 성공 (길이: {len(text)})")
        return text

    except Exception:
        log.error("Donut 추론 중 오류 발생", exc_info=True)
        return ""
