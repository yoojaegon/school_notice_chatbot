import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import re
import json

def test_donut_base_model(image_path: str):
    """
    'naver-clova-ix/donut-base' 모델을 사용하여 주어진 이미지의 텍스트를 읽습니다.
    GPU가 사용 가능하면 자동으로 GPU를 사용합니다.

    Args:
        image_path (str): 분석할 이미지 파일의 경로.
    """
    
    # ----------------------------------------------------
    # 1. 모델 및 장치 설정
    # ----------------------------------------------------
    try:
        # GPU 사용 가능 여부 확인 및 장치 설정
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[{device.upper()}] 장치를 사용하여 모델을 로드합니다...")
        
        # 범용 'donut-base' 모델과 프로세서 로드
        model_name = "naver-clova-ix/donut-base-finetuned-cord-v2"
        processor = DonutProcessor.from_pretrained(model_name)
        model = VisionEncoderDecoderModel.from_pretrained(model_name)

        # 모델을 설정된 장치(GPU 또는 CPU)로 이동
        model.to(device)
        print("모델 로드 완료.")

    except Exception as e:
        print(f"[오류] 모델 로드 중 문제가 발생했습니다: {e}")
        print("PyTorch가 CUDA 버전으로 올바르게 설치되었는지 확인해주세요.")
        return

    # ----------------------------------------------------
    # 2. 이미지 준비 및 전처리
    # ----------------------------------------------------
    try:
        print(f"이미지 파일을 엽니다: '{image_path}'")
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"[오류] 이미지 파일을 찾을 수 없습니다. 경로를 확인해주세요: '{image_path}'")
        return
        
    # 이미지를 모델 입력 형식(픽셀 텐서)으로 변환
    pixel_values = processor(image, return_tensors="pt").pixel_values

    # ----------------------------------------------------
    # 3. 추론 실행
    # ----------------------------------------------------
    # donut-base는 사전 학습 시 IIT-CDIP 데이터셋으로 텍스트 읽기 작업을 수행했습니다.
    # 따라서 해당 데이터셋의 프롬프트를 사용하여 텍스트 읽기를 요청합니다.
    task_prompt = "<s_iitcdip>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

    print("모델 추론을 시작합니다...")
    
    # 추론 시에는 gradient 계산이 필요 없으므로 no_grad() 사용
    model.eval()
    with torch.no_grad():
        # 모델과 입력 데이터를 모두 동일한 장치로 이동
        outputs = model.generate(
            pixel_values.to(device),
            decoder_input_ids=decoder_input_ids.to(device),
            max_length=model.decoder.config.max_position_embeddings,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )
    
    print("모델 추론 완료.")

    # ----------------------------------------------------
    # 4. 결과 후처리 및 출력
    # ----------------------------------------------------
    # 모델이 생성한 토큰 ID를 사람이 읽을 수 있는 텍스트로 변환
    sequence = processor.batch_decode(outputs.sequences)[0]
    
    # 불필요한 특수 토큰(프롬프트, pad, eos) 제거
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    result_text = re.sub(r"<.*?>", "", sequence).strip()

    print("\n" + "="*60)
    print("          <<< OCR-free 텍스트 추출 결과 >>>")
    print("="*60)
    print(result_text)
    print("="*60)


if __name__ == "__main__":
    # ⭐️⭐️⭐️ 1. 파일 경로 수정 ⭐️⭐️⭐️
    # 테스트하고 싶은 이미지 파일의 경로를 아래에 입력하세요.
    # 예시: "my_poster.png" 또는 "data/notice_01.jpg"
    IMAGE_TO_TEST = "C:/Resource/school_notice_chatbot/crawled_data/일반공지/1065997_2024학년도 미디어&콘텐츠학과 신입생 추가 모집(3차)/2023학년도 신입생 모집 포스터.jpg"

    # 2. 코드 실행
    if IMAGE_TO_TEST == "your_image_path_here.jpg":
        print("[경고] 테스트할 이미지 파일 경로를 'IMAGE_TO_TEST' 변수에 지정해주세요.")
    else:
        test_donut_base_model(IMAGE_TO_TEST)