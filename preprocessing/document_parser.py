# preprocessing/document_parser.py

from typing import Dict, List, Union, Tuple
import pyhwp
import fitz  # PyMuPDF
import docx
import olefile

def parse_document(file_path: str) -> Dict[str, Union[str, List[Tuple[int, bytes]]]]:
    """
    파일 경로를 받아 확장자에 맞는 파서를 호출하고, 텍스트와 이미지를 추출합니다.

    Args:
        file_path (str): 파싱할 문서 파일의 경로.

    Returns:
        Dict: 'text'와 'images' 키를 포함하는 딕셔너리.
              - 'text': 문서의 전체 텍스트.
              - 'images': (이미지 인덱스, 이미지 바이트) 튜플의 리스트.
    """
    if file_path.endswith(".pdf"):
        return _parse_pdf(file_path)
    elif file_path.endswith(".docx"):
        return _parse_docx(file_path)
    elif file_path.endswith(".hwp"):
        return _parse_hwp(file_path)
    else:
        print(f"지원하지 않는 파일 형식입니다: {file_path}")
        return {"text": "", "images": []}

def _parse_pdf(file_path: str) -> Dict[str, Union[str, List[Tuple[int, bytes]]]]:
    doc = fitz.open(file_path)
    full_text = ""
    images = []
    
    for page_num, page in enumerate(doc):
        full_text += page.get_text()
        
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            images.append((len(images) + 1, image_bytes))
            
    doc.close()
    return {"text": full_text, "images": images}

def _parse_docx(file_path: str) -> Dict[str, Union[str, List[Tuple[int, bytes]]]]:
    doc = docx.Document(file_path)
    full_text = "\n".join([para.text for para in doc.paragraphs])
    
    images = []
    for rel in doc.part.rels.values():
        if "image" in rel.target_ref:
            image_bytes = rel.target_part.blob
            images.append((len(images) + 1, image_bytes))
            
    return {"text": full_text, "images": images}

def _parse_hwp(file_path: str) -> Dict[str, Union[str, List[Tuple[int, bytes]]]]:
    try:
        hwp_file = pyhwp.HWPReader(file_path)
        full_text = hwp_file.get_text()

        # pyhwp는 이미지 추출을 직접 지원하지 않음. olefile을 사용해야 함.
        # 이 부분은 복잡하므로, 우선 텍스트만 추출하고 이미지는 비워둠.
        # TODO: olefile을 사용하여 HWP 내 이미지 추출 로직 구현
        images = []
        f = olefile.OleFileIO(file_path)
        for i, (dirs, _, _) in enumerate(f.listdir()):
            if dirs and dirs[0] == "Pictures":
                stream = f.openstream(dirs)
                image_bytes = stream.read()
                images.append((i+1, image_bytes))
        f.close()


    except Exception as e:
        print(f"HWP 파일 처리 중 오류 발생: {e}")
        return {"text": "", "images": []}
        
    return {"text": full_text, "images": images}