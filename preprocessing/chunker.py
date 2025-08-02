# preprocessing/chunker.py

from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_text(text: str) -> List[str]:
    """
    주어진 텍스트를 의미 있는 단위의 청크(chunk)로 분할합니다.

    Args:
        text (str): 분할할 전체 텍스트.

    Returns:
        List[str]: 분할된 텍스트 청크들의 리스트.
    """
    if not text:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    
    return text_splitter.split_text(text)