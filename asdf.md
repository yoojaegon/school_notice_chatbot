# 데이터 전처리 순서도

```mermaid
graph TD
    subgraph "입력"
        A["시작: 원본 파일 수신<br>(hwp, pdf, docx, png, jpg 등)"]
    end

    A --> B{"파일 유형이 문서인가?<br>(hwp, pdf, docx 등)}"}

    B -- "No (이미지 파일)" --> C["이미지 OCR 처리<br>(네이버 클로바 OCR)"]

    B -- "Yes (문서 파일)" --> D[1. 문서에서 텍스트 추출]
    D --> E[2. 문서 내 포함된 이미지 추출]
    E --> F{추출된 이미지가 있는가?}
    F -- Yes --> G[3. 각 이미지 OCR 처리]
    G --> H[4. 기본 텍스트와 OCR 결과 통합]
    F -- No --> H

    C --> I(통합 텍스트)
    H --> I

    subgraph "공통 후처리"
        I --> J["텍스트 정제<br>(불필요한 공백/특수문자 제거)"]
        J --> K["<b>의미 기반 분할 (Semantic Chunking)</b>"]
        K --> L["청크별 메타데이터 추가<br>(원본 파일명, 게시글 제목 등)"]
    end

    subgraph "출력"
        M(["종료: 색인 준비가 완료된<br>텍스트 청크(Chunk) 목록"])
    end

    L --> M  <-- 이 라인 (32행) 바로 다음에 닫는 코드 블록이 와야 합니다.
```             