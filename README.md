# 🏫 학교 공지사항 RAG 챗봇 시스템

> ⚠️ **현재 개발 중인 프로젝트입니다.**

## 📖 프로젝트 개요

"장학금 신청 기간이 언제까지였지?", "지난 학기 성적 증명서 발급 안내문이 어디 있더라?"

학교 공지사항은 많고, 복잡하며, 다양한 파일 형식(HWP, PDF, PPTX, 이미지 등)으로 흩어져 있어 필요한 정보를 제때 찾기 어렵습니다. 이 프로젝트는 이러한 문제를 해결하기 위해 RAG(Retrieval-Augmented Generation) 기술을 기반으로, 학교 공지사항에 대해 자연어로 질문하고 답변을 얻을 수 있는 챗봇 시스템을 구축합니다.

이 시스템은 여러 공지사항 게시판을 주기적으로 지능적으로 크롤링하여 최신 정보만 선별적으로 수집하고, 첨부된 문서나 이미지 속 글자까지 오픈소스 Donut OCR 모델로 인식하여 정보의 누락을 최소화합니다. 최종적으로 사용자는 OpenAI의 GPT 모델을 통해 사람과 대화하듯 정확한 답변을 얻으며, 이때 답변의 근거가 되는 원문 혹은 이미지 출처까지 함께 제공받아 정보의 신뢰성을 높였습니다.

## ✨ 이 프로젝트가 해결하려는 문제

1.  **정보의 파편화**: 공지사항이 여러 게시판과 다양한 파일 형식으로 흩어져 있어 통합적인 검색이 어렵습니다.
2.  **키워드 검색의 한계**: 단순 키워드 검색으로는 문맥을 이해하는 질문(예: "이번 학기 등록금 관련해서 제일 중요한 공지가 뭐야?")에 답변할 수 없습니다.
3.  **정보 접근의 비효율성**: 사용자가 직접 여러 게시물을 열어보고, 파일(PDF, HWP)을 다운로드하여 내용을 확인해야 하는 번거로움이 있습니다.
4.  **시각적 정보의 소외**: 공지사항 본문이나 첨부파일 속 이미지(포스터, 표)에 포함된 중요한 정보는 검색 대상에서 제외됩니다.

## ⚙️ 시스템 아키텍처

이 챗봇은 크게 두 가지 파이프라인으로 구성됩니다: **(1) 데이터 수집 및 색인 파이프라인**과 **(2) 사용자 질의응답 파이프라인**.

```
<< 1. 데이터 수집 및 색인 파이프라인 (주기적/자동 실행) >>

[웹 크롤러 (crawler.py)] --(신규 공지만 선별)--> [다운로드된 원본 파일 (crawled_data/)]
     (증분 수집)                                       |
                                                       v
                                    [색인 실행기 (indexer.py)] --calls--> [전처리기 (preprocessing/)]
                                                                                |
               +----------------------------------------------------------------+
               |
               v
[핸들러 (handler.py)] --(파일 종류 판단)--> [문서 파서 (document_parser.py)] 또는 [OCR (ocr.py) - Donut]
               |                                                   |
               |                                                   +---> [추출된 텍스트/이미지]
               |                                                             |
               +<----------------------(모든 텍스트)--------------------------+
               |
               v
          [청커 (chunker.py)] --(텍스트 분할)--> [텍스트 조각 (청크) + 상세 메타데이터]
               |
               +------------------------(결과 반환)----------------------> [색인 실행기 (indexer.py)]
                                                                                |
                                                                                v
                                                [임베딩 (OpenAI) + 배치/재시도/캐싱] ----> [🧠 Vector DB]
                                                         |      ^                      (ChromaDB)
                                                         v      |
                                                  [Cache (embeddings.sqlite)]


<< 2. 사용자 질의응답 파이프라인 (실시간 실행) >>

[사용자] <---> [챗봇 UI (app.py)] <--- (질문) ---> [RAG 백엔드]
              (Streamlit/Gradio)                      |      ^
                                                      |      | (답변 + 상세 출처)
                                                      v      |
                                   [Retriever] --(검색)--> [🧠 Vector DB]
                                       |                     |
                                       +----(질문 + 검색된 컨텍스트)----> [LLM]
                                                                      (OpenAI GPT-4o)
```

## 🚀 시작하기

### 1. 사전 요구사항

-  Python 3.9 이상
- Git
- OpenAI API 키:
  텍스트 임베딩과 사용자의 질문에 대한 최종 답변 생성에 사용됩니다.
  OpenAI Platform에서 API 키를 발급받으세요.
- HWP 파일 처리:
  HWP 파일의 텍스트를 추출하기 위해 pyhwp 라이브러리가 필요하며, 내부적으로 hwp5txt 커맨드라인 도구를 사용합니다.

### 2. 설치

1.  **저장소 복제:**
    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```

2.  **가상 환경 생성 및 활성화:**
    ```bash
    python -m venv venv
    # Windows: .\venv\Scripts\activate | macOS/Linux: source venv/bin/activate
    ```

3.  **필요한 라이브러리 설치:**
    크롤링, 문서 파싱, OCR(PyTorch 포함) 등 필요한 모든 라이브러리가 포함된 requirements.txt를 사용합니다.
    ```bash
    pip install -r requirements.txt
    ```

### 3. API 키 및 환경 설정

프로젝트 루트 디렉토리에 `.env` 파일을 생성하고 아래 내용을 채워주세요.

**`.env` 파일:**
```
##크롤링할 학교 공지사항 게시판 URL 목록
##형식: "게시판이름1|URL1,게시판이름2|URL2"
##예시: SCHOOL_ANNOUNCEMENT_URLS="학사공지|https://.../notice,장학공지|https://.../scholarship"
SCHOOL_ANNOUNCEMENT_URLS="공지사항|https://www.your-school.ac.kr/community/notice"

##OpenAI API (임베딩 및 답변 생성용)
OPENAI_API_KEY=sk-...

##(선택) 크롤링 수집 연도 설정 (기본값: 작년)
##예: 2023으로 설정 시, 2023년 1월 1일 이전 게시물은 수집하지 않음
TARGET_YEAR=2023

##(선택) 추출된 텍스트를 별도 디렉토리에 저장할지 여부 (디버깅용)
SAVE_EXTRACTED_TEXT=true
```

### 4. 실행 순서

이 시스템은 각 단계별로 스크립트를 실행하여 구축합니다.

**1단계: 공지사항 크롤링**
학교 웹사이트 구조에 맞게 `crawler.py`를 수정한 후, 아래 명령어를 실행하여 공지사항과 첨부파일을 로컬에 다운로드합니다.
```bash
python crawler.py
```

**2단계: 데이터 색인 (전처리 및 DB 저장)**
다운로드된 파일들을 처리하여 벡터 DB에 저장합니다. 이 과정은 새로운 공지사항이 생길 때마다 실행해주어야 합니다.
```bash
python indexer.py
```

**3단계: 챗봇 애플리케이션 실행**
사용자와 상호작용할 수 있는 웹 기반 챗봇 인터페이스를 실행합니다.
```bash
streamlit run app.py
```
이제 웹 브라우저에 표시된 주소(예: `http://localhost:8501`)로 접속하여 챗봇을 사용할 수 있습니다.

## 📁 프로젝트 구조

```
.
├── crawler.py # 학교 공지사항 게시판 웹 크롤러
├── indexer.py # 전처리 핸들러를 호출하여 최종 색인을 수행하는 스크립트
├── app.py # Streamlit 기반의 챗봇 UI 애플리케이션
│
├── preprocessing/ # 파일 파싱, OCR, 텍스트 분할 등 모든 전처리 로직을 담은 패키지
│ ├── init.py
│ ├── handler.py # 파일 종류를 식별하고 적절한 처리 로직으로 위임하는 컨트롤러
│ ├── document_parser.py # HWP, PDF, DOCX 등 다양한 문서 파일의 구조를 분석하는 로직을 통합한 모듈
│ ├── chunker.py # 텍스트를 의미 단위로 분할(Chunking)하는 공용 모듈
│ └── ocr.py # 이미지 내 텍스트를 추출하는 공용 OCR 모듈
│
├── crawled_data/ # 크롤러가 다운로드한 원본 파일 저장소 (gitignore)
├── chroma_db/ # ChromaDB 데이터 저장소 (gitignore)
│
├── .env # API 키 및 환경 변수 설정 파일 (gitignore)
├── .gitignore # Git 버전 관리 제외 목록
├── README.md # 프로젝트 설명서 (바로 이 파일)
└── requirements.txt # 프로젝트 의존성 목록
```

## 🛠️ 커스터마이징

-   **크롤러 수정**: `crawler.py` 내부를 대상 학교 웹사이트 구조에 맞게 수정해야 합니다.
-   **파일 파서 확장**: 새로운 문서 형식(예: `.pptx`)을 지원하려면 `preprocessing/document_parser.py`에 파싱 함수를 추가하고, `preprocessing/handler.py`의 `DOCUMENT_EXTENSIONS`에 해당 확장자를 등록하면 됩니다.
-   **모델 변경**: `indexer.py`에서 사용하는 임베딩 모델이나 `app.py`에서 사용하는 LLM을 다른 모델로 변경할 할 수 있습니다.
-   **자동화**: `crawler.py`와 `indexer.py` 스크립트를 `APScheduler` 라이브러리나 OS의 `cron` 작업을 사용하여 주기적으로 자동 실행되도록 설정할 수 있습니다.