# check_db.py
import os
import sqlite3
import json
import argparse
import pprint

import chromadb
from dotenv import load_dotenv

# --- RAG 테스트를 위한 추가 임포트 ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# --- 설정 불러오기 ---
load_dotenv()
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "school_announcements")
SQLITE_CACHE_DB = "./cache/embeddings.sqlite"

def check_chromadb(limit: int):
    """ChromaDB에 저장된 데이터를 확인합니다."""
    print("=" * 50)
    print(f"🔍 ChromaDB 확인 중... (경로: {CHROMA_DB_DIR})")
    print("=" * 50)

    if not os.path.exists(CHROMA_DB_DIR):
        print(f"오류: ChromaDB 디렉토리('{CHROMA_DB_DIR}')를 찾을 수 없습니다.")
        print("'indexer.py'를 먼저 실행하여 데이터베이스를 생성해주세요.")
        return

    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        collection = client.get_collection(name=COLLECTION_NAME)

        total_count = collection.count()
        print(f"✅ 총 {total_count}개의 벡터(청크)가 저장되어 있습니다.")

        if total_count == 0:
            return

        print(f"\n📄 샘플 데이터 {min(limit, total_count)}개 미리보기:")
        print("-" * 20)

        # include에 "embeddings"를 추가하면 벡터 값도 볼 수 있습니다.
        data = collection.get(limit=limit, include=["metadatas", "documents"])

        for i in range(len(data["ids"])):
            print(f"\n[{i+1}] ID: {data['ids'][i]}")
            print("  [메타데이터]:")
            pprint.pprint(data['metadatas'][i], indent=4, width=100)
            print("  [내용 (앞 100자)]:")
            document_preview = data['documents'][i][:100].replace('\n', ' ') + "..."
            print(f"    '{document_preview}'")
            print("-" * 20)

    except Exception as e:
        print(f"ChromaDB 확인 중 오류 발생: {e}")
        print(f"컬렉션 '{COLLECTION_NAME}'이 존재하는지 확인해주세요.")


def check_sqlite_cache(limit: int):
    """SQLite 임베딩 캐시 데이터베이스를 확인합니다."""
    print("\n" + "=" * 50)
    print(f"🔍 SQLite 캐시 DB 확인 중... (경로: {SQLITE_CACHE_DB})")
    print("=" * 50)

    if not os.path.exists(SQLITE_CACHE_DB):
        print(f"오류: SQLite 캐시 파일('{SQLITE_CACHE_DB}')을 찾을 수 없습니다.")
        return

    try:
        conn = sqlite3.connect(SQLITE_CACHE_DB)
        cursor = conn.cursor()

        # 전체 개수 확인
        cursor.execute("SELECT COUNT(*) FROM emb")
        total_count = cursor.fetchone()[0]
        print(f"✅ 총 {total_count}개의 임베딩 결과가 캐시되어 있습니다.")

        if total_count == 0:
            conn.close()
            return

        print(f"\n📄 샘플 데이터 {min(limit, total_count)}개 미리보기:")
        print("-" * 20)

        # 샘플 데이터 가져오기
        cursor.execute("SELECT k, v FROM emb LIMIT ?", (limit,))
        rows = cursor.fetchall()

        for i, (key, value) in enumerate(rows):
            vector = json.loads(value)
            vector_preview = str(vector[:3])[:-1] + ", ...]" # 벡터 앞 3개만 표시
            print(f"\n[{i+1}] Key (SHA256): {key[:20]}...")
            print(f"  - Vector (size: {len(vector)}): {vector_preview}")

        conn.close()

    except Exception as e:
        print(f"SQLite 캐시 확인 중 오류 발생: {e}")

def test_rag_query(query: str):
    """주어진 쿼리로 RAG 파이프라인을 테스트합니다."""
    print("\n" + "=" * 50)
    print(f"⚡ RAG 테스트 시작... (질문: '{query}')")
    print("=" * 50)

    if not os.path.exists(CHROMA_DB_DIR):
        print(f"오류: ChromaDB 디렉토리('{CHROMA_DB_DIR}')를 찾을 수 없습니다.")
        print("'indexer.py'를 먼저 실행하여 데이터베이스를 생성해주세요.")
        return

    try:
        # 1. Vector DB 로드
        print("1. Vector DB 로드 중...")
        embeddings = OpenAIEmbeddings(model=os.getenv("EMBED_MODEL", "text-embedding-3-small"))
        vectordb = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})
        print("✅ Vector DB 로드 완료")

        # 2. LLM 및 Prompt 설정
        print("2. LLM 및 Prompt 설정 중...")
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        prompt_template = """
        당신은 학교 공지사항 전문가입니다. 주어진 문맥(context)을 바탕으로 사용자의 질문에 대해 정확하고 근거 있는 답변을 생성해주세요.
        답변은 다음 지침을 따라야 합니다:
        1. 답변은 반드시 제공된 '문맥' 정보에만 기반해야 합니다. 문맥에 없는 내용은 답변에 포함하지 마세요.
        2. 답변이 어떤 문서에서 비롯되었는지 '출처'를 명시해야 합니다.
        3. 명확하고 간결하게 한국어로 답변해주세요.
        4. 문맥에서 답변을 찾을 수 없는 경우, "죄송합니다, 제공된 정보 내에서 질문에 대한 답변을 찾을 수 없습니다."라고 솔직하게 답변하세요.

        [문맥]
        {context}

        [질문]
        {question}

        [답변]
        """
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        print("✅ LLM 및 Prompt 설정 완료")

        # 3. RetrievalQA 체인 생성 및 실행
        print("3. RetrievalQA 체인 생성 및 실행 중...")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        response = qa_chain.invoke({"query": query})
        print("✅ 답변 생성 완료!")

        # 4. 결과 출력
        print("\n" + "-" * 20)
        print("🤖 [생성된 답변]:")
        print(response['result'])
        print("\n" + "-" * 20)
        print("📚 [참고한 문서]:")
        if response['source_documents']:
            for i, doc in enumerate(response['source_documents']):
                metadata = doc.metadata
                print(f"  [{i+1}] 소스: {metadata.get('source', 'N/A')}")
                print(f"      - 카테고리: {metadata.get('category', 'N/A')}")
                print(f"      - 내용 미리보기: '{doc.page_content[:80].replace(chr(10), ' ')}...'")
        else:
            print("  참고한 문서가 없습니다.")
        print("-" * 20)

    except Exception as e:
        print(f"\nRAG 테스트 중 오류 발생: {e}")
        print("OpenAI API 키가 .env 파일에 올바르게 설정되었는지 확인해주세요.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="데이터베이스 확인 및 RAG 테스트 스크립트")
    subparsers = parser.add_subparsers(dest="command", help="실행할 명령어", required=True)

    # 'check' 명령어: 기존 DB 확인 기능
    parser_check = subparsers.add_parser("check", help="데이터베이스 내용을 확인합니다.")
    parser_check.add_argument(
        "--db", type=str, choices=["chroma", "sqlite", "all"], default="all",
        help="확인할 데이터베이스 종류 (chroma: 벡터DB, sqlite: 임베딩 캐시, all: 둘 다)"
    )
    parser_check.add_argument("-n", "--limit", type=int, default=3, help="미리보기로 보여줄 데이터 개수")

    # 'query' 명령어: RAG 테스트 기능
    parser_query = subparsers.add_parser("query", help="간단한 RAG 테스트를 수행합니다.")
    parser_query.add_argument("query_text", type=str, help="챗봇에게 할 질문")

    args = parser.parse_args()

    if args.command == "check":
        if args.db in ["chroma", "all"]:
            check_chromadb(args.limit)
        if args.db in ["sqlite", "all"]:
            check_sqlite_cache(args.limit)
    elif args.command == "query":
        test_rag_query(args.query_text)