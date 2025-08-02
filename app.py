# app.py

import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# .env 파일에서 환경 변수 로드
load_dotenv()

# --- 상수 정의 ---
CHROMA_DB_DIR = "chroma_db"

# --- RAG 파이프라인 설정 ---

@st.cache_resource
def load_rag_pipeline():
    """RAG 파이프라인을 로드하고 캐시합니다."""
    # 1. Vector DB 로드
    if not os.path.exists(CHROMA_DB_DIR):
        st.error(f"Vector DB 폴더('{CHROMA_DB_DIR}')를 찾을 수 없습니다. 'indexer.py'를 먼저 실행해주세요.")
        st.stop()
        
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectordb = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    
    # 2. Retriever 설정
    retriever = vectordb.as_retriever(search_kwargs={"k": 3}) # 상위 3개 문서를 검색
    
    # 3. LLM 설정
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # 4. Prompt 템플릿 정의
    prompt_template = """
    당신은 학교 공지사항 전문가입니다. 주어진 문맥(context)을 바탕으로 사용자의 질문에 대해 정확하고 근거 있는 답변을 생성해주세요.
    답변은 다음 지침을 따라야 합니다:
    1. 답변은 반드시 제공된 '문맥' 정보에만 기반해야 합니다. 문맥에 없는 내용은 답변에 포함하지 마세요.
    2. 답변이 어떤 문서에서 비롯되었는지 '출처'를 명시해야 합니다. (예: [출처: 2024년 1학기 장학금 신청 안내.pdf])
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
    
    # 5. RetrievalQA 체인 생성
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain

# --- Streamlit UI ---

# 페이지 설정
st.set_page_config(page_title="🏫 학교 공지사항 RAG 챗봇", page_icon="🤖")

# 제목
st.title("🏫 학교 공지사항 RAG 챗봇")
st.caption("궁금한 학교 공지사항에 대해 무엇이든 물어보세요!")

# RAG 파이프라인 로드
try:
    qa_chain = load_rag_pipeline()
except Exception as e:
    st.error(f"API 키 설정에 문제가 있는 것 같습니다. .env 파일을 확인해주세요. 오류: {e}")
    st.stop()


# 채팅 기록 초기화
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 학교 공지사항에 대해 궁금한 점이 있으신가요?"}]

# 이전 채팅 기록 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 받기
if prompt := st.chat_input("예: 1학기 국가장학금 신청 기간 알려줘"):
    # 사용자 메시지 추가 및 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 챗봇 답변 생성
    with st.chat_message("assistant"):
        with st.spinner("답변을 생성하고 있습니다..."):
            try:
                response = qa_chain.invoke({"query": prompt})
                
                # 답변과 출처를 분리하여 표시
                answer = response['result']
                source_docs = response['source_documents']
                
                # 출처 정보 가공
                sources_text = ""
                if source_docs:
                    unique_sources = set()
                    for doc in source_docs:
                        title = doc.metadata.get('announcement_title', '알 수 없는 공지')
                        filename = os.path.basename(doc.metadata.get('source', '알 수 없는 파일'))
                        unique_sources.add(f"'{title}' 공지의 '{filename}' 파일")
                    
                    sources_text = "\n\n---\n*참고 자료:*\n- " + "\n- ".join(unique_sources)
                
                full_response = answer + sources_text
                st.markdown(full_response)
                
            except Exception as e:
                st.error(f"답변 생성 중 오류가 발생했습니다: {e}")
                full_response = "오류가 발생했습니다. 다시 시도해주세요."

    # 챗봇 메시지 추가
    st.session_state.messages.append({"role": "assistant", "content": full_response})