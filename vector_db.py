import pymupdf  # PyMuPDF 패키지
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# 임베딩 모델 로드
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_pdf_text(pdf_path):
    """PDF 문서를 로드하고 텍스트를 추출"""
    doc = pymupdf.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text.strip()

def split_text(text, chunk_size=500, chunk_overlap=100):
    """텍스트를 RAG에 적합한 크기로 분할"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_text(text)
    return chunks

def create_faiss_index(chunks):
    """텍스트 청크를 벡터화하고 FAISS 인덱스에 저장 (from_texts 사용)"""
    # FAISS.from_texts()는 텍스트 목록과 임베딩 함수를 받아 인덱스를 생성합니다.
    index = FAISS.from_texts(chunks, embedding_model)
    return index

# PDF 파일 경로 설정
pdf_path = "./digitech_qna.pdf"
text = load_pdf_text(pdf_path)
chunks = split_text(text)

vector_db = create_faiss_index(chunks)
vector_db.save_local("./faiss_index")
