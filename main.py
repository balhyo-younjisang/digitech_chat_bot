from fastapi import FastAPI, Request, HTTPException, BackgroundTasks, Depends, File, UploadFile
from fastapi.security import HTTPBasic
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path
import os
import secrets
import httpx
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import uvicorn
import openai
import os
import logging
from dotenv import load_dotenv
import pymupdf 
from starlette.middleware.sessions import SessionMiddleware
from starlette.requests import Request
from pydantic import BaseModel
from langchain_core.documents import Document
import uuid
import json

# 로그인 요청 바디 모델 정의
class LoginRequest(BaseModel):
    username: str
    password: str


load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
session_key = os.getenv("SESSION_KEY", "sessionsecret")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://digitech-chatbot-admin.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)

app.add_middleware(
    SessionMiddleware,
    secret_key=session_key,  # 세션에 사용할 비밀 키
    same_site="none",  # 운영에서는 none
    https_only=True # 운영에서는 True
)

security = HTTPBasic()

faiss_index_path = "./faiss_index"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.load_local(faiss_index_path, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"), allow_dangerous_deserialization=True)

# 환경 변수에서 인증서 파일 경로 읽기
ssl_cert_path = os.getenv("SSL_CERT_PATH")
ssl_key_path = os.getenv("SSL_KEY_PATH")

ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "secret123")

UPLOAD_DIR = Path("files")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True) 

# 기존 벡터 DB 로드 또는 새로 생성
if os.path.exists(faiss_index_path):
    vector_db = FAISS.load_local(faiss_index_path, embedding_model, allow_dangerous_deserialization=True)
else:
    # 벡터 DB가 없으면 새로 생성
    documents = []
    for file_path in UPLOAD_DIR.iterdir():
        if file_path.is_file() and (file_path.suffix == '.txt' or file_path.suffix == ".pdf"):  # 텍스트 파일 또는 PDF만 처리
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                documents.append({"content": content, "metadata": {"source": file_path.name}})
    
    # 벡터화하여 FAISS에 저장
    vector_db = FAISS.from_documents(documents, embedding_model)
    vector_db.save_local(faiss_index_path)

# 세션을 통해 인증된 사용자만 접근할 수 있도록 설정
def verify_user(request: Request):
    username = request.session.get("username")
    if not username or username != ADMIN_USER:
        raise HTTPException(status_code=401, detail="아이디나 비밀번호가 올바르지 않습니다")
    return username


def load_pdf_text(pdf_path):
    """PDF 문서를 로드하고 텍스트를 추출"""
    doc = pymupdf.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text.strip()

# 로그인 처리 엔드포인트
@app.post("/admin/login")
def login(request: Request, login_data: LoginRequest):
    if login_data.username == ADMIN_USER and login_data.password == ADMIN_PASSWORD:
        request.session["username"] = login_data.username
        return {"message": f"{login_data.username} 계정으로 로그인"}
    else:
        raise HTTPException(status_code=401, detail="아이디나 비밀번호가 올바르지 않습니다")


# 로그아웃 처리 엔드포인트
@app.post("/admin/logout")
def logout(session: dict = Depends(lambda: {})):
    session.pop("username", None)  # 세션에서 username 제거
    return {"message": "로그아웃 성공"}

# 문서 목록 조회 엔드포인트 (관리자만 접근)
@app.get("/admin/documents")
def get_documents(user: str = Depends(verify_user)):
    files = [file.name for file in UPLOAD_DIR.iterdir() if file.is_file()]
    return {"documents": files}

# 파일 업로드 엔드포인트
@app.post("/admin/upload")
async def upload_document(file: UploadFile = File(...), user: str = Depends(verify_user)):
    if not (file.filename.endswith(".txt") or file.filename.endswith(".pdf")):
        raise HTTPException(status_code=400, detail="텍스트 파일(.txt)과 PDF 파일(.pdf)만 업로드할 수 있습니다.")
    
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    if file.filename.endswith(".txt"):
        document_content = content.decode('utf-8')
    elif file.filename.endswith(".pdf"):
        document_content = load_pdf_text(file_path)

    # 새로 업로드된 파일을 벡터 DB에 추가
    doc_id = str(uuid.uuid4())
    new_document = Document(page_content=document_content, metadata={"source": file.filename}, id=doc_id)
    vector_db.add_documents([new_document])
    vector_db.save_local(faiss_index_path)

    id_map_path = Path("document_id_map.json")
    if not id_map_path.exists():
        id_map_path.write_text("{}")

    with open(id_map_path, "r+", encoding="utf-8") as f:
        id_map = json.load(f)
        id_map[file.filename] = doc_id
        f.seek(0)
        f.truncate()
        json.dump(id_map, f)

    return {"message": f"파일 '{file.filename}'이 업로드되었습니다."}    


# 문서 다운로드 엔드포인트
@app.get("/admin/documents/{filename}")
def get_document(filename: str, user: str = Depends(verify_user)):
    file_path = UPLOAD_DIR / filename
    if file_path.exists():
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다")

# 문서 삭제 엔드포인트
@app.delete("/admin/documents/{filename}")
def delete_document(filename: str, user: str = Depends(verify_user)):
    file_path = UPLOAD_DIR / filename
    if file_path.exists():
        os.remove(file_path)
 
	        # id 매핑 파일에서 id 조회
        with open("document_id_map.json", "r+") as f:
            id_map = json.load(f)
            doc_id = id_map.get(filename)

            if doc_id:
                vector_db.delete(ids=[doc_id])
                vector_db.save_local(faiss_index_path)

                # 매핑에서도 제거
                del id_map[filename]
                f.seek(0)
                f.truncate()
                json.dump(id_map, f)	

        return {"message": f"파일 '{filename}'이 삭제되었습니다."}
    raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다")

@app.post("/why-digitech")
def why_digitech(request : Request):
    return {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "basicCard": {
                        "title": "WHY 디지텍?",
                        "description": "빠르게 변화하는 글로벌 디지털 시대, \n 신산업 분야의 수요를 반영한 교육과 글로벌 취업 창업이 가능한 IT 분야 인재 육성 \n  학생이 주인공인 즐거운 학교, 재미있게 배우는 학교 서울디지텍고등학교",
                        "thumbnail": {
                            "imageUrl": "https://sdh.sen.hs.kr/dggb/module/image/selectDesignImageView.do?sitemapId=310558&usrimgId=75739"
                        },
                        "buttons": [
                            {
                            "label": "더 알아보기",
                            "action": "webLink",
                            "webLinkUrl": "https://sdh.sen.hs.kr/198663/subMenu.do"
                            },
                            {
                                "label": "전화하기",
                                "action": "phone",
                                "phoneNumber": "02-798-3641"
                            },
                                {
                                "label": "공유하기",
                                "action": "share"
                            }
                        ]
                    }
                }
            ]
        }
    }

@app.post("/apply-middleschool-experience")
def apply_middleschool_experience(request : Request):
    return {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "textCard": {
                        "title" : '2025학년도 서울디지텍고등학교 1학기 "꿈을 찾는 진로체험!!" 신청 안내',
                         "description": "2025학년도 서울디지텍고등학교 1학기에 진행하는 꿈을 찾는 진로 체험!! \n 4월 9일부터 시작하는 매주 수요일 14시~16시, 꿈을 찾는 체험 과정에 함께 해요",
                         "buttons": [
                            {
                                "action": "webLink",
                                "label": "신청하러 가기",
                                "webLinkUrl": "https://naver.me/GbDu79HR"
                            }
                         ]
                    }
                }
            ]
        }
    }

@app.post("/message")
async def message(request : Request, background_tasks : BackgroundTasks):
    """사용자 질문을 받아 RAG 기반 검색 및 LLM 응답 생성"""
    kakao_request = await request.json()
    user_message = kakao_request['userRequest']['utterance']
    callback_url = kakao_request['userRequest']['callbackUrl']

    # 벡터 DB에서 관련 문서 검색
    docs = vector_db.similarity_search(user_message, k=10)

    # 검색된 문서의 텍스트 가져오기
    context = "\n".join([doc.page_content for doc in docs])    

    background_tasks.add_task(process_llm_response, user_message, context, callback_url)

    return {
        "version": "2.0",
        "useCallback": True,
        "data": {
            "msg":"답변을 입력중입니다..."
        }
    }
    
async def process_llm_response(user_message : str, context : str, callback_url : str):
     # Llama 모델에 전달할 메시지 생성
    prompt = f"""
        너는 서울디지텍고등학교에 대한 정보를 제공하는 챗봇이야.  
        학생, 학부모, 교사들에게 학교에 대한 정확하고 유용한 정보를 제공해야 해.  
        반드시 **한국어**로만 답변해. 영어를 사용하지 마.

        **규칙:**  
        - 서울디지텍고등학교(Seoul Digitech High School)에 대한 정보를 제공해.  
        - 친절하고 이해하기 쉽게 설명해야 해.  
        - 질문이 모호하면 정중하게 다시 물어봐.  
        - 제공된 정보(context) 내에서만 답변해야 해.  

        **학교 정보:**  
        {context}


        ** 질문 **:
        {user_message}

        ** 답변 **:
    """

    # response = ollama.chat(model="llama3.2:1b", messages=[{"role": "user", "content": prompt}])\
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    try:
        httpx.post(callback_url, json={
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": response["choices"][0]["message"]["content"]
                        }   
                    }
                ]
            }
        })
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail="Failed to calback data")



@app.post("/get-banner")
async def getBannerList(request : Request):
    url = "https://sdh.sen.hs.kr"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")
        image_items = []

        div_bx_viewport = soup.find('div', class_='main_slide_banner')
        if not div_bx_viewport:
            raise HTTPException(status_code=404, detail="bx-viewport 클래스를 가진 <div>를 찾을 수 없습니다.")

        li_tags = div_bx_viewport.find_all('li')
        if not li_tags:
            raise HTTPException(status_code=404, detail="<li> 태그를 찾을 수 없습니다.")

        li_tags = li_tags[:10]

        for slide in li_tags:  
            img_tag = slide.find("img")
            if not img_tag:
                continue  # 이미지 태그가 없으면 스킵
            image_text = img_tag.get("alt", "").strip()
            image_url = img_tag.get("src", "").strip()
            if not image_text or not image_url:
                continue  # 필수 값이 없으면 스킵
            full_url = urljoin(url, image_url)

            item = {
                "title": f"서울디지텍고등학교 {image_text} 한 눈에 보기",
                "thumbnail": {
                    "imageUrl": full_url
                },
                "buttons": [
                    {
                        "action": "webLink",
                        "label": "자세히 보기",
                        "webLinkUrl": url
                    }
                ]
            }
            image_items.append(item)

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail="Failed to fetch page")
    
    
    return {
        "version": "2.0",
        "template": {
            "outputs": [
                {       
                    "carousel": {
                        "type": "basicCard",
                        "items": image_items
                    }
                }
            ]
        }
    }


if __name__ == '__main__' :
    uvicorn.run('main:app', host='0.0.0.0', port=8080, reload=True,
                ssl_keyfile=ssl_key_path,  # 개인 키 경로
                ssl_certfile=ssl_cert_path # 인증서 경로
                )
