from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
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

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI()

faiss_index_path = "./faiss_index"
vector_db = FAISS.load_local(faiss_index_path, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"), allow_dangerous_deserialization=True)

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
    uvicorn.run('main:app', host='0.0.0.0', port=8080, reload=True)