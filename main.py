from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import httpx
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import uvicorn

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KakaoRequest(BaseModel):
    userRequest : dict

app = FastAPI()

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
async def message(request : Request):
    kakao_request = await request.json()
    user_message = kakao_request['userRequest']['utterance']
    response_text = process_message(user_message)
    return {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": response_text
                    }
                }
            ]
        }
    }

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

def process_message(user_message: str) -> str:
    return f"사용자님이 {user_message}라고 말씀하셨어요"

if __name__ == '__main__' :
    uvicorn.run('main:app', host='0.0.0.0', port=8080, reload=True)