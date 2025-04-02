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