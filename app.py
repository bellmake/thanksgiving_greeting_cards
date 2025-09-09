# app.py
import os, uuid, shutil, time, re, random
from io import BytesIO
from datetime import datetime
from typing import List, Tuple, Optional
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw

from dotenv import load_dotenv
from google import genai

# ---------- .env 로드 ----------
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")  # .env에는 GOOGLE_API_KEY만 두세요
API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("환경변수 GOOGLE_API_KEY(.env) 가 필요합니다.")

# ---------- FastAPI ----------
app = FastAPI(title="BillGates + You in Korea (4 shots, multi-reference, fast-fail)")
STATIC_DIR = str(BASE_DIR / "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ---------- 장면 설정 ----------
BILLGATES_SCENES: List[Tuple[str, str]] = [
    ("경복궁 근정전 앞",
     "at Gyeongbokgung Palace (Geunjeongjeon), early morning soft light, traditional palace architecture in background"),
    ("명동 거리 카페",
     "at a trendy cafe in Myeongdong street, casual friendly atmosphere, standing close together with arms around each other's shoulders in a warm friendly pose"),
    ("한강공원 벤치",
     "at Hangang Park on a bench, afternoon golden hour lighting, relaxed casual setting with Seoul skyline in background"),
    ("N서울타워 전망대",
     "at N Seoul Tower observatory, sunset skyline view of Seoul"),
]

JOKER_SCENES: List[Tuple[str, str]] = [
    ("고담시티 거리",
     "on a Gotham City street at night, dramatic urban lighting, dark atmospheric setting"),
    ("아케디 아케이드",
     "in an old arcade, neon lights and vintage game machines in background, moody atmosphere"),
    ("극장 계단",
     "on iconic concrete stairs, dramatic lighting, urban decay background"),
    ("웨인 극장 앞",
     "in front of Wayne Theater, classic Gotham architecture, evening atmosphere"),
]

# ---------- 레이트리밋/재시도/데드라인 설정 (빠른 실패 지향) ----------
MIN_INTERVAL_BETWEEN_CALLS_SEC = 5     # 컷 사이 최소 간격 (짧게)
MAX_RETRIES_PER_SHOT = 1               # 재시도 1회로 제한
PER_REQUEST_DEADLINE_SEC = 35          # HTTP 요청 전체 대기 상한 (초)
_last_call_ts = 0.0                    # 마지막 호출 시각(전역)

# ---------- 유틸 ----------
def visible_watermark(img: Image.Image, tag="AI-Generated"):
    draw = ImageDraw.Draw(img)
    w, h = img.size
    pad = 12
    bar_h = 36
    overlay = Image.new("RGBA", (w, bar_h), (0, 0, 0, 90))
    img.paste(overlay, (0, h - bar_h), overlay)
    draw.text((pad, h - bar_h + 8), f"{tag} • {datetime.now():%Y-%m-%d}", fill=(255, 255, 255, 220))
    return img

def save_image_bytes(image_bytes: bytes, suffix=".png") -> str:
    """디스크에 저장하고, 브라우저가 접근할 URL 경로('/static/..')를 반환."""
    out_name = f"{uuid.uuid4().hex}{suffix}"
    fs_path = os.path.join(STATIC_DIR, out_name)
    with open(fs_path, "wb") as f:
        f.write(image_bytes)
    return f"/static/{out_name}"

def _parse_retry_delay_seconds(err: Exception) -> Optional[int]:
    m = re.search(r'"retryDelay"\s*:\s*"(\d+)s"', str(err))
    return int(m.group(1)) if m else None

def is_quota_error(err: Exception) -> bool:
    s = str(err)
    return ("RESOURCE_EXHAUSTED" in s) or ("429" in s) or ("rate" in s.lower())

def _sleep_until_min_interval():
    """컷 사이 최소 간격 확보하되, 최대 3초까지만 기다림(UX 보호)."""
    global _last_call_ts
    now = time.time()
    elapsed = now - _last_call_ts
    remaining = MIN_INTERVAL_BETWEEN_CALLS_SEC - elapsed
    if remaining > 0:
        remaining = min(remaining, 3)  # 캡: 3초
        time.sleep(remaining + random.uniform(0.1, 0.3))

def _update_last_call_ts():
    global _last_call_ts
    _last_call_ts = time.time()

def downscale_max_side(img: Image.Image, max_side: int = 768) -> Image.Image:
    """최대 변 길이를 제한해 토큰/비용을 줄임."""
    w, h = img.size
    scale = max(w, h) / max_side
    if scale > 1:
        img = img.resize((int(w / scale), int(h / scale)), Image.LANCZOS)
    return img

def compose_prompt(scene_label: str, scene_desc: str, character_type: str, use_exact_character: bool, num_refs: int) -> str:
    """정체성 유지 지시 강화 + 다중 참조 이미지 활용 프롬프트."""
    
    ref_instruction = ""
    if num_refs == 1:
        ref_instruction = "PERSON A (center): The same individual shown in the uploaded reference selfie."
    elif num_refs == 2:
        ref_instruction = "PERSON A (center): The same individual shown in BOTH uploaded reference selfies."
    elif num_refs == 3:
        ref_instruction = "PERSON A (center): The same individual shown in ALL THREE uploaded reference selfies."
    else:  # 4장 이상
        ref_instruction = f"PERSON A (center): The same individual shown in ALL {num_refs} uploaded reference selfies."
    
    if character_type == "billgates":
        character_phrase = (
            "Bill Gates" if use_exact_character
            else "a Bill Gates look-alike (middle-aged Caucasian male with glasses)"
        )
        
        return (
            "Create a single photorealistic candid smartphone photo of two people.\n"
            f"{ref_instruction} "
            "ULTRA-STRICT IDENTITY PRESERVATION: Keep PERSON A's face identity ABSOLUTELY IDENTICAL to the reference photo(s). "
            "If only one reference photo is provided, maintain the EXACT same face, head pose, gaze direction, facial expression, "
            "hair style, skin tone, and all facial features WITHOUT ANY MODIFICATIONS. Do not change anything about the person's appearance. "
            "If multiple references are provided, analyze ALL images comprehensively to extract the MOST CONSISTENT features. "
            "CRITICAL EYE PRESERVATION: Pay special attention to eye shape, eye color, eyelid structure, eyebrow shape and thickness, "
            "eye spacing, and gaze direction. Eyes must be IDENTICAL to the reference photo(s). "
            "Maintain EXACT facial features, bone structure, nose shape, mouth shape, jawline, and any distinctive characteristics. "
            "For clothing: Keep the same style, colors, and type of clothing shown in the reference photo(s). "
            "Do NOT change the outfit unless absolutely necessary for the scene context.\n"
            f"PERSON B: {character_phrase}.\n"
            f"Scene: {scene_desc}.\n"
            "Camera: Natural smartphone photo style, ~35mm equivalent, realistic lighting & shadows, proper hand/finger anatomy, "
            "casual appropriate outfits for the scene. Both people should look natural and candid.\n"
            "ABSOLUTELY NO text overlays, timestamps, location names, or any written elements in the image. "
            "No borders. Only one image in the result."
        )
    
    else:  # joker
        return (
            "Create a single photorealistic candid smartphone photo of THREE people standing together.\n"
            f"{ref_instruction} "
            "ULTRA-STRICT IDENTITY PRESERVATION: Keep PERSON A's face identity ABSOLUTELY IDENTICAL to the reference photo(s). "
            "If only one reference photo is provided, maintain the EXACT same face, head pose, gaze direction, facial expression, "
            "hair style, skin tone, and all facial features WITHOUT ANY MODIFICATIONS. Do not change anything about the person's appearance. "
            "If multiple references are provided, analyze ALL images comprehensively to extract the MOST CONSISTENT features. "
            "CRITICAL EYE PRESERVATION: Pay special attention to eye shape, eye color, eyelid structure, eyebrow shape and thickness, "
            "eye spacing, and gaze direction. Eyes must be IDENTICAL to the reference photo(s). "
            "Maintain EXACT facial features, bone structure, nose shape, mouth shape, jawline, and any distinctive characteristics. "
            "For clothing: Keep the same style, colors, and type of clothing shown in the reference photo(s). "
            "Do NOT change the outfit unless absolutely necessary for the scene context.\n"
            "PERSON B (left): Joaquin Phoenix as Joker from the 2019 movie - distinctive red suit, green hair, white face paint with red smile, "
            "thin build, intense eyes, standing on the left side.\n"
            "PERSON C (right): Heath Ledger as Joker from The Dark Knight - purple suit, messy green hair, white face paint with black around eyes "
            "and red Glasgow smile scars, standing on the right side.\n"
            "POSE: All three people are standing close together with arms around each other's shoulders in a warm, friendly group pose. "
            "PERSON A is in the CENTER between the two Jokers, with one arm around each Joker's shoulder. "
            "The two Jokers also have their arms around PERSON A's shoulders, creating a tight group embrace.\n"
            f"Scene: {scene_desc}.\n"
            "Camera: Natural smartphone photo style, ~35mm equivalent, realistic lighting & shadows, proper hand/finger anatomy. "
            "All three people should look natural and friendly despite the Jokers' makeup.\n"
            "ABSOLUTELY NO text overlays, timestamps, location names, or any written elements in the image. "
            "No borders. Only one image in the result."
        )

def call_gemini_generate(ref_images: List[Image.Image], prompt: str) -> bytes:
    """Gemini 호출: 참조 사진(다중)을 먼저, 프롬프트를 나중에. 후보 1개(기본). 빠른 실패/짧은 백오프."""
    client = genai.Client(api_key=API_KEY)
    model_name = "gemini-2.5-flash-image-preview"  # 최고 성능 모델

    # contents 구성: [ref1, ref2, ref3, ..., prompt]
    contents = []
    for im in ref_images:
        contents.append(im)
    contents.append(prompt)

    last_err = None
    start_ts = time.time()

    for attempt in range(1, MAX_RETRIES_PER_SHOT + 1):
        try:
            _sleep_until_min_interval()

            # NOTE: google-genai 최신 버전은 generation_config 파라미터를 받지 않습니다.
            response = client.models.generate_content(
                model=model_name,
                contents=contents
            )
            _update_last_call_ts()

            # 첫 번째 후보의 첫 번째 inline 이미지 한 장만 사용
            for part in response.candidates[0].content.parts:
                if getattr(part, "inline_data", None) is not None:
                    return part.inline_data.data
            raise RuntimeError("이미지 생성에 실패했습니다(텍스트 응답만 수신).")

        except Exception as e:
            last_err = e
            _update_last_call_ts()

            if is_quota_error(e) and attempt < MAX_RETRIES_PER_SHOT:
                delay = _parse_retry_delay_seconds(e) or MIN_INTERVAL_BETWEEN_CALLS_SEC
                delay = min(delay, 6)  # 너무 오래 기다리지 않도록 캡
                elapsed = time.time() - start_ts
                if elapsed + delay > PER_REQUEST_DEADLINE_SEC:
                    break  # 더 기다리면 타임아웃 위험 → 즉시 실패
                time.sleep(delay + random.uniform(0.2, 0.8))
                continue
            break

    raise last_err

# ---------- HTML ----------
HTML_INDEX = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>AI Photo Generator — 캐릭터 선택</title>
    <style>
      body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:32px;color:#111;background:#f8fafc}
      .card{max-width:800px;margin:0 auto;background:white;border-radius:16px;box-shadow:0 4px 6px -1px rgba(0,0,0,0.1);padding:32px}
      h1{margin:0 0 16px 0;font-size:32px;font-weight:700;text-align:center}
      .subtitle{color:#666;text-align:center;margin-bottom:32px;font-size:18px}
      .options{display:grid;grid-template-columns:1fr 1fr;gap:24px;margin-bottom:32px}
      .option{border:2px solid #e5e5e5;border-radius:12px;padding:24px;text-align:center;cursor:pointer;transition:all 0.2s}
      .option:hover{border-color:#111;transform:translateY(-2px);box-shadow:0 4px 12px rgba(0,0,0,0.1)}
      .option-title{font-size:24px;font-weight:600;margin-bottom:8px}
      .option-desc{color:#666;line-height:1.5}
      .emoji{font-size:48px;margin-bottom:16px;display:block}
      .note{background:#f0f9ff;border:1px solid #0284c7;border-radius:10px;padding:16px;color:#0c4a6e;text-align:center}
    </style>
  </head>
  <body>
    <div class="card">
      <h1>🤖 AI Photo Generator</h1>
      <div class="subtitle">어떤 캐릭터와 함께 사진을 생성하시겠습니까?</div>
      
      <div class="options">
        <div class="option" onclick="location.href='/billgates'">
          <span class="emoji">👔</span>
          <div class="option-title">Bill Gates와 함께</div>
          <div class="option-desc">마이크로소프트 창립자 빌 게이츠와 함께 한국 명소에서 찍은 듯한 사진을 생성합니다.</div>
        </div>
        
        <div class="option" onclick="location.href='/joker'">
          <span class="emoji">🃏</span>
          <div class="option-title">Joker들과 함께</div>
          <div class="option-desc">호아킨 피닉스 조커와 히스 레저 조커 사이에서 어깨동무하며 찍은 듯한 사진을 생성합니다.</div>
        </div>
      </div>
      
      <div class="note">
        <strong>주의사항:</strong> 본 서비스는 AI 합성 이미지를 생성합니다. 사칭이나 허위정보 목적으로 사용을 금지하며, 업로드된 이미지는 처리 후 즉시 삭제됩니다.
      </div>
    </div>
  </body>
</html>
"""

HTML_BILLGATES = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>BillGates + You in Korea</title>
    <style>
      body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:32px;color:#111}
      .card{max-width:900px;margin:0 auto}
      h1{margin:0 0 8px 0}
      .muted{color:#666}
      .grid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin-top:18px}
      .imgbox{border:1px solid #e5e5e5;border-radius:12px;overflow:hidden}
      .note{background:#f8fafc;border:1px solid #e5e7eb;border-radius:10px;padding:12px;margin-top:16px}
      .btn{background:#111;color:#fff;border:none;border-radius:10px;padding:10px 16px;cursor:pointer}
      .row{display:flex;gap:12px;align-items:center;margin:10px 0}
      footer{margin-top:24px;color:#666;font-size:13px}
      .pill{padding:2px 8px;border-radius:999px;border:1px solid #ddd;display:inline-block;font-size:12px}
      a{text-decoration:none}
      small{color:#555}
      .back-btn{background:#666;color:#fff;border-radius:6px;padding:6px 12px;text-decoration:none;font-size:14px;margin-bottom:16px;display:inline-block}
    </style>
  </head>
  <body>
    <div class="card">
      <a href="/" class="back-btn">← 캐릭터 선택으로 돌아가기</a>
      <h1>👔 Bill Gates와 함께 in Korea — 4컷</h1>
      <div class="muted">셀피를 올리면 빌 게이츠와 함께 한국 명소에서 찍은 듯한 사진을 생성합니다.</div>
      <form action="/generate" method="post" enctype="multipart/form-data">
        <input type="hidden" name="character_type" value="billgates">
        <div class="row">
          <label>셀피 업로드(최소 1장):
            <input type="file" name="selfies" accept="image/*" multiple required>
          </label>
        </div>
        <small>※ 다양한 각도의 사진일수록 정체성 일관성이 높아집니다.</small>
        <div class="row">
          <label><input type="checkbox" name="exact_character" checked>
            빌 게이츠 실존 인물로 시도 (정책 이슈 시 look-alike로 전환)</label>
        </div>
        <div class="note">
          <b>주의/윤리</b> · 본 서비스는 AI 합성 이미지를 생성합니다. 사칭/허위정보 사용은 금지.
          업로드 이미지는 처리 후 즉시 삭제됩니다.
        </div>
        <div class="row"><button class="btn" type="submit">4장 생성하기</button></div>
      </form>
      <footer>
        모델: Google <b>Gemini 2.5 Flash Image</b>
      </footer>
    </div>
  </body>
</html>
"""

HTML_JOKER = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Jokers + You in Gotham</title>
    <style>
      body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:32px;color:#111}
      .card{max-width:900px;margin:0 auto}
      h1{margin:0 0 8px 0}
      .muted{color:#666}
      .grid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin-top:18px}
      .imgbox{border:1px solid #e5e5e5;border-radius:12px;overflow:hidden}
      .note{background:#f8fafc;border:1px solid #e5e7eb;border-radius:10px;padding:12px;margin-top:16px}
      .btn{background:#111;color:#fff;border:none;border-radius:10px;padding:10px 16px;cursor:pointer}
      .row{display:flex;gap:12px;align-items:center;margin:10px 0}
      footer{margin-top:24px;color:#666;font-size:13px}
      .pill{padding:2px 8px;border-radius:999px;border:1px solid #ddd;display:inline-block;font-size:12px}
      a{text-decoration:none}
      small{color:#555}
      .back-btn{background:#666;color:#fff;border-radius:6px;padding:6px 12px;text-decoration:none;font-size:14px;margin-bottom:16px;display:inline-block}
    </style>
  </head>
  <body>
    <div class="card">
      <a href="/" class="back-btn">← 캐릭터 선택으로 돌아가기</a>
      <h1>🃏 Jokers와 함께 in Gotham — 4컷</h1>
      <div class="muted">셀피를 올리면 호아킨 피닉스 조커와 히스 레저 조커 사이에서 어깨동무하며 찍은 듯한 사진을 생성합니다.</div>
      <form action="/generate" method="post" enctype="multipart/form-data">
        <input type="hidden" name="character_type" value="joker">
        <div class="row">
          <label>셀피 업로드(최소 1장):
            <input type="file" name="selfies" accept="image/*" multiple required>
          </label>
        </div>
        <small>※ 다양한 각도의 사진일수록 정체성 일관성이 높아집니다.</small>
        <div class="note">
          <b>주의/윤리</b> · 본 서비스는 AI 합성 이미지를 생성합니다. 사칭/허위정보 사용은 금지.
          업로드 이미지는 처리 후 즉시 삭제됩니다.
        </div>
        <div class="row"><button class="btn" type="submit">4장 생성하기</button></div>
      </form>
      <footer>
        모델: Google <b>Gemini 2.5 Flash Image</b>
      </footer>
    </div>
  </body>
</html>
"""

# ---------- 라우트 ----------
@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_INDEX

@app.get("/billgates", response_class=HTMLResponse)
def billgates():
    return HTML_BILLGATES

@app.get("/joker", response_class=HTMLResponse)
def joker():
    return HTML_JOKER

@app.post("/generate", response_class=HTMLResponse)
async def generate(
    selfies: List[UploadFile] = File(...),
    character_type: str = Form(...),
    exact_character: bool = Form(False)
):
    # ---- 업로드 파일 저장 (모든 사진 사용) ----
    temp_paths = []
    ref_images: List[Image.Image] = []
    try:
        if not selfies:
            return HTMLResponse("<h3>셀피를 최소 1장 이상 업로드하세요.</h3>", status_code=400)

        for i, uf in enumerate(selfies):  # 모든 사진 사용
            temp_path = os.path.join(STATIC_DIR, f"upload_{i}_{uuid.uuid4().hex}")
            with open(temp_path, "wb") as f:
                shutil.copyfileobj(uf.file, f)
            temp_paths.append(temp_path)

            img = Image.open(temp_path).convert("RGB")
            img = downscale_max_side(img, max_side=768)
            ref_images.append(img)

        out_urls, errors = [], []
        
        # 캐릭터 타입에 따라 장면 선택
        scenes = BILLGATES_SCENES if character_type == "billgates" else JOKER_SCENES

        for scene_label, scene_desc in scenes:  # 4컷
            prompt = compose_prompt(scene_label, scene_desc, character_type, use_exact_character=exact_character, num_refs=len(ref_images))
            try:
                img_bytes = call_gemini_generate(ref_images, prompt)
            except Exception as e1:
                # 429/쿼터: 페일오버도 하지 않고 실패 기록 (다음 컷으로)
                if is_quota_error(e1):
                    errors.append(f"{scene_label}: 실패 — {e1}")
                    continue
                # 정책/콘텐츠 이슈 추정 시 look-alike로 1회 재시도 (빌게이츠만)
                if exact_character and character_type == "billgates":
                    try:
                        fallback_prompt = compose_prompt(scene_label, scene_desc, character_type, use_exact_character=False, num_refs=len(ref_images))
                        img_bytes = call_gemini_generate(ref_images, fallback_prompt)
                    except Exception as e2:
                        errors.append(f"{scene_label}: 실패 — {e2}")
                        continue
                else:
                    errors.append(f"{scene_label}: 실패 — {e1}")
                    continue

            # 이미지 저장 (워터마크 없이)
            saved_url = save_image_bytes(img_bytes, suffix=".png")
            out_urls.append(saved_url)

    finally:
        # 업로드 원본 즉시 삭제
        for p in temp_paths:
            try:
                os.remove(p)
            except:
                pass

    # ---- 결과 페이지 ----
    character_title = "Bill Gates" if character_type == "billgates" else "Jokers"
    thumbs = "".join(
        f'<div class="imgbox"><img src="{u}" style="width:100%;display:block"/></div>'
        for u in out_urls
    )
    err_html = ""
    if errors:
        err_list = "<br/>".join(errors)
        err_html = f'<div class="note" style="margin-top:16px;color:#b42318;border-color:#fecaca;background:#fff1f2"><b>일부 실패</b><br/>{err_list}</div>'

    html = f"""
    <html><head><meta charset="utf-8"><title>결과 — {character_title} 4컷</title>
    <style>
      body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:32px;color:#111}}
      .grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}}
      .bar{{display:flex;gap:12px;align-items:center;margin-bottom:16px}}
      a.btn{{background:#111;color:#fff;border-radius:10px;padding:8px 14px;text-decoration:none}}
      .muted{{color:#666}}
    </style></head>
    <body>
      <div class="bar">
        <a class="btn" href="/">← 캐릭터 선택</a>
        <a class="btn" href="/{character_type}">← 다시 만들기</a>
        <div class="muted">생성 {len(out_urls)}장</div>
      </div>
      <div class="grid">{thumbs}</div>
      {err_html}
      <p class="muted" style="margin-top:18px">
        모든 이미지는 Google Gemini 2.5 Flash Image로 생성되었습니다.
      </p>
    </body></html>
    """
    return HTMLResponse(html)
