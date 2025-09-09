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

# ---------- 장면(4컷) ----------
SCENES: List[Tuple[str, str]] = [
    ("08:00 경복궁 근정전 앞",
     "at Gyeongbokgung Palace (Geunjeongjeon), early morning soft light, traditional palace architecture in background"),
    ("12:00 명동 거리 카페",
     "at a trendy cafe in Myeongdong street, casual friendly atmosphere, standing close together with arms around each other's shoulders in a warm friendly pose"),
    ("16:00 한강공원 벤치",
     "at Hangang Park on a bench, afternoon golden hour lighting, relaxed casual setting with Seoul skyline in background"),
    ("19:00 N서울타워 전망대",
     "at N Seoul Tower observatory, sunset skyline view of Seoul"),
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

def compose_prompt(scene_label: str, scene_desc: str, use_exact_billgates: bool, num_refs: int) -> str:
    """정체성 유지 지시 강화 + 다중 참조 이미지 활용 프롬프트."""
    billgates_phrase = (
        "Bill Gates" if use_exact_billgates
        else "a Bill Gates look-alike (middle-aged Caucasian male with glasses)"
    )
    
    ref_instruction = ""
    if num_refs == 1:
        ref_instruction = "PERSON A: The same individual shown in the uploaded reference selfie."
    elif num_refs == 2:
        ref_instruction = "PERSON A: The same individual shown in BOTH uploaded reference selfies."
    elif num_refs == 3:
        ref_instruction = "PERSON A: The same individual shown in ALL THREE uploaded reference selfies."
    else:  # 4장 이상
        ref_instruction = f"PERSON A: The same individual shown in ALL {num_refs} uploaded reference selfies."
    
    return (
        "Create a single photorealistic candid smartphone photo of two people.\n"
        f"{ref_instruction} "
        "ULTRA-STRICT IDENTITY PRESERVATION: Keep PERSON A's face identity ABSOLUTELY IDENTICAL across all reference photos. "
        "Analyze ALL reference images comprehensively to extract the MOST CONSISTENT and STABLE facial features. "
        "Maintain EXACT facial features, bone structure, eye shape, nose shape, mouth shape, jawline, skin tone, age appearance, "
        "facial proportions, and any distinctive characteristics (moles, scars, dimples, etc.). "
        "CRITICAL: When multiple references show variations, prioritize the features that appear MOST FREQUENTLY and CONSISTENTLY "
        "across the majority of reference images. Do NOT average or blend features - select the most reliable, recognizable traits. "
        "For hair: use the style and color that appears in the clearest/most recent reference. "
        "For skin tone: match the most consistent tone across all references. "
        "Facial expressions can vary naturally but the underlying bone structure and facial identity must remain ABSOLUTELY UNCHANGED. "
        "The person must be immediately recognizable as the same individual from all reference photos.\n"
        f"PERSON B: {billgates_phrase}.\n"
        f"Scene: {scene_desc}; time/place label: {scene_label} in Seoul.\n"
        "Camera: Natural smartphone photo style, ~35mm equivalent, realistic lighting & shadows, proper hand/finger anatomy, "
        "casual appropriate outfits for the scene. Both people should look natural and candid.\n"
        "No text overlays. No borders. Only one image in the result."
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
    <title>BillGates + You in Korea — dual-reference</title>
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
    </style>
  </head>
  <body>
    <div class="card">
      <h1>Bill Gates와 함께 in Korea — 4컷 (참조 3장 이상)</h1>
      <div class="muted">셀피 <b>최소 3장 이상</b>을 올리면, 모든 사진을 참조로 사용해 <b>정체성 일관성</b>을 극대화하여 생성합니다.</div>
      <form action="/generate" method="post" enctype="multipart/form-data">
        <div class="row">
          <label>셀피 업로드(최소 3장, 개수 제한 없음):
            <input type="file" name="selfies" accept="image/*" multiple required>
          </label>
        </div>
        <small>※ 최소 3장 이상의 다양한 각도 사진을 권장합니다. 정면, 좌측, 우측, 위, 아래 각도 등 더 많은 사진이 정체성 일관성을 높입니다.</small>
        <div class="row">
          <label><input type="checkbox" name="exact_billgates" checked>
            빌 게이츠 실존 인물로 시도 (정책/콘텐츠 이슈 시 look-alike로 전환, 단 429/쿼터는 제외)</label>
        </div>
        <div class="note">
          <b>주의/윤리</b> · 본 서비스는 AI 합성 이미지를 생성하며, 모든 생성물에는
          <span class="pill">AI-Generated</span> 표시가 추가됩니다. 사칭/허위정보 사용은 금지.
          업로드 이미지는 처리 후 즉시 삭제됩니다.
        </div>
        <div class="row"><button class="btn" type="submit">4장 생성하기</button></div>
      </form>
      <footer>
        모델: Google <b>Gemini 2.5 Flash Image</b> · SynthID 워터마크 포함
      </footer>
    </div>
  </body>
</html>
"""

# ---------- 라우트 ----------
@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_INDEX

@app.post("/generate", response_class=HTMLResponse)
async def generate(
    selfies: List[UploadFile] = File(...),  # 개수 제한 없음
    exact_billgates: bool = Form(False)
):
    # ---- 업로드 파일 저장 (모든 사진 사용) ----
    temp_paths = []
    ref_images: List[Image.Image] = []
    try:
        if not selfies or len(selfies) < 3:
            return HTMLResponse("<h3>셀피를 최소 3장 이상 업로드하세요.</h3>", status_code=400)

        for i, uf in enumerate(selfies):  # 모든 사진 사용
            temp_path = os.path.join(STATIC_DIR, f"upload_{i}_{uuid.uuid4().hex}")
            with open(temp_path, "wb") as f:
                shutil.copyfileobj(uf.file, f)
            temp_paths.append(temp_path)

            img = Image.open(temp_path).convert("RGB")
            img = downscale_max_side(img, max_side=768)
            ref_images.append(img)

        out_urls, errors = [], []

        for scene_label, scene_desc in SCENES:  # 4컷
            prompt = compose_prompt(scene_label, scene_desc, use_exact_billgates=exact_billgates, num_refs=len(ref_images))
            try:
                img_bytes = call_gemini_generate(ref_images, prompt)
            except Exception as e1:
                # 429/쿼터: 페일오버도 하지 않고 실패 기록 (다음 컷으로)
                if is_quota_error(e1):
                    errors.append(f"{scene_label}: 실패 — {e1}")
                    continue
                # 정책/콘텐츠 이슈 추정 시 look-alike로 1회 재시도
                if exact_billgates:
                    try:
                        fallback_prompt = compose_prompt(scene_label, scene_desc, use_exact_billgates=False, num_refs=len(ref_images))
                        img_bytes = call_gemini_generate(ref_images, fallback_prompt)
                    except Exception as e2:
                        errors.append(f"{scene_label}: 실패 — {e2}")
                        continue
                else:
                    errors.append(f"{scene_label}: 실패 — {e1}")
                    continue

            # 워터마크 + 저장
            img = Image.open(BytesIO(img_bytes)).convert("RGBA")
            img = visible_watermark(img, tag="AI-Generated")
            buf = BytesIO()
            img.save(buf, format="PNG")
            saved_url = save_image_bytes(buf.getvalue(), suffix=".png")
            out_urls.append(saved_url)

    finally:
        # 업로드 원본 즉시 삭제
        for p in temp_paths:
            try:
                os.remove(p)
            except:
                pass

    # ---- 결과 페이지 ----
    thumbs = "".join(
        f'<div class="imgbox"><img src="{u}" style="width:100%;display:block"/></div>'
        for u in out_urls
    )
    err_html = ""
    if errors:
        err_list = "<br/>".join(errors)
        err_html = f'<div class="note" style="margin-top:16px;color:#b42318;border-color:#fecaca;background:#fff1f2"><b>일부 실패</b><br/>{err_list}</div>'

    html = f"""
    <html><head><meta charset="utf-8"><title>결과 — 4컷</title>
    <style>
      body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:32px;color:#111}}
      .grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}}
      .bar{{display:flex;gap:12px;align-items:center;margin-bottom:16px}}
      a.btn{{background:#111;color:#fff;border-radius:10px;padding:8px 14px;text-decoration:none}}
      .muted{{color:#666}}
    </style></head>
    <body>
      <div class="bar">
        <a class="btn" href="/">← 다시 만들기</a>
        <div class="muted">생성 {len(out_urls)}장</div>
      </div>
      <div class="grid">{thumbs}</div>
      {err_html}
      <p class="muted" style="margin-top:18px">
        모든 이미지는 Google Gemini가 삽입하는 <b>SynthID</b> 워터마크를 포함하며,
        화면 좌측하단의 <b>AI-Generated</b> 표시는 본 앱이 추가합니다.
      </p>
    </body></html>
    """
    return HTMLResponse(html)
