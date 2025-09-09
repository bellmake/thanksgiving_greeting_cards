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

# ---------- 장면(2컷, 고퀄리티) ----------
SCENES: List[Tuple[str, str]] = [
    ("12:00 명동 거리 카페",
     "at a trendy cafe in Myeongdong street, casual friendly atmosphere, standing close together with arms around each other's shoulders in a warm friendly pose, perfect natural lighting, professional photo quality"),
    ("19:00 N서울타워 전망대",
     "at N Seoul Tower observatory, sunset golden hour lighting with Seoul skyline in background, both looking relaxed and happy, cinematic composition with perfect depth of field"),
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
        ref_instruction = ("PERSON A: The EXACT same individual shown in the uploaded reference selfie. "
                         "ZERO MODIFICATION RULE - SINGLE REFERENCE: When only ONE reference photo is provided, make ABSOLUTELY NO CHANGES to PERSON A. "
                         "PRESERVE EVERYTHING EXACTLY AS SHOWN: "
                         "✓ IDENTICAL facial angle, head position, and neck posture "
                         "✓ EXACT gaze direction and eye focus point "
                         "✓ UNCHANGED facial expression (smile, neutral, etc.) "
                         "✓ PERFECT preservation of ALL eye characteristics (shape, size, iris color, eyelid structure) "
                         "✓ EXACT clothing style, fit, color, and texture from the reference "
                         "✓ IDENTICAL body posture and stance "
                         "✓ UNCHANGED hair style, texture, and any accessories "
                         "✓ EXACT skin tone and lighting on the person "
                         "DO NOT modify, adjust, enhance, or change ANY aspect of PERSON A's appearance. "
                         "Simply place this EXACT person into the new Korean scene without ANY alterations.")
    elif num_refs == 2:
        ref_instruction = ("PERSON A: The same individual shown in BOTH uploaded reference selfies. "
                         "ULTRA-CRITICAL EYE PRESERVATION: Study both references to identify the EXACT eye characteristics and maintain them perfectly.")
    elif num_refs == 3:
        ref_instruction = ("PERSON A: The same individual shown in ALL THREE uploaded reference selfies. "
                         "ULTRA-CRITICAL EYE PRESERVATION: Analyze all three references to extract the most consistent eye features and preserve them exactly.")
    else:  # 4장 이상
        ref_instruction = (f"PERSON A: The same individual shown in ALL {num_refs} uploaded reference selfies. "
                         "ULTRA-CRITICAL EYE PRESERVATION: Comprehensively analyze all references to identify the person's true eye characteristics and maintain them with absolute precision.")
    
    return (
        "Create a single photorealistic candid smartphone photo of two people.\n"
        f"{ref_instruction} "
        "ABSOLUTE MAXIMUM IDENTITY PRESERVATION - ULTIMATE PRIORITY: PERSON A must be PERFECTLY PRESERVED with 100% accuracy in ALL aspects. "
        "SINGLE REFERENCE SPECIAL RULE: If only ONE reference photo is provided, treat it as a PERFECT TEMPLATE that must NOT be modified in ANY way. "
        "Simply transplant this EXACT person into the Korean scene with ZERO changes to face, body, clothing, or pose. "
        "COMPREHENSIVE MULTI-REFERENCE ANALYSIS: Study ALL uploaded reference images with extreme precision to identify the person's TRUE identity. "
        "Extract the MOST RELIABLE and CONSISTENT features across ALL references, ignoring lighting/angle variations.\n"
        "MANDATORY PRESERVATION CHECKLIST - FACIAL FEATURES (CRITICAL FOR SINGLE REFERENCE):\n"
        "✓ EXACT bone structure and facial geometry (jaw shape, cheekbone prominence, forehead shape)\n"
        "✓ ULTRA-PRECISE EYE CHARACTERISTICS - HIGHEST PRIORITY:\n"
        "  • EXACT eye shape and size (round, almond, hooded, etc.)\n"
        "  • IDENTICAL eye spacing and positioning relative to nose bridge\n"
        "  • PERFECT eyelid structure (upper/lower lid fold patterns, thickness)\n"
        "  • EXACT iris color, size, and pupil characteristics\n"
        "  • IDENTICAL eyebrow shape, thickness, arch, and positioning\n"
        "  • PRECISE eye corner shape (inner/outer canthus angles)\n"
        "  • EXACT under-eye area characteristics (bags, lines, shadows)\n"
        "  • IDENTICAL eye expression and natural resting position\n"
        "✓ IDENTICAL nose features (bridge width, nostril shape, tip angle, overall proportions)\n"
        "✓ EXACT mouth and lip characteristics (shape, size, cupid's bow, lip thickness)\n"
        "✓ PERFECT facial proportions (eye-to-nose, nose-to-mouth ratios)\n"
        "✓ CONSISTENT skin tone (match the most representative tone across all references)\n"
        "✓ IDENTICAL age appearance and facial maturity level\n"
        "✓ ALL distinctive marks (moles, freckles, scars, dimples) in EXACT positions\n"
        "✓ HAIR style and color from the clearest/most recent reference\n"
        "MANDATORY PRESERVATION CHECKLIST - BODY & PHYSIQUE (CRITICAL FOR SINGLE REFERENCE):\n"
        "✓ EXACT body proportions and build type (slim, athletic, muscular, etc.)\n"
        "✓ IDENTICAL height proportions relative to the scene\n"
        "✓ CONSISTENT shoulder width and posture characteristics\n"
        "✓ MATCHING body shape and overall physique from reference images\n"
        "✓ PRESERVE natural body language and movement patterns visible in references\n"
        "✓ MAINTAIN consistent body type throughout all generated scenes\n"
        "MANDATORY PRESERVATION CHECKLIST - CLOTHING & STYLE (CRITICAL FOR SINGLE REFERENCE):\n"
        "✓ ANALYZE clothing style and fashion preferences from ALL reference images\n"
        "✓ MAINTAIN consistent personal style aesthetic (casual, formal, sporty, trendy, etc.)\n"
        "✓ PRESERVE color palette preferences shown in reference clothing\n"
        "✓ MATCH clothing fit and silhouette preferences (loose, fitted, oversized, etc.)\n"
        "✓ KEEP accessory style consistent (glasses style, jewelry preferences, etc.)\n"
        "✓ ADAPT reference clothing style appropriately to the new scene while maintaining personal aesthetic\n"
        "ULTRA-CRITICAL INSTRUCTION FOR SINGLE REFERENCE: When only ONE reference photo is provided, this is a PERFECT TEMPLATE. "
        "DO NOT change, modify, adjust, or enhance ANY aspect of PERSON A's appearance, facial features, clothing, or pose. "
        "The reference image represents the EXACT desired appearance that must be preserved completely. "
        "Simply place this IDENTICAL person into the Korean scene background while keeping ALL aspects of their appearance UNCHANGED. "
        "Think of it as copying and pasting the person exactly as they are into a new background.\n"
        "ULTRA-CRITICAL INSTRUCTION FOR MULTIPLE REFERENCES: Analyze ALL reference images to determine the person's CONSISTENT body type and physique. "
        "If references show variations due to clothing or angles, prioritize the MOST REPRESENTATIVE body characteristics. "
        "NEVER alter or idealize the person's natural body proportions - maintain their authentic physique exactly as shown. "
        "EYE CONSISTENCY ULTRA-PRIORITY: The eyes are the most important feature for identity recognition. "
        "Study each reference image to identify the EXACT eye characteristics that remain consistent across different angles and lighting. "
        "For single reference: maintain EXACT eye angle, gaze direction, and expression. "
        "For multiple references: identify the MOST RELIABLE eye features that appear consistently. "
        "Never modify eye shape, size, color, or spacing - these are identity-defining characteristics. "
        "Pay special attention to: eye symmetry, pupil size, iris patterns, eyelash density, and natural eye expression. "
        "CLOTHING ADAPTATION RULES: Study the clothing styles in ALL reference images to understand the person's fashion preferences. "
        "Adapt their clothing style to the scene while maintaining their personal aesthetic - if they wear casual clothes, keep it casual; "
        "if they prefer fitted clothing, maintain that preference; if they like certain colors or patterns, incorporate similar elements. "
        "The outfit should feel natural for that person while being appropriate for the Korean scene. "
        "The generated person must be INSTANTLY recognizable as the SAME individual from the references in face, body, AND personal style, "
        "even by people who know them personally. The underlying facial structure, body type, and style identity must remain COMPLETELY UNCHANGED.\n"
        f"PERSON B: {billgates_phrase}.\n"
        f"Scene: {scene_desc}; time/place label: {scene_label} in Seoul.\n"
        "Camera: Professional smartphone photography, ~35mm equivalent, perfect natural lighting with studio-quality shadows, "
        "flawless hand/finger anatomy, premium casual outfits appropriate for the scene. Both people should look naturally candid "
        "yet cinematically composed with magazine-quality aesthetics.\n"
        "TECHNICAL REQUIREMENTS: Ultra-high resolution details, perfect skin texture, natural color grading, professional depth of field, "
        "studio-quality lighting that enhances facial features without harsh shadows.\n"
        "No text overlays. No borders. Only one pristine image in the result."
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
      <h1>Bill Gates와 함께 in Korea — 2컷 (참조 1장 이상)</h1>
      <div class="muted">셀피 <b>최소 1장 이상</b>을 올리면, 모든 사진을 참조로 사용해 <b>정체성 일관성</b>을 극대화하여 고품질 2장을 생성합니다.</div>
      <form action="/generate" method="post" enctype="multipart/form-data">
        <div class="row">
          <label>셀피 업로드(최소 1장, 개수 제한 없음):
            <input type="file" name="selfies" accept="image/*" multiple required>
          </label>
        </div>
        <small>※ 1장만 업로드 시 얼굴과 시선 방향이 그대로 유지됩니다. 3장 이상의 다양한 각도 사진을 권장합니다.</small>
        <div class="row">
          <label><input type="checkbox" name="exact_billgates" checked>
            빌 게이츠 실존 인물로 시도 (정책/콘텐츠 이슈 시 look-alike로 전환, 단 429/쿼터는 제외)</label>
        </div>
        <div class="note">
          <b>주의/윤리</b> · 본 서비스는 AI 합성 이미지를 생성합니다. 
          사칭/허위정보 사용은 금지. 업로드 이미지는 처리 후 즉시 삭제됩니다.
        </div>
        <div class="row"><button class="btn" type="submit">2장 생성하기</button></div>
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
        if not selfies or len(selfies) < 1:
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

        for scene_label, scene_desc in SCENES:  # 2컷만
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

            # 이미지 저장 (워터마크 없음)
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
    thumbs = "".join(
        f'<div class="imgbox"><img src="{u}" style="width:100%;display:block"/></div>'
        for u in out_urls
    )
    err_html = ""
    if errors:
        err_list = "<br/>".join(errors)
        err_html = f'<div class="note" style="margin-top:16px;color:#b42318;border-color:#fecaca;background:#fff1f2"><b>일부 실패</b><br/>{err_list}</div>'

    html = f"""
    <html><head><meta charset="utf-8"><title>결과 — 2컷</title>
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
        모든 이미지는 Google Gemini가 삽입하는 <b>SynthID</b> 워터마크를 포함합니다.
      </p>
    </body></html>
    """
    return HTMLResponse(html)
