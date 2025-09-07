# Bill Gates + You in Korea - AI Photo Generator

빌 게이츠와 함께 한국 명소에서 찍은 듯한 사진을 AI로 생성하는 FastAPI 애플리케이션입니다.

## 기능

- 사용자 셀피 사진(최대 2장)을 업로드하면 빌 게이츠와 함께 한국 명소에서 찍은 듯한 사진 2장을 생성
- Google Gemini 2.5 Flash Image 모델 사용
- 정체성 일관성을 위한 듀얼 레퍼런스 시스템
- 빠른 실패 및 레이트 리밋 처리
- AI 생성 워터마크 자동 추가

## 생성 장면

1. **08:00 경복궁 근정전 앞** - 아침 햇살이 비치는 전통 궁궐
2. **19:00 N서울타워 전망대** - 서울 야경이 보이는 전망대

## 설치 및 실행

### 1. 저장소 클론

```bash
git clone https://github.com/your-username/billgates-korea-ai.git
cd billgates-korea-ai
```

### 2. 가상환경 설정

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

### 4. 환경변수 설정

`.env` 파일을 생성하고 Google API 키를 설정합니다:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

Google AI Studio에서 API 키를 발급받을 수 있습니다: https://makersuite.google.com/app/apikey

### 5. 애플리케이션 실행

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

브라우저에서 http://localhost:8000 으로 접속하면 사용할 수 있습니다.

## 사용 방법

1. 웹 인터페이스에서 본인의 셀피 사진을 최대 2장 업로드
2. "빌 게이츠 실존 인물로 시도" 옵션 선택 (선택사항)
3. "2장 생성하기" 버튼 클릭
4. AI가 생성한 사진 2장 확인

## 주요 특징

- **듀얼 레퍼런스**: 업로드한 2장의 사진을 모두 참조로 사용하여 정체성 일관성 향상
- **빠른 실패**: 긴 대기시간 방지를 위한 타임아웃 및 재시도 제한
- **레이트 리밋 처리**: API 호출 간격 조절 및 쿼터 에러 처리
- **자동 워터마크**: 모든 생성 이미지에 AI-Generated 표시 추가
- **즉시 삭제**: 업로드된 원본 이미지는 처리 후 즉시 삭제

## 기술 스택

- **Backend**: FastAPI, Python 3.11+
- **AI Model**: Google Gemini 2.5 Flash Image
- **Image Processing**: Pillow (PIL)
- **Frontend**: HTML, CSS (Vanilla)

## 라이선스

MIT License

## 주의사항

- 본 서비스는 AI 합성 이미지를 생성하며, 모든 생성물에는 AI-Generated 표시가 추가됩니다
- 사칭이나 허위정보 생성 목적으로 사용을 금지합니다
- 업로드된 이미지는 처리 후 즉시 삭제됩니다
- Google API 사용량에 따른 비용이 발생할 수 있습니다
