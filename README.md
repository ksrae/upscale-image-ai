# 1. 프로젝트 디렉터리 생성 및 가상환경 설정
프로젝트 디렉터리 생성 및 이동
원하는 위치에서 새 디렉터리를 만듭니다.

```
mkdir upscale-image-ai
cd upscale-image-ai

```

## pipenv로 가상환경 생성
Python 3.12를 사용하여 가상환경을 생성합니다.


### pipenv가 설치되어 있지 않다면 설치
```
pip install pipenv
```

### 가상환경 설정
```
pipenv --python 3.12
pipenv shell
```

위 명령을 실행하면 Pipfile이 생성되고, 가상환경이 활성화됩니다.

# 2. 필수 패키지 설치
프로젝트에 필요한 패키지를 설치합니다. 터미널에서 아래 명령어를 실행하세요:

```
pipenv install flask pillow numpy pytesseract torch torchvision realesrgan
pip install opencv-python
```

## 주의:

### Tesseract OCR:
pytesseract는 실제 Tesseract 프로그램이 필요합니다.
Windows: Tesseract 설치 가이드를 참고하여 설치하세요.
Real‑ESRGAN 모델 파일:
업스케일링에 사용할 모델 파일(RealESRGAN_x4plus.pth)을 공식 GitHub 또는 Hugging Face 페이지에서 다운로드 받아 프로젝트 루트에 배치하세요.


### RealESRGAN_x4plus.pth
https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth


## tesseract 설치
https://github.com/UB-Mannheim/tesseract/wiki


# 3. 실행

```
python app.py
```

http://localhost:5000에 접속


# Version
## 2.0
스타일 수정,
하위 폴더까지 선택

### 개선 필요
비동기 처리 필요
여전히 OCR 인식하고 적용하는 부분 개선은 필요함