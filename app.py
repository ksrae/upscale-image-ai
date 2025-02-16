import sys
import os
import types
import torchvision.transforms.functional as F

# torchvision 패치
if "torchvision.transforms.functional_tensor" not in sys.modules:
    functional_tensor = types.ModuleType("torchvision.transforms.functional_tensor")
    functional_tensor.rgb_to_grayscale = F.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = functional_tensor

import torch
from flask import Flask, request, jsonify, render_template
from PIL import Image, ImageDraw, ImageFont
import pytesseract
import numpy as np
import io
import base64

# 커스텀 RRDBNet 임포트
from custom_rrdbnet import RRDBNet

# Tesseract 경로 설정
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'



# Flask 애플리케이션 설정
app = Flask(__name__)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 모델 파일 경로
model_path = os.path.join(os.path.dirname(__file__), "RealESRGAN_x4plus.pth")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"모델 파일 '{model_path}'을(를) 찾을 수 없습니다.")

class RealESRGANer:
    def __init__(self, scale, model_path, device, half=False):
        self.scale = scale
        self.device = device
        self.half = half
        
        # RRDBNet 모델 초기화
        self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
        
        # 체크포인트 로드
        checkpoint = torch.load(model_path, map_location=device)
        if 'params_ema' in checkpoint:
            state_dict = checkpoint['params_ema']
        else:
            state_dict = checkpoint
            
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.model.to(device)
        
        if half:
            self.model = self.model.half()

    def enhance(self, img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
            
        # PIL Image를 텐서로 변환
        img_tensor = torch.from_numpy(np.array(img)).float()
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        img_tensor = img_tensor / 255.0
        img_tensor = img_tensor.to(self.device)
        
        if self.half:
            img_tensor = img_tensor.half()

        with torch.no_grad():
            output = self.model(img_tensor)
            output = output.squeeze(0).permute(1, 2, 0)  # [H, W, 3]
            output = output.clamp_(0, 1)
            output = (output * 255.0).cpu().numpy().astype(np.uint8)
            output = Image.fromarray(output)

        return output, None

# 모델 초기화
model = RealESRGANer(scale=4, model_path=model_path, device=device, half=False)

@app.route('/')
def index():
    return render_template('index.html')

import os

# 앱 시작 시 upscales 폴더 생성
UPLOAD_FOLDER = 'upscales'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    print("'upscales' 폴더 생성됨")

@app.route('/process', methods=['POST'])
def process_image():
    print("============ 이미지 처리 시작 ============")
    
    if 'file' not in request.files:
        print("오류: 파일이 업로드되지 않음")
        return jsonify({'error': '파일이 업로드되지 않았습니다.'}), 400
    
    file = request.files['file']
    if file.filename == '':
        print("오류: 선택된 파일 없음")
        return jsonify({'error': '선택된 파일이 없습니다.'}), 400

    print(f"처리할 파일: {file.filename}")

    try:
        print("1. 이미지 로딩 시작...")
        img = Image.open(file.stream).convert('RGB')
        print(f"   - 이미지 크기: {img.size}")
        print("   - 이미지 로딩 완료")
    except Exception as e:
        print(f"오류: 이미지 로딩 실패 - {str(e)}")
        return jsonify({'error': '유효하지 않은 이미지 파일입니다.'}), 400

    try:
        print("2. OCR 처리 시작...")
        # 한글과 영어 모두 인식하도록 'kor+eng' 설정
        ocr_text = pytesseract.image_to_string(img, lang='kor+eng')
        print(f"   - 추출된 텍스트: {ocr_text[:100]}..." if len(ocr_text) > 100 else f"   - 추출된 텍스트: {ocr_text}")
        print("   - OCR 처리 완료")
    except Exception as e:
        print(f"오류: OCR 처리 실패 - {str(e)}")
        ocr_text = "OCR 처리 실패"

    try:
        print("3. 이미지 업스케일링 시작...")
        upscaled_img, _ = model.enhance(img)
        print(f"   - 업스케일링 후 크기: {upscaled_img.size}")
        print("   - 업스케일링 완료")
    except Exception as e:
        print(f"오류: 업스케일링 실패 - {str(e)}")
        return jsonify({'error': '이미지 업스케일링 실패'}), 500

    try:
        print("4. OCR 텍스트 오버레이 시작...")
        draw = ImageDraw.Draw(upscaled_img)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
            print("   - Arial 폰트 로드 성공")
        except Exception as e:
            print(f"   - Arial 폰트 로드 실패, 기본 폰트 사용 - {str(e)}")
            font = ImageFont.load_default()
        
        text_position = (10, upscaled_img.height - 100)
        draw.text(text_position, ocr_text, fill=(255, 255, 255), font=font)
        print("   - 텍스트 오버레이 완료")
    except Exception as e:
        print(f"오류: 텍스트 오버레이 실패 - {str(e)}")

    # 업스케일된 이미지 저장
    try:
        print("5. 업스케일된 이미지 저장 시작...")
        # 파일명만 추출 (경로 제거)
        filename = os.path.basename(file.filename)
        # 저장할 파일 경로 생성 (os.path.join 사용)
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        
        # 필요한 경우 하위 디렉토리 생성
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 이미지 저장
        upscaled_img.save(save_path)
        print(f"   - 저장 완료: {save_path}")
    except Exception as e:
        print(f"오류: 이미지 저장 실패 - {str(e)}")
        return jsonify({'error': f'이미지 저장 실패: {str(e)}'}), 500

    try:
        print("6. 썸네일 생성 시작...")
        thumb_img = upscaled_img.copy()
        thumb_img.thumbnail((200, 200))
        print(f"   - 썸네일 크기: {thumb_img.size}")
        
        buffered = io.BytesIO()
        thumb_img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        print("   - 썸네일 생성 및 인코딩 완료")
    except Exception as e:
        print(f"오류: 썸네일 생성 실패 - {str(e)}")
        return jsonify({'error': '썸네일 생성 실패'}), 500

    print("============ 이미지 처리 완료 ============")
    return jsonify({'thumbnail': img_str})

if __name__ == '__main__':
    print("Flask 서버 시작...")
    print(f"실행 환경: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    try:
        print("Tesseract 버전 확인:", pytesseract.get_tesseract_version())
        # 사용 가능한 언어 목록 확인
        langs = pytesseract.get_languages()
        print("사용 가능한 OCR 언어:", langs)
        if 'kor' in langs:
            print("한글 OCR 사용 가능")
        else:
            print("경고: 한글 OCR을 위한 언어 패턴이 설치되지 않았습니다.")
    except Exception as e:
        print("Tesseract 설치 확인 필요:", str(e))
    print(f"업스케일된 이미지 저장 경로: {os.path.abspath('upscales')}")
    app.run(debug=True)