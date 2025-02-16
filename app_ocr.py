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
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import pytesseract
import numpy as np
import io
import base64
import cv2  # OpenCV 임포트

# 커스텀 RRDBNet 임포트
class ResidualDenseBlock(torch.nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = torch.nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = torch.nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = torch.nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(torch.nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)
        
    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x

class RRDBNet(torch.nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4):
        super(RRDBNet, self).__init__()
        self.scale = scale
        self.conv_first = torch.nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = torch.nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = torch.nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(feat))
        feat = self.upsample(feat)
        feat = self.lrelu(self.conv_up2(feat))
        feat = self.upsample(feat)
        feat = self.lrelu(self.conv_hr(feat))
        out = self.conv_last(feat)
        return out

# Tesseract 경로 설정 (Windows 예시)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Flask 애플리케이션 설정
app = Flask(__name__)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 모델 파일 경로
model_path = os.path.join(os.path.dirname(__file__), "RealESRGAN_x4plus.pth")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"모델 파일 '{model_path}'을(를) 찾을 수 없습니다.")

# 업스케일된 이미지를 저장할 폴더
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "upscales")
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    print("'upscales' 폴더 생성됨")

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
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor / 255.0
        img_tensor = img_tensor.to(self.device)
        
        if self.half:
            img_tensor = img_tensor.half()

        with torch.no_grad():
            output = self.model(img_tensor)
            output = output.squeeze(0).permute(1, 2, 0)
            output = output.clamp(0, 1)
            output = (output * 255.0).cpu().numpy().astype(np.uint8)
            output = Image.fromarray(output)

        return output, None

# 모델 초기화
model = RealESRGANer(scale=4, model_path=model_path, device=device, half=False)

def adaptive_threshold(img, window_size, c=2):
    height, width = img.shape
    result = np.zeros_like(img)
    half_window = window_size // 2
    padded_img = np.pad(img, half_window, mode='edge')
    
    for i in range(height):
        for j in range(width):
            window = padded_img[i:i+window_size, j:j+window_size]
            threshold = np.mean(window) - c
            result[i, j] = 255 if img[i, j] > threshold else 0
    
    return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/full/<filename>')
def full_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

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

    try:
        print("1. 원본 이미지 로딩 시작...")
        original_img = Image.open(file.stream).convert('RGB')
        print(f"   - 원본 이미지 크기: {original_img.size}")
    except Exception as e:
        print(f"오류: 이미지 로딩 실패 - {str(e)}")
        return jsonify({'error': '유효하지 않은 이미지 파일입니다.'}), 400

    try:
        print("2. 향상된 OCR 전처리 및 텍스트 추출 시작...")
        # OpenCV로 전처리 진행
        ocr_cv = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2GRAY)
        # 노이즈 제거를 위한 GaussianBlur 적용
        ocr_cv = cv2.GaussianBlur(ocr_cv, (5, 5), 0)
        # Otsu 이진화로 최적 임계값 적용
        _, ocr_cv = cv2.threshold(ocr_cv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 모폴로지 연산으로 문자 선명화
        kernel = np.ones((3, 3), np.uint8)
        ocr_cv = cv2.morphologyEx(ocr_cv, cv2.MORPH_OPEN, kernel)
        ocr_cv = cv2.morphologyEx(ocr_cv, cv2.MORPH_CLOSE, kernel)
        # PIL 이미지로 복원
        ocr_img = Image.fromarray(ocr_cv)
        # OCR 수행 (dpi 옵션 포함)
        custom_config = r'--oem 3 --psm 6 --dpi 300 -c preserve_interword_spaces=1'
        ocr_data = pytesseract.image_to_data(ocr_img, lang='kor+eng', config=custom_config, output_type=pytesseract.Output.DICT)
        print("   - OCR 처리 완료")
    except Exception as e:
        print(f"오류: OCR 전처리/처리 실패 - {str(e)}")
        ocr_data = None

    try:
        print("3. 이미지 업스케일링 시작...")
        upscaled_img, _ = model.enhance(original_img)
        print(f"   - 업스케일링 후 크기: {upscaled_img.size}")
    except Exception as e:
        print(f"오류: 업스케일링 실패 - {str(e)}")
        return jsonify({'error': '이미지 업스케일링 실패'}), 500

    try:
        print("4. OCR 텍스트 오버레이 시작...")
        draw = ImageDraw.Draw(upscaled_img)
        # 고화질 텍스트 출력을 위해 큰 폰트 사용 (scale factor에 따라 조정 가능)
        try:
            font = ImageFont.truetype("malgun.ttf", 24)
            print("   - 맑은 고딕 폰트 로드 성공")
        except Exception as e:
            try:
                font = ImageFont.truetype("NanumGothic.ttf", 24)
                print("   - 나눔고딕 폰트 로드 성공")
            except Exception as e:
                font = ImageFont.load_default()
                print("   - 기본 폰트 사용")
        
        # 원본과 업스케일 이미지 사이의 scale factor 계산
        scale_factor_w = upscaled_img.width / original_img.width
        scale_factor_h = upscaled_img.height / original_img.height
        
        min_confidence = 30
        if ocr_data:
            for i in range(len(ocr_data['text'])):
                text = ocr_data['text'][i].strip()
                try:
                    conf = float(ocr_data['conf'][i])
                except:
                    conf = 0
                if text and conf > min_confidence:
                    # OCR의 좌표를 업스케일 이미지 좌표로 변환
                    x = int(ocr_data['left'][i] * scale_factor_w)
                    y = int(ocr_data['top'][i] * scale_factor_h)
                    # 고화질 텍스트 오버레이: 텍스트 외곽에 약간의 그림자 효과를 줘서 가독성 향상
                    shadow_color = "black"
                    text_color = "white"
                    # 그림자 효과 (오프셋 적용)
                    offset = 1
                    draw.text((x+offset, y+offset), text, font=font, fill=shadow_color)
                    draw.text((x, y), text, font=font, fill=text_color)
        print("   - 텍스트 오버레이 완료")
    except Exception as e:
        print(f"오류: 텍스트 오버레이 실패 - {str(e)}")

    try:
        print("5. 업스케일된 이미지 저장 시작...")
        filename = os.path.basename(file.filename)
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        upscaled_img.save(save_path)
        print(f"   - 저장 완료: {save_path}")
    except Exception as e:
        print(f"오류: 이미지 저장 실패 - {str(e)}")
        return jsonify({'error': f'이미지 저장 실패: {str(e)}'}), 500

    try:
        print("6. 썸네일 생성 시작...")
        thumb_img = upscaled_img.copy()
        thumb_img.thumbnail((200, 200))
        buffered = io.BytesIO()
        thumb_img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        print("   - 썸네일 생성 및 인코딩 완료")
    except Exception as e:
        print(f"오류: 썸네일 생성 실패 - {str(e)}")
        return jsonify({'error': '썸네일 생성 실패'}), 500

    print("============ 이미지 처리 완료 ============")
    return jsonify({'thumbnail': img_str, 'filename': filename})


if __name__ == '__main__':
    print("Flask 서버 시작...")
    print(f"실행 환경: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    try:
        print("Tesseract 버전 확인:", pytesseract.get_tesseract_version())
        langs = pytesseract.get_languages()
        print("사용 가능한 OCR 언어:", langs)
        if 'kor' in langs:
            print("한글 OCR 사용 가능")
        else:
            print("경고: 한글 OCR을 위한 언어 패턴이 설치되지 않았습니다.")
    except Exception as e:
        print("Tesseract 설치 확인 필요:", str(e))
    print(f"업스케일된 이미지 저장 경로: {os.path.abspath(UPLOAD_FOLDER)}")
    app.run(debug=True)
