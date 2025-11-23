import os
import argparse
import glob
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
# [1] train.py에서 모델 생성 함수 가져오기
# ---------------------------------------------------------------------------
try:
    # train.py 파일에 정의된 build_model 함수를 가져옵니다.
    from train import build_model
except ImportError:
    print("[!] 'train.py'를 찾을 수 없거나 build_model 함수가 없습니다.")
    print("    이 파일(inference.py)을 train.py와 같은 위치에 두세요.")
    exit()

# Activation 객체 생성 헬퍼 (train.py의 ACT 로직 반영)
def get_activation(act_name):
    if act_name == 'relu': return nn.ReLU(inplace=True)
    elif act_name == 'leakyrelu': return nn.LeakyReLU(inplace=True)
    elif act_name == 'gelu': return nn.GELU()
    elif act_name == 'elu': return nn.ELU()
    else: return nn.ReLU(inplace=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Inference using Local Checkpoint")

    # --- 필수 경로 ---
    parser.add_argument('--checkpoint_path', type=str, required=True, 
                        help='학습된 모델 가중치 파일 경로 (.pth)')
    parser.add_argument('--image_path', type=str, required=True, 
                        help='추론할 이미지 파일 또는 폴더 경로')

    # --- 모델 설정 (Train 시 사용한 값과 동일해야 함) ---
    # build_model(kind=args.model_type, ...) 부분 반영
    parser.add_argument('--model_type', type=str, default='resnet', help='vgg, resnet 등')
    parser.add_argument('--feat_dim', type=int, default=128)
    parser.add_argument('--width', type=int, default=1)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.0)

    # --- 데이터 설정 ---
    # build_dataloaders(img_size=200, ...) 부분 반영
    parser.add_argument('--img_size', type=int, default=200, help='이미지 리사이즈 크기')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    return parser.parse_args()

def load_model(args):
    print(f"[*] 모델 생성 중: {args.model_type} (Activation: {args.activation})")
    
    # 1. Activation 객체 생성
    ACT = get_activation(args.activation)
    
    # 2. 모델 아키텍처 빌드 (train.py의 build_model 호출)
    model = build_model(
        kind=args.model_type,
        in_channels=3,
        feat_dim=args.feat_dim,
        width=args.width,
        activation=ACT,
        dropout=args.dropout
    ).to(args.device)

    # 3. 가중치 로드
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"체크포인트 파일 없음: {args.checkpoint_path}")
        
    print(f"[*] 가중치 로드 중: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=args.device)

    # state_dict 키 처리
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval() # 추론 모드 (Dropout 비활성화)
    return model

def preprocess_image(image_path, img_size):
    """
    단일 이미지를 학습 때와 동일한 규격(200x200)으로 전처리
    """
    image = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)), # args.img_size (200)
        transforms.ToTensor(),
        # 일반적인 ImageNet 정규화 값 (학습 코드 확인 필요)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 배치 차원 추가: (3, 200, 200) -> (1, 3, 200, 200)
    return transform(image).unsqueeze(0)

def main():
    args = parse_args()
    
    # 1. 모델 준비
    model = load_model(args)
    
    # 2. 이미지 파일 목록 준비
    if os.path.isdir(args.image_path):
        image_files = glob.glob(os.path.join(args.image_path, "*"))
        valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = [f for f in image_files if f.lower().endswith(valid_ext)]
    else:
        image_files = [args.image_path]

    print(f"[*] 총 {len(image_files)}장 추론 시작...")

    # 3. 추론 루프
    with torch.no_grad():
        for img_path in image_files:
            try:
                # 전처리
                input_tensor = preprocess_image(img_path, args.img_size).to(args.device)
                
                # 모델 예측
                output = model(input_tensor)
                
                # 결과 해석 (num_bins=91 고려)
                # Case A: 분류 (Classification) - 출력 크기가 1보다 큰 경우 (예: 91)
                probs = torch.softmax(output, dim=1)
                pred_age = torch.argmax(probs, dim=1).item()
                confidence = probs[0][pred_age].item()
                print(f"[{os.path.basename(img_path)}] 예측: {pred_age}세 (확률: {confidence:.2%})")

            except Exception as e:
                print(f"[ERROR] {os.path.basename(img_path)} 처리 실패: {e}")

if __name__ == '__main__':
    main()
