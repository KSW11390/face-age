import os
import argparse
import glob
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

try:
    from models.model_factory import build_model
    from models.head import SoftHead
except ImportError:
    try:
        from models.model_factory import build_model
        from models.head import SoftHead
    except ImportError:
        print("[!] 모델 정의 파일을 찾을 수 없습니다. (faceage 패키지 또는 models 폴더 확인 필요)")
        exit()


def get_activation(act_name): # 받은 인자에 따라 활성화함수 리턴

    if act_name == 'relu': return nn.ReLU(inplace=True)
    elif act_name == 'leakyrelu': return nn.LeakyReLU(0.1, inplace=True)
    elif act_name == 'gelu': return nn.GELU()
    elif act_name == 'elu': return nn.ELU()
    else: return nn.ReLU(inplace=True)

def parse_args():   # 모델 파싱 함수
    parser = argparse.ArgumentParser(description="Inference using Trained Checkpoint")

    parser.add_argument('--checkpoint_path', type=str, required=True, 
                        help='학습된 모델 가중치 파일 경로 (.pt)')
    parser.add_argument('--image_path', type=str, required=True, 
                        help='추론할 이미지 파일 또는 폴더 경로')

    parser.add_argument('--model_type', type=str, default='resnet', choices=["vgg", "resnet"])
    parser.add_argument('--feat_dim', type=int, default=128)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--dropout', type=float, default=0.0)
    
    parser.add_argument("--use_race_onehot", action="store_true", default=False)

    parser.add_argument('--img_size', type=int, default=200)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    return parser.parse_args()

def load_model_and_head(args):
    print(f"[*] 모델 생성 중: {args.model_type} (Width: {args.width}, Act: {args.activation})")
    print(f"[*] Head 생성 중: SoftHead (Feat: {args.feat_dim}, UseRace: {args.use_race_onehot})")
    
    # Backbone 생성
    ACT = get_activation(args.activation)
    model = build_model(
        kind=args.model_type,
        in_channels=3,
        feat_dim=args.feat_dim,
        width=args.width,
        activation=ACT,
        dropout=args.dropout
    ).to(args.device)

    # Head 생성
    head = SoftHead(
        args.feat_dim,   
        num_bins=91, 
        use_race=args.use_race_onehot
    ).to(args.device)

    # 가중치 로드
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"체크포인트 파일 없음: {args.checkpoint_path}")
        
    print(f"[*] 가중치 파일 로드: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=args.device)

    # Backbone 가중치 로드
    if "model" in checkpoint:
        model_state = checkpoint["model"]
        try:
            model.load_state_dict(model_state)
        except RuntimeError:
            new_state = {k.replace("module.", ""): v for k, v in model_state.items()}
            model.load_state_dict(new_state)
    else:
        print("[!] 경고: 체크포인트에 'model' 키가 없습니다.")

    # Head 가중치 로드
    if "head" in checkpoint:
        head_state = checkpoint["head"]
        try:
            head.load_state_dict(head_state)
        except RuntimeError:
            new_state = {k.replace("module.", ""): v for k, v in head_state.items()}
            head.load_state_dict(new_state)
    else:
        print("[!] 경고: 체크포인트에 'head' 키가 없습니다.")

    model.eval()
    head.eval()
    
    return model, head

def preprocess_image(image_path, img_size):    # 테스트 이미지를 전처리
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"[!] 이미지 읽기 실패: {image_path}")
        return None
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(image).unsqueeze(0)

def main():
    args = parse_args()
    
    # 모델과 헤드 로드
    model, head = load_model_and_head(args)
    
    # 이미지 파일 준비
    if os.path.isdir(args.image_path):
        image_files = glob.glob(os.path.join(args.image_path, "*"))
        valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        image_files = [f for f in image_files if f.lower().endswith(valid_ext)]
    else:
        image_files = [args.image_path]

    print(f"[*] 총 {len(image_files)}장 추론 시작...")
    print("-" * 50)
    
    # 나이 계산용 Bins (0~90)
    num_bins = 91
    bins = torch.arange(num_bins, dtype=torch.float32).to(args.device)

    # 추론 루프
    with torch.no_grad():
        for img_path in image_files:
            # 전처리
            input_tensor = preprocess_image(img_path, args.img_size)
            if input_tensor is None: continue
            
            input_tensor = input_tensor.to(args.device)
            
            # [Forward Pass]
            # 1. Backbone: 이미지 -> 특징 벡터 (1, 128)
            feats = model(input_tensor)
            
            # 2. Head: 특징 벡터 -> 나이 로짓 (1, 91)
            # 추론 시 인종(race) 정보는 없으므로 None 전달 (SoftHead 내부 처리 의존)
            if args.use_race_onehot:
                # 만약 학습때 Race를 썼다면, 추론때도 넣어줘야 Dimension 에러가 안남.
                # 임시로 '0'번 인종 등으로 채우거나, None 처리가 되는지 SoftHead 확인 필요.
                # 여기서는 None으로 시도.
                logits = head(feats, None) 
            else:
                logits = head(feats, None)

            # [결과 해석]
            probs = torch.softmax(logits, dim=1)
            
            # 1) Max Probability (가장 높은 확률의 나이)
            pred_age_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_age_idx].item()
            
            # 2) Expectation (기댓값, 가중 평균 나이) - train.py의 MAE 계산 방식과 동일
            expected_age = (probs * bins).sum(dim=1).item()
            
            print(f"[{os.path.basename(img_path)}] "
                  f"예측(Max): {pred_age_idx}세 | ")

if __name__ == '__main__':
    main()

