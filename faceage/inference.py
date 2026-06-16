import os
import argparse
import glob
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from models.model_factory import build_model
from models.head import SoftHead



def get_activation(act_name):  # 활성화 함수 리턴
    if act_name == 'relu':
        return nn.ReLU(inplace=True)
    elif act_name == 'leakyrelu':
        return nn.LeakyReLU(0.1, inplace=True)
    elif act_name == 'gelu':
        return nn.GELU()
    elif act_name == 'elu':
        return nn.ELU()
    else:
        return nn.ReLU(inplace=True)

def parse_args():  # 파싱 코드
    parser = argparse.ArgumentParser(description="Inference using Trained Checkpoint")

    parser.add_argument('--checkpoint_path', type=str, required=True, help='학습된 모델 가중치 파일 경로')
    parser.add_argument('--image_path', type=str, required=True, help='추론할 이미지 파일 또는 폴더 경로')
    parser.add_argument('--model_type', type=str, default='resnet', choices=["vgg", "resnet"])
    parser.add_argument('--feat_dim', type=int, default=128)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument("--use_race_onehot", action="store_true", default=False)
    parser.add_argument('--img_size', type=int, default=200)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    return parser.parse_args()
 
def load_model_and_head(args):  # 모델과 헤드를 로드
    print(f"[*] 모델 생성 중: {args.model_type} (Width: {args.width}, Act: {args.activation})")
    
    ACT = get_activation(args.activation)
    model = build_model(kind=args.model_type, in_channels=3, feat_dim=args.feat_dim, width=args.width, activation=ACT, dropout=args.dropout).to(args.device)
    head = SoftHead(args.feat_dim, num_bins=91, use_race=args.use_race_onehot).to(args.device)

    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"체크포인트 파일 없음: {args.checkpoint_path}")
        
    print(f"[*] 가중치 파일 로드: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=args.device)

    if "model" in checkpoint:
        model_state = checkpoint["model"]
        model.load_state_dict(model_state)

    if "head" in checkpoint:
        head_state = checkpoint["head"]
        head.load_state_dict(head_state)

    model.eval()
    head.eval()
    return model, head


def preprocess_image(image_path):  # 예측할 이미지 전처리
    image = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    return transform(image).unsqueeze(0)


def main():     # 모델과 헤드 로드 -> 정방향 패스 -> 추론 -> 결과출력
    args = parse_args()
    model, head = load_model_and_head(args)
    
    if os.path.isdir(args.image_path):
        image_files = glob.glob(os.path.join(args.image_path, "*"))
        valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        image_files = [f for f in image_files if f.lower().endswith(valid_ext)]
    else:
        image_files = [args.image_path]

    print(f"[*] 총 {len(image_files)}장 추론 시작...")
    
    num_bins = 91
    bins = torch.arange(num_bins, dtype=torch.float32).to(args.device)

    with torch.no_grad():
        for img_path in image_files:
            input_tensor = preprocess_image(img_path)
            if input_tensor is None:
                continue
            
            input_tensor = input_tensor.to(args.device)
            feats = model(input_tensor)
            
            logits = head(feats, None) 

            probs = torch.softmax(logits, dim=1)
            expected_age = (probs * bins).sum(dim=1).item()
            
    
            original_img = Image.open(img_path).convert("RGB")
                
            plt.figure(figsize=(12, 5))
                
            plt.subplot(1, 2, 1)
            plt.imshow(original_img)
            plt.axis('off')
            plt.title("Input Image", fontsize=14)
                
            plt.subplot(1, 2, 2)
            plt.axis('off')
                
            info_text = (
                f"Prediction : {expected_age:.1f} years\n"
            )
            plt.text(0.1, 0.5, info_text, fontsize=20, va='center', ha='left')
            plt.title("Inference Result", fontsize=14)
            
            # save_name = f"result_{os.path.basename(img_path)}"
            # plt.savefig(save_name)
            
            plt.tight_layout()
            plt.show()
            plt.close() # 메모리 누수 방지를 위해 닫기


if __name__ == '__main__':
    main()

