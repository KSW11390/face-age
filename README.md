# Face Age Estimation (UTKFace)

이 프로젝트는 **UTKFace** 얼굴 이미지 데이터셋을 이용해  
입력 이미지로부터 사람의 나이를 예측하는 **Face Age Estimation** 모델을 학습·추론하는 코드입니다.

dataset : https://www.kaggle.com/datasets/moritzm00/utkface-cropped

PyTorch 기반으로 구현되어 있으며,  
`ResNet` / `VGG` 백본과 **soft label + KLD loss**를 사용해 나이 분포를 회귀 형태로 학습합니다.


---

## 1. 환경 설정

```bash
git clone https://github.com/KSW11390/face-age.git
cd face-age

# (선택) 가상환경 생성 후 패키지 설치
pip install -r requirements.txt
# Face Age Estimation (UTKFace)

이 프로젝트는 **UTKFace** 얼굴 이미지 데이터셋을 이용해  
입력 이미지로부터 사람의 나이를 예측하는 **Face Age Estimation** 모델을 학습·추론하는 코드입니다.

PyTorch 기반으로 구현되어 있으며,  
`ResNet` / `VGG` 백본과 **soft label + KLD loss**를 사용해 나이 분포를 회귀 형태로 학습합니다.

---

## 1. 환경 설정

```bash
git clone https://github.com/KSW11390/face-age.git
cd face-age

pip install -r requirements.txt
```

---

## 2. 데이터셋 준비 (UTKFace)

UTKFace 데이터셋을 아래 경로에 압축 해제했다고 가정합니다.

```
/content/UTKFace/UTKFace
```

Colab 예시:

```bash
!cp "/content/drive/MyDrive/MLProject/Dataset/UTKFace.zip" "/content/"
!unzip -q /content/UTKFace.zip -d /content/UTKFace
```

---

## 3. 학습 실행 예시

```bash
!python -m faceage.train \
  --data_root="/content/UTKFace/UTKFace" \
  --epochs=10 \
  --batch_size=32 \
  --lr=1e-4 \
  --model_type=resnet \
  --activation=gelu \
  --dropout=0.0 \
  --optimizer=adamw \
  --weight_decay=1e-4 \
  --width=32 \
  --feat_dim=128 \
  --patience=30 \
  --wandb_project=face_age_augmentation \
  --save_dir="/content/drive/MyDrive/MLProject/checkpoints" \
  --label_type=soft \
  --loss_fn=kld \
  --sigma=2.0 \
  --aug_strength=medium \
  --aug_dup=1 \
  --random_erase 0.2
```

---

## 4. 주요 옵션 설명

### 📌 경로 설정

- `--data_root="/content/UTKFace/UTKFace"`  
  학습에 사용할 **UTKFace 데이터셋 경로**

- `--save_dir="..."`  
  체크포인트(`.pt`) 저장 디렉토리

- `--wandb_project`  
  W&B 로깅용 프로젝트 이름

---

### 📌 모델 / 하이퍼파라미터

- `--model_type resnet` : 백본 선택 (`resnet`, `vgg`)
- `--activation gelu` : 활성화 함수 (["relu", "leakyrelu", "gelu", "elu"])
- `--feat_dim 128` : feature embedding dimension
- `--width 32` : 채널 width scaling
- `--dropout 0.0` : dropout 비율

---

### 📌 학습 설정

- `--epochs 10` : Epoch 수  
- `--batch_size 32` : 배치 크기  
- `--lr 1e-4` : Learning rate  
- `--optimizer adamw` : Optimizer (["adam", "adamw", "sgd"])
- `--weight_decay 1e-4` : L2 regularization  
- `--patience 30` : Early stopping patience  

---

### 📌 Label / Loss 설정

- `--label_type soft` : Gaussian soft label  ("soft", "hard")
- `--loss_fn kld` : KL Divergence 기반 soft label 학습 ("kld", "ce")
- `--sigma 2.0` : soft label 분포 폭  

---

### 📌 Data Augmentation

- `--aug_strength medium` : 증강 강도  {weak, medium, strong}
- `--aug_dup N` : 이미지 N배 증강  
- `--use_random_erase 0.2` : Random Erasing 사용 여부, 확률

---

## 5. 추론(Inference)

### 5-1. Notebook (`test.ipynb`)에서 추론

`test.ipynb`를 실행하면  
- checkpoint 로드  
- 이미지 테스트  
- **입력 이미지 + 예측 나이 시각화**  
를 쉽게 확인할 수 있습니다.

---

### 5-2. 스크립트로 추론 (`inference.py`)

```bash
!python /content/face-age/faceage/inference.py \
    --checkpoint_path "/content/face-age/best_model.pt" \
    --image_path "/content/face-age/faceage/test_images" \
    --width 32 \
    --activation gelu \
    --model_type resnet \
    --feat_dim 128 \
    --img_size 200
```

---


---

## 6. 기타

- 실험 비교는 W&B 대시보드를 통해 확인할 수 있습니다.
- Soft label + KLD 조합은 단일 정답이 아닌 확률분포를 학습하므로  
  더 안정적인 Age Estimation이 가능합니다.