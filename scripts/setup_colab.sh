#!/bin/bash
echo "🔧 Setting up Face-Age environment..."

# 1️⃣ Google Drive 마운트
python3 - <<'PYCODE'
from google.colab import drive
drive.mount('/content/drive')
PYCODE

# 2️⃣ 코드 디렉토리 확인 (수동 clone만 1회)
cd /content
if [ ! -d "face-age" ]; then
    echo "⬇️ Cloning repo (처음 실행 시 1회만 필요)..."
    git clone https://github.com/KSW11390/face-age.git
else
    echo "📁 face-age 폴더 이미 존재 — git pull 생략"
fi

# 3️⃣ 의존성 설치
cd face-age
pip install -e .[dev] --quiet || pip install -r requirements.txt

echo "✅ Colab setup complete!"