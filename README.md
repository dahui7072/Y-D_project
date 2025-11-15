# Y:D_project
# 1. 프로젝트 개요

본 레포지토리는 주최 측이 제공한 Baseline 모델 구조를 기반으로 하여,

이미지기반 Visual Grounding Task를 수행하는 모델을 학습하고 추론할 수 있도록 구성되어 있습니다.

본 README는 다음 내용을 **완전 재현(Full Reproducibility)** 가능하도록 설명합니다:

- **모델 학습 방법 (Train)**
- **모델 추론 방법 (Inference)**
- **학습 및 추론 환경 명시**
- **데이터 구성 및 전처리(create_dataset.py / reorganize_dataset.py)**

---

# 2. 개발 및 실행 환경(Environment)

아래 환경에서 학습 및 추론을 검증하였습니다.

| 항목 | 사용 환경 |
| --- | --- |
| GPU | **NVIDIA RTX 3090 (24GB VRAM)** 1장 |
| CUDA | 11.8 |
| Python | 3.9+ |
| Framework | PyTorch 2.1 |
| 기타 | tqdm, numpy, opencv, regex 등 |

---

# 3. 데이터 구성

대회 Train 데이터 기준 아래 구조를 따름

```
dataset/train/json/
dataset/train/jpg/

dataset/val/json/
dataset/val/jpg/
```

**① create_dataset.py**

→ JSON/이미지 경로 매핑 및 학습용 샘플(DataSet) 생성

**② reorganize_dataset.py**

→ 폴더 구조 재정렬 및 파일 일관성 유지

---

# 4. 모델 구조

모델은 Cross-Attention 기반의 Visual-Language Model(`CrossAttnVLM`)이며:

- Text Token + Image Feature → Cross Attention
- Bounding Box (x, y, w, h) 예측
- Smooth L1 Loss 기반 회귀 학습

구조는 `model.py`에 정의되어 있습니다.

---

# 5. 학습 방법 (Training)

학습 실행 파일: **train.py**

학습 시 내부적으로:

- `create_dataset.py` → 데이터셋 생성
- vocab 자동 구축
- 이미지/텍스트 토큰화
- CrossAttnVLM 모델 초기화
- Smooth L1 Loss 기반 훈련

### **학습 실행 명령어**

```bash
python train.py \
  --json_dir ./dataset/train/json \
  --jpg_dir ./dataset/train/jpg \
  --batch_size 16 \
  --img_size 640 \
  --epochs 2 \
  --lr 1e-4 \
  --num_workers 4 \
  --save_ckpt ./ckpt/baseline.pth

```

### 주요 인자 설명

| 인자 | 설명 |
| --- | --- |
| --json_dir | train JSON 폴더 |
| --jpg_dir | 이미지(.jpg) 폴더 |
| --img_size | 모델 입력 이미지 크기 |
| --batch_size | 배치 크기 |
| --epochs | epoch 수 |
| --lr | learning rate |
| --save_ckpt | 모델 저장 경로 |

학습이 완료되면:

```
ckpt/baseline.pth

```

파일이 생성됨

---

# 6. 추론 방법 (Inference)

추론 실행 파일: **test.py**

### **추론 실행 명령어**

```bash
python test.py \
  --json_dir ./dataset/test/json \
  --jpg_dir ./dataset/test/jpg \
  --checkpoint ./ckpt/baseline.pth \
  --save_path ./submission.csv

```

### 추론 결과

아래 형태의 CSV 파일로 저장:

| query_id | query_text | pred_x | pred_y | pred_w | pred_h |
| --- | --- | --- | --- | --- | --- |
