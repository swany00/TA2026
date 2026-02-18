# TA 2D Pipeline (GK-2A Clay Transfer)

GK-2A 기반 **2D 패치 TA 회귀 파이프라인**입니다.  
Clay 사전학습 인코더를 가져와 **전이학습(fine-tuning)** 으로 TA를 예측합니다.

---

## 개요 한눈에 보기

| 항목 | 내용 |
| --- | --- |
| 입력 | \(x(19, 9, 9)\) + time7 + loc + landcover\_onehot |
| 타깃 | TA z-score (정규화된 TA) |
| 학습 방식 | Clay encoder transfer fine-tuning |
| 주요 산출물 | `best.pt`, `current.pt`, 평가 메트릭 |
| 데이터 포맷 | `patch_pt` (`*_chunk_XXXXX.pt`) |

---

## 디렉터리 구조 (요약)

- `shared/`  
  - 공용 모델/IO/데이터셋 모듈 (`model_clay.py`, `dataset_patch_pt.py`, `io.py` 등)
- `step01_make_index/`  
  - 인덱스 생성 (`index_train/val/test.csv`)
- `step02_make_patches/`  
  - 패치 생성 및 `patch_pt` 청크 저장
- `step03_train_clay/`  
  - Clay 기반 TA 회귀 학습 (`weights/` 폴더에 체크포인트 저장)
- `step04_infer_map/`  
  - 900×900 영역에 대해 슬라이딩 윈도우 추론
- `step05_eval_metrics/`  
  - 예측 결과의 정량 평가/메트릭 산출

`data/`, `outputs/`, 각 step의 `outputs/`·`weights/` 등 **대용량 결과물은 Git에 포함하지 않고 로컬에서만 관리**합니다.

---

## 실행 방법 (Step-by-Step)

가상환경 및 의존성 설치는 사용 중인 환경(`clay_env`) 기준으로 맞춰 두었다고 가정합니다.

```bash
# step01: 인덱스 생성
python step01_make_index/run.py --config step01_make_index/config.yaml

# step02: 패치 생성
python step02_make_patches/run.py --config step02_make_patches/config.yaml

# step03: Clay 전이학습
python step03_train_clay/run.py --config step03_train_clay/config.yaml

# step04: 지도 추론
python step04_infer_map/run.py --config step04_infer_map/config.yaml

# step05: 메트릭 계산
python step05_eval_metrics/run.py --config step05_eval_metrics/config.yaml
```

각 단계별 `config.yaml` 내부에서 **입력/출력 경로, 스플릿(train/val/test), 모델 설정**을 조정할 수 있습니다.

---

## 파이프라인 흐름

1. **인덱스 생성 (`step01_make_index`)**
   - 원천 TA + 메타데이터를 읽어 `index_train/val/test.csv` 생성
2. **패치 생성 (`step02_make_patches`)**
   - 인덱스를 바탕으로 2D 패치 추출
   - `x(19,9,9)`, `time7`, `loc`, `landcover_onehot`, `y(TA z-score)` 구성
   - `patch_pt` 청크(`*_chunk_XXXXX.pt`)로 저장
3. **Clay 전이학습 (`step03_train_clay`)**
   - Clay encoder + 회귀 헤드로 supervised fine-tuning
   - `weights/current.pt`, `weights/best.pt` 저장
4. **지도 추론 (`step04_infer_map`)**
   - 900×900 영역에 대해 \(9×9\) 슬라이딩 윈도우 중앙값 예측
5. **평가 (`step05_eval_metrics`)**
   - RMSE (norm/K), MSE 등 메트릭 산출 및 로그 정리

---

## 데이터 & 산출물 관리

- **GitHub에는 코드/설정만 올리고, 데이터·대용량 산출물은 제외**합니다.
- `.gitignore`에서 제외되는 대표 경로:
  - `data/` – 원시/전처리/통계 파일 등
  - `outputs/` – 루트 레벨 체크포인트/로그
  - `step01_make_index/outputs/`, `step02_make_patches/outputs/`
  - `step03_train_clay/weights/`
  - `step0*/outputs/` (각 단계별 산출물)

리포지토리를 **복제한 뒤에는** 위 경로에 맞게 데이터와 산출물을 준비한 다음,  
위의 Step-by-Step 명령을 순서대로 실행하면 동일한 파이프라인을 재현할 수 있습니다.
