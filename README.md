# TA 2D Pipeline (Step-by-Step)

데이터(`data`)는 그대로 두고, 단계별로 직접 실행하는 구조입니다.

## Step Folders
- `step01_make_index/`
- `step02_make_patches/`
- `step03_train_clay/`
- `step04_infer_map/`
- `step05_eval_metrics/`
- `shared/`

## Run
```bash
python step01_make_index/run.py --config step01_make_index/config.yaml
python step02_make_patches/run.py --config step02_make_patches/config.yaml
python step03_train_clay/run.py --config step03_train_clay/config.yaml
python step04_infer_map/run.py --config step04_infer_map/config.yaml
python step05_eval_metrics/run.py --config step05_eval_metrics/config.yaml
```

## Notes
- 현재 `step01/step02`는 기존 산출물 재사용 모드입니다.
- 핵심 학습은 `step03_train_clay`에서 수행합니다.
- 최종 추론은 `900x900`에 대해 `9x9` 슬라이딩 중앙값 예측입니다.

## GitHub 업로드 시 제외 항목
대용량 데이터·산출물은 `.gitignore`로 제외됩니다.
- `data/` — 원시/전처리 데이터
- `outputs/` — 루트 학습 체크포인트 등
- `step01_make_index/outputs/`, `step02_make_patches/outputs/` — 인덱스/패치 산출물
- `step03_train_clay/weights/` — 학습된 모델 가중치
- `step0*/outputs/` — 각 단계 산출물
복제 후 위 경로에 데이터·산출물을 준비한 뒤 파이프라인을 실행하면 됩니다.
