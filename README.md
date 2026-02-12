# TA_2D_VER1

## Structure

```text
TA_2D_VER1/
  main.py                    # 단일 진입점: stage orchestration
  configs/
    data_2d.yaml             # 데이터 경로/split/패치/채널/정규화 파일 경로
    train_ta.yaml            # 학습 하이퍼파라미터/체크포인트 설정
  src/
    data/
      build_index.py         # 라벨 기준 인덱스 CSV 생성 (streaming append)
      build_shards.py        # 인덱스로 2D 패치 shard NPY 생성
      dataset.py             # shard 로더
      transforms.py          # 정규화/one-hot 변환
      validators.py          # NaN/범위 검증
    models/
      clay_ta_head.py        # TA supervised 모델 헤드
      losses.py              # 손실 함수
    train/
      trainer.py             # 학습 루프 엔트리
      loops.py               # train/val step
      checkpoint.py          # checkpoint save/load
    utils/
      io.py                  # yaml/json/경로 유틸
      time_features.py       # UTC 시간 피처 인코딩
  outputs/
    index/                   # build_index 결과
    shards/                  # build_shards 결과
    logs/                    # build/train 로그
    checkpoints/             # 모델 상태 저장
```

## Run

```bash
python main.py --config configs/data_2d.yaml --stage build_index
python main.py --config configs/data_2d.yaml --stage build_patches_pt
python main.py --config configs/train_ta.yaml --stage train
```

`train_ta.yaml`의 `train.data_source: index`를 사용하면 `build_shards` 없이 `index.csv`를 직접 읽어 on-the-fly로 학습합니다.
대용량 인덱스에서는 `train.sample_fraction` 또는 `train.max_samples_per_split`를 반드시 설정해 시작하세요.
`build_index`는 `outputs/index/index.csv`와 함께 `index_train.csv`, `index_val.csv`, `index_test.csv`를 자동 생성합니다.
`train.data_source: patch_pt`로 설정하면 `build_patches_pt`에서 만든 `21x21` 패치를 바로 학습에 사용합니다.

Clay 전이학습을 사용하려면 `configs/train_ta.yaml`의 `model.clay_ckpt_path`에 Clay pretrained `.ckpt` 경로를 설정하세요.

## Notes

- Normalization is loaded from:
  - `data/statistics/input_statistics_2022.json`
  - `data/statistics/statistics_2022.json`
- No normalization-statistics computation code is included.
- Current training stage is TA supervised only.
- `build_index` 로그: `outputs/logs/build_index.log`
- `build_shards` 로그: `outputs/logs/build_shards.log`
