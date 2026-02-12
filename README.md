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

## 데이터 소개

### 입력 변수 구성

현재 모델 입력은 다음과 같이 구성됩니다:

- **GK-2A 밝기온도 (BT)**: 10차원
- **GK-2A 반사도 (Reflectance)**: 6차원
- **DEM (Digital Elevation Model)**: 1차원
- **시간 인코딩**: 7차원 (DOY, Hour, Month, SZA)
- **육지-해양 마스크 (LSMASK)**: 1차원
- **토지 피복 원-핫 인코딩**: 17차원 (MODIS MCD12C1 IGBP 분류 기준: 0~16)

**총 입력 차원**: 19 (스펙트럴) + 7 (시간) + 3 (위치/메타) + 17 (토지피복) = 46차원

### 데이터 저장 방식

- **인덱스**: CSV 파일로 샘플 메타데이터 저장 (`outputs/index/index_train.csv` 등)
- **실제 데이터**: 
  - `build_patches_pt` 단계: PyTorch `.pt` 파일로 패치 데이터 저장
  - `build_shards` 단계: NumPy `.npy` 파일로 샘플 저장
- 각 샘플은 패치 중심 픽셀 좌표 기준으로 처리됩니다.

## 개발 고민사항 Q&A

### Q1: 시간 인코딩은 어떻게 처리하나요?

**A**: 현재는 파일 매칭용 timestamp만 있고, 모델 입력 피처(예: sin/cos of day-of-year, minute-of-day)로는 아직 완전히 구현되지 않았습니다. 

- **현재 상태**: UTC 기준으로 시간 인코딩 (7차원)
- **구현 예정**: DOY, Hour, Month, SZA를 포함한 시간 피처 인코딩

### Q2: GSD 인코딩은 어떻게 하나요?

**A**: 아직 입력에 포함되지 않았습니다.

- **단일 센서 (GK2A ko020lc)**: 상수 GSD 2km 고정값 사용
- **멀티 센서 확장 시**: 샘플별 GSD를 입력에 포함해야 합니다.

### Q3: 밴드 파장(wavelength) 메타데이터는 어떻게 처리하나요?

**A**: Dynamic Embedding/Waves Transformer의 핵심인데, 현재는 채널값만 있고 "각 채널의 중심 파장" 전달 로직이 없습니다.

**필요한 파장 테이블**:

| Band | 명칭      | 중심파장 (μm) |
| ---- | ------- | --------- |
| 1    | VIS0.47 | 0.47      |
| 2    | VIS0.51 | 0.51      |
| 3    | VIS0.64 | 0.64      |
| 4    | VIS0.86 | 0.86      |
| 5    | NIR1.37 | 1.37      |
| 6    | NIR1.6  | 1.60      |
| 7    | SWIR3.8 | 3.80      |
| 8    | WV6.3   | 6.30      |
| 9    | WV6.9   | 6.90      |
| 10   | WV7.3   | 7.30      |
| 11   | IR8.7   | 8.70      |
| 12   | IR9.6   | 9.60      |
| 13   | IR10.5  | 10.50     |
| 14   | IR11.2  | 11.20     |
| 15   | IR12.3  | 12.30     |
| 16   | IR13.3  | 13.30     |

### Q4: MAE + Self-Distillation 학습은 언제 구현하나요?

**A**: 현재는 TA 감독학습용 데이터셋 생성 단계입니다. Clay의 본래 MAE(재구성 95%) + DINOv2 표현손실 5% 학습 파이프라인은 추후 구현 예정입니다.

**현재 단계**: 2D 회귀 데이터 준비
**다음 단계**: 시간/GSD/lat-lon/파장 메타를 포함한 데이터 스키마 확장

### Q5: 마스킹을 해도 괜찮은가요? TA가 지점자료인데...

**A**: 네, 마스킹은 문제없습니다. 

- **이유**: TA 값은 900×900 격자 전체가 아니라 일부 지점자료이지만, MAE 학습 시에는 **입력 이미지 패치를 재구성**하는 것이 목표입니다.
- **마스킹 비율**: 입력 패치의 일부 영역을 마스크하고, 모델이 마스크된 부분을 예측하도록 학습합니다. 이는 self-supervised learning의 핵심 기법입니다.
- **TA 지점자료와의 관계**: TA는 감독학습 레이블로만 사용되며, MAE 재구성 타겟과는 별개입니다.

### Q6: Self-Distillation (DINOv2)이 무엇인가요?

**A**: Self-Distillation은 "자기 자신을 가르치는" 학습 방법입니다.

**비유로 설명하면**:
- **학생 모델**: 현재 학습 중인 모델 (우리가 만드는 모델)
- **선생 모델 (Teacher)**: DINOv2 같은 사전학습된 강력한 모델
- **학습 과정**: 학생 모델이 선생 모델의 출력(표현)을 따라하도록 학습
  - 95%는 MAE 재구성 손실 (자기 자신을 예측)
  - 5%는 DINOv2 표현 손실 (선생 모델의 표현을 모방)

**왜 유용한가**: DINOv2는 대규모 이미지로 학습되어 강력한 표현을 가지고 있습니다. 이를 모방하면 더 나은 특징 추출이 가능합니다.

### Q7: 샘플은 CSV로 저장하나요, NPY로 저장하나요?

**A**: 둘 다 사용합니다.

- **CSV (인덱스)**: 샘플 메타데이터만 저장
  - 위치: `outputs/index/index_train.csv` 등
  - 내용: timestamp, 좌표, 파일 경로 등 메타정보
- **NPY/PT (실제 데이터)**: 실제 패치 데이터 저장
  - `build_patches_pt`: PyTorch `.pt` 파일 (패치 + 메타)
  - `build_shards`: NumPy `.npy` 파일 (샘플 배열)

### Q8: Landcover one-hot은 어떻게 처리하나요?

**A**: 패치 중심 픽셀 기준으로 처리합니다.

- 참고 코드: `/JBOD/Lab_data/suhwankim/jbnu_kma_ksh_ta/TA_FINAL_VER3/02_preprocessing/preprocess_pixels_numpy.py`
- 처리 방식: 각 샘플의 `(pixel_x, pixel_y)` 좌표에 해당하는 landcover 값을 사용하여 one-hot 인코딩
- 패치 내 다수결이 아닌, **중심 픽셀의 landcover 클래스**를 사용

### Q9: 결측 데이터는 어떻게 처리하나요?

**A**: 36채널 중 하나라도 NaN/이상치가 있으면 샘플을 제외합니다.

- 검증: `src/data/validators.py`의 `has_any_nan_or_inf` 함수 사용
- 처리: BT, Reflectance, SZA, DEM 중 하나라도 NaN/Inf가 있으면 해당 샘플 제외

### Q10: 정규화는 어떻게 하나요?

**A**: Train split (2022년) 기준으로 채널별 mean/std를 계산하여 고정 사용합니다.

- 입력 통계: `data/statistics/input_statistics_2022.json`
- 타겟 통계: `data/statistics/statistics_2022.json`
- 정규화 공식: `(x - mean) / std`

## 설계 결정사항

다음은 프로젝트 초기 설계 시 결정한 사항들입니다:

1. **시간 인코딩**: UTC 기준
2. **정규화**: 2022년 (train split) 기준 mean/std 고정
3. **학습 단계**: 현재는 TA supervised only, 추후 MAE+distillation 구현 예정
4. **Landcover 처리**: 패치 중심 픽셀 기준
5. **결측 처리**: NaN/Inf가 하나라도 있으면 샘플 제외

## Notes

- Normalization is loaded from:
  - `data/statistics/input_statistics_2022.json`
  - `data/statistics/statistics_2022.json`
- No normalization-statistics computation code is included.
- Current training stage is TA supervised only.
- `build_index` 로그: `outputs/logs/build_index.log`
- `build_shards` 로그: `outputs/logs/build_shards.log`
