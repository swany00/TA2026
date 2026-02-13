# TA_2D_VER1

<div align="center">

GK-2A 2D 패치 기반 TA 회귀 학습 파이프라인  
**Clay 사전학습 인코더 기반 전이학습**

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9-EE4C2C?logo=pytorch&logoColor=white)
![Model](https://img.shields.io/badge/Model-Clay%20Transfer-0A7EA4)
![Data](https://img.shields.io/badge/Data-2D%20Patch%20PT-2D7D46)
![Stage](https://img.shields.io/badge/Stage-Supervised%20TA-6C4AB6)

</div>

---

## 한눈에 보기

| 항목 | 현재 상태 |
| --- | --- |
| 학습 방식 | Clay encoder transfer fine-tuning |
| 데이터 포맷 | `patch_pt` (`*_chunk_XXXXX.pt`) |
| 입력 | `x(19,9,9) + time7 + loc + landcover_onehot` |
| 타깃 | TA z-score |
| 체크포인트 | `current.pt`(최신), `best.pt`(최저 val loss) |
| 메트릭 | `loss(MSE), RMSE(norm), RMSE(K)` |

---

## 파이프라인

```mermaid
sequenceDiagram
    participant Raw as 원천 데이터
    participant Index as build_index
    participant Patch as build_patches_pt
    participant PT as outputs/patch_pt
    participant Train as train (clay_transfer)
    participant Ckpt as outputs/checkpoints

    Note over Raw,Index: 1) 인덱스 생성
    Raw->>Index: TA 라벨 + BT/RF/SZA 경로 + 좌표
    Index-->>Raw: index_train/val/test.csv 저장

    Note over Index,Patch: 2) 패치 생성
    Index->>Patch: split별 index CSV 입력
    Patch->>Patch: 좌표 검증/결측 필터/정규화
    Patch->>Patch: time7, loc, landcover_onehot 생성
    Patch-->>PT: train/val/test_chunk_XXXXX.pt 저장

    Note over PT,Train: 3) 학습
    PT->>Train: patch_pt 로딩 (x, y, time7, loc, lc)
    Train->>Train: Clay encoder + 회귀 헤드 학습
    Train-->>Ckpt: current.pt (매 epoch)
    alt val loss 개선
        Train-->>Ckpt: best.pt 갱신
    end
```

### 실행 순서

```bash
python main.py --config configs/data_2d.yaml --stage build_index
python main.py --config configs/data_2d.yaml --stage build_patches_pt
python main.py --config configs/train_ta.yaml --stage train
```

---

## 패치 생성 상세 (`build_patches_pt`)

### 입력/출력

- 입력 인덱스:
  - `outputs/index/index_train.csv`
  - `outputs/index/index_val.csv`
  - `outputs/index/index_test.csv`
- 출력 청크:
  - `outputs/patch_pt/train_chunk_00000.pt` ...
  - `outputs/patch_pt/val_chunk_00000.pt` ...
  - `outputs/patch_pt/test_chunk_00000.pt` ...

### PT 청크 내부 스키마

- `x`: `(N, 19, 9, 9)` float32
- `y`: `(N,)` float32 (정규화된 TA)
- `time7`: `(N, 7)` float32
- `loc`: `(N, 3)` float32
- `landcover_onehot`: `(N, 17)` float32
- `meta`: 길이 `N` list (`timestamp_utc`, `stn`, `pixel_x`, `pixel_y`)

---

## 샘플 텐서 구조

```mermaid
flowchart TB
  S["Sample i"] --> X["x: [19,9,9]<br/>BT10 + RF6 + SZA1 + DEM1 + LSMASK1"]
  S --> T["time7: [7]<br/>doy/hour/month sin-cos + sza_center"]
  S --> L["loc: [3]<br/>lat, lon, gsd_m"]
  S --> C["landcover_onehot: [17]"]
  S --> Y["y: [1]<br/>normalized TA"]
```

정규화 통계 파일:
- `data/statistics/input_statistics_2022.json`
- `data/statistics/statistics_2022.json`

---

## Clay 전이학습 설정

`configs/train_ta.yaml`

- `model.type: clay_transfer`
- `model.clay_ckpt_path`: Clay 체크포인트 경로
- `model.bt_waves_um`, `model.rf_waves_um`: 파장 메타데이터
- `model.freeze_encoder: true` (기본)

현재 학습 범위:
- 구현됨: supervised TA fine-tuning
- 미구현: MAE reconstruction, DINOv2 self-distillation

---

## 체크포인트 정책

저장 위치: `outputs/checkpoints`

- `current.pt`: 최신 epoch 상태
- `best.pt`: 최저 validation loss 상태

포함 항목:
- `model`, `optimizer`, `scheduler`, `epoch`, `best_val`

---

## 로그/메트릭

epoch 로그 출력:

- `train_loss`, `val_loss` (MSE)
- `train_rmse_norm`, `val_rmse_norm`
- `train_rmse_K`, `val_rmse_K`

`RMSE_K = RMSE_norm * TA_std`

---

## 문서

- Q&A: `docs/FAQ.md`
- 문서 구성 안내: `docs/README.md`

## Utilities

패치 파일 빠른 확인:

```bash
python src/data/check_patch_pt.py
```

특정 파일/인덱스 확인:

```bash
python src/data/check_patch_pt.py outputs/patch_pt/train_chunk_00010.pt --idx 10
```
