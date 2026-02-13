# FAQ (현재 구현 기준)

## Q1. 지금 Clay를 그대로 구현한 건가요?

아니요. 현재는 **Clay encoder 기반 TA 전이학습(fine-tuning)** 입니다.

- 구현됨:
  - pretrained ckpt 로드
  - 파장(waves), 시간(time), 위치(lat/lon), GSD 입력
  - supervised TA 회귀 학습
- 미구현:
  - MAE 재구성 loss
  - DINOv2 self-distillation loss

## Q2. 시간/위경도/GSD/파장은 실제로 모델에 들어가나요?

네, 들어갑니다.

- 시간: `time7`에서 Clay 입력 조합으로 사용
- 위치/GSD: `loc = [lat, lon, gsd_m]`
- 파장: `model.bt_waves_um + model.rf_waves_um`

## Q3. loss, rmse, rmse_K는 어떻게 다른가요?

- `loss`: MSE
- `rmse`: `sqrt(MSE)` (정규화 스케일)
- `rmse_K`: 정규화 해제 RMSE
  - `RMSE_K = RMSE_norm * TA_std`

참고: RMSE에는 `mean`을 더하지 않습니다.

## Q4. 왜 학습 중간중간 끊기나요?

주요 원인은 연산보다 `patch_pt` 대용량 파일 I/O(`torch.load`) 병목입니다.

영향 요소:
- 파일 크기/개수
- worker 수
- shuffle 전략

## Q5. shuffle은 어떻게 두는 게 좋나요?

속도 우선:
- `patch_shuffle_files: false`
- `patch_shuffle_within_file: false`

일반화 우선:
- 두 값을 `true`로 올릴 수 있으나 I/O 부담이 증가할 수 있습니다.

## Q6. 왜 GPU 메모리가 덜 차 보이나요?

현재 기본이 `freeze_encoder: true`라 activation 저장량이 작고, 패치 크기(`9x9`)도 작아서 배치 증가 대비 메모리 증가가 제한적입니다.

## Q7. freeze_encoder를 지금 풀어야 하나요?

현재 단계에서는 보통 `true` 유지가 권장됩니다.

1. 먼저 `freeze_encoder=true`로 안정 수렴
2. 필요 시 후반에 `freeze_encoder=false` + 낮은 LR로 짧게 미세조정

## Q8. validation이 train보다 느려 보일 수 있나요?

가능합니다. `val` 직후 체크포인트 저장 I/O 시간이 체감상 포함되어 느리게 보일 수 있습니다.

## Q9. check_patch_pt.py는 인자 없이 실행 가능한가요?

가능합니다.

```bash
python src/data/check_patch_pt.py
```

자동으로 `outputs/patch_pt`에서 첫 청크를 찾아 출력합니다.

## Q10. 현재 마스킹(MAE) 학습도 하고 있나요?

아니요. 현재 학습 루프는 **TA supervised fine-tuning** 입니다.

## Q11. 샘플은 CSV로 저장하나요, NPY/PT로 저장하나요?

둘 다 씁니다.

- 인덱스 메타: CSV (`outputs/index/index_*.csv`)
- 학습 데이터: PT 청크 (`outputs/patch_pt/*_chunk_*.pt`)

현재 기본 학습 경로는 `patch_pt`입니다.

## Q12. Landcover one-hot은 어떻게 처리하나요?

패치 중심 픽셀 클래스 기준으로 17차원 one-hot을 생성합니다.

## Q13. 결측 데이터는 어떻게 처리하나요?

`build_patches_pt` 단계에서 유효성 검사를 수행하고, NaN/Inf/범위 조건을 통과한 샘플만 저장합니다.

## Q14. 정규화는 어떻게 하나요?

2022 통계 파일 기반 z-score 정규화를 사용합니다.

- 입력 통계: `data/statistics/input_statistics_2022.json`
- 타깃 통계: `data/statistics/statistics_2022.json`

## Q15. UTC 기준인가요?

네. timestamp는 UTC 기준으로 처리합니다.

## Q16. split 기준은 어떻게 되나요?

현재 설정(`configs/data_2d.yaml`)은 다음과 같습니다.

- Train: 2022
- Val: 2023 (1~12월)
- Test: 2024

## Q17. 파장 테이블(16채널)은 어디서 쓰이나요?

학습 시 `configs/train_ta.yaml`의 아래 값이 사용됩니다.

- `model.bt_waves_um` (10)
- `model.rf_waves_um` (6)

즉 총 16개 중심파장이 Clay encoder 입력으로 전달됩니다.

## Q18. MAE + Self-Distillation(0.95/0.05)은 지금 동작하나요?

아니요. 현재는 동작하지 않습니다.

- 현재: supervised TA 회귀 학습
- 추후 과제: MAE reconstruction + DINOv2 distillation 결합 학습
