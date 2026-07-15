# Pilot-3 계산 예산 결정 근거

이 문서는 논문의 실험 틀과 예상 경향을 먼저 확인하기 위한 exploratory pilot의
계산 예산을 동결한다. 확증 실험의 seed 수나 최종 통계 규약으로 인용하지 않는다.

## 5,000 updates 필요성 점검

중단·보존된 artifact
`sagodi_source_repaired_baselines_v6-noisefree-8fbd7fd`에서 완주한 78개
sentinel/fanout training trace를 사용했다. 각 값은 해당 update에서 가장 좋은 LR
cell의 validation MSE 중앙값이다. LSTM은 fanout 전이라 seed 100만 반영되어 해석이
제한적이다.

| model | 1k | 2k | 3k | 4k | 5k | 해석 |
|---|---:|---:|---:|---:|---:|---|
| RNN | 0.02193 | 0.00991 | 0.00754 | 0.00603 | 0.00496 | 5k에서 처음 NMSE -20 dB 기준 부근 |
| GRU | 0.00567 | 0.00277 | 0.00236 | 0.00135 | 0.00163 | 2–3k부터 pilot screening 가능 |
| LSTM | 0.00321 | 0.00056 | 0.00061 | 0.00043 | 0.00036 | 2–3k부터 충분해 보이나 1 seed 근거 |

RNN의 best LR도 3k의 0.001에서 5k의 0.003으로 바뀌었고, GRU의 best LR도
update에 따라 달라졌다. 따라서 모든 최종 checkpoint를 2–3k로 자르면 Ságodi
eligibility와 LR 선택이 함께 달라질 수 있다. 이전 LRU pilot에서도 validation MSE가
1k에서 0.056–0.098, 5k에서 0.022–0.033으로 계속 개선되어 짧은 최종 학습은 특히
LRU 계열에 불리했다.

## 동결한 pilot 예산

- LR grid screening: 각 cell seeds 100,101, 2,000 updates.
- State-noise std screening: 각 cell seeds 100,101, 2,000 updates.
- 선택된 baseline main: fresh seeds 0,1,2, 5,000 updates.
- Baseline Ságodi core analysis: model × noise 조건별 대표 seed 0만 분석하고,
  slow manifold·full local spectrum·projected drift만 계산한다.
- CA-LRU RP search: seed 100 sentinel과 seeds 101,102 fanout을 유지한다.
- CA-LRU 2×2 main: 조건별 fresh seeds 0,1,2, 5,000 updates.

5,000은 grid의 모든 cell에 쓰지 않고 선택된 main에만 사용한다. Main trace는 500
updates마다 validation을 기록하므로, pilot 종료 뒤 3k/4k/5k 결과를 다시 비교해
확증 실험에서 공통 5k가 실제로 필요한지 결정한다.

## 실행 수

| stage | 이전 | pilot-3 |
|---|---:|---:|
| RNN/GRU/LSTM LR tuning | 120 | 48 |
| RNN/GRU/LSTM main | 30 | 9 |
| LRU LR tuning | 40 | 16 |
| LRU main | 10 | 3 |
| Four-baseline noise tuning | 100 | 40 |
| Four-baseline noise main | 40 | 12 |
| Baseline Ságodi analysis | 80 | 8 |
| CA-LRU RP sentinel/fanout | 47 | 37 |
| CA-LRU 2×2 main | 40 | 12 |

이 예산의 목적은 효과 크기와 실패 양상을 빠르게 확인해 논문의 그림·표·주장 구조를
정하는 것이다. 최종 신뢰구간이나 seed 안정성 주장은 이후 확증 실험에서 별도로
검증한다.
