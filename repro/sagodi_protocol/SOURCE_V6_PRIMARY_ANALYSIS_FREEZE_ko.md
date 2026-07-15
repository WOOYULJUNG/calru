# Source-v6 baseline noise/no-noise Ságodi-core pilot 분석 동결

RNN, GRU, LSTM, LRU의 pilot main seed 0을 noise-free와 strictly-positive
state-noise-training 조건에서 등록한다. 총 denominator는 8 checkpoint다. 학습된
seeds 1,2 checkpoint는 보존하지만 이번 논문 구조 확인용 분석에는 쓰지 않는다.

등록된 모든 checkpoint에는 task 성능 통과 여부와 무관하게 approximate continuous
attractor 주장을 구분하는 데 필요한 세 가지 구조 분석을 적용한다.

- task-aligned slow manifold: 256 direct T128 task trajectories
- 각 endpoint에서 16T=2,048 blank-input rollout
- trajectory별 최대 speed의 1e-3 이하 후보
- output ring nearest point와 128-point periodic cubic spline
- residual drift: spline 위 output-projected flow와 uniform norm
- timescale separation: spline 모든 점의 full local Jacobian eigenspectrum과
  leading-mode gap

이번 core pilot에서는 flow-reversal fixed-point topology, stable/saddle 개수,
finite-time/asymptotic basin·capacity, carrier ambient-normal recovery를 계산하지
않는다. 이 항목들은 핵심 CA 구조가 확인된 뒤 확증·supplementary 분석에서만
확대한다.

Checkpoint의 clean NMSE가 -20 dB보다 낮은지는 `task-performance eligible` 라벨로
기록하지만 structural analysis의 gate로 사용하지 않는다. 미통과 checkpoint도
slow manifold, full local spectrum, projected drift를 동일하게 계산한다. 분석 시
state noise는 항상 꺼진다. 따라서 비교 요인은 inference noise가 아니라 training
state-noise의 효과다. 이 1-seed 결과는 효과 방향과 논문 도표를 결정하기 위한
exploratory evidence이며, seed 안정성이나 최종 통계의 근거로 사용하지 않는다.
