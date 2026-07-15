# Source-v6 baseline noise/no-noise 확장 동역학 pilot 분석 동결

RNN, GRU, LSTM, LRU의 pilot main seed 0을 noise-free와 strictly-positive
state-noise-training 조건에서 등록한다. 총 denominator는 8 checkpoint다. 학습된
seeds 1,2 checkpoint는 보존하지만 이번 논문 구조 확인용 분석에는 쓰지 않는다.

등록된 모든 checkpoint에는 task 성능 통과 여부와 무관하게 approximate continuous
attractor 구조와 기억 topology를 구분하는 분석을 적용한다.

- task-aligned slow manifold: 256 direct T128 task trajectories
- 각 endpoint에서 16T=2,048 blank-input rollout
- trajectory별 최대 speed의 1e-3 이하 후보
- output ring nearest point와 128-point periodic cubic spline
- residual drift: spline 위 output-projected flow와 uniform norm
- timescale separation: spline 모든 점의 full local Jacobian eigenspectrum과
  leading-mode gap
- memory topology: signed angular flow의 cyclic reversal에서 stable/saddle을
  검출하고 basin 폭, entropy, effective basin count를 계산
- finite-time memory: spline memory를 2,048 blank steps 동안 rollout하여
  angular error를 계산
- finite normal-kick recovery: 32개 anchor, anchor당 4개 seeded ambient
  tangent-complement 방향과 2개 in-plane radial 방향, manifold RMS scale 대비
  반경 0.01/0.05/0.1, horizon 0/1/4/16/64/256/1024/4096

Slow-manifold reconstruction, projected flow, full local eigenspectrum,
flow-reversal topology, finite/asymptotic memory는 Ságodi-matched 축으로 기록한다.
Finite normal-kick recovery는 Ságodi 원 분석이라고 부르지 않고, threshold나 binary
gate가 없는 CA-LRU project-defined supplementary diagnostic으로 명시한다. 거리는
causal carrier state에서 sampled spline까지의 최근접 Euclidean 거리이며, 동일
anchor의 clean blank rollout을 paired control로 함께 기록한다.

Checkpoint의 clean NMSE가 -20 dB보다 낮은지는 `task-performance eligible` 라벨로
기록하지만 structural analysis의 gate로 사용하지 않는다. 미통과 checkpoint도
slow manifold, full local spectrum, projected drift를 동일하게 계산한다. 분석 시
state noise는 항상 꺼진다. 따라서 비교 요인은 inference noise가 아니라 training
state-noise의 효과다. 이 1-seed 결과는 효과 방향과 논문 도표를 결정하기 위한
exploratory evidence이며, seed 안정성이나 최종 통계의 근거로 사용하지 않는다.
