# Source-v6 baseline noise/no-noise Ságodi-primary 분석 동결

RNN, GRU, LSTM, LRU의 pilot main seeds 0--2를 noise-free와 strictly-positive
state-noise-training 조건에서 모두 등록한다. 총 denominator는 24 checkpoint다.

각 checkpoint에는 Ságodi-primary 분석만 적용한다.

- 1,024 direct T128 task trajectories
- 각 endpoint에서 16T=2,048 blank-input rollout
- trajectory별 최대 speed의 1e-3 이하 후보
- output ring nearest point와 1,024-point periodic cubic spline
- output-projected flow와 cyclic flow-reversal fixed-point topology
- spline 모든 점의 full local Jacobian eigenspectrum
- finite-time angular error와 asymptotic basin/capacity

Checkpoint의 clean NMSE가 -20 dB보다 낮지 않으면 structural analysis를 실행하지
않되 denominator에서 제거하지 않는다. 분석 시 state noise는 항상 꺼진다. 따라서
비교 요인은 inference noise가 아니라 training state-noise의 효과다. 기존
carrier ambient-normal recovery는 threshold 없는 project-defined descriptive
extension으로 별도 표기하며 CA gate로 사용하지 않는다.
