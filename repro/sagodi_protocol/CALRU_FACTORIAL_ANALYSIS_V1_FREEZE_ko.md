# CA-LRU 2×2 factorial 확장 동역학 분석 동결

`calru_factorial_v1`의 최종 4조건 × seed 3개, 총 12 checkpoint를 전부 분석한다.

- CA-LRU no-RP, state-noise training 없음
- CA-LRU no-RP, state-noise training 있음
- CA-LRU RP, state-noise training 없음
- CA-LRU RP, state-noise training 있음

각 checkpoint에는 baseline 확장 pilot과 동일한 256 trajectory, 128 spline point,
T=128 task, 16T=2,048 blank rollout을 적용한다. Slow-manifold reconstruction,
output-projected flow, full local Jacobian eigenspectrum, flow-reversal fixed-point
topology, finite-time angular memory, stable/saddle basin 및 asymptotic capacity를
Ságodi-matched 분석 축으로 기록한다.

추가로 32개 anchor에서 tangent-complement Gaussian 방향 4개와 in-plane radial
방향 2개를 사용한다. Perturbation 반경은 carrier-manifold RMS scale의
0.01/0.05/0.1이며, 0/1/4/16/64/256/1024/4096 blank step에서 sampled carrier
spline까지의 거리, 초기 거리 대비 비율, decoded same-memory error, matched-clean
excess error를 기록한다. 이 finite normal-kick recovery는 Ságodi 원 분석이 아니라
threshold-free CA-LRU-specific supplementary analysis다.

학습 때의 state noise는 분석 rollout에서 재생하지 않는다. NMSE -20 dB 기준은
라벨일 뿐 structural analysis의 gate가 아니다. 어떤 seed도 성능에 따라 교체하거나
제외하지 않는다.
