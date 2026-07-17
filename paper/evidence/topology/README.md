# Topology evidence

현재 topology 결과는 두 묶음만 논문 후보 evidence로 취급한다.

## `all_models_v1`

RNN, GRU, LSTM, CA-LRU, H-C × \(S^1,T^2,S^2\) × seeds 10–12의
45-checkpoint 비교다.

- [`summary.csv`](all_models_v1/summary.csv): 모델×topology 3-seed 요약
- [`seed_metrics.csv`](all_models_v1/seed_metrics.csv): figure 제작에 우선
  사용하는 45행 wide table
- [`task_metrics.csv`](all_models_v1/task_metrics.csv): task 성능
- [`blank_metrics.csv`](all_models_v1/blank_metrics.csv): horizon별 blank 기억
- [`geometry_metrics.csv`](all_models_v1/geometry_metrics.csv): fiber와 local geometry
- [`dynamics_metrics.csv`](all_models_v1/dynamics_metrics.csv): sampled tangent/normal
- [`topology_radial_metrics.csv`](all_models_v1/topology_radial_metrics.csv):
  topology radius 방향의 hidden/output recovery
- [`RESULTS_ko.md`](all_models_v1/RESULTS_ko.md): 비교 해석과 제한
- [`manifest.json`](all_models_v1/manifest.json): 원 분석 root와 비교 조건

`recovery_q512_median`은 sampled hidden-normal 진단이고,
`output_radial_recovery_q512_median`은 topology radius 진단이다. 이름이 비슷해도
같은 열로 합치지 않는다.

## `persistence_v1`

동일 45 checkpoint의 원 hidden state에서 계산한 Vietoris–Rips persistent
homology다.

- [`seed_metrics.csv`](persistence_v1/seed_metrics.csv): seed×horizon 결과
- [`summary.csv`](persistence_v1/summary.csv): 3-seed 요약
- [`threshold_sensitivity_seed.csv`](persistence_v1/threshold_sensitivity_seed.csv):
  bar threshold 민감도
- [`reference.json`](persistence_v1/reference.json): 이상적 topology 기준
- [`completion.json`](persistence_v1/completion.json): source/config hash와 실행 완료 기록

Persistent homology는 global topology 진단이지 normal attraction이나 smooth
homeomorphism의 단독 증명이 아니다.

## 업데이트

이 디렉터리의 파일은 직접 편집하지 않는다.

~~~bash
make artifacts
make check-artifacts
~~~

정식 source mapping은
[`configs/paper_artifacts.json`](../../../configs/paper_artifacts.json)에 있다.
