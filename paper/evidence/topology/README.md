# Topology evidence

현재 topology 결과는 `all_models_v1`, `persistence_v2`, `ood_v1`을 논문
후보 evidence로 취급한다. `persistence_v1`은 이전 all-seed
\(H\leq2048\) snapshot으로 보존한다.

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

## `persistence_v2`

원 hidden state에서 계산한 Vietoris–Rips persistent homology다. Primary
표와 그림은 모델×topology별 task-success seed 중 validation error가 가장
작은 checkpoint를 사용한다. 성공 seed가 없으면 전체 seed 중 validation
error가 가장 작은 checkpoint를 명시적인 task-failed fallback으로 사용한다.
전체 45 checkpoint 집계는 robustness 보조 분석으로 남긴다.

- [`representative_selection.csv`](persistence_v2/representative_selection.csv):
  성공 representative 또는 명시적 실패 fallback 선택 근거
- [`representative_metrics.csv`](persistence_v2/representative_metrics.csv):
  primary seed×horizon 결과
- [`seed_metrics.csv`](persistence_v2/seed_metrics.csv): 전체 seed×horizon 결과
- [`summary.csv`](persistence_v2/summary.csv): 전체 3-seed robustness 요약
- [`threshold_sensitivity_seed.csv`](persistence_v2/threshold_sensitivity_seed.csv):
  bar threshold 민감도
- [`reference.json`](persistence_v2/reference.json): 이상적 topology 기준
- [`completion.json`](persistence_v2/completion.json): source/config hash와 실행 완료 기록

Persistent homology는 global topology 진단이지 normal attraction이나 smooth
homeomorphism의 단독 증명이 아니다.

## `ood_v1`

동결된 RNN, GRU, LSTM, CA-LRU checkpoint를 재학습하지 않고 동일한 frozen
parent에서 만든 17개 paired 조건으로 평가한다. 길이, velocity scale, dwell
분포, GP smoothness, length×velocity stress와 각 조건 뒤 blank retention을
분리한다.

- [`RESULTS_ko.md`](ood_v1/RESULTS_ko.md): 수치와 논문 해석
- [`representative_selection.csv`](ood_v1/representative_selection.csv):
  task-success 우선 대표 seed와 실패 fallback
- [`representative_metrics.csv`](ood_v1/representative_metrics.csv):
  primary figure의 조건별 수치
- [`seed_metrics.csv`](ood_v1/seed_metrics.csv): 36 checkpoint × 17 조건
- [`ood_all_seed_summary.csv`](ood_v1/ood_all_seed_summary.csv):
  모델×topology×조건의 3-seed 요약
- [`checkpoint_provenance.csv`](ood_v1/checkpoint_provenance.csv):
  checkpoint와 training metadata checksum
- [`banks_manifest.json`](ood_v1/banks_manifest.json): 51개 paired bank의
  exact audit와 pairing 검사

현재 결과의 가장 일관된 CA-LRU 이점은 blank retention이다. Input-driven
transport OOD는 topology 의존적이며, 특히 \(S^2\)와 combined stress에서는
LSTM이 더 강하다. 따라서 OOD figure는 모든 조건의 일괄 우위를 주장하는
자료가 아니라 retention과 transport 병목을 분리하는 자료로 사용한다.

## 업데이트

이 디렉터리의 파일은 직접 편집하지 않는다.

~~~bash
make artifacts
make check-artifacts
~~~

정식 source mapping은
[`configs/paper_artifacts.json`](../../../configs/paper_artifacts.json)에 있다.
