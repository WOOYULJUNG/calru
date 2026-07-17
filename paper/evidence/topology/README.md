# Topology evidence

현재 topology 결과는 `all_models_v1`, `persistence_v2`, `ood_v2`를 논문
후보 evidence로 취급한다. `persistence_v1`과 `ood_v1`은 이전 pilot
snapshot으로 보존한다.

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

## `ood_v2`

동결된 RNN, GRU, LSTM, CA-LRU checkpoint를 재학습하지 않고 동일한 frozen
parent에서 만든 21개 paired 조건으로 평가한다. Primary temporal OOD는 ID의
128개 command와 endpoint 및 누적 path를 그대로 유지하고 command 사이에
exact blank만 삽입해 elapsed time을 최대 16배 늘린다. 누적 path/winding,
velocity scale, dwell 분포, GP smoothness, length×velocity stress와 각 조건
뒤 blank retention은 별도 축으로 분리한다.

- [`RESULTS_ko.md`](ood_v2/RESULTS_ko.md): 수치와 논문 해석
- [`representative_selection.csv`](ood_v2/representative_selection.csv):
  task-success 우선 대표 seed와 실패 fallback
- [`representative_metrics.csv`](ood_v2/representative_metrics.csv):
  primary figure의 조건별 수치
- [`seed_metrics.csv`](ood_v2/seed_metrics.csv): 36 checkpoint × 21 조건
- [`ood_all_seed_summary.csv`](ood_v2/ood_all_seed_summary.csv):
  모델×topology×조건의 3-seed 요약
- [`checkpoint_provenance.csv`](ood_v2/checkpoint_provenance.csv):
  checkpoint와 training metadata checksum
- [`banks_manifest.json`](ood_v2/banks_manifest.json): 63개 paired bank의
  exact audit와 pairing 검사

동일 path temporal 16×에서 CA-LRU의 3-seed median error는 최선 baseline보다
\(S^1,T^2,S^2\)에서 각각 1.78배, 18.00배, 2.34배 낮다. 반면 누적
path/winding stress와 combined stress의 우위는 topology 의존적이다. 따라서
논문의 primary 주장은 elapsed-time generalization이고, path stress는 반복
transport의 failure limit를 보여주는 supplementary 분석이다.

## `ood_v1` (archived pilot)

이 pilot은 horizon과 함께 active command 수와 누적 이동량도 증가시켰다.
따라서 기존 `length` 곡선은 순수 temporal OOD가 아니라 elapsed time과
cumulative path가 섞인 stress test다. 파일은 provenance를 위해 보존하지만
headline temporal 결과로 사용하지 않는다.

## 업데이트

이 디렉터리의 파일은 직접 편집하지 않는다.

~~~bash
make artifacts
make check-artifacts
~~~

정식 source mapping은
[`configs/paper_artifacts.json`](../../../configs/paper_artifacts.json)에 있다.
