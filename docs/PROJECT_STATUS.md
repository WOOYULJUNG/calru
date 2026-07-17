# Project status

이 문서는 논문 작업의 현재 결정을 한 곳에서 고정한다. 과거 README나 branch
이름보다 이 문서를 우선한다.

## 모델

| 모델 | 상태 | 사용처 |
|---|---|---|
| CA-LRU + RP | current primary | ring, topology, OOD, blank-memory 분석 |
| CA-LRU no-RP | current ablation | RP의 matched control |
| RNN, GRU, LSTM, LRU | current baselines | task와 동역학 비교 |
| H-C | archived exploratory | state-dependent retention의 가능성과 failure mode |
| legacy PAN/CAMN/AM-LRU tags | provenance only | 기존 JSON/checkpoint 식별 |

논문에서 새로 만든 명칭은 CA-LRU와 Retention Plasticity 두 개만 사용한다.
H-C 결과는 CA-LRU 결과와 합치지 않는다.

## 현재 evidence

| ID | 상태 | 의미 |
|---|---|---|
| `all_models_v1` | current exploratory | 모델×topology task·blank·geometry 비교 |
| `persistence_v2` | current exploratory | 성공 seed 우선 persistent topology |
| `ood_v3` | current exploratory | baseline 동시 실패 경계를 보정한 OOD |
| `perturbation_v1` | current exploratory | finite hidden-normal recovery |
| `ood_v2` | auxiliary | 동일 path temporal-dilation 진단 |
| `ood_v1` | archived | elapsed time과 cumulative path가 섞인 pilot |
| `persistence_v1` | archived | 실패 seed를 포함한 초기 all-seed snapshot |
| `hc_local_grid_v2` | archived development | H-C 설정 선택 과정 |

현재 artifact status는
[`configs/paper_artifacts.json`](../configs/paper_artifacts.json)의 `status`와
일치해야 한다.

## 현재 OOD 강도

ID task gate를 통과한 checkpoint만 OOD 생존 분모에 포함한다.

| 축 | headline 범위 | stress-only |
|---|---:|---|
| length | \(T\le1536\) | \(T=2048\) |
| velocity | \(0.5\times\)–\(2\times\) | \(3\times,4\times\) |
| GP correlation | \(\ell=0.5\times\)–\(4\times\) | 시험 범위 내 없음 |
| contiguous dwell | 96 steps | all-blank 128 |
| post-blank | \(H=512\) | \(H\ge768\) |
| hidden-normal kick | \(\rho=0.25,H=2048\) | 시험 범위 내 없음 |

`chance_collapsed`와 자기 ID error 대비 8배를 넘는
`severely_degraded`를 구분한다. 강한 stress에서 모든 baseline이 무너지면
headline 비교로 사용하지 않는다.

## Claim boundary

- CA-LRU가 모든 topology와 모든 지표에서 baseline보다 우수하다고 주장하지 않는다.
- finite hidden-normal kick에서 baseline은 nearest-manifold distance를 더 강하게
  줄이고, CA-LRU는 same-memory error를 더 작게 유지한다.
- persistent homology는 global topology 진단이지 normal attraction의 증명이 아니다.
- PCA는 시각화이며 topology의 증명이 아니다.
- \(S^2\) CA-LRU는 현재 ID-qualified seed가 하나이므로 confirmatory claim이 아니다.
- Ságodi-matched 분석과 프로젝트가 추가한 finite perturbation 진단을 구분한다.

## 다음 confirmatory 단계

1. frozen protocol과 새 seed를 먼저 commit한다.
2. 모델별 ID eligibility를 충분히 확보한다.
3. 현재 calibration에서 선택한 headline 강도를 변경하지 않는다.
4. seed-level 결과와 실패 seed를 모두 보고한다.
5. 새 결과는 별도 experiment ID에 저장한 뒤 artifact manifest로 승격한다.
