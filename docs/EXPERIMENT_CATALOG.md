# Experiment catalog

현재 논문 artifact에 연결된 frozen experiment만 기록한다. smoke, failed cell,
중간 sweep의 전체 디렉터리 목록이 아니다. 로컬 경로의 공통 prefix는
`${CALRU_EXPERIMENT_ROOT}/`이며 GitHub에는 checkpoint를 커밋하지 않는다.

## Current topology/OOD sources

| Artifact ID | 상태 | 로컬 source root | 역할 |
|---|---|---|---|
| `all_models_v1` | current | `manifold_topology_hparam_v1-c6dc758/comparison_all_models_v1` | RNN/GRU/LSTM/CA-LRU/H-C 45-checkpoint 비교 |
| `persistence_v2` | current | `manifold_topology_persistence_success_v2-20260717` | 성공 seed 우선 \(H\le4096\) persistent topology |
| `ood_v3` | current | `manifold_topology_ood_calibrated_v3-20260717` | 36 checkpoint × 27 calibrated conditions |
| `ood_v3 banks` | current | `manifold_topology_ood_banks_calibrated_v3-20260717` | 81 frozen paired banks |
| `perturbation_v1` | current | `manifold_topology_perturbation_calibrated_v1-20260717` | \(\rho\le0.25,H\le2048\) hidden-normal recovery |

`all_models_v1`에는 H-C가 역사적 비교 대상으로 남아 있지만 현재 논문 중심 모델은
CA-LRU다.

## Supporting sources

| Artifact ID | 상태 | 로컬 source root | 역할 |
|---|---|---|---|
| `generator_v1_rodrigues` | frozen input | `manifold_benchmark_generator_v1_rodrigues-20260716` | validation ID banks |
| `generator_v1_rodrigues-test` | frozen input | `manifold_benchmark_generator_v1_rodrigues-test-20260716` | sequestered test parent/banks |
| `baseline_all_v2` | source analysis | `manifold_topology_transfer_v1-pilot-1f4a80d/analysis_all_v2` | RNN/GRU/LSTM topology 분석 |
| `selected_hparam_v1` | source analysis | `manifold_topology_hparam_v1-c6dc758/analysis_final_v1` | 선택 CA-LRU/H-C checkpoint 분석 |
| `ood_v2` | auxiliary | `manifold_topology_ood_v2-20260717` | exact same-path temporal dilation |
| `ood_v2 banks` | auxiliary | `manifold_topology_ood_banks_v2-20260717b` | v2 paired banks |

## Archived sources

| Artifact ID | 이유 |
|---|---|
| `ood_v1` | horizon과 cumulative movement가 동시에 증가한 pilot |
| `persistence_v1` | all-seed \(H\le2048\) 초기 snapshot |
| `hc_local_grid_v2` | 폐기된 H-C 중심 모델의 development selection |

Archived 결과는 provenance를 위해 paper tree에 남지만 현재 headline figure를
고르는 source가 아니다.

## Canonical exports

| 질문 | 먼저 볼 파일 |
|---|---|
| project/claim 상태 | `docs/PROJECT_STATUS.md` |
| 전체 모델 수치 | `paper/evidence/topology/all_models_v1/summary.csv` |
| persistent topology | `paper/evidence/topology/persistence_v2/representative_metrics.csv` |
| calibrated OOD 판정 | `paper/evidence/topology/ood_v3/calibration_decisions.json` |
| OOD seed-level 수치 | `paper/evidence/topology/ood_v3/seed_metrics.csv` |
| finite normal kick | `paper/evidence/topology/perturbation_v1/model_survival.csv` |
| figure 목록 | `paper/figures/README.md` |
| source/checksum 대응 | `paper/artifact_checksums.json` |

## 동기화

~~~bash
export CALRU_EXPERIMENT_ROOT=/path/to/calru-experiments
make artifacts-list
make artifacts
make check-artifacts
~~~

정확한 source/destination 목록은
[`configs/paper_artifacts.json`](../configs/paper_artifacts.json)이 유일한
machine-readable source of truth다. 이 문서와 manifest가 다르면 manifest를
기준으로 문서를 수정한다.
