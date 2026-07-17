# Code map

디렉터리를 탐색해서 실행 파일을 추측하지 않도록, 공개 진입점과 상태를 여기에
고정한다. 기존 import와 frozen receipt를 깨지 않기 위해 historical file을
대규모 rename하지 않는다.

## Current

### `repro/manifold_benchmark`

Topology \(S^1,T^2,S^2\), CA-LRU 비교, persistent topology, OOD와 finite
normal-kick 분석의 현재 경로다.

| 단계 | 진입점 | config |
|---|---|---|
| ID bank | `build_id_banks.py` | `generator_v1_freeze.json` |
| topology training | `launch_topology_transfer.py` | `topology_transfer_v1.json` |
| CA-LRU selection | `launch_topology_hparam.py` | `topology_hparam_v1.json` |
| task/blank/geometry | `aggregate_topology_pilot.py` | `topology_*_analysis_v*.json` |
| persistent topology | `analyze_persistent_topology.py` | `topology_persistence_success_v2.json` |
| OOD bank | `build_ood_banks.py` | `topology_ood_calibrated_v3.json` |
| OOD evaluation | `analyze_ood_generalization.py` | `topology_ood_calibrated_v3.json` |
| OOD strength decision | `calibrate_ood_strength.py` | `topology_ood_calibrated_v3.json` |
| finite normal kick | `analyze_tangent_normal.py` | analysis config + CLI overrides |
| perturbation decision | `calibrate_perturbation_strength.py` | `topology_perturbation_calibrated_v1.json` |

`topology_ood_v1.json`과 `topology_ood_v2.json`은 호환성과 provenance 때문에
남아 있지만 새 primary OOD는 v3를 사용한다.

### `repro/sagodi_protocol`

Ring angular-integration baseline 재현, CA-LRU/RP factorial, Ságodi-based
slow-manifold 분석 코드다. 파일명에 v3–v6가 함께 있으므로 새 실행은 해당
freeze 문서와 config를 반드시 함께 지정한다.

| 목적 | 진입점/문서 | 상태 |
|---|---|---|
| public-code-centered baseline v6 | `run_source_repaired_v6_pipeline.sh` | supporting |
| LRU→CA-LRU v6 | `source_repaired_lru_calru_v6.py` | supporting |
| noise search | `run_state_noise_search_v1.sh` | supporting optimization |
| CA-LRU factorial | `run_calru_factorial_v1.sh` | current ring ablation |
| Ságodi core analysis | `source_v6_primary_analysis_campaign.py` | current evaluation tool |
| primary v3/v4 configs | `primary_v3_pipeline.py`, `primary_v4.py` | historical provenance |

## Archived exploratory

| 디렉터리 | 이유 |
|---|---|
| `repro/state_dependent_retention/` | H-C 탐색. 논문 중심 모델에서 제외됨 |
| `repro/experimental_v2/` | 초기 P0/confirmatory scaffold. 현재 topology pipeline으로 대체됨 |
| `repro/legacy_code/` | 최초 논문 수치를 만든 코드 snapshot. 새 실험 기반으로 사용하지 않음 |

Archived는 삭제 대상이라는 뜻이 아니다. 기존 artifact를 설명하고 재감사하는 데
필요하지만, 새 논문 figure의 기본 source로 승격하려면 별도 freeze와 PR이 필요하다.

## Lightweight repository tools

| 명령 | 역할 |
|---|---|
| `make status` | artifact group과 canonical 문서 위치 출력 |
| `make evidence` | legacy raw에서 9개 표 재생성 |
| `make check` | 표의 byte-level 재현 검사 |
| `make provenance` | raw manifest 갱신 |
| `make artifacts` | local frozen experiment에서 paper export 동기화 |
| `make check-artifacts` | paper export checksum 검사 |
| `make check-layout` | artifact manifest와 문서 구조 검사 |
| `make test` | lightweight aggregation/repository tests |
| `make test-experiment` | PyTorch/NumPy가 필요한 full protocol tests |
| `make verify` | local experiment를 요구하지 않는 전체 검사 |

GPU 학습은 `make` target으로 숨기지 않는다. 실행한 module, config, output root,
commit hash를 completion receipt에 명시적으로 남긴다.
