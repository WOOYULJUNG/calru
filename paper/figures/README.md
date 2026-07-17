# Paper figure catalog

논문 후보 figure는 주제별 하위 디렉터리에서 찾는다. 이 디렉터리에 figure를
평평하게 추가하지 않는다.

## 바로 볼 figure

### 모델 비교

경로: [`topology/model_comparison/`](topology/model_comparison/)

| 파일 | 질문 | 입력 evidence |
|---|---|---|
| `fig_A_task_and_blank` | task 정확도와 장기 blank 기억은 어떤가? | `task_metrics.csv`, `blank_metrics.csv` |
| `fig_B_blank_curves` | blank horizon에 따라 기억이 어떻게 변하는가? | `blank_metrics.csv` |
| `fig_C_geometry` | fiber, local rank, PCA PR은 어떤가? | `geometry_metrics.csv` |
| `fig_D_dynamics` | tangent와 sampled hidden-normal gain은 어떤가? | `dynamics_metrics.csv` |
| `fig_E_endpoint_pca` | 1,024점 transported hidden atlas가 PC1–3에서 어떤 3D 구조를 이루는가? | frozen bank와 checkpoint 재계산 |
| `fig_F_sampled_normal_recovery` | 무작위 hidden-normal kick에서 복귀하는가? | `dynamics_metrics.csv`와 per-run NPZ |
| `fig_G_topology_radial_recovery` | ring/torus/sphere의 radius 방향으로 복귀하는가? | `topology_radial_metrics.csv`와 per-run JSON |
| `fig_H1_s1_pca_evolution` | \(S^1\) hidden atlas가 blank 시간에 따라 수축·분열하는가? | 모델별 best-validation task-success seed |
| `fig_H2_t2_pca_evolution` | \(T^2\) 주기 격자가 blank 시간에 따라 접히거나 붕괴하는가? | 성공 seed 우선; 없으면 best-available 실패 fallback |
| `fig_H3_s2_pca_evolution` | \(S^2\) hidden atlas가 blank 시간에 따라 구면 topology를 유지하는가? | 성공 seed 우선; 없으면 best-available 실패 fallback |

표 입력은
[`../evidence/topology/all_models_v1/`](../evidence/topology/all_models_v1/)에
있다. `fig_F`와 `fig_G`는 다른 진단이다. `fig_F`는 추정 tangent에 직교한
무작위 hidden 방향이고, `fig_G`는 decoder가 정의하는 topology radius 방향이다.

### Persistent topology

현재 경로: [`topology/persistence_v2/`](topology/persistence_v2/)

| 파일 | 내용 |
|---|---|
| `fig_signature_representatives` | 성공 우선 모델별 representative의 expected \(H_1/H_2\) signature |
| `fig_signature_all_seeds` | 실패 seed를 포함한 3-seed robustness 보조 분석 |
| `fig_diagram_distance` | 이상적 persistence diagram까지 bottleneck distance |
| `fig_diagrams_h0` | blank 이전 representative persistence diagrams |
| `fig_diagrams_h4096` | 4,096 blank step 이후 diagrams |

수치는
[`../evidence/topology/persistence_v2/`](../evidence/topology/persistence_v2/)에
있다.

### OOD와 일반화

현재 경로: [`topology/ood_v2/`](topology/ood_v2/)

| 파일 | 내용 |
|---|---|
| `fig_O1_temporal_generalization` | 동일한 128개 command·endpoint·path를 최대 16배 긴 시간에 배치한 primary temporal OOD |
| `fig_O2_cumulative_path_stress` | active density를 유지하며 누적 path/winding을 늘린 별도 stress test |
| `fig_O3_velocity_generalization` | 동일 trajectory primitive를 0.5×–2×로 스케일한 paired velocity OOD |
| `fig_O4_dwell_generalization` | dense, fixed activity probability, 64-step middle blank 비교 |
| `fig_O4b_smoothness_generalization` | GP length scale 변화에 대한 control smoothness OOD |
| `fig_O5_combined_and_postblank` | length×velocity 복합 stress와 이후 512-step blank 기억 |

대표선은 task-success seed 우선이며, 성공 seed가 없는 모델×topology는 점선과
fallback 이름으로 표시한다. 수치와 전체 3-seed 범위는
[`../evidence/topology/ood_v2/`](../evidence/topology/ood_v2/)에 있다.
`ood_v1`은 elapsed time과 누적 path를 분리하지 못한 pilot으로만 보존한다.

## 재생성·동기화

Generator:

- 모델 비교:
  [`compare_all_topology_models.py`](../../repro/manifold_benchmark/compare_all_topology_models.py)
- persistent topology:
  [`analyze_persistent_topology.py`](../../repro/manifold_benchmark/analyze_persistent_topology.py)
- OOD bank와 평가:
  [`build_ood_banks.py`](../../repro/manifold_benchmark/build_ood_banks.py),
  [`analyze_ood_generalization.py`](../../repro/manifold_benchmark/analyze_ood_generalization.py)

Frozen experiment에서 논문 디렉터리로 복사할 때는 수동 `cp` 대신 다음을 쓴다.

~~~bash
make artifacts
make check-artifacts
~~~

source와 destination의 전체 대응은
[`configs/paper_artifacts.json`](../../configs/paper_artifacts.json), 파일별
checksum은 [`paper/artifact_checksums.json`](../artifact_checksums.json)에 있다.

## 추가 규칙

Figure를 추가할 때 한 artifact group에 다음을 함께 등록한다.

- PDF와 GitHub preview용 PNG
- generator 경로
- seed-level CSV 또는 원 분석 파일
- concept/measured 구분과 seed 수
- frozen experiment ID

PCA는 시각적 진단이지 topology의 증명이 아니다. `fig_H*`는 1,024개 hidden
endpoint를 \(H=\{0,128,512,1024,2048,4096\}\)에서 추적한다. 각 모델 행에서
\(H=0\)의 PCA 중심과 PC1--PC3 축을 모든 horizon에 고정하고 공통 축 범위를
사용하므로 bulk translation, 수축, 팽창, 분열을 함께 보존한다. 각
모델×topology에서 task gate를 통과한 seed 중 validation error가 가장 작은
checkpoint를 사용한다. 성공 seed가 하나도 없으면 validation error가 가장
작은 checkpoint를 fallback으로 보여주되 빨간색 `failed task gate`로
표시한다. AAAI 공개 전 PDF metadata와 절대경로를 다시 검사한다.
