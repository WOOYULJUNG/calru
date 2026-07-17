# Current experiment catalog

이 문서는 현재 논문 후보 결과만 가리킨다. 과거 smoke/sweep의 전체 목록이 아니다.

## Topology benchmark

| ID | 역할 | 로컬 source |
|---|---|---|
| `generator_v1_rodrigues` | frozen \(S^1,T^2,S^2\) validation bank | `experiments/manifold_benchmark_generator_v1_rodrigues-20260716` |
| `generator_v1_rodrigues-test` | frozen test bank | `experiments/manifold_benchmark_generator_v1_rodrigues-test-20260716` |
| `baseline_all_v2` | RNN/GRU/LSTM 3 topology × 3 seeds 분석 | `experiments/manifold_topology_transfer_v1-pilot-1f4a80d/analysis_all_v2` |
| `selected_hparam_v1` | CA-LRU/H-C topology별 선택 checkpoint 분석 | `experiments/manifold_topology_hparam_v1-c6dc758/analysis_final_v1` |
| `all_models_v1` | 위 두 분석의 45-checkpoint 통합 비교 | `experiments/manifold_topology_hparam_v1-c6dc758/comparison_all_models_v1` |
| `persistent_topology_v1` | 동일 checkpoint의 \(H_0,H_1,H_2\) 분석 | `experiments/persistent_topology_all_models_v1` |

로컬 경로의 공통 prefix는 `/home/biadmin/ca_rnn/`이다.

## 논문용 canonical export

| 질문 | 먼저 볼 파일 |
|---|---|
| 전체 모델 수치 | `paper/evidence/topology/all_models_v1/summary.csv` |
| seed별 figure 입력 | `paper/evidence/topology/all_models_v1/seed_metrics.csv` |
| radius-normal 수축 | `paper/evidence/topology/all_models_v1/topology_radial_metrics.csv` |
| global topology 유지 | `paper/evidence/topology/persistence_v1/summary.csv` |
| 현재 figure 목록 | `paper/figures/README.md` |

## 비교 재생성

~~~bash
python -m repro.manifold_benchmark.compare_all_topology_models \
  --baseline-analysis \
  /home/biadmin/ca_rnn/experiments/manifold_topology_transfer_v1-pilot-1f4a80d/analysis_all_v2 \
  --selected-analysis \
  /home/biadmin/ca_rnn/experiments/manifold_topology_hparam_v1-c6dc758/analysis_final_v1 \
  --output \
  /home/biadmin/ca_rnn/experiments/manifold_topology_hparam_v1-c6dc758/comparison_all_models_v1

make artifacts
make check-artifacts
~~~

`fig_F_sampled_normal_recovery`와 `fig_G_topology_radial_recovery`를 혼용하지
않는다. 전자는 hidden normal complement의 sampled direction, 후자는
ring/sphere 한 개와 torus 두 개의 decoded radius 방향이다.
