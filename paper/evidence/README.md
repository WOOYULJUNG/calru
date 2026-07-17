# Paper evidence bundle

이 디렉터리는 저장된 seed-level 결과와 그 결과에서 만든 논문용 표를 함께
보존한다. 목표는 **학습을 다시 실행하지 않고도 논문 표의 집계 과정을 감사하고
재현**하는 것이다.

## 디렉터리

- [`raw/`](raw/): legacy 실험에서 가져온 불변 JSON/CSV snapshot
- [`tables/`](tables/): raw snapshot을 평균과 표본 표준편차로 집계한 9개 CSV
- [`topology/`](topology/): 45-checkpoint topology 비교와 persistent-topology
  분석의 frozen export
- [`hc_local_grid_v2/`](hc_local_grid_v2/): 최종 H-C 후보를 고른 24-run 개발
  sweep과 checkpoint-only CA 분석 snapshot
- [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md): metric, task, column의 정확한 뜻
- [`PROVENANCE.md`](PROVENANCE.md): legacy source, 변환 과정, evidence tier와 한계
- [`raw/file_manifest.csv`](raw/file_manifest.csv): raw 파일의 크기·SHA-256·legacy
  source path
- [`raw/checksums.sha256`](raw/checksums.sha256): `sha256sum` 호환 checksum 목록
- [`../../docs/LEGACY_NAME_MAP.md`](../../docs/LEGACY_NAME_MAP.md): 논문명과 코드명
  매핑

Raw 파일은 수동 편집하지 않는다. 잘못된 source record가 발견되면 기존 snapshot을
조용히 고치는 대신 정정 이유, 교체 파일과 checksum을 provenance에 추가한 새
snapshot으로 다룬다.

`topology/`도 수동으로 복사하지 않는다. 루트에서 `make artifacts`로
`configs/paper_artifacts.json`에 고정된 source만 동기화하고,
`make check-artifacts`로 source와 committed export의 SHA-256 일치를 검사한다.

## 집계 재현

루트에서 개발 dependency를 설치하고 다음을 실행한다.

~~~bash
python -m pip install -e ".[dev]"
make check
~~~

`make check`는 임시 위치에서 표를 다시 만든 뒤 committed tables와 비교하는 검증
경로다. 실제 committed tables를 재생성할 때만 다음을 사용한다.

~~~bash
make evidence
~~~

Raw provenance를 갱신하거나 현재 상태만 검사하려면 다음을 사용한다.

~~~bash
make provenance
make check-provenance
~~~

집계의 source of truth는 `configs/evidence_manifest.json`,
`src/calru_paper/evidence.py`, 그리고 `raw/`다. 기본 규칙은 seed 0/1/2의
산술평균과 표본 표준편차(`statistics.stdev`, `ddof=1`)다. 필수 aggregate row가
세 seed를 모두 포함하지 않으면 검증은 실패해야 한다.

## 생성되는 표

| 파일 | 내용 | 기본 증거 등급 |
|---|---|---|
| `main_manifold_metrics.csv` | 8개 manifold task의 CA-LRU와 recurrent baseline | primary |
| `update_ablation_metrics.csv` | state-dependent/input-only/linear update | primary |
| `ring_integrate_ood_metrics.csv` | temporal, velocity, perturbation OOD | primary |
| `epsilon_sensitivity.csv` | 8 tasks × 3 RP thresholds | supporting |
| `rp_control_metrics.csv` | final-scaffold score shuffle와 legacy linear controls | supporting |
| `retention_scaling_metrics.csv` | 별도 line-integration protocol의 d=1…16 | primary |
| `ring_transport_seed_values.csv` | ring transport의 seed-level 값 | diagnostic |
| `ring_transport_summary.csv` | ring transport의 3-seed 요약 | diagnostic |
| `discrete_control_metrics.csv` | K-way/flip-flop scope control | exploratory |

증거 등급은 우열 점수가 아니라 주장 가능한 범위를 나타낸다.

- **primary:** main empirical claim에 직접 연결하는 3-seed 결과
- **supporting:** main claim을 한정하거나 ablation/sensitivity를 보조하는 결과
- **diagnostic:** main battery보다 좁은 post-hoc dynamical protocol
- **exploratory:** 논문의 scope를 정하지만 main positive claim으로 쓰지 않는 control

`rp_control_metrics.csv`는 table-level로 supporting이지만, 그 안의 no-RP/all-slow
행은 final CA-LRU가 아닌 legacy linear scaffold다. Table-level tier가 이 행들을
final-scaffold causal ablation으로 바꾸지는 않는다. Smoke/pilot과 집계가 불일치한
과거 결과는 이 bundle에 claim-bearing table로 import하지 않았다.

`hc_local_grid_v2/`는 seeds 0–2를 사용해 architecture hyperparameter를 선택한
development/diagnostic evidence다. 이 snapshot의 1위 조건을 독립 confirmatory
결과처럼 취급하거나 trajectory를 독립 replicate로 세면 안 된다.

## 집계와 전체 학습은 다르다

여기의 명령은 이미 저장된 metric을 다시 계산해 표로 만들 뿐, 신경망을
재학습하거나 checkpoint에서 진단을 다시 실행하지 않는다. 학습 코드 snapshot은
[`repro/`](../../repro/)에 있지만 checkpoint와 trace가 없고 전체 실행이 아직
공개 환경에서 검증되지 않았다. 따라서 표가 정확히 재생성되는 것과 모델 학습이
완전히 재현되는 것을 구분해 보고한다.

## 알려진 제한

- task별 선택 epsilon은 held-out validation으로 정하지 않았다.
- 모델 간 evaluation sample은 고정·paired되지 않았다.
- normal-kick main metric은 clean drift와 perturbation recovery가 섞인 target
  RMSE이며 pure normal contraction rate가 아니다.
- final-scaffold matched no-RP/all-slow control은 없다.
- modern baseline(Mamba, RG-LRU, LinOSS 등)은 포함되지 않았다.
