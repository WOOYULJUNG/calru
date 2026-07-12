# Evidence provenance

## Snapshot 성격

이 evidence bundle은 2026-07-12에 기존 연구 작업공간에서 선별한 결과 snapshot이다.
원 작업공간은 Git worktree가 아니었으므로 원 실행을 가리키는 commit hash가 없다.
따라서 이 저장소의 최초 import commit은 **정리된 artifact의 시작점**일 뿐, 과거
학습 run 당시 코드 revision을 증명하지 않는다.

Raw 디렉터리 이름과 filename은 source 추적을 위해 legacy 형태를 유지한다. 문서와
aggregate table은 canonical paper-facing 이름을 사용한다. 두 이름의 매핑은
[`docs/LEGACY_NAME_MAP.md`](../../docs/LEGACY_NAME_MAP.md)에 고정한다.

## Raw snapshot 구성

| repo-relative source | 파일 수 | 역할 |
|---|---:|---|
| `raw/exp88_manifold_main_results/` | 240 | standard baselines, linear update, legacy RP controls |
| `raw/exp88_writer_sweep_results/` | 162 | final CA-LRU, input-only update, RP score shuffles |
| `raw/exp72_line_integrate_10k_d*_camn_rnw_eps1e-4_results/` | 15 | d=1…16 retention scaling |
| `raw/exp89_discrete_kway_quick_results/` | 37 | K-way와 flip-flop scope controls |
| `raw/analysis_exp88_ring_transport/` | 3 | seed 0/1/2 ring transport CSV |

이 snapshot은 checkpoint와 lambda trace를 포함하지 않는다. 따라서 JSON/CSV에서
표를 재집계할 수는 있지만 checkpoint를 다시 로드해 모든 진단을 재계산할 수는
없다.

총 457개 raw input의 repo-relative 경로, byte 수, SHA-256과 legacy source path는
[`raw/file_manifest.csv`](raw/file_manifest.csv)에 있다. 동일 digest는
[`raw/checksums.sha256`](raw/checksums.sha256)에서도 검사할 수 있다. 두 파일은
`scripts/build_raw_manifest.py`가 결정적으로 생성하며 checksum 계산 대상에서는
제외된다.

## Raw에서 table까지의 변환

1. `configs/evidence_manifest.json`이 기대 seed와 task별 선택 epsilon을 정의한다.
2. `src/calru_paper/evidence.py`가 repo-relative glob으로 raw record를 읽는다.
3. 각 metric을 float로 변환해 seed 산술평균과 표본 표준편차를 계산한다.
4. identity, seed IDs와 `source_pattern`을 함께 `tables/*.csv`에 쓴다.
5. 재현성 검사는 임시 출력과 committed tables의 schema와 값을 비교한다.

Raw metric을 다시 계산하거나 보간하지 않으며, aggregate 단계에서 model ranking이나
통계적 유의성을 새로 추론하지 않는다. Table을 원고에 옮길 때 반올림하더라도
committed CSV의 full-precision 값을 근거로 남긴다.

## Model lineage

- final CA-LRU는 `PAN-RNW-full`과 `am_lru_rnw_*`/일부 `camn_*` raw tag에서 왔다.
- input-only control은 `PAN-NW-full`과 `am_lru_nw_*`다.
- linear-update control은 `PAN-full`과 `am_lru_*`다.
- `am_lru_eta0`와 `am_lru_allslow`는 final state-dependent CA-LRU가 아니라
  `PAN-full` linear scaffold다.
- `camn_shufmatch`는 fixed score permutation이지만 persistent count가 matched된
  control은 아니다.

PAN, CAMN, AM-LRU라는 문자열은 모델의 현재 이름이 아니라 역사적 filename과 코드
호환성을 위한 provenance다.

## Evidence tier

Tier 이름과 table-level 배정의 source of truth는 `configs/evidence_manifest.json`이다.

### Primary

- 8-task main CA-LRU/baseline 결과, 각 seed 0/1/2
- 4 integration task의 update-term ablation
- ring integration temporal/velocity/repeated-kick OOD
- 별도 Exp72 retention scaling

### Supporting

- final scaffold의 aligned/fixed-permutation/fresh-shuffle RP-score controls
- 8-task epsilon sensitivity
- `rp_control_metrics.csv`의 legacy linear-scaffold 행은 같은 table에 있지만 더 좁은
  해석 제한을 유지함

### Diagnostic

- 3-seed ring transport: 단일 base angle, 1,440-point grid, 비정규화 state distance

### Exploratory

- discrete K-way/flip-flop scope control

### Legacy row와 protocol의 추가 제한

- RP-control table에 포함된 `PAN-full` no-RP/all-slow controls
- final CA-LRU가 아닌 linear-update scaffold의 Jacobian/selective-persistence 결과
- main battery와 분포·metric이 다른 과거 line/ring protocol

No-RP/all-slow seed rows는 RP-control table에 provenance 목적으로 포함되지만 final
scaffold causal ablation은 아니다. Jacobian/selective-persistence와 old protocol은
현재 9개 committed table의 main claim으로 집계하지 않는다.

### Import에서 제외한 결과

- smoke와 pilot run
- seed aggregation rule이 일관되지 않은 과거 표
- seed-0 qualitative plot만으로 모든 manifold의 일반적 정량 주장을 만든 경우

`rp_control_metrics.csv`에는 final-scaffold shuffle과 legacy linear-scaffold 행이
함께 있으므로 table 전체에 하나의 causal 해석을 적용하면 안 된다.

## 기록된 실행 환경

원 실험 환경에서 확인된 핵심 버전은 다음과 같다.

| 항목 | 값 |
|---|---|
| Python | 3.10.20 |
| PyTorch | 2.4.0+cu121 |
| NumPy | 1.26.4 |
| Matplotlib | 3.5.3 |
| GPU | NVIDIA TITAN RTX 24 GB, 최대 6장 사용 |
| driver | 550.107.02 |

GPU가 공유되었으므로 raw `seconds`는 관측 wall time일 뿐 공정한 throughput
benchmark가 아니다. CUDA/cuDNN deterministic 설정과 모든 system package는 완전한
lockfile로 보존되지 않았다.

## 재현성과 주장에 남은 간극

- train/validation/test split과 fixed evaluation set이 없다.
- task별 epsilon을 held-out rule로 선택하지 않았다.
- architecture마다 RNG 소비가 달라 모델 간 evaluation sample이 paired되지 않았다.
- final `PAN-RNW-full`에서 matched no-RP/all-slow ablation이 없다.
- main normal kick은 overwrite되는 stream을 포함한 full state에 적용된다.
- final CA-LRU의 carrier-only blank Jacobian이 없다.
- Mamba, RG-LRU, LinOSS 같은 현대 baseline이 없다.

이 제한은 데이터 결함을 숨기기 위한 것이 아니라, 각 표가 허용하는 주장 범위를
명확히 하기 위한 것이다.

## 변경 정책

- `raw/`는 불변 snapshot으로 취급한다.
- `tables/`는 집계 코드와 manifest를 통해서만 갱신한다.
- source record를 교체하면 변경 이유, 이전/새 파일 식별자와 checksum을 함께
  기록한다.
- 새 학습 run은 기존 raw 디렉터리에 바로 섞지 않고 별도 snapshot으로 import한다.
- 라이선스가 확정되기 전에는 이 bundle의 공개 배포와 재사용 조건이 정해진 것으로
  간주하지 않는다.
