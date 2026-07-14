# Ságodi-based CA-LRU evaluation protocol

결과 생성에 사용할 현재 경로는 `Ságodi-primary v3`다. Ságodi et al.의
방법은 CA-LRU의 제안 방법이 아니라 continuous-attractor 특성을
검사하는 평가 도구로만 사용한다. 정확한 범위·모호성 해소·수치 규칙은
`SAGODI_PRIMARY_V3_FREEZE_ko.md`에 고정했다.

```text
Phase 0: six-model full-Markov-state / blank-map audit
    ↓
120-run LR selector: 6 models × 5 selection seeds × 4 LRs × 100 updates
    ↓ verified selector receipt + atomic resolved-main freeze
Primary main training: 6 models × 10 seeds × 5,000 updates
    ↓ validation NMSE < -20 dB (all attempted seeds still reported)
Ságodi-based primary analysis: slow spline, full spectrum, projected drift,
fixed-point topology, finite/asymptotic memory
    ↓
Engineering benefit: temporal/velocity OOD and state-perturbation retention
    ↓
Descriptive dynamics–utility association (no causal claim or binary gate)
```

Primary six-model comparison is parameter-near-matched:
`RNN-206`, official-style `GRU-135`, `LSTM-109`, existing `LRU full-96`,
`CA-LRU without RP-96`, and `CA-LRU-96`. RNN/LSTM/LRU are explicitly project
baselines, not bit-exact Ságodi architectures.

현재 v3 파일:

- `sagodi_primary_lr_selection_v3.yaml` / `.json`: 120-run selector freeze
- `lr_selection_v3.py`: strict, resumable selector campaign
- `primary_main_template_v3.json` / `primary_main_campaign.py`: selector-bound
  60-run main training
- `sagodi_primary_analysis.py`: Ságodi-primary numerical definitions
- `sagodi_primary_runner.py`: one-checkpoint primary analysis
- `primary_analysis_campaign.py`: 60-checkpoint analysis and descriptive aggregation
- `engineering_benefit_freeze_v1.json`, `engineering_benefit_runner.py`,
  `engineering_benefit_campaign.py`: 별도 공학 효용 평가
- `dynamics_utility_association_freeze_v1.json`,
  `dynamics_utility_association.py`: seed-level 기술적 연관 분석
- `primary_v3_pipeline.py`: 위 다섯 단계를 같은 clean commit에서 순차 실행·재개
- `phase1_analysis.py`: historical CA-LRU-specific **supplementary** diagnostics only

아래 v1/v2 설명은 과거 artifact의 provenance와 재분석을 위해 보존한다.

## v3 실행

전체 v3는 하나의 clean commit에서 selector → main → primary
analysis → engineering benefit → descriptive association 순으로
실행한다. 어느 단계에서든 코드를 바꾸면 exact-commit 검증이
후속 실행을 중단한다.

```bash
python -m repro.sagodi_protocol.lr_selection_v3 \
  --selector repro/sagodi_protocol/sagodi_primary_lr_selection_v3.json \
  --protocol repro/sagodi_protocol/sagodi_primary_lr_selection_v3.yaml \
  --artifact-root /path/to/v3/selector \
  --python /path/to/python \
  --gpus 0,1,2,3,4,5

python -m repro.sagodi_protocol.primary_main_campaign \
  --selector-root /path/to/v3/selector \
  --template repro/sagodi_protocol/primary_main_template_v3.json \
  --artifact-root /path/to/v3/main \
  --python /path/to/python \
  --gpus 0,1,2,3,4,5

python -m repro.sagodi_protocol.primary_analysis_campaign \
  --main-root /path/to/v3/main \
  --artifact-root /path/to/v3/primary_analysis \
  --python /path/to/python \
  --gpus 0,1,2,3,4,5

python -m repro.sagodi_protocol.engineering_benefit_campaign \
  --main-root /path/to/v3/main \
  --selector-root /path/to/v3/selector \
  --freeze repro/sagodi_protocol/engineering_benefit_freeze_v1.json \
  --artifact-root /path/to/v3/engineering_benefit \
  --python /path/to/python \
  --gpus 0,1,2,3,4,5

python -m repro.sagodi_protocol.dynamics_utility_association \
  --primary-root /path/to/v3/primary_analysis \
  --engineering-root /path/to/v3/engineering_benefit \
  --main-root /path/to/v3/main \
  --selector-root /path/to/v3/selector \
  --freeze repro/sagodi_protocol/dynamics_utility_association_freeze_v1.json \
  --artifact-root /path/to/v3/dynamics_utility_association
```

권장 실행법은 영수증 검증과 재개를 포함한 단일 launcher다.

```bash
python -m repro.sagodi_protocol.primary_v3_pipeline \
  --artifact-root /path/to/v3 \
  --python /path/to/python \
  --gpus 0,1,2,3,4,5
```

`--dry-run`은 artifact를 만들거나 학습을 실행하지 않고 정확한 다섯 단계
명령만 출력한다.

Selector smoke는 120개 경로와 receipt를 검사하지만 2 updates로
축소되므로 LR winner를 생성하지 않는다. Main은 반드시 100-step
full selector의 120개 verified receipt와 6개 winner가 있어야 시작한다.
따라서 pipeline launcher는 과학적으로 잘못된 smoke 체인을 제공하지 않는다.

## Legacy v1/v2 (provenance only)

아래 설명은 과거 artifact를 재현·감사하기 위한 legacy v1/v2 경로다.
Legacy freeze는 선행 상태 감사와 ring pilot만 허용했다.

```text
Phase 0: actual Markov state / blank map audit
    ↓ pass
Phase 1: angular-integration pilot (CA-LRU, No-RP, GRU; 5 pilot seeds)
    ↓ human review + new freeze
Phase 2–4: disabled
```

핵심 파일:

- `analysis_protocol.yaml`: 실행 가능한 단일 freeze와 claim/QA threshold
- `calru_native_sagodi_ring_pilot_v1.yaml`: 기존 성공 CA-LRU recipe를
  Ságodi ring task에 이식한 별도 pilot freeze (sentinel 1 + 자동 fan-out 14)
- `calru_native_sagodi_ring_analysis_v2.yaml`: immutable v1 checkpoint를
  별도 artifact root에서 교정된 진단으로만 재분석하는 parent-bound freeze
- `sagodi_ring_lr_selection_v2.yaml`: Ságodi paper-aligned 100-update
  learning-rate selection 전용 freeze (4 models x 5 seeds x 4 LRs = 80 runs)
- `CORRECTIVE_ACTION_PLAN_V2_ko.md`: 외부 검토 항목과 v1/v2 실행 경계
- `EXPERIMENT_RECONSTRUCTION_ko.md`: 범위와 해석 규칙
- `CALRU_NATIVE_RECIPE.md`: Exp88 성공 recipe를 Ságodi task에 이식하는
  별도 pilot의 provenance, Protocol-A/B 경계, sentinel-first 실행 규칙
- `tasks.py`: Ságodi MGS 및 single/double angular-integration bank
- `models.py`, `state.py`: 모델 registry와 최소 Markov-state adapter
- `audit.py`, `phase0.py`: 실제 `step(0,state)` 선행 감사
- `train.py`: freeze-driven Adam/AdamW, paired online batches, optional state
  noise, gradient clipping, RP schedule와 atomic progress telemetry
- `phase1_analysis.py`: task endpoint에서 시작하는 Track-A slow-state
  reconstruction을 primary manifold로 사용하고, 8-path settling/fiber를 독립
  correspondence QA로 유지하며, multi-horizon manifold-distance C3와
  매-step projected-normal cocycle를 계산
- `manifold_diagnostics.py`: `D_clean`, `Q_recovery`, same-memory와 양방향
  settling/expansion/manifold-adherence의 순수 진단 함수
- `analysis_freeze.py`: v2 분석 freeze의 strict loader와 canonical fingerprint
- `reanalyze.py`: 완료된 immutable v1 checkpoint만 별도 root에서 교정 진단
- `lr_selection.py`: Phase 0 뒤 80개 training-only selector run을 GPU queue로
  실행하고 update 100 loss의 5-seed 산술평균으로 모델별 LR을 freeze
- `orchestrate.py`, `status.py`: GPU queue, source lock, atomic attempt,
  checkpoint-bound receipt와 독립 상태 검증
- `aggregation.py`: 15개 pilot run의 all-started/task-success-conditional 기술통계

## 검증

저장소 root에서 다음을 실행한다.

```bash
PYTHONNOUSERSITE=1 python -m pytest -q repro/sagodi_protocol/tests
python -m repro.sagodi_protocol.config
```

전체 캠페인 전에 별도 artifact root로 smoke를 실행한다.

```bash
python -m repro.sagodi_protocol.orchestrate \
  --artifact-root /tmp/calru_sagodi_smoke \
  --python /path/to/python \
  --gpus 0,1,2,3,4,5 \
  --smoke
```

## 전체 pilot

```bash
python -m repro.sagodi_protocol.orchestrate \
  --artifact-root /path/to/calru_sagodi_phase01 \
  --python /path/to/python \
  --gpus 0,1,2,3,4,5
```

읽기 전용 상태 확인:

```bash
python -m repro.sagodi_protocol.status /path/to/calru_sagodi_phase01
```

완료는 파일 존재만으로 인정하지 않는다. 각 training/analysis job과 Phase 0의
completion receipt가 모든 필수 artifact SHA-256과 일치해야 하며, analysis receipt의
checkpoint SHA-256은 현재 training receiptㆍmanifestㆍ`checkpoint.pt`와 모두 같아야 한다. 중단·손상된 output은
`attempts/` 아래 보존하고 새 attempt만 원자적으로 final 경로에 publish한다. 같은
artifact root에는 PID뿐 아니라 process 시작 시점과 boot ID까지 확인하는 한
orchestrator만 lock을 획득할 수 있다. 죽은 owner의 stale lock은 삭제하지 않고
`attempts/campaign_setup/orchestrator_lock/`에 보존한 뒤 재시도한다. lock은 삭제하지
않는 `.orchestrator.guard` inode에 `flock`을 건 상태에서 token payload를 원자적으로
쓴다. 따라서 payload 생성과 쓰기 사이의 빈 파일을 다른 process가 stale lock으로
오인할 수 없다.

모든 Phase-1 analysis receipt가 검증된 뒤에만 `pilot_aggregation/`을 원자적으로
publish한다. `pilot_summary.json`은 모델별 all-started 분모와 task-success 조건부
분모, 각 gate status/value 및 유한 scalar의 mean/sample standard deviation/median/IQR을
기록하며 `pilot_run_matrix.csv`는 run별 gate matrix다. 이는 pilot seed 5개에 대한
비확증적 기술통계이고 p-value를 계산하지 않는다. 두 파일의 SHA-256은 `COMPLETE`와
`status` 검증에 포함되므로 완료 후 파일이 바뀌면 campaign은 더 이상 complete가 아니다.

현재 pilot은 path/fiber와 sampled normal gate까지 평가하지만 C4, 10 main seeds,
strict worst-normal operator와 전체 radius/horizon sensitivity sweep을 수행하지 않는다.
따라서 `claim_gate.json`의 L3는 명시적으로 false이며, pilot 수치만으로
approximate-CA 최종 주장을 하지 않는다.

v2 C3의 primary 값은 고정된 Track-A manifold까지의 거리다. 등록 horizon
`1,5,20,100,500,1024`에서 clean adherence를 확인하고, H=500에서 radial 및
ambient-normal `Q_recovery`를 따로 gate한다. 과거 clean-paired endpoint-normal
deviation은 diagnostic으로만 저장되며 C3를 만족시킬 수 없다.

## v2 실행 명령

LR selection은 manifold 분석이나 CA evidence를 만들지 않는다. smoke는 80개
경로와 receipt를 검증하지만 winner를 선택하지 않고 `freeze_eligible=false`로
기록한다. 정식 선택에서도 CA-LRU의 RP는 100-update selector 동안 0회로
고정한다. 모델 4종, 선택 seed 5개, LR 4개의 80개 run 모두가 검증된 성공
receipt를 가져야만 모델별 winner를 선택한다. nonzero exit, OOM, kill 또는
손상된 receipt는 과학적 LR 실패로 세지 않고 캠페인을 중단한 뒤 재개 대상으로
남긴다.

```bash
python -m repro.sagodi_protocol.lr_selection \
  --protocol repro/sagodi_protocol/sagodi_ring_lr_selection_v2.yaml \
  --artifact-root /path/to/calru_sagodi_lr_selection_v2 \
  --python /path/to/python \
  --gpus 0,1,2,3,4,5
```

v1 checkpoint 재분석은 parent campaign을 수정하지 않으며, 기본 모드는 frozen
15-run matrix가 모두 완료되어야 시작한다. 진행 중인 parent를 점검하는
`--available-only` 결과는 부분 결과이고 `COMPLETE`를 만들 수 없다.

```bash
python -m repro.sagodi_protocol.reanalyze \
  --parent-root /path/to/calru_native_sagodi_ring_pilot_v1-668867dfa36a \
  --artifact-root /path/to/calru_native_sagodi_ring_analysis_v2 \
  --protocol repro/sagodi_protocol/calru_native_sagodi_ring_pilot_v1.yaml \
  --analysis-freeze repro/sagodi_protocol/calru_native_sagodi_ring_analysis_v2.yaml \
  --python /path/to/python \
  --gpus 0,1,2,3,4,5
```

Sampled normal gate의 primary 값은 actual clean rollout의 projector frame에서 매 step
`P_N`을 적용한 radial+ambient cocycle이다. Tangent 값은 endpoint block
`T(q_H)^T J_{0:H} T(q_0)`을 사용한다. Normal을 endpoint에서 한 번만 투영한 값과
중간 `P_T`를 넣은 tangent 값은 비주장 diagnostic으로 분리된다. 또한 sampled
normal의 tangent 내적 최대가 frozen QA `1e-6`을 넘으면 관련 gate는
`inconclusive`/`passed=null`이다.

CA-LRU/No-RP의 현재 scaffold에서 blank carrier map은 정확히
`F0(h) = Lambda h`다. 비선형 writer는 input-conditioned이며, 결과를 nonlinear
autonomous restoring field 또는 exact nonzero continuum로 해석하면 안 된다.

## v3 동역학–효용 연관 분석 (descriptive only)

`dynamics_utility_association.py`는 완료된 Ságodi-primary 60개 결과와
engineering-benefit 60개 결과를 정확한 `(model_id, model_seed)`로 결합한다.
등록된 60개 pair를 하나도 삭제하거나 대체하지 않으며, ineligibleㆍnot-estimableㆍ
metric-missing 상태를 각 연관 분석의 row와 분모에 그대로 기록한다. Pearson과
average-rank Spearman 및 고정 seed paired-bootstrap 구간은 기술통계일 뿐이며,
인과 주장, p-value, CA gate 또는 예상 방향 pass threshold로 사용하지 않는다.

Top-two real-part gap은 spline tangent 정렬을 직접 측정한 값이 아니라
slow-leading-mode 가정 아래의 Ságodi-style timescale-separation proxy로 명시한다.
전체 단계는 selector/main/두 parent campaign과 동일한 clean Git commit에서만
실행된다.

```bash
python -m repro.sagodi_protocol.dynamics_utility_association \
  --primary-root /path/to/v3/primary_analysis \
  --engineering-root /path/to/v3/engineering_benefit \
  --main-root /path/to/v3/main \
  --selector-root /path/to/v3/selector \
  --freeze repro/sagodi_protocol/dynamics_utility_association_freeze_v1.json \
  --artifact-root /path/to/v3/dynamics_utility_association
```
