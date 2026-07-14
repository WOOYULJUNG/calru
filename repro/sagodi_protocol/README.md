# Ságodi-aligned CA-LRU protocol

이 패키지는 `CA_LRU_Sagodi_Experimental_Protocol_ko.md`의 전체 Cartesian
product를 한 번에 실행하지 않는다. 현재 freeze는 선행 상태 감사와 ring pilot만
허용한다.

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
- `EXPERIMENT_RECONSTRUCTION_ko.md`: 범위와 해석 규칙
- `CALRU_NATIVE_RECIPE.md`: Exp88 성공 recipe를 Ságodi task에 이식하는
  별도 pilot의 provenance, Protocol-A/B 경계, sentinel-first 실행 규칙
- `tasks.py`: Ságodi MGS 및 single/double angular-integration bank
- `models.py`, `state.py`: 모델 registry와 최소 Markov-state adapter
- `audit.py`, `phase0.py`: 실제 `step(0,state)` 선행 감사
- `train.py`: freeze-driven Adam/AdamW, paired online batches, optional state
  noise, gradient clipping, RP schedule와 atomic progress telemetry
- `phase1_analysis.py`: Track-A slow-state reconstruction, 8-path settling/fiber,
  독립 8-path known-q projection QA, local-SVD C1 rank, 매-step projected-normal
  cocycle와 C1–C3 pilot 분석
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

Sampled normal gate의 primary 값은 actual clean rollout의 projector frame에서 매 step
`P_N`을 적용한 radial+ambient cocycle이다. Tangent 값은 endpoint block
`T(q_H)^T J_{0:H} T(q_0)`을 사용한다. Normal을 endpoint에서 한 번만 투영한 값과
중간 `P_T`를 넣은 tangent 값은 비주장 diagnostic으로 분리된다. 또한 sampled
normal의 tangent 내적 최대가 frozen QA `1e-6`을 넘으면 관련 gate는
`inconclusive`/`passed=null`이다.

CA-LRU/No-RP의 현재 scaffold에서 blank carrier map은 정확히
`F0(h) = Lambda h`다. 비선형 writer는 input-conditioned이며, 결과를 nonlinear
autonomous restoring field 또는 exact nonzero continuum로 해석하면 안 된다.
