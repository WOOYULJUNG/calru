# Experimental v2: P0 confirmatory training

이 디렉터리는 기존 Exp88 결과와 체크포인트를 건드리지 않고 추가 P0 실험을
실행하기 위한 격리된 launcher다. 공유 model builder에는 opt-in `RG-LRU-full`과
`GRU-full`만 추가했으며, 기존 variant와 historical default sweep은 그대로 유지한다.

## 무엇을 실행하는가

기본 campaign은 `ring_hold`, `torus_integrate`와 seeds 0/1/2에서 다음 30개
training run을 만든다.

| family | condition | runs | retention 학습 | H=500 정보 |
|---|---|---:|---|---|
| RP | `rp_aligned` | 6 | ablation-damage RP | 있음 |
| RP control | `rp_off_frozen_retention` | 6 | 초기 λ spectrum 고정 | 없음 |
| RP control | `uniform_retention_theta_cap` | 6 | 모든 λ를 RP parameter cap에 고정 | 없음 |
| horizon control | `gradient_lambda_aux` | 6 | θ를 ordinary gradient로 학습 | 있음 |
| horizon control | `lru_aux_h500` | 6 | LRU의 ordinary parameters 학습 | 있음 |

`gradient_lambda_aux`는 RP를 사용하지 않는다. RP와 같은 warm-up 이후 cadence
(100 steps), probe batch 96, probe sequence 260, blank horizon 500으로 auxiliary
target loss를 준다. PAN의 θ는 `requires_grad_(True)`로 바꾸고 매 optimizer step 후
`[-18,18]`로 제한한다. 이 조건을 frozen-retention auxiliary와 혼동하면 안 된다.

### 이것은 compute matching이 아니다

Auxiliary control은 **장기 horizon target 정보**를 맞추지만, RP가 96개 carrier
coordinate를 ablation하는 계산량까지 맞추지 않는다. 결과 JSON에는 다음이 항상
기록된다.

- `aux_is_compute_matched: false`
- auxiliary weight, cadence, batch, sequence/blank horizon
- auxiliary update/example/recurrent-transition 수
- auxiliary graph forward wall time와 전체 wall time

`compute_matched_controls`를 config에 넣으면 launcher가 실행 전에 거부한다. 실제
FLOPs 또는 wall-clock을 맞춘 별도 알고리즘이 구현되기 전에는 “compute-matched”라는
표현을 사용하지 않는다.

## all-slow control의 수정

기존 λ=0.999는 “maximum retention”이 아니다.

| fixed λ | \(λ^{500}\) | \(λ^{1000}\) |
|---:|---:|---:|
| 0.999 | 0.6064 | 0.3677 |
| 0.9999 | 0.9512 | 0.9048 |
| 0.99999 | 0.9950 | 0.9900 |
| \(\sqrt{\sigma(18)}\approx0.9999999924\) | 0.999996 | 0.999992 |

기본 uniform control은 RP가 사용하는 θ clip의 실제 상한
\(\sqrt{\sigma(18)}\)을 사용한다. λ=0.999는 필요하면 legacy-reproduction sweep의
한 점으로만 추가한다. Float32에서는 `sigmoid(18)`이 1.0으로 포화되지만,
`train_fixed_retention.py`가 checkpoint의 θ를 `inf`가 아닌 정확히 18.0으로
고정하고 requested/effective λ를 모두 결과 JSON에 기록한다.

## 안전성과 resume 규칙

- 모든 output은 marker가 있는 별도 artifact root 아래에만 생성된다.
- legacy output/checkpoint/trace/log 경로를 사용하지 않는다.
- worker command에 `--force`를 절대 넣지 않는다.
- result/checkpoint/필수 trace의 구조와 SHA-256이 immutable completion receipt와
  일치할 때만 완료 job을 skip한다.
- 일부 파일만 존재하면 overwrite하지 않고 `blocked_partial`로 중단한다.
- 재시도 log는 `attempt001`, `attempt002`, …로 새 파일을 만든다.
- manifest는 canonical config, source hashes, command와 expected artifact를 담으며
  같은 campaign에서 byte-identical하지 않으면 overwrite하지 않는다.

## 결정론적 RNG stream

v2 campaign은 model initialization과 별도로 세 RNG stream을 고정한다.

- `train_data_seed_base`: 같은 task/replicate/step이면 모든 조건에 같은 training batch;
- `probe_seed_base`: 같은 task/replicate/step이면 RP와 auxiliary에 같은 H=500 probe;
- `eval_seed_base`: 모든 model과 training seed에 공통인 task별 fixed evaluation batch.

각 legacy worker가 probe를 추가로 생성해도 다음 training batch의 RNG는 바뀌지 않는다.
세 seed base와 실제 derived evaluation seed는 result JSON과 manifest에 기록된다. 모든
legacy/fixed-retention job은 10,000-step loss trajectory도 `.npz`로 보존한다.
Confirmatory config는 `torch.use_deterministic_algorithms(True)`, deterministic
cuDNN 및 `CUBLAS_WORKSPACE_CONFIG=:4096:8`도 켜며, 실제 값은 result JSON에 기록한다.
검증에 사용한 전체 Python package snapshot은 `environment.freeze.txt`에 있다.

## 실행 명령

저장소 root에서 먼저 dry-run한다.

```bash
python repro/experimental_v2/launch_p0.py \
  --config repro/experimental_v2/campaign.example.json \
  --gpus 0,1,2 \
  --dry-run
```

기본 30-run P0 anchor campaign 실행:

```bash
python repro/experimental_v2/launch_p0.py \
  --config repro/experimental_v2/campaign.example.json \
  --gpus 0,1,2
```

같은 명령을 다시 실행하면 완료 job은 skip되고, 아직 결과가 전혀 없는 job만
resume된다.

짧은 auxiliary code-path smoke check:

```bash
python repro/experimental_v2/launch_p0.py \
  --config repro/experimental_v2/campaign.example.json \
  --gpus 0 \
  --seeds 0 \
  --only gradient_lambda_aux \
  --smoke
```

Uniform-retention sensitivity를 추가하려면:

```bash
python repro/experimental_v2/launch_p0.py \
  --config repro/experimental_v2/campaign.example.json \
  --gpus 0,1,2 \
  --uniform-lambdas 0.999,0.9999,0.99999
```

이 명령은 theta-cap 조건을 유지하면서 세 fixed-λ 조건을 더한다.

## Opt-in modern baseline variants

Config에는 다음 variant가 accidental large sweep를 막기 위해 disabled 상태로
예약되어 있다.

- `rg-lru-full`: `rg_lru_main`, `rg_lru_aux_h500`
- `matched-gru-full` (worker variant `GRU-full`): `matched_gru_anchor`,
  `matched_gru_aux_h500`
- RG-LRU learning-rate anchor pilot: `rg_lru_lr3e4_anchor`,
  `rg_lru_lr3e3_anchor` (main setting `1e-3`의 양쪽 sensitivity)

Launcher는 task별 input/output dimension을 사용해 CA-LRU width 96의 parameter
수에 가장 가까운 recurrent width를 결정하고, 5% mismatch를 넘으면 실행을
거부한다. 현재 구현에서는 RG-LRU width 176, scaffold-matched GRU width 64가
선택된다. Ring 기준 전체 parameter는 CA-LRU 57,122, RG-LRU 58,066(+1.65%),
matched GRU 57,122(정확히 일치)다. Manifest에는 전체 parameter와 ordinary-gradient
trainable parameter를 둘 다 기록한다. 다음처럼 현대 baseline과 그
horizon-information control을 활성화한다.

```bash
python repro/experimental_v2/launch_p0.py \
  --config repro/experimental_v2/campaign.example.json \
  --gpus 0,1,2,3,4,5 \
  --enable rg_lru_main \
  --enable rg_lru_aux_h500 \
  --enable rg_lru_lr3e4_anchor \
  --enable rg_lru_lr3e3_anchor \
  --dry-run
```

Matched GRU까지 함께 계획하는 전체 dry-run 명령은 다음과 같다.

```bash
python repro/experimental_v2/launch_p0.py \
  --config repro/experimental_v2/campaign.example.json \
  --gpus 0,1,2,3,4,5 \
  --enable rg_lru_main \
  --enable rg_lru_aux_h500 \
  --enable matched_gru_anchor \
  --enable matched_gru_aux_h500 \
  --dry-run
```

현재 권장 confirmatory queue는 필수 비교와 RG-LRU learning-rate pilot을 합친 72
runs다. 추가 LRU/GRU auxiliary를 무조건 늘리지 않고, CA-LRU와 RG-LRU의 H=500
정보 대조를 우선한다.

```bash
python repro/experimental_v2/launch_p0.py \
  --config repro/experimental_v2/campaign.example.json \
  --gpus 0,1,2,3,4,5 \
  --enable rg_lru_main \
  --enable rg_lru_aux_h500 \
  --enable matched_gru_anchor \
  --enable rg_lru_lr3e4_anchor \
  --enable rg_lru_lr3e3_anchor \
  --disable lru_aux_h500 \
  --dry-run
```

이 행렬은 RP 세 조건 18, CA/RG horizon-information control 12, RG-LRU main 24,
matched GRU 6, RG-LRU learning-rate sensitivity 12 runs로 구성된다.

학습과 다섯 checkpoint-only 분석을 한 번에 순차 실행하려면 환경 경로를 지정해
`run_confirmatory_pipeline.sh`를 사용한다. 학습이 성공한 뒤 fixed performance,
retention permutation, campaign/legacy dynamics, ring transport가 GPU별로 병렬 실행된다.

```bash
ARTIFACT_ROOT=/path/to/p0-training \
ANALYSIS_ROOT=/path/to/p0-analysis \
PYTHON=/path/to/python \
bash repro/experimental_v2/run_confirmatory_pipeline.sh
```

`--dry-run`에서 모든 조건이 `ready`이고 manifest의 `parameter_match`가 5% 이내인
것을 확인한 뒤에만 같은 명령에서 `--dry-run`을 제거한다. 향후 variant가 현재
builder에서 사라지거나 parameter matching이 실패하면 실제 실행은 어떤 job도
시작하기 전에 중단된다.

## Read-only 진행 상황 확인

`status_p0.py`는 artifact를 쓰거나 고치지 않고 manifest, `status.json`, completion
receipt와 artifact hash를 검증한다. Artifact root를 주면 그 아래 campaign 전체를,
campaign directory를 주면 해당 campaign만 요약한다.

```bash
python repro/experimental_v2/status_p0.py /path/to/v2-artifacts
python repro/experimental_v2/status_p0.py /path/to/v2-artifacts/CAMPAIGN_ID --json
```

사람용 출력에는 state별 개수, 완료 비율, 실행 중 GPU/PID/log, failed/partial log가
표시된다. `--json`은 같은 내용을 자동화 가능한 구조로 출력한다. 검증 오류가 있으면
exit code 1, 입력 경로가 campaign/artifact root가 아니면 exit code 2를 반환한다.

## 검증

```bash
python -m py_compile \
  repro/experimental_v2/launch_p0.py \
  repro/experimental_v2/train_aux_blank.py \
  repro/experimental_v2/train_fixed_retention.py

python -m pytest -q repro/experimental_v2/tests
```

학습 후 checkpoint 전용 평가는 다음 문서를 따른다.

- `README_fixed_performance.md`: 공통 fixed assets와 H=1000 성능, retention permutation;
- `README_dynamics.md`: clean family, carrier perturbation, zero-input JVP;
- `README_transport.md`: stabilized multi-point ring transport.
