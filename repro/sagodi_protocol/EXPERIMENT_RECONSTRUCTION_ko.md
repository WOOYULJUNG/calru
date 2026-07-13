# CA-LRU × Ságodi 실험 재구성 동결안

이 디렉터리는 `CA_LRU_Sagodi_Experimental_Protocol_ko.md`를 바로 전체 실행하지 않고, 재현성과 해석 가능성을 먼저 확인하는 **Phase 0 → Phase 1 pilot**만 동결한다. 실행의 단일 사양은 `analysis_protocol.yaml`이며, 확장과 검증은 표준 라이브러리만 사용하는 `config.py`가 담당한다.

이 freeze의 목적은 논문 결론을 내리는 것이 아니다. 실제 recurrent state와 blank map을 감사하고, 작은 ring 조건에서 task 학습·manifold 분석·clean-paired perturbation 분석이 끝까지 재현되는지 확인하는 것이다. 기존 campaign 결과는 exploratory evidence로만 보존하며 이 pilot 또는 향후 confirmatory 결과와 합산하지 않는다.

## 현재 허용된 범위

| Phase | 상태 | 내용 | 다음 단계 조건 |
|---|---|---|---|
| Phase 0 | 활성 | 최소 Markov state, 실제 (F_0), overwrite/reset, deterministic/Jacobian audit | 모든 audit artifact와 `phase0_gate.json` 통과 |
| Phase 1 ring pilot | 활성 | Protocol A식 angular-velocity integration, 3 models × 5 pilot seeds | pilot 결과 검토 후 별도 confirmatory freeze |
| Phase 1 confirmatory | 비활성 | main seeds 0–9 | 새 freeze 필요 |
| Phase 2 torus | 비활성 | (d=2), mixed/coupled, C4 | radial attribution과 clean-paired recovery 통과 필요 |
| Phase 3 scaling | 비활성 | (d=4,8) | Phase 2 C1/C2/ambient recovery 통과 필요 |
| Phase 4 seed/mechanism | 비활성 | main seeds, task-map seeds, 전체 ablation | Phase 3 검토와 새 freeze 필요 |
| Exact CA axis | 비활성 | all-point fixedness와 stable normal bundle | architecture audit가 exact continuum을 배제하지 않을 때만 |

Phase 0이 실패하면 Phase 1을 실행하지 않는다. 특히 auxiliary stream이 다음 transition에 feedback되는데 primary state에서 빠졌거나, zero-input rollout에 cue/reset이 섞였거나, autograd Jacobian과 finite difference가 맞지 않으면 CA metric을 만들지 않는다.

## Phase 1 동결 run matrix

| 항목 | 동결값 |
|---|---|
| Task | Ságodi angular-velocity integration, ring (S^1) |
| Initialization | learned hidden initialization from ((\cos\theta_0,\sin\theta_0)) |
| Input/target | raw (omega_t); token 처리 후 (q_{t+1}) target |
| (T,Delta t) | 256, 0.1 |
| GP | normalized grid `linspace(-1,1,256)`, (ell=1,sigma_v=1) |
| Models | `ca_lru`, final-scaffold `no_rp`, `gru` |
| Width | 96, width-matched pilot only |
| Pilot seeds | 100, 101, 102, 103, 104 |
| Batch / updates | 64 / 5,000 |
| Optimizer | Adam, betas (0.9, 0.999), weight decay 0 |
| Training state noise | recurrent-coordinate standard deviation 0.1 |
| Active LR | (10^{-2}), **pilot default** |
| Frozen LR grid | (10^{-2},10^{-3},10^{-4},10^{-5}) |
| Authorized runs | 3 models × 5 seeds × 1 active LR = 15 |

(10^{-2})는 현재 결과를 보고 고른 값이 아니다. Ságodi default에 맞춘 초기 pipeline pilot 값이다. 네 LR의 정식 pilot selection은 아직 비활성이고, 활성화할 경우 각 LR마다 5 pilot runs를 실행한 뒤 update 100 평균 loss로 선택하는 새 freeze가 필요하다.

Width 96은 원 논문의 64/128/256 sweep 재현값이 아니라 이번 ring pilot을 위한 요청된 고정 폭이다. 따라서 이 결과를 원 논문의 width sweep이나 parameter-matched 비교로 표현하지 않는다.

`No-RP`는 generic LRU가 아니다. CA-LRU의 encoder, writer, readout, nonlinear scaffold, 초기 retention spectrum과 retention의 task-gradient 정책을 유지하고 RP hook만 비활성화한다.

### CA-LRU blank map의 실제 의미

이 freeze의 CA-LRU와 No-RP는 `PAN-RNW-full`, one layer, `carry_stream=False`,
`encoder_bias=False`, `use_norm_in=False`다. Recurrent writer는
$g(h,u)-g(h,0)$이므로 실제 blank input $u=0$에서는 정확히 상쇄된다. 따라서
최소 Markov carrier의 autonomous map은

\[
F_0(h)=\Lambda h
\]

인 homogeneous diagonal linear map이다. 비선형성은 input-conditioned writer에
있으며 blank recovery 자체를 `nonlinear restoring field`라고 부르지 않는다.
$\lambda_j<1$인 동안 nonzero exact fixed-point continuum는 구조적으로 배제되지만,
유한 시간의 approximate slow-memory/attraction 조건은 별도로 검사할 수 있다.

## Phase 0 필수 audit

모델마다 다음을 독립적으로 남긴다.

1. 미래 transition을 결정하는 모든 tensor의 목록과 `carried / overwritten / decoder-only` 분류
2. 최소 Markov primary state의 무손실 pack/unpack
3. 실제 `model.step(zero_velocity, state)`로 계산한 $F_0^{\mathrm{primary}}$
4. 해당되는 경우 carrier-only 및 reported/full map
5. 학습 전 architecture trace에서 20-step 실제 input/state/reset/decoder tensor
6. 학습 checkpoint의 task atlas가 만들어진 뒤 $0.1R_s$ radial perturbation에 대한 별도 20-step autonomous trace와 nearest-manifold distance
7. dropout·training noise를 끈 동일-call determinism
8. float64 subset 비교
9. 16개 방향에서 autograd JVP와 centered finite difference 비교
10. 함수 외부 step index 또는 hidden cache가 없다는 확인

학습 전에는 task manifold와 radial direction이 정의되지 않는다. 따라서 Phase 0의
architecture trace는 해당 두 항목을 `not_available_before_training`으로 기록하고,
각 trained checkpoint 분석의 preflight가 이를 실제 atlas 기준으로 보완한다.

복원이 실제 zero input과 reset 없는 full Markov map에서 일어날 때만 `autonomous`라고 쓴다. Stream overwrite만 줄고 task-active carrier error가 유지되면 autonomous radial attraction의 증거로 사용하지 않는다.

## Phase-1 ring atlas 재구성

Task NMSE가 -20 dB cutoff를 통과한 checkpoint에만 manifold fitting을 수행한다.
실패 seed는 작업에서 삭제하지 않고 `not_applicable_task_failure` receipt와 함께 전체
seed 분모에 남긴다. 각 checkpoint는 fitting 전에 task-evoked primary state로 실제
`step(x=0, state)` audit를 다시 통과해야 한다.

두 atlas track을 서로 바꾸어 부르지 않는다.

- **Ságodi Track A:** 균일한 1,024 angle의 task-defined initialization을 noise 없이
  blank 16T=4,096 steps 전개한다. Trajectory별
  `speed_t < 1e-3 * max_t speed_t` candidate를 모으고, target angle별 decoded angle이
  가장 가까운 candidate를 전역 circular search로 고른 뒤 periodic cubic spline으로
  1,024 points를 재표본화한다. Candidate 부족이나 spline 실패는 정상적인
  reconstruction failure artifact다.
- **Task-conditioned track:** q=0에서 각 target q로 순변위가 정확히 같은 8개 velocity
  path(분산 pulse, early/late/middle pulse, shape/order variant, zero-net excursion)를
  실제 모델에 통과시킨다. Canonical initialization, 8개 endpoint, 그리고
  Hsettle={0,5,20,100} state를 모두 저장한다. 가장 작은 유효 H의 8-path mean만
  settled primary atlas 후보가 된다.

Track A와 settling이 모두 유효할 때만 settled atlas를 claim용 primary atlas로 쓴다.
어느 한쪽이 실패해도 canonical/path bank와 diagnostic 수치는 저장하지만 projection에
의존하는 gate는 `inconclusive`다. Fiber gate는 동일 q의 8개 path variance와 인접 q
separation으로 계산한 chi_fiber를 사용한다.

Ring projection은 original anchor의 exact NN만 쓰지 않는다. Periodic cubic spline을
8배 dense하게 표본화하고 exact dense NN 뒤 local 1-D tangent-plane refinement를 한다.
Known-q QA는 이 spline에서 state를 다시 뽑지 않는다. Original anchor와 겹치지 않는
stratified mid-cell angle마다 독립적인 8-path task trajectory를 새로 실행하고, primary
atlas가 선택한 것과 **동일한** settling horizon에서 path mean을 만든다. 이 held-out
task state와 0.05 Rs synthetic radial point에서 projection error를 별도로 측정하며,
두 family 각각 q95<=.002여야 invariance, drift,
same-memory, tangent/JVP gate를 판정한다. 세 finite-difference epsilon
{.001,.005,.01}의 derivative 차이는 모두 저장하되, 별도 사전등록 cutoff가 없으므로
`diagnostic_only_no_preregistered_cutoff`로 표시한다.

Primary atlas의 첫 anchor를 0.1 Rs radial로 민 뒤 20-step autonomous trace에는 actual
input tensor, pre/post carrier와 stream, overwrite mask와 별도의 all-false reset mask,
실제 decoder input/output, F0 residual, dense-spline manifold distance를 저장한다.

## Claim gate 동결

아래 값은 원 논문의 보편 threshold가 아니라 원 프로토콜 §3.13.2의 operational gate다. Pilot은 이 metric을 계산하지만 confirmatory L3 결론을 내리지 않는다.

| Gate | Pilot 구현 |
|---|---|
| Task | masked NMSE $<-20$ dB |
| C1 decode | held-out **linear** diagnostic decoder, mean geodesic ≤ .05, q95 ≤ .10 |
| C1 rank | atlas 99%에서 $\sigma_d/\sigma_1\ge10^{-3}$ |
| C1 neighborhood | trustworthiness와 continuity 각각 ≥ .95 |
| C1 sheet/fiber | $\chi_{\mathrm{fiber}}\le0.1$, 아니면 validated fiber-tangent branch |
| Invariance | q95 $r_{\mathrm{inv}}\le0.01$ |
| C2 drift | **정확한 (4T=1,024)** step에서 mean ≤ .05, q95 ≤ .10 |
| C3 recovery | $\rho=0.1R_s/\sqrt d$, $H=500$; radial과 ambient가 **각각** median $R_N\le0.5$, q95 $R_N<1$ |
| C3 same-memory | $H=500$, mean $E_{\mathrm{excess}}\le0.05$, q95 ≤ .10 |
| C3 paper noise | evaluation coordinate std .1에서 masked NMSE $<-20$ dB |
| Tangent equivariance | primary shift .01 rad, mean $E_T\le0.05$, q95 ≤ .10 |
| Tangent non-expansion | (H=50), atlas 내 **q95** amplification ≤ 1.05 |
| Sampled normal gap | (H=50), atlas 95%에서 (ell_N^{max}<0)와 (gamma_{50}>0) 동시 만족 |
| C4 | primary recurrent-map draws 90%가 모든 L2 gate 유지; **현재 미실행** |
| Main seed gate | main seeds 10개 중 8개 L3; **pilot seed에 적용 금지** |

### 모호점의 보수적 해소

- Random normal 평균으로 radial evidence를 대체하지 않는다. Radial과 ambient recovery를 따로 aggregate하고 둘 다 통과해야 한다.
- (T=256)의 C2 gate는 목록에 있던 근삿값 1,000이 아니라 정확한 1,024 step을 사용한다. 1,000도 descriptive horizon으로 보존한다.
- Tangent non-expansion은 평균이 아니라 seed 내부 atlas q95로 판정한다.
- Ring의 C1 rank는 각 anchor의 local chart Jacobian을 `[carrier,1]`로 만들어
  normalized $\sigma_{\min}$, $\sigma_{\max}$, 그 비를 모두 저장한다. d=1이라 resolved
  chart의 비는 1이지만 normalized $\sigma_1\le100\epsilon_{mach}$인 0/0 chart는 비를
  0으로 두어 실패시킨다. 단순 tangent-speed cutoff로 gate를 대체하지 않는다.
- Sampled normal gap의 primary normal block은 실제 clean $F_0$ trajectory를 projector로
  매 step 재투영해 $P_N(q_t)$를 모든 JVP step 뒤에 적용한다. Tangent block은 freeze대로
  $T(q_H)^T J_{0:H}T(q_0)$이며 중간 $P_T$를 넣지 않는다. Endpoint에서만 normal을
  투영한 $J_{0:H}$ 값은 `non_primary endpoint-only proxy`로 별도 저장한다.
- C1 primary decoder는 held-out linear decoder다. 2-layer decoder는 후속 sensitivity로 남기며 linear failure를 덮는 용도로 사용할 수 없다.
- Tangent equivariance primary small shift는 목록 중 가장 큰 .01 rad로 두고 .001/.005 rad를 sensitivity로 남긴다.
- C4와 8/10 model-seed gate는 정의를 config에 보존하지만 이 pilot에서는 실행하거나 주장하지 않는다.

## QA threshold와 claim threshold의 분리

`analysis_protocol.yaml`의 `claim_gates`만 논문 claim 판정에 사용한다. `qa_thresholds`는 분석 수치가 신뢰 가능한지 판단하는 품질관리 규칙이다.
Radial/ambient 표본마다 unit tangent와의 절대 내적을 저장하고 전체 최대가 $10^{-6}$
이하여야 한다. 이 QA가 실패하면 normal-dependent gate는 실패로 세지 않고
`status=inconclusive, passed=null`로 기록한다. Projection QA 등 다른 선행 eligibility가
실패한 경우에도 동일하게 `passed=null`이며, L1/L2 aggregate boolean만 false다.

- Settling 후보: 0, 5, 20, 100 steps
- 다음 5-step within-path variance 감소 <1%, settling drift q95 <.01
- “transverse residual이 체계적으로 감소”는 다음 5 step에서 residual이 1% 넘게 감소하는 anchor 비율로 operationalize하고, 그 비율이 0.5 이하일 때만 plateau로 본다. 이는 pilot QA이지 claim threshold가 아니다.
- Tangent/normal orthogonality error ≤ (10^{-6})
- projection q95 error ≤ .002, 즉 invariance gate .01의 20%
- projection QA 실패 시 metric은 fail로 억지 변환하지 않고 `inconclusive`로 표시
- clean rollout과 perturb rollout을 항상 pair
- 실패 seed도 success-rate 분모에서 제거하지 않음

Settling을 100 step 안에 만족하지 않으면 `transient_task_sheet`로 표시하고 L2/L3 manifold claim에 사용하지 않는다.

## Bank와 분석 범위

- validation 2,048 trials
- fixed ID test 4,096 trials
- ring atlas 1,024 anchors
- Jacobian 및 finite-kick 256 stratified anchors
- 같은 memory endpoint당 서로 다른 path 8개
- primary blank drift gate: **1,024=4T**
- primary finite Jacobian gate: $H=50$
- primary finite kick gate: $H=500$, radius $.10R_s/\sqrt d$
- ambient random normal: anchor당 8개

원 protocol의 blank horizon 1–4,096, Jacobian $H\in\{1,10,50\}$,
kick horizon 0–1,000, radius $.05$–$1.0R_s$ 전체 sensitivity sweep은 현재
core-pipeline pilot의 active Cartesian product가 아니다. Primary gate가 정상 작동하고
결과를 검토한 뒤 새 freeze에서 활성화한다. C4, strict worst normal, torus와 scaling도
같은 이유로 후속 phase에 남긴다.

Task, data stream, model, evaluation bank, perturbation bank seed를 분리하고, 같은 condition의 모든 모델에 같은 online batch key와 fixed evaluation/perturbation bank를 제공한다. 통계적 독립 단위는 trajectory나 anchor가 아니라 trained model seed다.

## Pilot 해석과 중단 규칙

- Pilot 결과의 레이블은 `pilot_evidence_only`다.
- Pilot seed 5개로 8/10 L3 gate, confirmatory p-value 또는 최종 model superiority를 주장하지 않는다.
- C1/C2가 반복 실패하면 framing을 `slow recurrent memory`로 낮춘다.
- Radial은 수축하지만 ambient가 실패하면 `direction-selective radial recovery`만 주장한다.
- `nonlinear`은 gain이 radius 또는 state에 따라 실제로 달라진 경우에만 쓴다.
- Fixedness가 실패해도 추후 C1–C4가 통과하면 approximate CA는 가능하지만 exact CA는 아니다.

## 재현 산출물

각 run manifest는 freeze/config hash, source protocol hash, clean code commit,
Python/Torch/CUDA/GPU environment fingerprint, complete state definition, parameter
count, 모든 seed, 공통 evaluation/perturbation bank hash, checkpoint hash를 포함한다.
Analysis receipt는 현재 training checkpoint hash와 일치해야 한다. 완료는 모든
receipt와 artifact hash, Phase 0 gate를 다시 계산한 뒤에만 인정한다. 실패·중단
attempt는 `attempts/` 아래 보존하고 새 attempt를 원자적으로 publish한다.

Phase 1의 최소 artifact는 config/manifest/checkpoint/receipt 외에 task metrics, atlas states, tangent basis, flow residual, Jacobian summary, S-type perturbation, `claim_gate.json`이다. 각 claim gate record는 metric, threshold, value, pass/fail, source artifact 경로를 가진다.

15개 pilot analysis receipt가 모두 검증된 뒤에는 run별 gate status/value를 보존하는
`pilot_aggregation/pilot_run_matrix.csv`와 모델별 기술통계를 담은
`pilot_summary.json`을 원자적으로 생성한다. 분모는 (i) task failure를 제거하지 않은
all-started pilot run과 (ii) task gate 성공 run 조건부를 함께 제시한다. 유한 scalar
gate value에는 mean, sample standard deviation, median, IQR을 보고하되, 이 5-seed
pilot에서 p-value나 확증적 추론은 계산하지 않는다. 이 두 aggregate의 hash도
`COMPLETE` 검증 대상이다.

구성 검증과 run matrix 확인:

```bash
python repro/sagodi_protocol/config.py
python repro/sagodi_protocol/config.py --print-runs
python repro/sagodi_protocol/config.py --print-fingerprint
```

이 명령이 승인하는 학습 run은 정확히 15개다. 이후 phase나 LR grid sweep은 config 파일을 몰래 확장하지 않고 별도 freeze와 검토를 거쳐야 한다.
