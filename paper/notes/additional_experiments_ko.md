# CA-LRU 추가 실험 실행 계획

이 문서는 2026-07-12 living draft와 GitHub evidence snapshot
`822b40a`를 대조해, AAAI 제출 전에 필요한 추가 실험을 실행 순서대로 정리한
계획서다. 목적은 실험 수를 무작정 늘리는 것이 아니라, 현재 논문의 핵심 주장에
직접 연결되는 비교만 남기는 것이다.

## 0. 결론

권장하는 논문 범위는 다음과 같다.

> **8개 continuous-memory task에서의 long-horizon retention 비교와, ring 및 두 개
> anchor task를 중심으로 한 mechanism 분석.**

이 범위라면 가장 중요한 추가 작업은 네 가지다.

1. 모든 모델에 공통인 고정 validation/test trajectory와 평가 규칙을 먼저 만든다.
2. CA-LRU와 같은 outer scaffold 및 비슷한 parameter 수를 쓰는 RG-LRU를 추가한다.
3. 최종 state-dependent CA-LRU scaffold에서 RP-off와 uniform-retention control을
   다시 학습한다.
4. 기존 checkpoint를 이용해 carrier-only perturbation, clean-state-family,
   zero-input Jacobian을 다시 분석한다.

초안의 모든 P0 항목을 곧바로 전 task와 5 seeds로 실행할 필요는 없다. 두 anchor
task에서 먼저 결과를 확인한 뒤 전체 task와 seed로 확장하는 편이 안전하다.

## 1. 새 GPU 학습 없이 먼저 할 일

### 1.1 기존 raw에서 표를 확장한다

현재 9개 aggregate table이 사용하는 열보다 Exp88 JSON에 더 많은 값이 이미 있다.

- perturbation radius: 0.25, 0.5, 1.0;
- recovery horizon: 0, 20, 100, 500;
- perturbed-versus-clean state deviation ratio;
- repeated perturbation;
- temporal horizon 500, 1000, 2000;
- integration velocity 1, 1.5, 2배;
- sparse/alternating velocity profile;
- PCA, vector RMSE, `lambda_gt_0p9`, `lambda_gt_0p95`, `lambda_gt_0p99`.

따라서 기존 OOD 값의 endpoint를 통일하거나 radius/horizon 곡선을 만드는 일은 새
학습이 아니라 집계기 확장으로 먼저 해결한다. 기존 raw snapshot은 덮어쓰지 않는다.

### 1.2 task별 epsilon 선택을 중단한다

가장 단순하고 방어하기 쉬운 main setting은 모든 task에

\[
\epsilon=3\times10^{-5}
\]

를 공통으로 쓰는 것이다. 현재 저장 결과에서 이 고정값의 post-H1000 RMSE는 기존
best baseline보다 8개 task 모두 낮고, 개선 배수는 4.2배에서 52.2배다. 세 후보
`0`, `3e-5`, `1e-4` 중 최악의 조합까지 포함해도 기존 baseline 대비 최소 개선은
약 2.6배다.

- main table: global `epsilon=3e-5`;
- appendix: 세 epsilon 전체 sensitivity;
- 삭제할 표현: `validation-selected epsilon`;
- RG-LRU가 추가된 뒤에도 같은 결론이 유지되는지 다시 확인한다.

이 선택은 기존 결과를 development sweep으로 취급하고, 이후 고정 test trajectory를
새로 만들어 평가한다는 전제다. task별 최적 수치를 계속 쓰려면 별도의 held-out
선택 규칙과 untouched test가 필요하다.

### 1.3 평가 asset과 primary metric을 고정한다

학습을 다시 시작하기 전에 task별로 다음을 결정적으로 생성하거나 저장한다.

- initial intrinsic coordinate;
- segment schedule과 velocity path;
- hold/dwell schedule;
- tangent/normal perturbation seed;
- validation/test split;
- asset SHA-256.

모든 모델은 동일 trajectory를 사용한다. training RNG, RP-probe RNG, validation RNG,
test RNG를 분리한다.

권장 primary metric은 **post-H1000 ambient component RMSE**다. 함께 보고할 secondary
metric은 endpoint RMSE, H=0 대비 paired drift, carrier-only clean-subtracted
perturbation error, nearby-memory error다. continuous metric을 임의 threshold로
pass/fail로 바꾸지 않는다.

## 2. P0 새 학습 실험

### E1. 같은 scaffold의 RG-LRU baseline

가장 가까운 현대 baseline 하나를 제대로 추가하는 것이 여러 SSM 이름을 얕게
추가하는 것보다 중요하다.

| 항목 | protocol |
|---|---|
| model | attention 없는 RG-LRU recurrent layer |
| scaffold | CA-LRU와 같은 encoder, projection/GLU, residual, LayerNorm, head |
| parameter | CA-LRU 약 57k의 ±5% 이내 |
| training | 10,000 updates, batch 256, horizon 80–260 |
| main range | 8 tasks × seeds 0,1,2 |
| 새 학습 | **24 runs** |

learning rate, gate/retention initialization, width를 조정한다면 CA-LRU와 동일한
development search budget을 부여한다. convergence curve도 저장해 10,000 step에서
baseline만 덜 학습된 것이 아닌지 확인한다.

현재 `SSM` 표기는 단순 real diagonal recurrence다. RG-LRU 결과가 들어오기 전에는
이를 selective SSM, Mamba, S4/S5와 같은 범주로 서술하지 않는다.

### E2. 최종 scaffold의 RP causal ablation

모든 조건은 `PAN-RNW-full`, 즉 최종 state-dependent CA-LRU scaffold를 사용한다.

| condition | 의미 | 현재 상태 |
|---|---|---|
| aligned RP | 실제 damage score로 retention update | 3개 anchor task × 3 seeds 있음 |
| RP off | 초기 linspace retention을 고정 | final scaffold 결과 없음 |
| uniform maximum retention | 모든 좌표를 RP가 도달하는 최대 retention에 고정 | 없음 |
| fixed-permutation scores | 한 permutation으로 damage 좌표를 어긋나게 학습 | 결과 있음, count-matched 아님 |
| spectrum permutation | 학습된 lambda spectrum을 평가 시 좌표 간 permutation | evaluation-only, 정확한 budget match |
| gradient-learned lambda | ordinary task gradient로 retention 학습 | 권장 control, 필수 최소선은 아님 |

Anchor task는 다음 두 개를 hard floor로 사용한다.

- `ring_hold`: 순수 1차원 continuous retention;
- `torus_integrate`: 2차원 retention과 driven update.

`ring_integrate`를 추가하는 것이 권장된다. hard floor의 RP-off와 uniform-retention은
2 tasks × 2 conditions × 3 seeds = **12 new runs**다. `ring_integrate`를 포함하면
**18 new runs**다. 기존 aligned/fixed-permutation checkpoint는 우선 재사용한다.

#### all-slow control의 중요한 수정

`lambda=0.999` 하나만을 all-slow 대표값으로 사용하면 안 된다.

\[
0.999^{1000}\approx0.368
\]

이므로 1000 blank step 뒤 carrier amplitude의 약 37%만 남는다. 반면 현재 aligned
RP run의 serialized `lambda_max`는 1.0이다. 최소한 다음 중 하나를 사용한다.

- RP의 theta upper clamp와 동일한 uniform maximum retention;
- validation에서 고른 `0.999`, `0.9999`, `0.99999`, maximum sweep;
- `sum_j lambda_j^H` 또는 `sum_j lambda_j^{2H}`를 맞춘 effective-retention-budget
  control.

결과가 같더라도 “RP is necessary”라고 쓰지 않는다. 안전한 질문은 **RP가 동일한
장기 유지 성능을 더 작은 effective retention budget으로 얻는가**다.

#### fixed permutation의 정확한 해석

현재 코드 주석은 `shuffle_fixed`를 count-matched라고 부르지만, 실제 결과의
`lambda>0.99` 수는 aligned 3–6개에 비해 약 11–27개다. 따라서 이 조건은
score-multiset permutation이지 final-support count match가 아니다.

정확한 budget control은 aligned checkpoint의 최종 lambda spectrum을 그대로
permutation해 평가하면 추가 학습 없이 만들 수 있다. 이는 학습 과정 전체의 인과
효과가 아니라 **최종 retention 좌표 정렬의 효과**를 검사한다.

### E3. 장기-horizon 정보량과 compute fairness

현재 CA-LRU는 RP 과정에서 100 optimizer step마다 H=500 blank probe를 본다. 일반
baseline은 horizon 80–260의 task loss만 본다. 따라서 post-H1000 우위가 구조와 RP
rule 때문인지, CA-LRU만 추가 장기 정보를 받았기 때문인지 분리해야 한다.

두 anchor task에서 최소 다음을 비교한다.

1. CA-LRU + RP;
2. CA-LRU, RP off + H=500 auxiliary blank loss;
3. RG-LRU + 같은 H=500 auxiliary blank loss;
4. 필요하면 GRU/LRU + 같은 auxiliary loss.

auxiliary batch, 빈도, target, blank horizon을 RP probe와 맞추고, wall time과 forward
budget을 함께 기록한다. ordinary BPTT의 비용이 RP coordinate ablation과 다르므로
완전한 FLOP equality를 가장하지 말고 **동일 장기 정보**와 **관측 compute**를 각각
보고한다.

CA-LRU auxiliary와 RG-LRU auxiliary만 사용하면 2 tasks × 2 models × 3 seeds =
**12 new runs**다. 이 비교 없이 “training horizon 밖으로 generalize한다”는 문장은
RP가 이미 본 H=500과, 실제로 보지 않은 H>500을 구분해야 한다.

### E4. parameter/scaffold-matched GRU anchor

현재 GRU는 width-matched지만 CA-LRU보다 parameter가 적고 outer scaffold도 다르다.
가장 강한 classical baseline 하나만이라도 CA-LRU와 같은 scaffold 및 약 57k
parameter로 맞춰 두 anchor task에 실행한다.

- 2 tasks × 3 seeds = **6 new runs**;
- task별 input/output 차원을 반영해 가장 가까운 hidden width를 자동 선택;
- 기존 단순 GRU 결과도 함께 남겨 scaffold 효과를 분리.

### P0 학습량 요약

| 범위 | 새 training runs |
|---|---:|
| E1 RG-LRU, 8 tasks × 3 seeds | 24 |
| E2 RP controls, 2 anchor tasks | 12 |
| E3 horizon-information controls | 12 |
| E4 matched GRU anchors | 6 |
| **hard floor 합계** | **54** |
| ring integrate를 E2에 추가 | +6 |
| baseline tuning pilot | 최대 +12 |

처음부터 54–72개를 한꺼번에 실행하지 않는다. 두 anchor task의 36–42 run pilot을
먼저 확인한 뒤 RG-LRU를 나머지 6 tasks로 확장한다.

## 3. P0 checkpoint-only state-dynamics 분석

다음 분석은 새 학습이 아니라 로컬 Exp88 checkpoint 재평가다. 다만 현재 GitHub
artifact에는 checkpoint가 없으므로, 결과를 채택할 때 checkpoint hash와 loader를
같이 보존해야 한다.

### E5. clean-state family와 blank flow

현재 post-H1000 RMSE는 출력이 유지된다는 것은 보여주지만 recurrent state가 하나의
learned family 근처에 남는지는 직접 보여주지 않는다.

각 intrinsic coordinate `q`에 여러 입력 history로 도달한 carrier state를 수집한다.

- 같은 q에서 history-conditioned state dispersion;
- 서로 다른 q 사이 separation;
- carrier와 full state의 intrinsic rank;
- dense clean-state reference family;
- blank rollout 중 nearest-family distance;
- nearest clean coordinate의 along-manifold drift;
- nearby-state separation ratio.

핵심 측정은

\[
d_\perp(t)=\min_{q'} d\bigl(h_t,\mathcal M_h(q')\bigr)
\]

와 nearest `q'`의 변화다. distance는 clean-manifold diameter 또는 local tangent
scale로 정규화한다.

동시에 raw carrier norm, normalized carrier direction, full-state norm, output을
기록한다. carrier norm은 원점으로 줄지만 LayerNorm/readout이 방향만 증폭해 출력을
유지한다면, 이는 흥미로운 directional memory이지만 “state manifold 자체가
고정된다”는 주장과는 다르다.

### E6. carrier-only tangent/normal perturbation

현재 full-state kick은 다음 step에 다시 계산되는 stream까지 perturb한다. 새
분석에서는 persistent recurrent state만 perturb한다.

| 항목 | 최소 protocol |
|---|---|
| models | CA-LRU, RG-LRU, LRU, matched GRU |
| tasks | ring hold, torus integrate |
| seeds | 3 |
| clean states | task당 최소 64 |
| directions | state당 tangent 8개, normal 8개 또는 최소 normal 8개 |
| radii | local tangent scale의 0.1, 0.3, 1.0배 |
| recovery | H = 0, 20, 100, 500, 1000 |

모델별 persistent state 정의를 명시한다. CA-LRU는 96차원 carrier, full LRU는 complex
recurrent state, LSTM을 포함할 경우 `h,c` 전체가 recurrent state다.

clean rollout과 sample-wise pair해 다음을 보고한다.

\[
\Delta_{\rm kick}(H)
=
\operatorname{MSE}_{\rm kick}(H)
-
\operatorname{MSE}_{\rm clean}(H).
\]

추가 metric은 perturbed-clean decoded distance, carrier deviation ratio,
nearest-clean-family distance, represented-coordinate drift다. direct tangent kick은
normal kick과 같은 norm으로 넣어, tangent displacement는 기억값의 변화로 남고 normal
displacement는 줄어드는지 비교한다.

평가 행렬은 4 models × 2 tasks × 3 seeds = **24 checkpoints**다.

### E7. final-model zero-input Jacobian

기존 Jacobian 결과는 구형 linear-write scaffold이고 integration에서는 driven
midpoint를 사용했다. 새 분석은 최종 state-dependent checkpoint의 endpoint에서
`x=0`으로 계산한다.

- carrier-only와 full-state Jacobian 분리;
- tangent-restricted singular values;
- random-normal gain distribution;
- maximum normal gain과 gain>1 비율;
- tangent span과 high-retention coordinate span의 principal angles;
- 1-step뿐 아니라 finite horizon의 gain.

CA-LRU carrier는 blank input에서 `J=Lambda`이므로 항등식을 다시 발견하는 것이
목적이 아니다. 핵심은 learned tangent가 high-retention directions와 정렬되고,
task-relevant normal directions보다 느리게 변하는지다. “모든 normal direction이
strict contraction”을 성공 조건으로 두지 않는다.

### E8. multi-point ring transport

논문의 transport 주장을 ring으로 제한한다면 모든 geometry로 확장할 필요는 없다.
현재 single-base probe를 다음처럼 고친다.

- cue 직후가 아니라 stabilized clean base state에서 drive 시작;
- base angle 32개;
- 양방향;
- velocity scale 1, 2;
- move length 5, 20;
- state-dependent, input-only, linear write;
- seeds 0,1,2;
- blank H = 0, 20, 100, 500, 1000;
- carrier distance를 local tangent scale 또는 manifold diameter로 정규화.

기본 drive 조건은 32 × 2 × 2 × 2 × 3 writes × 3 seeds = **2,304 probe
conditions**다. decoded displacement, target clean-state distance, nearest clean angle
error를 함께 보고한다.

모든 geometry에서 transport한다고 쓰려면 torus/curve/surface에도 같은 검사를
추가해야 한다. 추가하지 않는다면 원고 문장은 “in the ring transport analysis”로
제한한다.

## 4. P1 강력 권장 실험

### 4.1 Anchor task를 5 seeds로 확장

3 seeds는 hard floor다. ring hold와 torus integrate의 headline 모델만 5 seeds로
확장하고 seed-level dots, mean ± sample SD, seed별 paired difference를 보인다.
trajectory를 training replicate처럼 취급해 pseudo-replication하지 않는다.

### 4.2 write ablation parameter matching

현재 대략적인 parameter 수는 state-dependent 57k, input-only 48k, linear 38k다.
따라서 현재 결과는 update form은 맞지만 capacity까지 matched라고 부를 수 없다.
main claim을 write mechanism의 인과 효과로 강하게 유지하려면 width를 조정해 parameter
또는 compute를 맞춘다. 최소 ring integrate와 torus integrate에서 3–5 seeds를 쓴다.

### 4.3 OOD 요인을 분리한다

- sequence length만 증가;
- dwell distribution만 변화;
- 두 요인을 함께 변화;
- 동일 velocity trajectory를 1, 1.5, 2배로 paired scaling;
- repeated kick의 마지막 event 뒤 recovery 100/500을 별도 보장.

현재 repeated-100 조건은 마지막 kick이 evaluation endpoint에 들어가므로 recovery
실험이라고 부르면 안 된다.

### 4.4 intrinsic error를 추가한다

ambient component RMSE는 유지하되 다음을 appendix 또는 보조 panel에 추가한다.

- ring angular error;
- torus wrapped-coordinate/geodesic error;
- complex curve arc-length 또는 nearest-parameter error;
- surface intrinsic-coordinate error.

### 4.5 RP mechanism과 effective half-life

`lambda>0.99` count 하나만으로 compact retention을 정의하지 않는다.

- 전체 lambda spectrum;
- `lambda^H`와 `-1/log(lambda)` half-life/time constant;
- `sum lambda^H` 또는 `sum lambda^(2H)` budget;
- held-out ablation damage;
- carrier variance, decoder sensitivity, tangent loading;
- training 중 damage와 lambda trajectory.

최종 lambda와 같은 training damage를 다시 상관시키는 것은 순환적일 수 있으므로,
별도 held-out probe 또는 post-training coordinate swap을 사용한다.

### 4.6 Scaling과 외부 task

현재 d=1,2,4,8,16 결과는 final `PAN-RNW-full`이지만 main Exp88과 다른 line
protocol이다. broad capacity claim을 유지할 때만 main protocol과 held-out 설정으로
다시 실행한다. width 48/96/192 또는 noisy delayed-estimation task는 좋은 확장이나,
현재 핵심 주장보다 우선하지 않는다.

## 5. 실행 전 반드시 고칠 해석

| 현재 표현/설정 | 정확한 해석 또는 수정 |
|---|---|
| `epsilon=0` | no-RP가 아니다. threshold가 0일 뿐 RP update는 계속된다. |
| `shuffle_fixed = count-matched` | 거짓이다. score permutation이며 final support count는 맞지 않는다. |
| `all-slow lambda=0.999` | H1000에서 amplitude가 0.368이므로 단독 strong control이 아니다. |
| `RP off` | frozen initial linspace retention이다. gradient-learned retention과 다르다. |
| `training horizon 80–260 only` | CA-LRU RP는 별도로 H=500 blank 정보를 본다. |
| `matched write ablation` | scaffold는 유사하지만 parameter 수 57k/48k/38k는 다르다. |
| `normal recovery` | 현재 값은 full-state kick과 clean blank drift가 섞인 target RMSE다. |
| `nearby-memory tangent perturbation` | 실제로는 neighboring memory를 새로 cue한 protocol이다. |
| `multi-geometry transport` | 현재 strongest quantitative result는 single-base ring이다. |
| `modern SSM baseline` | 현재 SSM은 simple real diagonal recurrence다. |

## 6. 결과에 따른 decision gate

1. **RG-LRU가 CA-LRU의 post-H1000 격차를 대부분 없애면:** mechanism 실험을 더
   늘리기 전에 contribution을 RP compactness 또는 blank-map interpretability로
   다시 좁힌다.
2. **uniform maximum retention이 aligned RP와 같은 성능을 내면:** RP를 retention에
   필수라고 쓰지 않고, compactness/compute/coordinate allocation만 주장한다.
3. **H=500 auxiliary baseline이 격차를 없애면:** 우위의 원인을 architecture보다
   long-horizon supervision으로 재해석한다.
4. **carrier-only normal perturbation이 clean-subtracted recovery를 보이지 않으면:**
   normal contraction과 attractor-basin 문장을 삭제하고 finite-time retention으로
   논문 범위를 줄인다.
5. **carrier norm은 붕괴하지만 normalized output만 유지되면:** stable state manifold가
   아니라 direction-coded memory로 기술한다.
6. **multi-point ring transport만 성공하면:** 모든 geometry가 아니라 ring-centered
   mechanism evidence로 명시한다.

## 7. 이번 라운드에서 보류할 실험

다음은 anchor pilot 결과를 보기 전에는 실행하지 않는다.

- Mamba/S4/S5를 여러 개 넣는 baseline zoo;
- discrete flip-flop/K-way task 확대;
- 모든 RP control의 8-task sweep;
- 모든 기존 baseline의 8-task × 5-seed 재학습;
- final-scaffold scaling의 전체 width/dimension grid;
- protocol과 source table이 확정되기 전의 최종 figure 제작.

RG-LRU 하나와 공정성 control을 정확히 만드는 것이 baseline 수를 늘리는 것보다
먼저다. multi-geometry transport와 scaling은 main claim을 그 범위까지 유지할 때만
승격한다.

## 8. 권장 실행 순서

1. **Protocol freeze:** global epsilon, fixed validation/test assets, metric schema, hashes.
2. **Anchor pilot:** ring hold와 torus integrate에서 RG-LRU, matched GRU, RP controls,
   horizon-information controls.
3. **Gate review:** 결과에 따라 주장과 남은 실험 범위를 다시 확정.
4. **Eight-task extension:** 통과한 RG-LRU와 최종 CA-LRU setting을 전체 task로 확장.
5. **Checkpoint diagnostics:** clean family, carrier perturbation, Jacobian, ring transport.
6. **Seed extension:** headline anchor만 5 seeds.
7. **Artifact update:** 새 raw/checkpoint hash, aggregate tables, figure provenance, clean
   retraining smoke test.

기존 관측 시간으로 단순 추산하면 RG-LRU 24 runs와 final-scaffold RP control 18
runs만 약 35 GPU-hours 규모다. horizon/scaffold fairness control까지 포함한 P0는 약
45–60 GPU-hours로 예상한다. 현재 6개의 Titan RTX를 모두 사용할 수 있다면 코드와
I/O가 준비된 뒤 이상적인 wall time은 약 8–12시간이지만, 먼저 2-task pilot로
실패 가능성을 확인하는 것이 좋다.
