# Approximate-CA 진단 교정 계획 (v2)

이 문서는 `f322b66` 파일럿 구현에 대한 외부 검토를 코드 변경과 실행
freeze로 연결한다. 기존 `calru_native_sagodi_ring_pilot_v1`은 학습 안정성과
artifact 파이프라인을 확인하는 **비확증적 native recipe-transfer 파일럿**으로
보존한다. 그 분석 결과는 approximate continuous attractor(CA)의 증거로 사용하지
않는다.

## 1. 교정 범위

### C3: clean-paired 차이와 manifold recovery를 분리

기존 값

\[
\|P_N(F_0^H(m+\delta)-F_0^H(m))\|/\rho
\]

은 두 궤적이 함께 manifold 밖으로 이동하거나 원점으로 수축해도 작아질 수 있다.
v2의 primary 값은 복원된 manifold \(\mathcal M\)까지의 거리로 정의한다.

\[
D_{\mathrm{clean}}(H)=d(F_0^H(m),\mathcal M)/R_s,
\qquad
Q_{\mathrm{recovery}}(H)=
\frac{d(F_0^H(m+\delta),\mathcal M)}
     {\max(d(m+\delta,\mathcal M),\epsilon_d)}.
\]

등록 horizon은 `1, 5, 20, 100, 500, 1024`이며 radial perturbation과 ambient
normal perturbation을 따로 보고한다. clean adherence, recovery, 같은 memory
coordinate 보존을 서로 다른 gate로 유지한다. 기존 clean-paired 값은
`paired_endpoint_normal_deviation_over_rho`라는 비주장 diagnostic으로만 남긴다.

### Track A: primary manifold로 승격

v2 Track A는 고정 평가 bank의 task trajectory endpoint에서 시작해 장시간
`F0` rollout을 수행한다. 느린 후보를 angle별로 선택해 periodic cubic spline으로
재구성한 `resampled_state`를 C1--C3, JVP와 projector의 유일한 primary manifold로
사용한다.

다음 QA를 통과하지 못하면 manifold-dependent gate는 `inconclusive`로 처리한다.

- 모든 후보의 속도가 정확히 0인 경우도 유효한 identity ring 후보로 처리
- 전체 원주에 대한 bin occupancy와 maximum circular gap
- candidate-to-target angle 오차
- 32개 coarse circular bin 전부의 occupancy
- (\min\|dm/dq\|/R_s\ge 10^{-3}) tangent norm 하한
- (10^{-3}) rad seam probe에서 spline C0 오차 (\le10^{-5}R_s),
  C1 상대차 (\le10^{-2})
- finite state scale과 projector 품질

기존 8-path task-conditioned atlas는 primary manifold를 대체하지 않는다. Track A와
task-reachable sheet 사이의 angle correspondence, normal/fiber distance 및 대칭
Hausdorff distance를 측정하는 독립 검증으로 사용한다.

### Settling: 절댓값 변화와 expansion을 동시에 제한

분산 또는 manifold distance가 증가하는데 음수 상대변화로 통과하는 문제를 막기
위해 다음 두 값을 함께 계산한다.

\[
A_t=\frac{|V_{t+5}-V_t|}{\max(V_t,V_{t+5},\epsilon_V)},
\qquad
E_t=\frac{\max(V_{t+5}-V_t,0)}{\max(V_t,\epsilon_V)}.
\]

두 값의 q95와 실제 `F0^5` endpoint-to-primary-manifold distance를 각각 gate한다.

### Task contract와 baseline

YAML task 항목은 하나의 resolved task specification으로 변환하고 training, RP,
smoke, evaluation bank, analysis fallback에 모두 명시적으로 전달한다. resolved
payload와 SHA-256을 manifest/receipt에 결합해 설정 fingerprint와 실제 생성기가
달라질 수 없게 한다.

GRU는 다음을 구분한다.

- 기존 `GRUCell + MLP readout`: custom pipeline control(기존 v1 명칭만 유지)
- Ságodi-style width 96: `tanh(output_to_hidden)` 초기화와 direct linear readout
- Ságodi-style parameter-matched: CA-LRU trainable parameter 수에 가장 가까운 width

공개 코드의 초기화되지 않은 `output_to_hidden` weight는 재현 가능한
`Normal(0, 1/sqrt(N))` 초기화로 명시적으로 교정하며, bit-exact reproduction이
아님을 기록한다.

## 2. 실행 경계

1. v1 학습은 원래 commit/worktree에서만 실행한다.
2. v2 코드는 별도 branch/worktree에서 테스트한다.
3. v1 checkpoint를 v2로 분석할 때는 parent protocol, manifest, checkpoint SHA-256을
   검증하고 별도 artifact root에 쓴다. 기존 v1 분석 artifact를 덮어쓰지 않는다.
4. Ságodi paper-aligned track은 LR-selection seed와 최종 평가 seed를 분리한다.
5. LR receipt가 freeze되기 전에는 main seed를 시작하지 않는다.
6. d=1 교정 파일럿과 human review를 통과하기 전에는 torus/dimension scaling을
   시작하지 않는다.

## 3. 주장 경계

CA-LRU carrier의 blank map이 `F0(h) = Lambda h`인 현재 구현에서 모든
`lambda_i < 1`이면 무한시간에는 원점으로 수축한다. 따라서 v2도 유한 horizon에서의
기억 다양체 adherence/recovery만 평가하며, exact nonzero fixed continuum 또는
비선형 autonomous restoring field를 주장하지 않는다. float32에서 `lambda == 1`은
우선 dtype saturation 여부를 검사한다.
