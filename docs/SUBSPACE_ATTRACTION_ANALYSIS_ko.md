# Slow subspace와 manifold attraction 분리 분석

## 결론

현재 대표 checkpoint에서 CA-LRU는 세 topology 모두 **task-aligned
high-retention subspace와 그 바깥 방향의 강한 contraction**을 보인다.
그러나 high-retention subspace 내부의 manifold-normal perturbation은
복원되지 않는다.

따라서 이 결과만으로 지지되는 해석은 다음이다.

> CA-LRU는 거의 불변인 task-aligned slow carrier를 만들고 ambient
> fast directions를 제거하지만, 현재 checkpoint는 slow subspace 내부에서
> manifold-specific basin을 형성하지 않는다.

즉 현재 결과는 일반 slow-subspace memory보다 강하고 안정적인 구조를
보이지만, genuine ring/torus/sphere attractor의 결정적 조건인
within-subspace normal attraction은 통과하지 못한다. 이 결과를 유지한다면
논문의 표현은 `approximate continuous attractor`보다 `task-aligned slow
invariant carrier` 또는 `directional/slow-coordinate memory`가 더 정확하다.

## 분석 계약

학습은 새로 하지 않고 기존 representative checkpoint만 분석했다.

- 비교 모델: RNN, GRU, LSTM, CA-LRU
- topology: \(S^1,T^2,S^2\)
- 대표 seed: task-success seed 중 validation error가 가장 작은 seed
- 성공 seed가 없는 모델-topology 조합: 가장 나은 failed-task fallback을
  별표로 명시
- empirical slow-response subspace:
  \[
  G_{16}=\frac1A\sum_{h\in\mathcal A}
  J_{16}(h)^\top J_{16}(h)
  \]
  의 eigen-direction 중 finite response gain
  \(\sqrt{\mu_j}\ge 0.9\)인 방향
- CA-LRU explicit retention subspace:
  \[
  S_\lambda=\operatorname{span}\{e_j:\lambda_j\ge0.99\}
  \]
- local tangent: 기존 transported-atlas finite difference
- normal split:
  \[
  N_{\rm in}(h)=S\cap T_h\mathcal M^\perp,\qquad
  N_{\rm out}(h)=(S+T_h\mathcal M)^\perp
  \]
- kick radius: atlas scale의 1%, 5%, 10%
- primary figure: 5%
- blank horizon:
  \[
  H\in\{0,128,512,1024,2048,4096\}
  \]
- nearest-manifold reference:
  \[
  M_H=F_0^H(M_0)
  \]

`N_out`에 tangent를 함께 제거한 것은 baseline에서 tangent가 empirical
subspace에 완전히 포함되지 않을 수 있기 때문이다. expanding direction도
normal로 잘못 분류하지 않도록 response gain 0.9 이상이면 empirical
subspace에 포함했다.

## CA-LRU 대표 checkpoint 결과

아래 값은 5% kick과 \(H=4096\) 결과다.

| topology | seed | rank \(S_{\rm dyn}\) | tangent containment | \(N_{\rm in}\) recovery | \(N_{\rm out}\) recovery | carrier norm ratio | geometry distortion std | decoded error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| \(S^1\) | 10 | 6 | 0.980 | 0.99995 | \(<10^{-7}\) | 0.99994 | 0.00348 | 0.00279 |
| \(T^2\) | 10 | 8 | 0.997 | 1.00002 | \(<10^{-7}\) | 1.00000 | 0.00055 | 0.00161 |
| \(S^2\) | 10 | 7 | 0.988 | 1.00000 | \(<10^{-7}\) | 1.00000 | 0.00278 | 0.00486 |

CA-LRU의 empirical slow-response rank와 explicit
\(\lambda_j\ge0.99\) rank는 각각 \(6,8,7\)로 일치했다. 두 subspace로
분리한 recovery curve도 사실상 동일하다. 따라서 관측된 split은 PCA
선택의 우연이 아니라 학습된 retention axes와 직접 연결된다.

핵심 패턴은 다음과 같다.

1. `N_out` perturbation은 \(H=128\)까지 수치 floor로 사라진다.
2. `N_in` perturbation은 \(H=4096\)에도 크기가 그대로다.
3. tangent perturbation은 약 \(0.015\)의 memory shift로 유지된다.
4. `N_in` perturbation의 same-memory error는 topology별 약
   \(0.0010,0.0012,0.0048\)로 작다.

따라서 high-retention subspace 안에는 원래 clean atlas보다 두꺼운
near-neutral fiber가 존재한다. 모델은 그 fiber의 여러 상태를 거의 같은
memory로 decode하지만 clean manifold로 복원하지는 않는다.

## Hidden manifold 자체의 변화

CA-LRU의 \(H=4096\) raw stationarity는 atlas diameter 기준
\(0.113,0.029,0.063\)이다. 그러나 변화의 대부분은 초기 fast-coordinate
transient다.

- full-state norm ratio: \(0.976,0.998,0.987\)
- slow-carrier norm ratio: \(0.99994,1.00000,1.00000\)
- normalized carrier direction drift:
  \(4.9\times10^{-4},0,0\) rad
- pairwise log-distance distortion std:
  \(0.00348,0.00055,0.00278\)

즉 CA-LRU의 기억은 단순히 전체 hidden vector가 계속 축소되고 decoder가
방향만 읽는 경우와는 다르다. 학습된 slow carrier 자체는 거의 stationary고
geometry도 유지된다. 다만 그 carrier 안의 clean topology로 되돌리는
restoring field가 없다.

## Baseline과의 차이

성공한 baseline 대표 checkpoint도 일부 normal kick의 distance ratio가
0에 가까워진다. 이것을 attraction으로 해석하면 안 된다. clean atlas
자체가 함께 collapse하거나 크게 변형되기 때문이다.

- 성공 baseline의 \(H=4096\) decoded clean-memory error:
  약 \(0.145\)–\(0.439\)
- CA-LRU:
  약 \(0.0016\)–\(0.0049\)
- 성공 baseline의 pairwise distortion std:
  약 \(0.884\)–\(1.782\)
- CA-LRU:
  약 \(0.00055\)–\(0.00348\)

또한 baseline의 empirical slow subspace tangent containment는 대표
checkpoint에서 약 \(0.12\)–\(0.48\)인 반면 CA-LRU는
\(0.98\)–\(1.00\)이다. CA-LRU의 장점은 “모든 normal을 복원한다”가
아니라, **task tangent를 정확히 포함하는 안정적인 slow carrier를 만들고
그 바깥 방향을 제거한다**는 데 있다.

## 현재 주장에 미치는 영향

현재 checkpoint로 허용되는 강한 주장:

> CA-LRU learns a task-aligned, nearly invariant slow carrier with strong
> contraction outside the retained subspace and substantially more stable
> long-horizon geometry than recurrent baselines.

현재 checkpoint로는 피해야 하는 주장:

> CA-LRU exhibits manifold-specific normal attraction or a genuine
> continuous-attractor basin.

후자의 주장을 유지하려면 새로운 학습 또는 구조 수정 후 여러 seed에서
\(N_{\rm in}\) recovery ratio가 1보다 명확히 작아지고, 동시에 tangent
memory shift와 clean-manifold stationarity가 유지되는 결과가 필요하다.

## 산출물

실행 결과 root:

`/home/biadmin/ca_rnn/experiments/manifold_subspace_attraction_v1-current`

주요 파일:

- `representative_summary.csv`
- `stationarity_metrics.csv`
- `subspace_kick_metrics.csv`
- `fig_A_subspace_normal_recovery`
- `fig_B_directional_same_memory`
- `fig_C_hidden_manifold_character`
- `fig_D_carrier_state_memory`
- `fig_E_calru_explicit_retention_split`
