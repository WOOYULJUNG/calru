# H-C local-grid v2 후보 선택 근거

## 결론

개발용 H-C 후보는

\[
\boxed{a=0.05,\qquad b_{\mathrm{gate}}=-0.10}
\]

으로 고정한다. 이 선택은 단순히 task NMSE가 가장 작거나 normal recovery ratio가
가장 작은 셀을 고른 것이 아니다. approximate continuous attractor에는 다음 조건이
동시에 필요하다고 보았다.

1. 모든 개발 seed가 2,048-step blank rollout에서 안정할 것
2. tangent memory direction이 가능한 한 중립적일 것
3. radial deviation은 tangent보다 빠르게 줄어들 것
4. finite kick 뒤 raw state가 복원되면서 기억 각도가 과도하게 바뀌지 않을 것
5. manifold 위 residual flow와 2,048-step memory drift가 작을 것
6. task error가 충분히 낮을 것

따라서 강한 수축 하나만으로는 충분하지 않다. 모든 state가 소수의 discrete fixed
point로 빠르게 모이면 normal recovery ratio는 매우 작아질 수 있지만 analog memory
capacity는 나빠진다.

## 1. 먼저 적용한 필수 stability gate

두 조건은 3개 중 2개 seed만 blank-stable하여 full CA 후보에서 제외했다.

- `a0p035_bm0p05`: 2/3 stable
- `a0p05_bm0p15`: 2/3 stable

나머지 7개 조건과 기존 중앙 reference는 동일한 checkpoint-only CA protocol로
비교했다. 통계적 독립 단위는 trajectory가 아니라 model seed이며, 표의 대표값은
세 seed 중앙값이다.

## 2. 선택된 조건의 결과

| 지표 | Seed 0 | Seed 1 | Seed 2 | 중앙값 |
|---|---:|---:|---:|---:|
| Task NMSE | 0.000145 | 0.003415 | 0.001090 | 0.001090 |
| 2,048-step screen 각도오차 (rad) | 0.0931 | 0.2078 | 0.2145 | 0.2078 |
| Manifold 2,048-step memory error (rad) | 0.0908 | 0.2067 | 0.1977 | 0.1977 |
| Tangent one-step gain | 0.999476 | 0.999458 | 0.999195 | 0.999458 |
| In-plane radial one-step gain | 0.7994 | 0.8547 | 0.8064 | 0.8064 |
| Tangent − radial gain | 0.2000 | 0.1448 | 0.1928 | 0.1928 |
| 4,096-step matched-clean recovery ratio | 0.02255 | 0.00843 | 0.02237 | 0.02237 |
| Uniform projected-flow norm | 0.000159 | 0.007279 | 0.000483 | 0.000483 |
| Effective basin count | 10.57 | 8.58 | 7.80 | 8.58 |

핵심은 tangent gain이 약 0.9995로 거의 중립적인 동시에 radial gain이 약 0.806으로
충분히 작다는 점이다. 즉 기억 방향 전체를 강하게 줄이는 것이 아니라, memory
direction과 radial correction 사이에 약 0.193의 분리가 생긴다. 4,096-step 뒤
perturbation은 초기 matched-clean distance의 약 0.8–2.3%로 감소한다.

Seed 1의 uniform flow `0.007279`는 나머지 두 seed보다 크다. 따라서 seed variability가
완전히 사라진 것은 아니며, 이 결과를 fresh-seed 검증 없이 최종 inferential claim으로
사용해서는 안 된다.

## 3. 가장 가까운 대안과의 비교

### `a=0.05, b=-0.05`

이 조건은 radial recovery가 더 강하다.

- radial gain: 0.775 대 선택 후보 0.806
- 4,096-step recovery ratio: 0.0040 대 0.0224
- uniform flow: 0.000443 대 0.000483

하지만 analog memory 쪽 비용이 더 컸다.

- task NMSE: 0.00375 대 0.00109
- 2,048-step memory error: 0.311 대 0.198 rad
- effective basin count: 3.93 대 8.58

즉 normal contraction은 강하지만 ring을 따라 보존되는 장기 memory resolution이 더
낮다. approximate CA의 목적은 perturbation을 무조건 가장 빨리 지우는 것이 아니라,
normal error를 지우면서 tangent memory를 유지하는 것이므로 중앙 조건을 선택했다.

### `a=0.035, b=-0.10`

이 조건은 task NMSE가 0.000887로 가장 낮지만 dynamical evidence가 중앙 조건보다
일관되게 약하다.

- 2,048-step memory error: 0.228 대 0.198 rad
- tangent gain: 0.999230 대 0.999458
- radial gain: 0.841 대 0.806
- 4,096-step recovery ratio: 0.0525 대 0.0224
- uniform flow: 0.00103 대 0.000483
- effective basin count: 6.26 대 8.58

이 논문의 선택 목표는 최저 supervised task loss가 아니라 attracting slow-memory
manifold이므로 task NMSE의 작은 이점을 우선하지 않았다.

### recovery ratio만 매우 작은 조건들

`a0p035_bm0p15`와 `a0p065_bm0p15`는 recovery ratio가 각각 0.00460과 0.00348로
작다. 그러나 tangent gain은 0.9948과 0.9936, radial gain은 0.947과 0.942이며,
2,048-step memory error는 0.388과 0.364 rad이다. Uniform flow도 0.00777과
0.0228로 크다.

작은 recovery ratio가 여기서는 좋은 tangent–normal separation을 뜻하지 않는다.
clean state와 perturbed state가 함께 같은 discrete basin으로 이동하면서 paired
distance가 작아질 수 있으므로, memory drift와 Jacobian을 같이 봐야 한다.

### 큰 modulation `a=0.065`

`a=0.065` 계열은 전반적으로 task error, tangent neutrality, memory drift 또는
seed variability가 악화됐다. 특히 `a0p065_bm0p10`은 task NMSE 0.0122, tangent gain
0.9834, 2,048-step memory error 0.365 rad로 중앙 조건보다 명확히 나빴다.

## 4. 자동 rank와 과학적 선택의 관계

`comparison.json`의 diagnostic rank sum은 다음 여섯 지표의 동등 가중 순위합이다.

- task NMSE
- blank screen error
- 2,048-step memory error
- tangent neutrality error
- 4,096-step recovery ratio
- uniform flow

선택 조건은 rank sum 13으로 1위였고, 다음 조건들은 17, 18이었다. 이 순위는 검토
순서를 정하는 보조값이다. 최종 선택은 위의 tangent–radial 분리, basin capacity와
seed별 failure를 함께 확인한 뒤 내렸다.

## 5. 주장 가능한 범위와 다음 단계

현재 선택은 seeds 0–2에서 hyperparameter를 고른 development 결과다. 따라서 다음은
말할 수 있다.

> Among the development configurations, `a=0.05` and gate bias `-0.10`
> provided the best balance between near-neutral tangential dynamics, finite
> radial recovery, bounded memory drift, and task accuracy.

하지만 아직 다음은 말할 수 없다.

- 독립 seed에서도 동일 조건이 최적이다.
- exact continuous ring attractor이다.
- 모든 topology와 dimension에서 같은 hyperparameter가 최적이다.

다음 실험에서는 이 설정을 더 이상 tuning하지 않고 freeze한 뒤, 겹치지 않는 fresh
model seeds에서 ring mechanism을 확인해야 한다. 그 후에만 topology benchmark로
이동한다.
