# CA-LRU RP search와 2×2 factorial v1 동결

## 실험 순서

1. Noise-free LRU LR winner를 No-RP와 CA-LRU의 공통 LR로 고정한다.
2. 네-baseline state-noise search의 LRU winner를 `+Noise` 조건의 공통 std로
   고정한다. CA 계열에서 state noise를 다시 tuning하지 않는다.
3. Noise-free CA-LRU에서만 CA 전용 변수
   `eta_lambda × damage_epsilon × intervention_interval_updates`를 탐색한다.
4. 선택한 RP 설정 하나를 고정하고 아래 2×2 main을 fresh seeds 0--9로 학습한다.

| RP | state-noise training | condition |
|---|---:|---|
| 없음 | 없음 | `no_rp_no_noise` |
| 없음 | 있음 | `no_rp_with_noise` |
| 있음 | 없음 | `rp_no_noise` |
| 있음 | 있음 | `rp_with_noise` |

## 선택과 공정성

- RP grid는 eta `[300,1000,3000]`, epsilon `[1e-5,3e-5,1e-4]`, intervention
  interval `[25,50,100]` updates다.
- 27-cell sentinel seed 100 뒤 상위 5개 cell을 seeds 101--104에서 평가한다.
  sentinel pruning이 있다는 사실과 전체 denominator를 결과에 명시한다.
- 선택은 NMSE eligibility, task MSE success, 4,096-step blank MSE, task MSE,
  frozen grid order 순이다.
- 네 main 조건은 seed별 초기화와 online task batch를 공유한다. 양의 state-noise
  두 조건은 동일 standard-normal stream과 동일 std를 공유한다.
- RP 두 조건은 같은 eta, epsilon, intervention interval을 공유한다.
- target noise와 output dropout은 항상 0이며 평가는 clean이다.

이 설계는 RP와 noise의 main effect 및 interaction을 비교하기 위한 것이다. 원하는
순위는 사전 가설일 뿐이며 `CA-LRU+Noise > CA-LRU > baseline+Noise > baseline`을
selection rule이나 실패 seed 제거로 강제하지 않는다.
