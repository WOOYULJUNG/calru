# CA-LRU 포함 동일 기준 비교

## 구성

- 모델: RNN, GRU, LSTM, CA-LRU, split-field
- topology: \(S^1, T^2, S^2\)
- 모델·topology당 3개 checkpoint
- CA-LRU: topology-tuning-v2에서 validation으로 선택한 5,000-update cell
- 공통 분석: task error, blank \(H=2048\), recurrent-carrier Jacobian,
  finite normal kick, blank shape distortion, persistent homology

CA-LRU의 reported state에는 recurrent carrier와 decoder stream이 함께 저장된다.
Jacobian과 persistent homology는 실제 자율 recurrence를 나타내는 primary recurrent
carrier에서 계산했고, decoder stream은 출력 계산 시에만 blank input으로
재구성했다.

## 핵심 결과

| Topology | 후보 | Task rad | Blank-2048 rad | Tangent error | Worst normal gain | Gap |
|---|---|---:|---:|---:|---:|---:|
| \(S^1\) | CA-LRU | 0.0135 | 0.0248 | 0.0131 | 1.0000 | -0.0131 |
| \(S^1\) | split-field | 0.0167 | 0.2700 | 0.1829 | 1.0145 | -0.1944 |
| \(T^2\) | CA-LRU | 0.0275 | 0.0432 | 0.0077 | 1.0000 | -0.0077 |
| \(T^2\) | split-field | 0.0318 | 0.5560 | 0.1890 | 1.0215 | -0.2104 |
| \(S^2\) | CA-LRU | 0.0683 | 0.1091 | 0.0099 | 1.0000 | -0.0099 |
| \(S^2\) | split-field | 0.0898 | 0.7704 | 0.3903 | 1.0645 | -0.4562 |

CA-LRU는 \(S^1,T^2\) task와 세 topology의 long blank memory에서 conventional
baseline보다 강하다. \(S^2\) task에서는 LSTM의 0.0474 rad가 CA-LRU의 0.0683
rad보다 낮다. Split-field와 직접 비교하면 CA-LRU가 세 topology 모두에서 task와
blank error가 더 낮다.

그러나 CA-LRU의 worst normal gain과 finite normal recovery ratio는 거의 정확히
1이다. Tangent는 매우 잘 보존하지만 manifold-normal perturbation을 복원하지
않는다. 따라서 이 결과가 지지하는 해석은

> CA-LRU는 안정적인 task-aligned slow-coordinate memory이지만, 현재 형태로는
> manifold-specific normal attraction을 갖는 continuous attractor가 아니다.

이다. Split-field도 모든 topology에서 positive tangent-normal gap을 만들지
못했으므로 strict approximate-CA 판정은 통과하지 못했다.

RNN·GRU·LSTM만 baseline이라고 부르면 전체적으로 가장 강한 baseline은 LSTM이다.
단, \(T^2\) blank \(H=2048\)만 보면 GRU가 LSTM보다 근소하게 낮다. CA-LRU는
proposed candidate이므로 “best baseline”에는 포함하지 않고 별도로 비교한다.
