# Calibrated finite normal-kick recovery

ID task gate를 통과한 checkpoint만 사용했다. ID부터 실패한 모델은 OOD 실패로 세지 않았다.

선택 강도는 `ρ=0.25`, 회복 horizon은 `H=2048`이다. 시험한 어느 topology에서도 모든 eligible baseline이 동시에 실패하지 않았으므로 강도를 낮출 필요가 없었다.

| Topology | Model | eligible seeds | Q (normal distance ratio) | same-memory error (rad) | status |
|---|---|---:|---:|---:|---|
| S1 | RNN | 1 | 0.0000 | 0.0526 | survived |
| S1 | GRU | 3 | 0.0036 | 0.0286 | survived |
| S1 | LSTM | 3 | 0.0018 | 0.0203 | survived |
| S1 | CA-LRU | 3 | 0.2565 | 0.0065 | survived |
| T2 | RNN | 0 | — | — | ineligible_id |
| T2 | GRU | 3 | 0.0573 | 0.0502 | survived |
| T2 | LSTM | 3 | 0.0176 | 0.0453 | survived |
| T2 | CA-LRU | 3 | 0.3307 | 0.0069 | survived |
| S2 | RNN | 0 | — | — | ineligible_id |
| S2 | GRU | 0 | — | — | ineligible_id |
| S2 | LSTM | 3 | 0.0219 | 0.0510 | survived |
| S2 | CA-LRU | 1 | 0.2971 | 0.0233 | survived |

`Q<1`은 초기 normal distance보다 manifold에 가까워졌음을 뜻한다. 단, 작은 Q만으로 같은 memory로 돌아왔다고 할 수 없으므로 same-memory angular error를 함께 본다.

이 결과에서는 baseline이 CA-LRU보다 더 강한 normal-distance 수축을 보이지만, CA-LRU가 더 작은 same-memory error를 보인다. 따라서 이 축은 CA-LRU의 일방적 우위가 아니라 `memory-preserving recovery`와 `geometric contraction`의 trade-off로 보고해야 한다.
