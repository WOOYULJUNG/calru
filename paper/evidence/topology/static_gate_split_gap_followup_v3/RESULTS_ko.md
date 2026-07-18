# Split-field 10,000-update duration follow-up

2,000-update screen에서 topology별로 task/blank 성능, normal gain, gap 관점의 후보를
각 3개씩 골라 optimizer state를 유지한 채 총 10,000 updates까지 연장했다.

## 결론

- 9개 후보 모두 tangent-normal gap이 음수였다.
- \(S^1\)의 best-normal 후보는 normal gain 0.9989였지만 tangent gain이
  0.4959로 함께 수축했다.
- \(T^2\)의 best-performance 후보는 normal gain 0.9795였지만 tangent gain도
  0.8664여서 gap은 -0.1132였다.
- \(S^2\) 세 후보의 normal gain은 모두 1보다 컸다.
- 어떤 topology에서도 “neutral tangent + contracting normal” 조합이 나타나지
  않았다.

따라서 2,000-update screen의 실패는 단순히 학습 시간이 짧아서 생긴 현상으로
보기 어렵다. 현재 split-field parameterization과 task-only objective에서는 더
오래 학습하는 것만으로 strict approximate continuous attractor가 형성되지 않았다.
