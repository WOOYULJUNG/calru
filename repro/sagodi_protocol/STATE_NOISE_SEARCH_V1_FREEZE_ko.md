# Model-specific state-noise search v1 freeze

## 목적

이 campaign은 noise-free primary에서 선택된 **모델별 learning rate를 고정**한 뒤,
학습 시 recurrent state에 더하는 coordinate-wise Gaussian noise std만 탐색하는
보조 hyperparameter search다. noise-free primary를 대체하거나 Ságodi 분석 결과로
간주하지 않는다.

## 고정 순서

1. parent baseline의 verified main과 downstream의 verified LRU main completion을 요구한다.
2. RNN, GRU, LSTM, LRU의 noise-free parent-selected LR을 그대로 상속한다.
3. state-noise std `[0, 0.003, 0.01, 0.0316228, 0.1]`를 pilot seeds
   `100,101`에서 각각 2,000 updates로 평가한다.
4. NMSE `< -20 dB` seed 수, MSE `< 0.01` seed 수, median/mean MSE,
   grid order 순으로 (a) 0 포함 overall winner와 (b) strictly-positive winner를
   함께 기록한다.
5. noise/no-noise 분석용 fresh main seeds `0..2`는 strictly-positive winner로
   처음부터 5,000 updates 학습한다. overall winner가 0이면 noise가 최적화 관점에서 이롭지
   않았다는 사실을 별도로 보고한다.

총 run 수는 tuning `4 × 5 × 2 = 40`, main `4 × 3 = 12`이다.

## noise 계약

- noise는 transition 이후 full recurrent state 각 좌표에 iid Gaussian으로 더한다.
- 표의 값은 per-coordinate std이며 state dimension에 따른 total energy를 같게 만든
  값이 아니다.
- target noise와 output dropout은 모든 모델에서 0이다.
- evaluation은 state noise 0의 clean fixed bank다.
- std 0 cell은 generator를 만들거나 random draw를 하지 않는다.
- std가 양수인 cell은 같은 model seed에서 동일한 standard-normal stream을 사용하고
  std만 곱한다.

## 범위 제한

이 campaign에는 No-RP와 CA-LRU를 포함하지 않는다. 먼저 네 baseline의 LR과
state-noise std를 순서대로 정한 뒤, 별도 campaign에서 LRU LR을 고정하고 CA-LRU에만
존재하는 RP hyperparameter를 탐색한다. 이 분리는 baseline noise tuning이 CA-LRU
설계 선택에 오염되는 것을 막는다.

## 재현성

artifact root는 parent completion receipts, selection files, fixed-bank SHA, code commit,
config/freeze/runtime hashes를 묶는다. 각 run은 initial state-dict hash, data/noise RNG
identity, selected parent LR provenance, result/checkpoint/completion receipt를 기록한다.
