# Legacy name map

논문에서 새로 만든 명칭은 **CA-LRU**와 **Retention Plasticity (RP)** 두 개다.
PAN, CAMN, AM-LRU는 개발 과정에서 사용한 이름이며 논문 본문에는 새 개념처럼
노출하지 않는다. 다만 raw filename, checkpoint tag와 코드 재현성을 위해 이
저장소에서는 삭제하지 않는다.

## Final model과 update controls

| paper-facing name | canonical ID | code variant | 대표 legacy raw tag | update |
|---|---|---|---|---|
| CA-LRU | `ca_lru` | `PAN-RNW-full` | `am_lru_rnw_eps*`, discrete의 `camn_eps*` | `g([h;u])-g([h;0])` |
| CA-LRU, input-only update | `ca_lru_input_only_update` | `PAN-NW-full` | `am_lru_nw_eps*` | input-only nonlinear update, blank에서 0 |
| CA-LRU, linear update | `ca_lru_linear_update` | `PAN-full` | `am_lru_eps*` | `Bu` |

논문의 CA-LRU main row는 반드시 `PAN-RNW-full`에서 가져온다. `PAN-full`을 final
model로 부르면 안 된다.

## RP controls

| paper-facing 설명 | code/raw tag | scaffold | 해석 제한 |
|---|---|---|---|
| aligned RP | `am_lru_rnw_eps3e-5` | final `PAN-RNW-full` | damage score가 원 좌표에 정렬됨 |
| fixed-permutation RP scores | `camn_shufmatch_eps3e-5` | final `PAN-RNW-full` | 고정 permutation이지만 count-matched 아님 |
| freshly shuffled RP scores | `camn_shuffle_eps3e-5` | final `PAN-RNW-full` | probe마다 shuffle되어 대체로 dense support |
| no RP, linear scaffold | `am_lru_eta0` | `PAN-full` | final CA-LRU matched ablation 아님 |
| all-slow λ=0.999, linear scaffold | `am_lru_allslow` | `PAN-full` | λ=0.99가 아니며 final matched ablation 아님 |

과거 launcher의 `count-matched shuffle` 설명은 실제 persistent count와 일치하지
않는다. Paper-facing 표현은 `fixed-permutation RP scores`가 정확하다.

## Standard baselines

| paper-facing name | legacy variant/tag | 주의 |
|---|---|---|
| RNN | `RNN`, `rnn` | tanh RNN |
| GRU | `GRU`, `gru` | main 결과에서 가장 강한 standard baseline인 경우가 많음 |
| LSTM | `LSTM`, `lstm` | decoder는 hidden state를 읽음 |
| SSM | `SSM`, `ssm` | 단순 learned real diagonal recurrence |
| LRU | `lru-full`, `lru` | complex diagonal full block |

여기서 SSM을 최신 selective SSM/Mamba와 동일시하면 안 된다. 현재 battery에는
Mamba, RG-LRU, LinOSS가 없다.

## 더 오래된 이름

- `PAN`: Retention Plasticity가 정식 명칭으로 확정되기 전 persistence update를
  가리킨 내부 이름이다.
- `AM-LRU`: CA-LRU 직전 모델 후보명이다.
- `CAMN`: 초기 모델 후보명이며 일부 discrete/shuffle tag에 남아 있다.
- `P-LRU`: static spectral-polarization baseline으로 CA-LRU의 옛 이름이 아니다.

## Filename을 읽는 예

`ring_integrate_am_lru_rnw_eps1e-4_seed2.json`은 ring integration, final CA-LRU,
RP threshold `1e-4`, seed 2를 뜻한다.

`ring_hold_am_lru_allslow_seed0.json`은 ring hold의 all-slow control이지만
`PAN-full` linear scaffold에서 얻은 값이다.

`flipflop_n4_camn_eps1e-4_seed1.json`의 `camn`은 역사적 tag이고 실제 code variant는
`PAN-RNW-full`, paper-facing name은 CA-LRU다.

Raw tag를 바꾸거나 파일을 일괄 rename하지 않는 이유는 기존 result와 집계 pattern의
추적 가능성을 보존하기 위해서다.
