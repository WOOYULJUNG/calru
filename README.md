# Learning Approximate Attractor Dynamics for Recurrent Memory

이 저장소는 CA-LRU와 Retention Plasticity(RP) 논문의 원고 조각, 실험 결과
스냅샷, 그리고 논문 표를 재생성하는 코드를 정리한 연구 artifact다. 기존의
대규모 실험 작업공간 전체를 옮긴 것이 아니라, 논문 수치의 출처를 감사하고
재집계할 수 있도록 필요한 파일만 선별한 패키지다.

현재 보장하는 재현 범위는 **저장된 raw JSON/CSV에서 9개의 논문 증거표를
결정적으로 다시 집계하는 것**이다. 학습 코드는 [`repro/`](repro/)에 보존되어
있지만 원 checkpoint와 trace가 포함되어 있지 않다. 새 P0 confirmatory launcher와
축소 GPU end-to-end 검증은 [`repro/experimental_v2/`](repro/experimental_v2/)에
보존되어 있다. Ságodi식 state audit·ring 재학습·manifold 분석을 다시 구성한 현재
pipeline은 [`repro/sagodi_protocol/`](repro/sagodi_protocol/)에 격리했다. 따라서 이
저장소는 아직 “모든 모델을
처음부터 재학습하여 동일 수치를 얻는 완전한 학습 재현 패키지”가 아니다.

## 먼저 확인할 상태

- 현재 원고는 [`paper/manuscript/sections_01_02.md`](paper/manuscript/sections_01_02.md)의
  Introduction과 Related Work까지만 최신 협업본이다.
- 완성된 LaTeX 원고, bibliography, 컴파일된 최종 PDF는 이 저장소에 없다.
- 이전 작업 기록에 언급된 최종 FIG-1–FIG-5 파일도 현재 snapshot에서 찾지
  못했다. [`paper/figures/README.md`](paper/figures/README.md)에 이 상태를 기록했다.
- [`paper/archive/`](paper/archive/)의 초안에는 현재 감사 결과와 충돌하는 수치와
  과장된 문장이 있으므로 최신 원고나 수치의 근거로 사용하면 안 된다.
- 코드와 데이터의 공개 라이선스는 아직 확정되지 않았다. `LICENSE`가 추가되기
  전에는 이 저장소의 존재를 재사용 허가로 해석하면 안 된다.

## 재현 수준

### 1. 표 재집계: 현재 지원

커밋된 raw 결과는 [`paper/evidence/raw/`](paper/evidence/raw/)에, 생성된 표는
[`paper/evidence/tables/`](paper/evidence/tables/)에 있다. 집계 규칙과 metric의
정확한 뜻은 다음 문서를 따른다.

- [`paper/evidence/README.md`](paper/evidence/README.md)
- [`paper/evidence/DATA_DICTIONARY.md`](paper/evidence/DATA_DICTIONARY.md)
- [`paper/evidence/PROVENANCE.md`](paper/evidence/PROVENANCE.md)
- [`docs/LEGACY_NAME_MAP.md`](docs/LEGACY_NAME_MAP.md)

개발 환경을 만든 뒤 표를 재생성하고 검증한다.

~~~bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

make evidence
make provenance
make check
make test
~~~

`make evidence`는 committed table 디렉터리를 다시 쓴다. 단순 검증만 할 때는
먼저 `make check`를 사용한다. `make provenance`는 raw file manifest와 SHA-256
목록을 갱신한다. 세 개의 seed(0, 1, 2), 산술평균, 표본 표준편차(`ddof=1`)가
기본 집계 규칙이다.

### 2. 전체 학습 재현: 아직 검증되지 않음

[`repro/legacy_code/`](repro/legacy_code/)에는 논문 수치를 만든 당시의 파일명과
legacy model tag를 보존한 코드가 있다. 환경 버전과 대표 명령은
[`repro/README.md`](repro/README.md)에 기록되어 있다. 다만 다음 항목이 빠져 있다.

- 원 checkpoint와 lambda trace
- 모든 모델에 공통인 고정 evaluation set
- held-out validation으로 정한 RP threshold
- 새 환경에서 검증된 한 번의 end-to-end full sweep
- 원 작업공간의 Git commit identifier

따라서 legacy launcher를 실행한 결과는 기존 raw snapshot을 덮어쓰지 않는 별도
디렉터리에 저장해야 한다. 새 결과가 기존 표를 재현한다고 주장하려면 환경,
명령, seed, hardware, wall time과 차이를 별도로 보고해야 한다.

### 3. Ságodi public-code-centered controlled adaptation v6: 현재 실행 경로

현재 재학습 경로는 Ságodi 공개 저장소를 commit
`cbd7404e9baca4b2dc291560cfc6576bb7b1f078`로 고정하되, 서로 다른 공개
RNN/GRU/LSTM 경로를 T128/B64/5k 공통 비교 계약으로 옮긴 **controlled
adaptation with documented repairs, not exact**다. main seed 10개를 모두
보고하고 MSE `<0.01`은 descriptive yield, NMSE `<-20 dB`는 seed별 분석
eligibility로 사용한다. 모든 모델의 actual post-transition coordinate state-noise
std, target-noise std, output dropout은 여섯 모델 모두 0으로 고정하고
`[0.03,0.01,0.003,0.001,0.0003,0.0001,0.00003,0.00001]`에서 LR만 5-seed
선택한다. 모든 모델은 clean q1과 clean loss target을 쓴다. 논문의 state std
`0.1`, 공개 GRU/LSTM target noise/dropout은 provenance로만 기록한다. shipped
공개 RNN config의 state noise는 0이며, nominal 0.1을 가정할 때의
`0.1*sqrt(0.1)=0.0316228`은 hypothetical formula provenance일 뿐이다.
이전 내부 `source-resolved-v1` recipe의 0.1/0.0316228 쌍은 v6가 0/0으로
명시적으로 override하며, shipped 공개 config와 구분해 provenance로 남긴다.
baseline 모델 하나의 zero eligibility는 downstream을
막지 않는다. 이후 LRU를 같은 fixed bank에서 독립 tuning/main하고, LRU가
eligible seed를 하나 이상 만들 때만 No-RP/CA-LRU pair를 시작한다. CA는
No-RP의 LR과 공통 noise를 상속한다. 실행 명령과 차이/repair 내역은
[`repro/sagodi_protocol/README.md`](repro/sagodi_protocol/README.md)와
[`SAGODI_SOURCE_REPAIRED_BASELINES_V6_FREEZE_ko.md`](repro/sagodi_protocol/SAGODI_SOURCE_REPAIRED_BASELINES_V6_FREEZE_ko.md)에 있다.

### 4. Model-specific state-noise search v1: 보조 최적화

Noise-free v6에서 선택된 모델별 LR을 고정한 뒤 RNN, GRU, LSTM, LRU에서만
training state-noise std
`[0,0.003,0.01,0.0316228,0.1]`를 5 seeds 모두 탐색한다. Target noise와
dropout은 계속 0이고 clean evaluation을 사용한다. Tuning은 100 runs, 선택된
std의 fresh main은 40 runs다. No-RP/CA-LRU와 RP 전용 hyperparameter는 후속
campaign으로 분리한다. 세부 계약은
[`STATE_NOISE_SEARCH_V1_FREEZE_ko.md`](repro/sagodi_protocol/STATE_NOISE_SEARCH_V1_FREEZE_ko.md)에 있다.

### 5. CA-LRU RP search와 2×2 factorial v1

LRU의 noise-free LR과 strictly-positive state-noise winner를 고정한 뒤 CA-LRU에만
존재하는 `eta_lambda`, `damage_epsilon`, RP intervention interval만 탐색한다.
선택한 RP 설정으로 No-RP/RP × state-noise 없음/있음의 4조건을 fresh seeds 0--9에서
학습한다. 세부 계약은
[`CALRU_FACTORIAL_V1_FREEZE_ko.md`](repro/sagodi_protocol/CALRU_FACTORIAL_V1_FREEZE_ko.md)에 있다.

### 6. Historical Ságodi-based primary v3.1: provenance only

[`repro/sagodi_protocol/analysis_protocol.yaml`](repro/sagodi_protocol/analysis_protocol.yaml)은
Phase 0 state/blank-map audit와 Phase 1 ring pilot만 활성화한다. 공통 fixed bank,
5개 pilot seed × 3개 모델, Track-A reconstruction, 8-path settling/fiber,
clean-paired radial/ambient perturbation, source/checkpoint-bound receipt를 사용한다.
이는 non-confirmatory pilot이며 C4·10 main seeds·torus·dimension scaling은 결과
검토 후 새 freeze가 있어야 실행한다. 세부 해석 규칙은
[`repro/sagodi_protocol/EXPERIMENT_RECONSTRUCTION_ko.md`](repro/sagodi_protocol/EXPERIMENT_RECONSTRUCTION_ko.md)를
따른다.

현재 primary 경로의 규범 문서는
[`SAGODI_PRIMARY_V3_FREEZE_ko.md`](repro/sagodi_protocol/SAGODI_PRIMARY_V3_FREEZE_ko.md)다.
v3.1은 6개 모델 × 10개 main seed의 Ságodi-based 분석에 minimum causal carrier
state의 ambient-normal finite-perturbation recovery를 추가한다. normal distance
ratio와 same-memory angular error를 함께 보고하되 threshold/pass gate는 두지
않는다. 이 항목은 Ságodi 원문의 bit-exact 재현이 아니라 normal attraction을
검사하는 project-defined primary extension이다. Isotropic perturbation은 별도
engineering robustness로만 해석한다. 평가 시에는 학습 중 additive recurrent-state
noise를 끄므로 `noise-free task`가 아니라 `state-noise-disabled deterministic
evaluation`이라고 표기한다.

## 저장소 구조

~~~text
configs/                 evidence 집계 설정
docs/                    명칭과 공개 artifact 설명
paper/
  manuscript/            현재 원고 조각
  archive/               동결된 과거 초안
  notes/                 수식·방법론·데이터 감사본
  evidence/
    raw/                 불변 입력 snapshot
    tables/              재생성 가능한 CSV
  figures/               최종 figure를 위한 자리와 규칙
repro/
  legacy_code/           실험 당시 파일명 그대로의 코드 snapshot
  experimental_v2/       격리된 P0 학습·fixed test·동역학 분석
  sagodi_protocol/        Phase-gated Ságodi ring 재학습·분석 pipeline
src/calru_paper/          결정적 evidence 집계 구현
scripts/                  사용자용 집계·검증 진입점
tests/                    집계 재현성 검사
~~~

## 논문 수치를 읽을 때의 핵심 주의점

- 논문의 최종 모델 CA-LRU는 legacy variant `PAN-RNW-full`이다.
- `PAN-full`은 linear-update control이며 최종 CA-LRU가 아니다.
- 기존 no-RP/all-slow 결과는 linear-update scaffold에서 얻었으므로 최종
  CA-LRU의 matched causal ablation으로 부를 수 없다.
- main RMSE는 manifold-coordinate 또는 각도 오차가 아니라 ambient output
  component RMSE다.
- task별 RP threshold는 held-out validation으로 선택되지 않았다.
- ring transport는 세 seed를 포함하지만 단일 base angle과 양자화된 clean grid를
  사용한다.

더 자세한 제한은
[`paper/notes/methods_equations_data_ko.md`](paper/notes/methods_equations_data_ko.md)와
[`paper/evidence/PROVENANCE.md`](paper/evidence/PROVENANCE.md)를 따른다.

## AAAI double-blind 주의

공개 GitHub 저장소는 저자명, 소속, 개인 계정, 파일 metadata, commit history,
checkpoint metadata를 통해 저자를 드러낼 수 있다. 목표 연도의 AAAI 익명성 및
supplementary-material 규정을 저자가 직접 확인하기 전에는 이 저장소를 private로
유지하거나 완전히 익명화해야 한다. 이 저장소를 생성했다는 사실만으로 공개가
허용되는 것은 아니다.

## 인용과 라이선스

논문 서지정보와 `CITATION.cff`는 제목·저자·게재 상태가 확정된 뒤 추가한다.
코드 라이선스와 raw result/figure 라이선스도 저자와 기관이 별도로 확정해야 한다.
현재는 `LICENSE`가 없으며, 제3자 스타일 파일이나 자산도 라이선스를 확인하기
전에는 포함하지 않는다.
