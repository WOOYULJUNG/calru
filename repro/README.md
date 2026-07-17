# Experiment code

이 디렉터리는 수명주기가 다른 실험 코드를 함께 보존한다. 새 실험의 진입점을
파일명이나 version 숫자로 추측하지 말고 [`docs/CODE_MAP.md`](../docs/CODE_MAP.md)를
먼저 확인한다.

## 상태별 디렉터리

| 디렉터리 | 상태 | 역할 |
|---|---|---|
| [`manifold_benchmark/`](manifold_benchmark/) | current | \(S^1,T^2,S^2\), topology, OOD, perturbation |
| [`sagodi_protocol/`](sagodi_protocol/) | current/supporting | ring baseline, CA-LRU factorial, Ságodi evaluation |
| [`state_dependent_retention/`](state_dependent_retention/) | archived exploratory | H-C writer와 retention sweep |
| [`experimental_v2/`](experimental_v2/) | archived scaffold | 초기 P0 fixed-bank pipeline |
| [`legacy_code/`](legacy_code/) | provenance only | 최초 Exp71–Exp89 코드 snapshot |

현재 논문 모델은 CA-LRU다. H-C는 탐색 과정과 failure mode를 보존하기 위해
남기지만 새 headline 실험의 기본 비교군으로 사용하지 않는다.

## 환경

Lightweight evidence 재집계에는 루트 package만 설치한다.

~~~bash
python -m pip install -e ".[dev]"
~~~

GPU 분석 환경의 확인된 핵심 버전은 Python 3.10, PyTorch 2.4.0+cu121,
NumPy 1.26.4, Matplotlib 3.5.3이다.

~~~bash
python -m venv .venv-experiment
source .venv-experiment/bin/activate
python -m pip install -r repro/requirements-experiment.txt
~~~

Persistent homology만 필요하면 다음 optional dependency를 추가한다.

~~~bash
python -m pip install -e ".[topology-analysis]"
~~~

## 실행 원칙

1. code와 config를 먼저 commit한다.
2. output은 `experiments/<descriptive-id>`의 새 디렉터리에 쓴다.
3. checkpoint, config, code, bank SHA-256와 seed를 receipt에 저장한다.
4. tuning/validation과 final test bank를 분리한다.
5. failed seed를 삭제하거나 성공 seed로 교체하지 않는다.
6. 논문 export는 `configs/paper_artifacts.json`에 등록한 뒤 `make artifacts`로
   동기화한다.

원 checkpoint와 trace는 용량과 익명성 때문에 Git에 포함하지 않는다. GitHub의
CSV/JSON/figure는 frozen experiment의 작은 export이며, 전체 training run의
byte-for-byte 복제물이 아니다.

## Legacy 이름

논문 표준 이름은 CA-LRU지만 기존 checkpoint와 raw JSON에는 PAN, CAMN, AM-LRU가
남아 있다. mapping은 [`docs/LEGACY_NAME_MAP.md`](../docs/LEGACY_NAME_MAP.md)를
따르며 raw provenance 밖에서 legacy 이름을 새 모델명처럼 사용하지 않는다.
