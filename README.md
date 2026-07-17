# Learning Approximate Attractor Dynamics for Recurrent Memory

CA-LRU와 Retention Plasticity(RP)의 논문 코드, frozen 결과 export, 재집계
도구를 관리하는 연구 저장소다. 대규모 checkpoint와 training trace는 Git에
넣지 않고, 논문에서 실제로 참조하는 작은 CSV·JSON·figure와 그 checksum만
커밋한다.

## 현재 기준

- 논문의 중심 모델은 **CA-LRU**다.
- H-C/state-dependent-retention 계열은 폐기하지 않지만
  **archived exploratory study**로 분류한다.
- 현재 topology/OOD 결과는 `ood_v3`, `perturbation_v1`,
  `persistence_v2`, `all_models_v1`이다.
- Ságodi et al.의 방법은 CA 특성을 평가하는 도구이며 제안 모델 자체가 아니다.
- 현재 결과는 exploratory 3-seed evidence다. 특히 \(S^2\) CA-LRU는 ID task
  gate를 통과한 seed가 하나이므로 confirmatory claim으로 쓰지 않는다.

세부 상태는 [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md), 코드별 역할은
[`docs/CODE_MAP.md`](docs/CODE_MAP.md), frozen run과 export 대응은
[`docs/EXPERIMENT_CATALOG.md`](docs/EXPERIMENT_CATALOG.md)에서 관리한다.

## 먼저 볼 곳

| 목적 | 위치 |
|---|---|
| 현재 논문 figure | [`paper/figures/`](paper/figures/) |
| 현재 topology/OOD 수치 | [`paper/evidence/topology/`](paper/evidence/topology/) |
| legacy raw에서 재집계한 표 | [`paper/evidence/tables/`](paper/evidence/tables/) |
| figure·표의 source/checksum | [`paper/artifact_checksums.json`](paper/artifact_checksums.json) |
| 현재 topology/OOD 코드 | [`repro/manifold_benchmark/`](repro/manifold_benchmark/) |
| ring/Ságodi 실험 코드 | [`repro/sagodi_protocol/`](repro/sagodi_protocol/) |
| 논문 원고·노트 상태 | [`paper/README.md`](paper/README.md) |

## 빠른 검증

표 재집계와 저장소 테스트는 Python 3.9 이상에서 실행한다.

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

make status
make verify
~~~

`make verify`는 committed raw evidence에서 표를 다시 만들고, raw provenance,
repository layout, lightweight 테스트를 검사한다. CSV float는 Python minor version의
마지막 비트 차이가 checksum을 바꾸지 않도록 12 significant digits로
직렬화한다.

PyTorch/NumPy/Matplotlib이 설치된 실험 환경에서는 전체 protocol regression도
실행한다.

~~~bash
make test-experiment
~~~

로컬 frozen experiment가 있는 환경에서는 논문 artifact도 검증한다.

~~~bash
export CALRU_EXPERIMENT_ROOT=/path/to/calru-experiments
make artifacts-list
make artifacts
make check-artifacts
~~~

`make artifacts`는 `configs/paper_artifacts.json`에 등록된 source만
`paper/`로 동기화한다. `paper/figures/`나 `paper/evidence/topology/`에 결과를
수동 복사하지 않는다.

## 저장소 구조

~~~text
configs/                 evidence·paper artifact manifest
docs/                    현재 상태, code map, experiment catalog, workflow
paper/
  manuscript/            현재 원고 조각
  notes/                 수식·방법·실험 감사
  evidence/
    raw/                 legacy 불변 입력 snapshot
    tables/              raw에서 결정적으로 재생성한 표
    topology/            현재와 archived topology/OOD export
  figures/               checksum으로 관리되는 논문 후보 그림
repro/
  manifold_benchmark/    현재 topology·OOD·perturbation pipeline
  sagodi_protocol/       ring/Ságodi-based 비교와 historical freezes
  state_dependent_retention/  archived H-C exploration
  experimental_v2/      archived 초기 P0 pipeline
  legacy_code/           최초 실험 코드 snapshot
src/calru_paper/          lightweight evidence aggregation package
scripts/                  공개 진입점과 검증 도구
tests/                    aggregation·protocol regression tests
~~~

호환성 때문에 기존 Python package와 config 이름은 한 번에 이동하지 않는다.
현재/보조/보관 여부는 디렉터리명이 아니라 [`docs/CODE_MAP.md`](docs/CODE_MAP.md)의
상태를 따른다.

## 재현 범위

현재 지원하는 것은 두 수준이다.

1. 커밋된 legacy raw JSON/CSV에서 9개 evidence table을 byte-identical하게
   재생성한다.
2. 로컬 frozen experiment가 있으면 topology/OOD export 131개를 source
   checksum과 대조한다.

모든 checkpoint를 GitHub에서 내려받아 전체 sweep을 처음부터 재학습하는 완전한
artifact는 아직 아니다. GPU 실험은 새 output directory에 실행하고, code commit,
config hash, seed, bank hash를 completion receipt에 기록해야 한다.

## GitHub 운영

- 장기 branch는 `main` 하나만 유지한다.
- protocol 변경은 짧은 `experiment/*` 또는 `agent/*` branch와 draft PR에서 한다.
- merge 뒤 remote 작업 branch는 삭제하고 중요한 freeze는 tag로 남긴다.
- 동시에 돌리는 GPU snapshot은 branch를 늘리는 대신 detached worktree를 쓴다.
- PR에는 experiment ID, config/code hash, seed denominator, artifact check와
  claim limitation을 기록한다.

구체적인 절차는 [`docs/REPOSITORY_WORKFLOW.md`](docs/REPOSITORY_WORKFLOW.md)와
[PR template](.github/pull_request_template.md)을 따른다.

## 공개 전 주의

이 저장소에는 아직 `LICENSE`가 없다. 코드·데이터·figure의 재사용 허가는
저자와 기관이 확정해야 한다. 또한 공개 GitHub 계정과 commit history는 저자를
노출하므로 AAAI double-blind 규정을 확인하기 전에는 repository visibility와
supplementary 공개 시점을 별도로 검토해야 한다.
