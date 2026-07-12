# Experiment reproduction

이 폴더는 논문 수치를 만든 실험 코드의 스냅샷을 보존한다. 논문에서 사용할 표준 이름은 **CA-LRU**이지만, 기존 checkpoint·JSON 파일과의 호환성을 유지하기 위해 실험 코드 내부의 `PAN`, `CAMN`, `AM-LRU` 식별자는 그대로 두었다. 이 구분은 [`docs/LEGACY_NAME_MAP.md`](../docs/LEGACY_NAME_MAP.md)에 기록한다.

## 범위

- `legacy_code/`: main manifold, writer/update ablation, RP control, retention scaling, discrete control, ring transport 및 Jacobian 분석에 필요한 Python/shell 스냅샷
- `requirements-experiment.txt`: 기존 실험에서 확인된 핵심 패키지 버전

원본 discrete launcher의 로컬 절대경로·특정 conda interpreter·기존 K-way
seed 0 가정은 public artifact import 과정에서 제거했다.
`legacy_code/run_e8_discrete_full.sh`는 script 위치에서 실행하고, `PYTHON`
환경변수를 사용하며, K-way와 flip-flop 모두 seed 0/1/2를 돌린다.

체크포인트와 trace는 용량 및 저작자 식별 가능성 때문에 이 폴더에 복사하지 않았다. 따라서 현재 저장소는 저장된 raw metric에서 논문 표를 **완전히 재집계**할 수 있지만, 초기 checkpoint까지 포함한 모든 학습 run을 byte-for-byte 재현한다고 주장하지 않는다.

## 환경

기존 run에서 확인된 환경은 Python 3.10.20, PyTorch 2.4.0+cu121, NumPy 1.26.4, Matplotlib 3.5.3이다. CUDA 및 GPU 종류, cuDNN 버전, deterministic algorithm 설정은 최종 artifact metadata에 추가해야 한다.

~~~bash
python -m venv .venv-experiment
source .venv-experiment/bin/activate
python -m pip install -r repro/requirements-experiment.txt
cd repro/legacy_code
~~~

## 대표 명령

~~~bash
python run_exp88_manifold_main.py --seeds 0,1,2
python run_exp88_surface_main.py --seeds 0,1,2
python run_exp88_writer_sweep.py --seeds 0,1,2 --include-surface
python run_exp88_shuffle_matched.py --seeds 0,1,2
python run_e7_line_dim_rnw.py --seeds 0,1,2
bash run_e8_discrete_full.sh 5
~~~

Discrete launcher에 사용할 interpreter는 필요하면
`PYTHON=/path/to/python bash run_e8_discrete_full.sh 0`처럼 지정한다. 인자는
`CUDA_VISIBLE_DEVICES`에 넘길 GPU index이며 기본값은 0이다.

실행 전에 launcher의 GPU 지정, output overwrite 옵션, output/checkpoint/trace 경로를 반드시 확인한다. 표에 쓸 결과는 바로 덮어쓰지 말고 새 artifact 디렉터리에 생성한 뒤 기존 수치와 비교한다.
