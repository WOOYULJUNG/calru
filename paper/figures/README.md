# Paper figures

현재 이 디렉터리에는 persistent-topology 분석으로 생성한 measured-data
figure가 있다. 아직 본문 figure 번호는 확정하지 않았다.

## Persistent topology figures

- `fig_topology_signature_heatmap.{pdf,png}`: 모델·topology·blank horizon별
  3-seed persistent \(H_1/H_2\) signature 통과율. 셀 숫자는 강한
  persistent \(H_1/H_2\) bar 수의 seed median이다.
- `fig_topology_diagram_distance.{pdf,png}`: 이상적 topology persistence
  diagram에 대한 bottleneck distance. 선은 3-seed median, band는 seed
  range이며 y축은 log scale이다.
- `fig_topology_persistence_diagrams_h0.{pdf,png}`: blank 이전 representative
  seed-10 persistence diagrams.
- `fig_topology_persistence_diagrams_h2048.{pdf,png}`: 2,048 blank step 이후
  representative seed-10 persistence diagrams.

모두 실제 checkpoint와 hidden state로부터 측정한 결과이며 concept diagram이
아니다. 생성 코드는
[`repro/manifold_benchmark/analyze_persistent_topology.py`](../../repro/manifold_benchmark/analyze_persistent_topology.py),
동결된 계약은
[`topology_persistence_v1.json`](../../repro/manifold_benchmark/topology_persistence_v1.json)에
있다. 숫자는 `paper/evidence/tables/persistent_topology_*.csv`와
`persistent_topology_reference.json`에서 추적할 수 있다. 분석일은
2026-07-17이고, RNN/GRU/LSTM은 ring-selected zero-retuning, CA-LRU는
validation-selected, H-C는 test 접근 전 고정한 CA-aware finalist를 사용했다.

## 아직 복구되지 않은 기존 figures

과거 작업 요약에는 teaser, architecture, on-manifold movement, OOD, retention
scaling의 FIG-1–FIG-5와 LaTeX용 PDF가 생성되었다고 기록되어 있다. 그러나 이
repository를 정리할 때 접근 가능한 snapshot에서는 해당 최종 파일과 원래의 figure
source 디렉터리를 찾지 못했다. 따라서 진단용으로 남아 있던 다른 PNG/SVG를 최종
figure인 것처럼 복사하지 않았다.

## 향후 추가 규칙

Figure 하나를 추가할 때 최소한 다음을 함께 둔다.

- `figNN_short_name.pdf` 또는 `.svg`: 논문용 vector export
- 필요하면 동일 stem의 `.png`: GitHub preview
- 생성 script 또는 정확한 외부 생성 절차
- 입력 table/raw pattern과 seed
- concept diagram인지 measured data figure인지 밝힌 caption note
- 생성 환경과 날짜

Figure 내부 숫자는 [`paper/evidence/tables/`](../evidence/tables/) 또는 명시된
seed-level raw file에서 추적할 수 있어야 한다. Seed-0 post-hoc probe는 전체
3-seed 결과처럼 표시하지 않는다.

PDF/SVG metadata, embedded author name과 절대 경로는 AAAI double-blind 공개 전에
검사한다. 최종 figure가 없다는 현재 상태는 라이선스가 확정되었다는 뜻도 아니다.
