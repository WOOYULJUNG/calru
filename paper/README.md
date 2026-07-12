# Paper workspace

이 디렉터리는 논문 문장, 감사 노트, 수치 증거와 향후 figure를 서로 다른
수명주기로 관리한다. 한 파일을 “전체 논문의 최신본”으로 간주하지 말고 아래의
우선순위를 따른다.

## 내용과 우선순위

1. [`evidence/raw/`](evidence/raw/)와 집계 코드가 수치의 1차 근거다.
2. [`evidence/tables/`](evidence/tables/)는 raw snapshot에서 재생성한 논문용
   요약표다.
3. [`notes/methods_equations_data_ko.md`](notes/methods_equations_data_ko.md)는
   수식, 실험 방법, 수치 주장과 한계를 코드에 대조한 감사본이다.
4. [`manuscript/sections_01_02.md`](manuscript/sections_01_02.md)는 현재 최신인
   Introduction과 Related Work 조각이다.
5. [`archive/`](archive/)는 역사 보존용이며 현재 주장이나 수치의 근거가 아니다.

## 현재 빠진 논문 자산

- Sections 3–8의 통합 최신 원고
- 최종 Abstract
- LaTeX source와 bibliography
- 컴파일된 최종 PDF
- 최종 FIG-1–FIG-5 및 그 생성 스크립트

과거 작업 기록에 이러한 파일이 있었다는 언급은 남아 있지만, 이 repository를
만든 시점의 접근 가능한 snapshot에서는 확인되지 않았다. 존재하지 않는 파일을
README 링크나 재현 명령에 넣지 않는다.

## 편집 원칙

- 논문에서 새로 만든 용어는 **CA-LRU**와 **Retention Plasticity (RP)** 두 개만
  사용한다.
- PAN, CAMN, AM-LRU는 raw filename과 코드 provenance에만 남긴다.
- 모든 숫자는 [`evidence/tables/`](evidence/tables/)의 행과 연결한다.
- approximate continuous attractor는 finite-horizon behavioral/local 의미로만
  사용하며 exact invariant manifold나 global stability를 주장하지 않는다.
- archive의 문장을 현재 원고로 가져올 때는 먼저 감사 노트와 data dictionary에
  대조한다.

공개 전에는 루트 [`README.md`](../README.md)의 AAAI double-blind 주의를 반드시
확인한다.
