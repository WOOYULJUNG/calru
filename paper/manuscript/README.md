# Current manuscript fragments

[`sections_01_02.md`](sections_01_02.md)는 현재 협업에서 확정한 Introduction과
Related Work의 Markdown 원고다. 제목은 *Learning Approximate Attractor
Dynamics for Recurrent Memory*이다.

이 파일은 완성 논문이 아니다. Sections 3–8, Appendix의 통합 최신본과 최종
Abstract는 아직 이 디렉터리에 없고, LaTeX/PDF source도 없다. 따라서 해당 파일의
빈 section heading이나 과거 초안의 본문을 최신 완성본으로 해석하면 안 된다.

## 문장과 수치 규칙

- paper-facing 명칭은 CA-LRU와 Retention Plasticity (RP)를 사용한다.
- legacy identifier는 [`docs/LEGACY_NAME_MAP.md`](../../docs/LEGACY_NAME_MAP.md)에
  연결할 때만 노출한다.
- 실험 숫자를 넣기 전에 [`paper/evidence/tables/`](../evidence/tables/)과
  [`paper/evidence/DATA_DICTIONARY.md`](../evidence/DATA_DICTIONARY.md)를 확인한다.
- seed 수, 평균과 표준편차, evaluation endpoint를 함께 기록한다.
- on-manifold movement의 정량 일반화는 현재 ring 중심이며, 이를 모든 manifold의
  동일한 3-seed 검증으로 확대해서 쓰지 않는다.

## 향후 파일 규칙

통합 원고를 만들 때는 `main.tex` 또는 `main.md`처럼 명확한 단일 진입점을 만들고,
bibliography와 section 파일을 repo-relative include로 연결한다. 빌드 산출물은
`paper/build/`에 두며 source와 최종 제출본을 구분한다.

AAAI double-blind 제출 전 공개 repository, commit author, PDF metadata와 문서 내
소속 정보가 익명성을 깨지 않는지 별도로 확인한다.
