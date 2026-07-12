# Frozen manuscript archive

이 디렉터리는 모델명이 PAN, AM-LRU, CAMN을 거쳐 CA-LRU로 바뀐 과정과 논문
구조의 변화를 보존한다. 파일은 역사적 참고 자료이며 수정하지 않는다.

## 사용 금지 범위

과거 초안에는 이후 코드 감사에서 정정된 내용이 포함되어 있다. 특히 다음 문장을
현재 주장으로 재사용하면 안 된다.

- final CA-LRU와 no-RP/all-slow가 동일 scaffold라는 표현
- all-slow retention이 0.99라는 표현; 실제 control은 0.999다.
- 모든 manifold가 동일한 방식으로 다섯 진단을 3-seed 통과했다는 표현
- persistent coordinate 범위를 3–9/96으로 쓰는 표현; 선택된 main rows의 실제
  seed별 범위는 3–8/96이다.
- final CA-LRU에서 Jacobian이 tangent-neutral/normal-contracting임을 이미
  입증했다는 표현
- CA-LRU가 discrete flip-flop을 일관되게 완벽히 해결했다는 표현

최신 수치와 허용 가능한 주장은
[`paper/notes/methods_equations_data_ko.md`](../notes/methods_equations_data_ko.md)와
[`paper/evidence/`](../evidence/)를 따른다.

Archive 파일에서 유용한 문장을 가져와야 할 때는 원문을 직접 덮어쓰지 말고 현재
manuscript에 새로 작성한 뒤, 연결된 table과 evidence tier를 검토한다.
