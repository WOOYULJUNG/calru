# Repository and branch workflow

## 기준

- `main`은 GitHub의 유일한 장기 브랜치이자 현재 source of truth다.
- 실험 결과는 Git branch가 아니라 `/home/biadmin/ca_rnn/experiments/<run-id>`에
  저장한다.
- 논문에 쓰는 작은 export만 `paper/evidence/`와 `paper/figures/`에 커밋한다.
- branch 이름으로 결과 provenance를 표현하지 않는다. 각 run manifest에 source
  commit을 기록한다.

## 새 작업

작은 문서·figure 갱신은 `main`에서 한 묶음으로 커밋한다. 학습 코드나 frozen
protocol을 바꾸는 작업만 짧은 branch를 만든다.

~~~bash
git switch main
git pull --ff-only
git switch -c experiment/<short-topic>
~~~

branch 하나에는 protocol 변경 하나만 둔다. 완료되면:

1. 테스트와 artifact check를 통과시킨다.
2. `main`에 fast-forward 또는 리뷰된 PR로 합친다.
3. remote experiment branch를 삭제한다.
4. 중요한 freeze는 tag로 남긴다.

`analysis/<topic>` branch를 별도로 만들지 않는다. 분석 코드는 해당 실험 branch에
함께 두고, 결과는 experiment ID로 구분한다.

## GPU run용 worktree

동시에 실행할 코드 snapshot은 branch가 아니라 detached worktree로 만든다.

~~~bash
git worktree add --detach ../worktrees/run-<short-sha> <full-commit>
~~~

학습을 시작하기 전에 code/config를 커밋하고, run manifest에 다음을 저장한다.

- full source commit
- dirty-worktree 여부
- config SHA-256
- model, topology, seed
- output root

job이 끝나고 결과가 검증되면 detached worktree는 제거한다. 사용자 수정이 남은
worktree는 정리하지 않는다.

## Figure 업데이트

`paper/figures/`에 수동으로 파일을 던지지 않는다.

1. generator가 frozen experiment output에 figure와 표를 쓴다.
2. `configs/paper_artifacts.json`에 source/destination을 등록한다.
3. `make artifacts`로 canonical paper tree를 동기화한다.
4. `make check-artifacts`와 테스트 후 한 commit으로 올린다.

이 방식이면 GitHub에서는 `main → paper/figures → paper/evidence`만 보면 되고,
로컬의 수천 개 run directory를 알 필요가 없다.
