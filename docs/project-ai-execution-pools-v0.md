# 프로젝트별 AI 실행 풀 v0

## 목적

Shared Platform이 `CODING_WORKER`와 `AI_REVIEW`를 하나의 프로젝트별 자원 배분기와 공통 실행 영수증 장부로 관리한다. 이 계층은 실행 자원을 배정할 뿐, 작업 권한을 새로 만들거나 넓히지 않는다.

## Dashboard 이관 결과 반영

`shared-execution-contract-handoff-v0` 결과는 `DONE`으로 반환됐고, 공유 실행 정본의 경계가 확정됐다. 따라서 다음 계약은 `jeonghun917/opened-arm`의 Shared Platform 정본으로 유지한다.

- Coding Worker의 호출자 고정 `repository` / `branch` / `baseSha`
- 허용·금지 경로, 삭제 허용, 변경 개수·파일 크기·총 크기 제한
- 정규화된 계약의 SHA-256 증거 결합
- 추론기나 변경 제안이 대상·경로 권한을 덮어쓰지 못하는 규칙
- 실행 영수증의 대상 신원 결합과 exact base / branch HEAD / changed-path 독립 검증
- 결과 권위 `CANDIDATE_ONLY`, 독립 검토 필요, Continuity 자기완료·병합 권한 없음
- 범용 실행면 registry / route / preflight와 수동 승인·비용·재시도 실패닫힘 규칙

Dashboard 전용 기본 실행면 구성과 GitHub Console 도구는 이 정본에 포함하지 않는다. Dashboard의 기존 저장·실행 코드는 Shared Platform 동등성 및 영구 저장이 확인되기 전까지 호환 거울로 남는다.

## 공통 실행 식별자와 비용 장부

모든 실행은 `projectId`, `workstreamId`, `taskId`, `runId`, `authorityRef`에 묶인다. `AI_REVIEW`는 exact `candidateRef`도 필수다. 작업과 영수증의 값이 하나라도 다르면 실패한다.

프로젝트마다 `mode`, `slotCount`, `budgetUsdMicros`를 독립 설정한다. 임의 기본 슬롯·예산은 만들지 않는다. 유료 실행은 명시 승인, 프로젝트 예산, 실행 전 비용 추정, 예산 여유를 모두 만족해야 한다. 자동 유료 재시도는 금지한다.

완료 비용의 admission charge는 `max(사전 예약, provider 실측)`을 사용한다. provider 실측 비용은 별도 필드에 보존하므로 청구액과 보수적 admission 회계를 섞지 않는다.

## Coding Worker 정본 계약

`runner/ai_execution_pool.py`는 Dashboard에서 이관된 Coding Worker 불변조건을 저장소 중립 계약으로 구현한다.

- `repository`, `branch`, `baseSha`는 호출자가 고정한다.
- `allowedPaths`, `forbiddenPaths`, `allowDelete`와 변경량 상한은 계약에 고정한다.
- 계약은 `coding-worker-contract:v0:sha256:<digest>` 증거로 결합한다.
- 제안은 대상·경로 정책을 덮어쓸 수 없다.
- 성공 영수증은 exact 대상, base SHA, 새 commit SHA, changed paths, 계약 digest를 다시 결합한다.
- 결정론 검증은 observed base, branch HEAD, changed paths, evidence를 독립 확인한다.
- 성공해도 `CANDIDATE_ONLY`이며 독립 검토가 필요하고 병합·Continuity 완료 권한은 없다.

실제 코딩 모형 실행 공급자는 아직 승인·고정되지 않았으므로 `CONTRACT_READY_EXECUTION_CONFIG_REQUIRED` 상태다. 계약 이관 완료와 실제 유료/공급자 실행 승인은 별개다.

## 범용 실행면 정본 계약

같은 코어는 Dashboard 제품 설정을 제외한 범용 실행면 의미를 제공한다.

- registry의 plane과 capability route를 정규화한다.
- 경로 선택은 `EXACT` 또는 `ORDERED`다.
- 위임되지 않은 capability, 잘못된 plane, 비활성 plane은 실패 닫힘한다.
- `PAID`, `MANUAL_ONLY`, `MANUAL_REQUIRED`는 명시 수동 승인 없이는 실행하지 않는다.
- 자동 재시도는 route·plane·task policy 모두 허용해야 하며 기본 상한은 0이다.

Dashboard 전용 plane ID나 소비자별 경로 상수는 Shared Platform 공용 정본에 넣지 않는다.

## 기존 AWS 의미 검수기

실제 의미 검수 실행 정본은 계속 다음 경로다.

- `infra/aws/semantic-review/pool.py`
- `runner/semantic_review_pool_adapter.py`
- `infra/aws/semantic-review/model-selection.json`

adapter는 exact project/workstream/task/run/candidate/authority 신원을 공통 장부에 결합한다. `AI_REVIEW`는 `HYPOTHESIS_ONLY`, 병합 권한 없음, 자동 재시도 없음 경계를 유지한다. 이 작업은 새 AWS 권한이나 비밀정보를 만들지 않는다.

## 남은 CONFIG_REQUIRED

### Shared Platform 영구 슬롯·비용 저장

여러 실행 사이 슬롯과 비용 상태를 원자적으로 보존할 Shared Platform 소유 영구 저장이 아직 없다. 코어는 다음 조건을 만족하지 않으면 영구 저장을 `CONFIG_REQUIRED` 또는 `DENY_STORE`로 닫는다.

- 소유 프로젝트가 `shared-platform`
- 원자성 계약이 `LEASE_CAS` 또는 `SERIALIZABLE`
- 실제 `backendRef`가 명시됨

Dashboard Neon/Vercel OIDC 저장을 자동 상속하지 않는다. 기존 semantic-review AWS 역할에도 영구 저장 권한을 추가하지 않는다. 파일 잠금·process-local lock·artifact·전역 FIFO로 원자 저장을 가장하지 않는다.

### 프로젝트별 실제 실행 설정

초기 슬롯·예산·유료 실행 승인은 각 프로젝트가 명시해야 한다. Coding Worker의 실제 실행 공급자도 별도 승인 계약이 필요하다.

## 결정론 검증

```bash
python3 runner/ai_execution_pool.py self-test
python3 runner/semantic_review_pool_adapter.py self-test
```

자체검증은 기존 프로젝트별 배정·예산·공통 장부 회귀시험과 함께 다음을 확인한다.

- Coding Worker exact target / path policy / 계약 digest 고정
- 대상·경로 권한 덮어쓰기 거부
- exact base / branch HEAD / changed-path 결정론 검증
- `CANDIDATE_ONLY`, 독립 검토, no-merge / no-self-close
- 실행면 위임 capability, wrong-plane, paid approval, retry 실패닫힘
- Shared Platform 소유가 아닌 영구 저장 거부

## 비범위

- Dashboard 저장소 수정·삭제
- 소비자 저장소 권한 확대
- 새 공급자 권한·비밀정보 생성
- 자동 유료 검수·자동 유료 재시도
- 자동 병합·배포·Continuity 완료
