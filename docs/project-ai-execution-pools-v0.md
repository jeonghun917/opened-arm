# 프로젝트별 AI 실행 풀 v0

## 목적

공유 플랫폼이 `CODING_WORKER`와 `AI_REVIEW`를 서로 다른 스케줄러·비용 원장으로 운영하지 않고, 하나의 프로젝트별 자원 배분기와 하나의 실행 영수증 원장으로 관리한다.

이 계층은 **실행 자원 배분**만 담당한다. 저장소·브랜치·기준 커밋·파일 범위·병합 권한 같은 작업 권한을 새로 만들거나 넓히지 않는다.

## v0 경계

- 소유 프로젝트: `shared-platform`
- 구현 저장소: `jeonghun917/opened-arm`
- 작업 유형: `CODING_WORKER`, `AI_REVIEW`
- 프로젝트별 대기열과 프로젝트별 동시 실행 슬롯을 사용한다.
- 하나의 전역 FIFO 선두 작업이 다른 프로젝트를 막는 구조를 사용하지 않는다.
- 프로젝트별 예산과 슬롯 수는 **설정값**이며 v0가 임의 기본값을 만들지 않는다.
- 유료 실행에서 예산이 없으면 `CONFIG_REQUIRED`로 닫힌다.
- 유료 모델 호출은 명시 승인 없이는 시작하지 않는다.
- 자동 유료 재시도는 금지한다.
- 새 비밀키·provider 권한·외부 서비스 연결을 만들지 않는다.

## 공통 실행 식별자

모든 실행은 다음 다섯 값을 고정한다.

- `projectId`
- `workstreamId`
- `taskId`
- `runId`
- `authorityRef`

`AI_REVIEW`는 여기에 exact `candidateRef`를 추가로 고정한다. 공통 영수증도 같은 `authorityRef`를 반드시 싣고, AI 검수 영수증은 같은 `candidateRef`를 싣는다. task와 receipt 사이 값이 하나라도 다르면 실패한다.

모델 호출 영수증은 이 식별자와 함께 입력 토큰, 출력 토큰, 모델 호출 수, 추정 비용, 실측/provider 비용이 있으면 그 값, 결과, 재시도 횟수, 증거 참조를 기록한다.

## 프로젝트별 자원 배분

각 프로젝트는 독립적으로 다음 설정을 가진다.

- `mode`: `ENFORCED` 또는 `OBSERVE_ONLY`
- `slotCount`
- `budgetUsdMicros`

`slotCount`가 없으면 실행하지 않는다. `OBSERVE_ONLY`도 새 실행을 시작하지 않는다.

유료 작업은 추가로 다음 조건을 모두 만족해야 한다.

1. 명시 승인
2. 프로젝트 예산 설정
3. 실행 전 비용 추정값
4. 현재 완료 비용 + 실행 중 예약 비용 + 새 예약 비용이 예산 이내

프로젝트 A의 슬롯이 모두 차 있어도 프로젝트 B의 슬롯이 비어 있으면 B의 작업은 시작할 수 있다. 같은 후보에 대한 여러 `AI_REVIEW`도 해당 프로젝트 슬롯이 허용하는 범위에서 병렬 실행할 수 있다.

## 공통 비용 원장

`runner/ai_execution_pool.py`의 영수증 경로는 코딩 작업자와 AI 검수자가 공유한다. 별도 AI 검수 전용 비용 원장을 만들지 않는다.

비용은 권위 수준을 섞지 않는다.

- provider가 실제 비용을 반환하면 `authoritativeCostUsdMicros`에 기록한다.
- 토큰 단가로 계산한 값은 `estimatedCostUsdMicros`이며 provider 청구액이라고 부르지 않는다.
- 실제 비용과 사전 예약은 따로 보존한다.
- **예산 admission에는 완료 후에도 `max(사전 예약, provider 실측)`을 사용한다.** 영수증이 사전 예약보다 낮은 값을 제시해 다음 유료 실행 여유를 인위적으로 만드는 우회를 허용하지 않는다.

이 보수적 예산 차감은 provider 청구액을 왜곡하는 것이 아니다. 원장에는 `authoritativeCostUsdMicros`를 별도로 유지하고, admission 한도만 사전 예약보다 낮아지지 않게 한다.

## Coding Worker 권한 보존

Dashboard의 현행 Coding Worker v0 계약을 공유 실행 풀의 상위 권한으로 취급한다. 자원 배분기가 아래 값을 선택하거나 변경할 수 없다.

- 저장소
- 브랜치
- exact base SHA
- 허용/금지 파일 경로
- 삭제 허용 여부
- 변경량 제한

모델은 제안자일 뿐이다. 실행 결과도 `CANDIDATE_ONLY`이며 병합·배포·Continuity 완료 권한을 얻지 않는다.

Dashboard 소유 계약의 exact 현재 인터페이스는 `shared-platform-ai-pool-interface-request-v0` 작업요청으로 반환받는다. Shared Platform은 Dashboard 저장소를 직접 수정하지 않는다.

## AI 검수 권한과 토큰 긴축

기존 AWS Bedrock semantic-review 풀은 `infra/aws/semantic-review/pool.py`에 있으며, 현재 선택 모델은 `qwen.qwen3-coder-30b-a3b-v1:0`이다. 기존 풀은 reviewer를 병렬 실행하고 provider가 반환한 입력/출력 토큰을 집계한다.

AI 검수는 `HYPOTHESIS_ONLY`다. 검수자 수가 많거나 여러 검수자가 동의해도 자동 PASS/FAIL 또는 병합 권한이 생기지 않는다.

또한 **AI 검수는 기본 필수 게이트가 아니다.** 실행 풀은 여러 검수를 병렬로 돌릴 능력을 제공할 뿐, 모든 후보마다 고급 모델 검수를 자동 호출하지 않는다. 명시 승인 또는 프로젝트가 정한 위험 조건이 있을 때만 유료 `AI_REVIEW` 작업을 큐에 넣는 것이 v0 기본 정책이다.

따라서 일반 경로는 결정론적 검증과 Primary 판단을 유지할 수 있고, 별도 고비용 독립 검수는 필요한 경우에만 자원을 배정한다.

## 기존 AWS 실행 기반

새 AWS 연결을 만들지 않는다. 현재 opened-arm의 기존 OIDC 기반 semantic-review 역할/워크플로를 재사용 후보로 둔다.

- `infra/aws/semantic-review/pool.py`
- `infra/aws/semantic-review/model-selection.json`
- `.github/workflows/aws-semantic-review-pool-smoke.yml`

기존 실행기는 `automatic_retry=false`를 유지한다. v0 자원 배분기는 실제 유료 호출 전에 프로젝트 예산/승인을 추가로 확인하는 상위 계층이다.

공통 원장으로 가져올 때는 semantic-review 입력의 `candidate_ref`와 `authority_ref`를 결과에 그대로 echo하고, adapter가 실행 task의 exact `candidateRef`/`authorityRef`와 일치하는지 결정론적으로 확인한다. 기존 semantic-review 단독 실행은 이 필드가 없어도 동작하지만, 공통 원장 adapter는 두 값이 없거나 다르면 실패한다.

## 아직 CONFIG_REQUIRED인 부분

### 1. 프로젝트 초기 설정

v0는 달러 예산이나 슬롯 수를 임의로 정하지 않는다. 프로젝트가 명시적으로 설정하기 전 유료 실행은 닫혀 있다.

### 2. Coding Worker 모델 실행 어댑터

Coding Worker의 권한 계약은 Dashboard에 있고, 실제 모델 실행 기반은 Shared Platform에서 아직 승인·고정되지 않았다. Dashboard의 인터페이스 반환을 받은 뒤 권한을 복제하지 않는 어댑터만 연결한다.

### 3. 실시간 다중 실행용 영속 슬롯 저장소

`runner/ai_execution_pool.py`는 저장소 중립적인 결정론적 코어다. 여러 GitHub runner가 동시에 슬롯을 점유하려면 compare-and-set 또는 lease 성격의 승인된 영속 저장 어댑터가 필요하다.

현재 opened-arm의 승인된 semantic-review AWS 역할은 선택 Bedrock 모델 호출만 허용하며 durable store 권한은 없다. Dashboard의 Neon/Vercel OIDC 저장 계층도 Dashboard 소유이므로 Shared Platform이 권한을 가정해 가져오지 않는다. 따라서 새 비밀키·provider 권한 없이 재사용 가능한 원자적 저장 계층이 반환되기 전에는 이 항목을 `CONFIG_REQUIRED`로 유지한다.

v0는 이 문제를 파일 락, process-local lock, GitHub artifact 또는 단일 전역 FIFO로 가장하지 않는다.

## 자체 검증

다음 명령은 provider 호출 없이 코어 규칙을 검증한다.

```bash
python3 runner/ai_execution_pool.py self-test
```

검증 항목:

- 프로젝트 A가 막혀 있어도 B가 실행되는가
- 예산 없는 유료 실행이 닫히는가
- 같은 후보의 AI 검수 2개가 슬롯 2개에서 병렬 배정되는가
- Coding Worker와 AI Review가 같은 영수증 원장을 쓰는가
- AI Review에 병합 권한이 생기지 않는가
- 실행 영수증의 프로젝트/작업분기/실행/authority identity 불일치가 거부되는가
- AI Review의 exact candidate 불일치가 거부되는가
- 낮은 사후 영수증 비용으로 사전 예약 예산을 해제할 수 없는가
- 자동 재시도가 거부되는가

## 비범위

- 프로젝트 권한 정책 자체의 재설계
- consumer 저장소 직접 수정
- provider 권한·비밀키 생성
- 자동 병합/배포
- 자동 유료 검수
- 중앙 단일 FIFO 구축
