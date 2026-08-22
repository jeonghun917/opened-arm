# VELA microbudget campaign v6

상태: **PREPARED / NO PROVIDER JOB SUBMITTED**

## 예산
- Kaggle: 무료 quota 범위에서 별도 사용자 금액 상한 없음.
- Lightning 기본: $2.50.
- Modal 기본: $2.50.
- 추가 공용 reserve: **$2.00**.
- 새 승인 전 총 유료 hard ceiling: **$7.00**.
- 단일 유료 provider 절대 ceiling: **$4.50**.

추가 reserve는 단순 반복보다 실패 원인 식별, 새 후보 축 확인, provider 재현 실패 해소에 우선 사용한다.

## 격리
- `c3-voice-lab`은 건드리지 않는다.
- GitHub 실험 코드는 `Ars-Mentis` 전용 repo가 우선이며, 초기화가 막히면 `opened-arm:vela-experiment-infra` branch만 사용한다.
- 다른 프로젝트 main branch는 수정하지 않는다.
