# 02. SmolVLA (450M) — Zero-shot 주력

> **역할**: 본 프로젝트의 주력 모델. **SO-ARM 계열 데이터로 사전학습된 유일한 모델**이라
> zero-shot 3종 중 가장 높은 성공 확률 기대 — 측정 프로토콜의 리허설 효과 겸 심리적 앵커.
> SmolVLA zero-shot 정량치는 공개 자료에 없음 → **우리 측정치가 1차 자료가 된다.**

## 모델 개요

| 항목 | 내용 |
|---|---|
| 파라미터 | 450M |
| 사전학습 체크포인트 | ✅ `lerobot/smolvla_base` |
| 사전학습 데이터 | LeRobot 커뮤니티 데이터 (주로 **SO-100** tabletop, 481 dataset / ~23k ep) → SO-ARM101과 사실상 동일 계열 임바디먼트 (**in-distribution에 가까운 유일 모델**) |
| 학습 입력 | 이미지 + state + **언어** (LeRobotDataset) |
| 추론 입력 | 카메라 프레임 + joint state + 자연어 명령 |
| 출력 | flow-matching action chunk (joint) |
| 특성 | 컨슈머 GPU 설계, 언어 조건부 |
| 5070 Ti 16GB | ✅ Zero-shot 추론 및 Full FT 모두 가능 (로컬 완결) |

## 데이터 요구사항 (향후 FT 대비)

- **포맷**: LeRobotDataset + **에피소드별 자연어 task 설명 필드** (수집 시 포함하면 끝)
- 참고: SmolVLA 문서가 "25 에피소드로는 성능 나쁨"을 명시 — 향후 few-ep FT 비교(zero-shot / 25ep / 50ep+)의 기준점

## 진행 절차 (D5)

- [ ] **1. 셋업**
  - [ ] `lerobot/smolvla_base` 로드
  - [ ] 카메라 key / `observation.state` shape 정합 확인
  - [ ] dry-run
- [ ] **2. 측정**
  - [ ] T1 10회 ("Pick up the cube and place it in the box")
  - [ ] T2 10회 ("Put the [object] in the container") — **언어 명령 변경 반응 관찰 포함**
- [ ] **3. 산출물 완결** → π0로

## 완결 체크리스트 (게이트)

- [ ] 시행 로그 20회분 (2task × 10회)
- [ ] task별 대표 클립 각 1개 + 실패 대표 클립 1개
- [ ] 모델 요약 카드 (반 페이지)
- [ ] 로그·클립 백업

## 리스크 & 대응

| 리스크 | 대응 |
|---|---|
| 카메라 key mismatch | dry-run에서 선확인 |
| ACT T2 측정이 D5로 밀린 경우 | D5 오전 1시간 편입 — SmolVLA 완결은 유지 |

## 결과 기록

### T1. Pick-and-Place (ZS)
| 시행 | SR | PR | 지연(ms) | VRAM | 실패유형 | 메모 |
|---|---|---|---|---|---|---|
| 1 | | | | | | |
| 2 | | | | | | |
| 3 | | | | | | |
| 4 | | | | | | |
| 5 | | | | | | |
| 6 | | | | | | |
| 7 | | | | | | |
| 8 | | | | | | |
| 9 | | | | | | |
| 10 | | | | | | |

### T2. Table Cleanup (ZS)
| 시행 | SR | PR | 지연(ms) | VRAM | 실패유형 | 언어 반응 메모 |
|---|---|---|---|---|---|---|
| 1 | | | | | | |
| 2 | | | | | | |
| 3 | | | | | | |
| 4 | | | | | | |
| 5 | | | | | | |
| 6 | | | | | | |
| 7 | | | | | | |
| 8 | | | | | | |
| 9 | | | | | | |
| 10 | | | | | | |

### 요약 카드
- **셋업 특이사항**:
- **수치 요약**: T1 SR __/10, T2 SR __/10, 평균 PR __, 지연 __ms, VRAM 피크 __GB
- **관찰 메모** (언어 명령 변경 반응 포함):
- **다음 단계 시사점**:

## 참고 자료

- SmolVLA 논문: arXiv:2506.01844 — "SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics"
- 블로그: https://huggingface.co/blog/smolvla
- 공식 예제 데이터셋: `lerobot/svla_so101_pickplace`
