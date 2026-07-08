# 01. ACT (~52M) — FT 참조군 (⚠️ Zero-shot 아님)

> **역할**: 성능 하한 베이스라인 + 수집·학습·배포 파이프라인 전구간 검증용. "VLA가 정말 더 나은가"의 대조군.
> **중요 고지**: ACT는 사전학습 체크포인트가 없는 task-specific 모델이라 **zero-shot 시연이 원천적으로 불가능**.
> 예외적으로 **파인튜닝(학습) 후 시연**하며, 모든 산출물에 **"FT 참조군"** 라벨을 명시한다.

## 모델 개요

| 항목 | 내용 |
|---|---|
| 파라미터 | ~52M |
| 사전학습 체크포인트 | ✕ (태스크별 학습) |
| 학습 입력 | 이미지 + state (LeRobotDataset) — **언어 필드 불필요** |
| 추론 입력 | 카메라 프레임 + 현재 joint state |
| 출력 | joint action chunk (향후 k-step 관절값, action chunking) |
| 특성 | 가벼움·빠름·단일태스크 강함, 언어 조건 불가 |
| 5070 Ti 16GB | ✅ 완전 여유 (Full FT 가능) |
| SO-ARM 궁합 | SO-ARM101은 ALOHA와 같은 leader–follower·joint-space 시연 방식이라 ACT와 궁합이 좋음 |

## 데이터 요구사항

- **포맷**: LeRobotDataset 그대로, 추가 요구사항 없음
- **수집량**: T1 ~30ep + T2 ~30ep = 60ep (teleop)
- **수집 규칙**:
  - 3지점 그리드(테이핑) 커버
  - **카메라 뷰만 보고 조작** (cheating 방지)
  - 레퍼런스 데이터셋(`lerobot/svla_so101_pickplace`, `youliangtan/so101-table-cleanup`)의
    물체·배치·조명을 최대한 재현 — zero-shot 3종과 조건 통일
  - 카메라 이름은 공통 규칙(`top`/`wrist`) 준수 — 이후 4모델 재사용 전제

## 진행 절차

- [x] **1. 데이터 수집 (T1)**
  - [x] T1 Pick-and-Place 30ep 완료 (7/8) — 13,032 frames, 평균 14.5초/ep, NaN 0
  - [ ] T2 Table Cleanup ~30ep (D3)
  - [x] T1 데이터셋 HF Hub push: `heongyu/so101_t1_pickplace`
- [ ] **2. 학습** (task별 2회, 각 ~3시간)
  - [x] ACT-T1 학습 기동 (7/8 밤) — loss 6.74→2.47 (600step) 하강 확인, 9.2 steps/s, VRAM 5.3GB
    - 완료 시 자동 push: `heongyu/act_so101_t1`
  - [ ] ACT-T2 학습 (D4 오전)
- [ ] **3. 시연 준비**
  - [ ] 체크포인트 로드 sanity check (key/shape 정합)
  - [ ] dry-run
- [ ] **4. 측정**
  - [ ] T1 10회 (SR/PR/지연/VRAM/실패유형 기록)
  - [ ] T2 10회
- [ ] **5. 산출물 완결** → SmolVLA로

## 완결 체크리스트 (게이트)

- [ ] 시행 로그 20회분 (2task × 10회)
- [ ] task별 대표 클립 각 1개 + 실패 대표 클립 1개
- [ ] 모델 요약 카드 (반 페이지)
- [ ] **학습 로그·체크포인트 HF Hub 업로드, 수집 데이터셋 push** (ACT 전용 항목)
- [ ] 로그·클립 백업

## 리스크 & 대응

| 리스크 | 대응 |
|---|---|
| 학습 시간이 일정 압박 | 야간 학습 슬롯 활용 (T1 밤, T2 다음날 오전), 30ep 파일럿 규모 유지. 학습 중 SmolVLA 준비 병행 가능 |
| FT 결과가 zero-shot과 오해석 | 모든 표·클립에 FT/ZS 라벨 명시 |
| 학습이 D4 오전까지 미완 | T2 학습을 D4 야간으로 밀고 D4 오후는 T1 측정만 완결 → T2 측정은 D5 오전 편입 |

## 결과 기록

### T1. Pick-and-Place (FT)
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

### T2. Table Cleanup (FT)
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

### 요약 카드
- **셋업 특이사항**:
- **수치 요약**: T1 SR __/10, T2 SR __/10, 평균 PR __, 지연 __ms, VRAM 피크 __GB
- **관찰 메모**:
- **다음 단계 시사점**:

## 참고 자료

- ACT 논문: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ALOHA)
- LeRobot SO-101 문서: https://huggingface.co/docs/lerobot/so101
- Dataset Visualizer: https://huggingface.co/spaces/lerobot/visualize_dataset
