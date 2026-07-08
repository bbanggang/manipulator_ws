# ACT 모델 진행 보고서 — SO-ARM101 Pick-and-Place

> 작성일: 2026-07-08 | 작성 시점: ACT-T1 학습 진행 중 (step ~9,400/100,000, loss 0.178)
> 프로젝트: 단일 SO-ARM101 × 4모델(ACT/SmolVLA/π0/GR00T) 학습 및 zero-shot 비교

---

## 1. ACT의 역할과 배경

### 1.1 왜 ACT부터 시작하는가

ACT(Action Chunking with Transformers)는 본 프로젝트의 **FT(파인튜닝) 참조군**이다.
비교 대상 4개 모델 중 유일하게 **사전학습 체크포인트가 없는 task-specific 모델**이라
zero-shot 시연이 원천적으로 불가능하며, 예외적으로 "데이터 수집 → 학습 → 시연" 전체
파이프라인을 거친다. 이를 맨 앞에 배치한 이유는 세 가지다:

1. **파이프라인 전구간 검증** — 수집·학습·배포를 한 번 완주하면 이후 zero-shot 3종의 셋업 리스크가 줄어든다
2. **성능 상한 참조** — "소량 데이터로 파인튜닝하면 이 정도 나온다"는 기준선 (zero-shot 3종의 하한 기준선과 대비)
3. **데이터 자산화** — 수집한 에피소드는 이후 SmolVLA/π0/GR00T 파인튜닝 단계에 그대로 재사용

SO-ARM101은 ALOHA와 같은 **leader–follower·joint-space 시연 방식**이라 ACT와 궁합이
좋고, ~52M 파라미터의 경량 모델이라 RTX 5070 Ti(16GB)에서 여유 있게 full 학습이 가능하다.

### 1.2 참고 자료

| 자료 | 용도 |
|---|---|
| ACT 논문: *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (ALOHA) | 모델 구조·loss·action chunking 개념 |
| [LeRobot SO-101 공식 문서](https://huggingface.co/docs/lerobot/so101) | 조립·캘리브레이션·teleop·record 절차 |
| [LeRobotDataset v3 문서](https://huggingface.co/docs/lerobot/lerobot-dataset-v3) | 데이터셋 저장 구조 |
| 레퍼런스 데이터셋 `lerobot/svla_so101_pickplace` | task 셋업(물체·배치·조명) 재현 기준, [Visualizer](https://huggingface.co/spaces/lerobot/visualize_dataset)로 관찰 |
| Notion: 모델 비교 조사 보고서 / Zero-shot 시연 계획서 / 일별 계획 | 실험 설계·평가 프로토콜·일정 |

---

## 2. 하드웨어 및 환경 구성

### 2.1 로봇·카메라

```
[Leader 팔]  ── 사람이 조작 ──▶ 관절값 실시간 복사 ──▶ [Follower 팔] ── 실제 작업 수행
   /dev/ttyLEADER (시리얼 5AE6054948)              /dev/ttyFOLLOWER (시리얼 5AE6083400)

[top 카메라]   /dev/cam_top   — 작업 공간 전경 부감(俯瞰) 뷰: 그리드 3지점·큐브·박스·팔 전체
[wrist 카메라] /dev/cam_wrist — 그리퍼 1인칭 뷰: 파지 정밀도·접근 각도
```

- 두 카메라는 **서로 보완적**이다: top은 "어디로 가야 하는가"(전역 위치), wrist는
  "지금 잡을 수 있는가"(국소 정밀)를 담당한다. ACT는 두 뷰를 모두 입력받아 학습한다.
- 640×480 @ 30fps, **MJPG 강제** — 기본 YUYV 무압축은 2캠 295Mbps로 USB 허브(480Mbps)를
  포화시켜 wrist 영상이 깨지는 문제를 실측으로 확인, MJPG(~10분의 1 대역폭)로 해결했다.
- udev rules(`setup/99-lerobot.rules`)로 장치 경로를 고정 — 팔은 어댑터 시리얼 번호,
  카메라는 물리 포트 기준. 재부팅·재연결에도 경로가 유지된다.

### 2.2 소프트웨어

| 구성 | 값 | 비고 |
|---|---|---|
| OS / Python | Ubuntu 24.04 / Python 3.10 (uv로 고정) | 시스템 3.12는 LeRobot 생태계와 충돌 여지 |
| 환경 관리 | `envs/lerobot` uv 프로젝트 | `uv.lock`으로 2인 환경 재현 |
| 프레임워크 | lerobot 0.4.4, torch 2.10.0+cu128 | RTX 5070 Ti(Blackwell sm_120)는 CUDA 12.8+ 필수 |
| 영상 코덱 | 저장 AV1 / 디코딩 torchcodec(+시스템 ffmpeg) | ffmpeg 미설치 시 학습 데이터 로더가 실패 |
| 공유 채널 | HF Hub private repo | 데이터셋·체크포인트 왕복 |

---

## 3. 데이터 수집

### 3.1 Task 정의

**T1. Pick-and-Place**: 빨간 큐브를 집어 검은 박스에 넣는다.
언어 명령(메타데이터): `"Pick up the red cube and place it in the black box"`
— ACT 자체는 언어를 사용하지 않지만, **수집 시점에 포함해 두면 같은 데이터셋을 이후
VLA 3종(SmolVLA/π0/GR00T) 파인튜닝에 재사용**할 수 있다 ("한 번 수집, 4모델 재사용" 원칙).

### 3.2 수집 프로토콜

- **teleop 방식**: 사람이 leader를 조작하면 follower가 실시간으로 따라 하며, 이때
  follower의 관절 궤적 + 2캠 영상이 30fps로 기록된다
- **3지점 그리드 순환**: 큐브 시작 위치 3곳을 테이프로 표시, 에피소드마다 1→2→3 순환
  배치. 위치 다양성 확보(과적합 방지) + 이후 4모델 측정에서 동일 초기조건 재현 목적
- **카메라 뷰만 보고 조작** (cheating 방지): 모델이 보는 것과 같은 정보만으로 시연
- **에피소드 경계**: 배치 완료 즉시 종료(→키) — 복귀 동작 등 불필요한 꼬리를 데이터에서 배제
- 리셋 시간 15초 동안 큐브 재배치·팔 대기 자세 복귀, 손이 카메라에 찍히지 않게 관리

### 3.3 수집 결과 및 품질 지표

| 지표 | 값 |
|---|---|
| 에피소드 | **30개** (3세션 × 10ep) |
| 총 프레임 | 13,032 (약 7.2분) |
| 에피소드 길이 | 12.5~19.9초 (평균 ~14.5초, 60초 타임아웃 0건) |
| 데이터 무결성 | action/state NaN 0개 |
| 정지 프레임 비율 | 세션별 24.4% → 20.8% → 18.2% (조작 숙련도 향상) |
| 시각 검증 | 그리드 순환·시작 자세 일관성·task 완결(큐브가 박스 안)·손 침입 없음 확인 |

### 3.4 데이터셋 저장 구조 (LeRobotDataset v3)

로컬 `~/.cache/huggingface/lerobot/heongyu/so101_t1_pickplace/`에 저장되고
HF Hub private repo `heongyu/so101_t1_pickplace`로 push되어 있다.

```
so101_t1_pickplace/
├─ meta/
│  ├─ info.json                  # 스키마: fps=30, feature 정의, 총 에피소드/프레임 수
│  ├─ stats.json                 # feature별 통계 (정규화용 mean/std 등)
│  ├─ tasks.parquet              # 언어 명령 ("Pick up the red cube ...")
│  └─ episodes/chunk-000/*.parquet  # 에피소드별 길이·인덱스
├─ data/chunk-000/file-00X.parquet  # 프레임 단위 테이블:
│     • action              (6,)  — leader가 지시한 목표 관절값 (모델이 배울 정답)
│     • observation.state   (6,)  — follower의 현재 관절값
│     • timestamp / episode_index / task_index ...
└─ videos/
   ├─ observation.images.top/chunk-000/file-00X.mp4    # AV1 인코딩
   └─ observation.images.wrist/chunk-000/file-00X.mp4
```

핵심 설계: **관절·메타데이터는 Parquet(열지향 테이블), 영상은 MP4(AV1)로 분리** 저장.
프레임 인덱스로 두 매체가 동기화되며, 학습 시 torchcodec이 해당 타임스탬프의 프레임을
디코딩해 텐서로 합쳐준다. 6-DOF는 shoulder_pan, shoulder_lift, elbow_flex,
wrist_flex, wrist_roll, gripper.

---

## 4. 학습 방식

### 4.1 ACT 아키텍처와 학습 원리

ACT는 **CVAE(조건부 변분 오토인코더) + Transformer** 구조의 행동 복제(Behavior Cloning) 모델이다:

```
입력: top 이미지 ─┐
      wrist 이미지 ─┼─ ResNet18 백본 ×2 ─┐
      관절 state(6) ─────────────────────┼─ Transformer 인코더/디코더
      잠재변수 z  ────────────────────────┘        │
                                                  ▼
출력: action chunk — 미래 100스텝의 관절값 시퀀스 (chunk_size=100)
```

- **Action Chunking**: 매 스텝 1개 액션이 아니라 **미래 100스텝을 한 번에 예측**한다.
  30fps 기준 약 3.3초 분량의 궤적을 계획하는 셈이라 조작이 매끄럽고, 시연의 복합 오차
  (compounding error)에 강하다. 추론 시에는 chunk를 겹쳐 실행하며 부드럽게 이어간다.
- **CVAE 잠재변수 z**: 같은 상황에서도 사람 시연은 매번 조금씩 다르다(스타일 변동).
  학습 시 인코더가 시연의 "스타일"을 z로 압축하고, 추론 시에는 z=0(평균 스타일)으로
  결정론적으로 실행한다.

### 4.2 Loss 함수

```
total loss = L1 loss + 10 × KL divergence        (lerobot modeling_act.py:144-158)
```

| 항 | 역할 |
|---|---|
| **L1 loss** | 예측 action chunk vs 실제 시연 action의 절대 오차 — "시연을 얼마나 정확히 재현하는가". 패딩 프레임은 마스킹 |
| **KL divergence** (×10) | CVAE 잠재분포를 표준정규분포로 규제 — 잠재공간을 매끄럽게 유지해 추론 시 z=0 샘플링이 유효하도록 |

### 4.3 학습 설정 및 진행

| 항목 | 값 |
|---|---|
| 명령 | `lerobot-train --policy.type=act` (`setup/train_act_t1.sh`) |
| 파라미터 수 | 51.6M |
| batch / lr / steps | 8 / 1e-5 / 100,000 |
| 속도 | 9.3 steps/s (스텝당 0.107초) → **총 약 3시간** |
| VRAM | 5.3GB / 16GB |
| 체크포인트 | 20k 스텝마다 저장, 완료 시 `heongyu/act_so101_t1`로 자동 push |
| 데이터 흐름 | Parquet+MP4 → torchcodec 디코딩 → (2캠 이미지, state, action chunk) 배치 |

**Loss 진행** (2026-07-08 야간 학습, 실측):

| step | total loss |
|---|---|
| 200 | 6.742 |
| 600 | 2.466 |
| 2,000 | ~1.0 |
| 3,400 | 0.639 |
| 9,400 | **0.178** (계속 하강 중) |

실시간 모니터링: `setup/loss_monitor.py`가 학습 로그를 15초마다 파싱해
`logs/loss_curve.png`(log 스케일 곡선 + ETA)를 갱신한다.

### 4.4 학습 중 발생한 이슈와 해결

| 이슈 | 원인 | 해결 |
|---|---|---|
| wrist 영상 가로선 깨짐 | 카메라 기본 YUYV 무압축 → USB 허브 대역폭 포화 | 전 명령에 `fourcc: MJPG` 강제 |
| torchcodec 로드 실패 | 데이터셋 영상이 AV1인데 시스템 ffmpeg 부재 | `apt install ffmpeg` |
| 쓰레기 에피소드 혼입 | 카메라 조정 중 record 실행 | 데이터셋 초기화 후 재수집 (`setup/reset_dataset.sh`) |

---

## 5. 현재 상태와 다음 단계

```
[완료] 환경 구축 (D1) ─ [완료] T1 30ep 수집·검증·Hub push ─ [진행 중] ACT-T1 학습 (~3h)
                                                                    │
[예정] T2(Table Cleanup) 30ep 수집 ─ ACT-T2 학습 ─ T1·T2 각 10회 측정 ─ ACT 완결 게이트
```

- **측정 프로토콜** (예정): task당 10회, 3지점 그리드 순환, SR/PR/추론지연/VRAM/실패유형
  기록. ACT 수치는 모든 산출물에 "FT 참조군" 라벨을 명시해 zero-shot 3종과 구분한다.
- ACT 완결 게이트(시행 로그 20회 + 대표 클립 + 요약 카드 + Hub 백업)를 채운 뒤에만
  다음 모델(SmolVLA)로 넘어간다.

관련 문서: [model_markdown/README.md](../model_markdown/README.md) (전체 가이드라인),
[model_markdown/01_ACT.md](../model_markdown/01_ACT.md) (ACT 체크리스트·결과 기록표)
