# manipulator_ws — SO-ARM101 × 4모델 학습/Zero-shot 비교

단일 SO-ARM101 매니퓰레이터로 ACT / SmolVLA / π0 / GR00T N1.5 4개 모델의
학습 및 zero-shot 성능을 비교하는 프로젝트 워크스페이스.

## 구조

```
manipulator_ws/
├─ model_markdown/   # 진행 가이드라인 + 모델별 체크리스트·결과 기록
│  ├─ README.md      #   전체 계획·평가 프로토콜·일별 체크리스트 (시작점)
│  ├─ 01_ACT.md      #   ACT (FT 참조군)
│  ├─ 02_SmolVLA.md  #   SmolVLA (zero-shot 주력)
│  ├─ 03_Pi0.md      #   π0 (zero-shot + 5090 원격 FT)
│  └─ 04_GR00T_N1.5.md
├─ report/           # 진행 보고서 (ACT 학습 전 과정 등)
├─ setup/            # 재현용 스크립트
│  ├─ 99-lerobot.rules      # udev: 카메라·시리얼 경로 고정
│  ├─ identify_devices.sh   # 장치 식별
│  ├─ check_cameras.sh      # 카메라 점검 (링크·fps·스냅샷)
│  ├─ teleop.sh             # teleoperation (MJPG 강제)
│  ├─ record_t1.sh          # T1 데이터 수집
│  ├─ reset_dataset.sh      # 데이터셋 초기화 (로컬+Hub)
│  ├─ train_act_t1.sh       # ACT-T1 학습
│  ├─ loss_monitor.py       # 학습 loss 실시간 그래프
│  └─ wrist_cam_tune.py     # 카메라 초점/아티팩트 튜닝
└─ envs/
   ├─ lerobot/       # uv 프로젝트 (lerobot 0.4.4 + torch cu128) — uv sync로 재현
   └─ gr00t/         # uv 프로젝트 (Isaac-GR00T용, 격리)
```

## 환경 재현

```bash
# 1. uv 설치
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 의존성 설치 (Python 3.10 자동 고정)
cd envs/lerobot && uv sync

# 3. udev rules 설치 (장치 경로 고정 — 포트가 다르면 identify_devices.sh로 재확인)
sudo cp setup/99-lerobot.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger

# 4. 시스템 의존성
sudo apt install ffmpeg   # torchcodec(학습 영상 디코딩) 필수
```

## 데이터·모델 (HF Hub private)

| repo | 내용 |
|---|---|
| `heongyu/so101_t1_pickplace` | T1 Pick-and-Place 30ep (teleop 시연) |
| `heongyu/so101_t2_cleanup` | T2 Table Cleanup (예정) |
| `heongyu/act_so101_t1` | ACT-T1 학습 체크포인트 |

## 주의사항

- 카메라 설정에 **반드시 `fourcc: MJPG`** — YUYV는 USB 대역폭 포화로 영상 깨짐
- RTX 5070 Ti(Blackwell sm_120)는 CUDA 12.8+ / torch 2.10+cu128 필요
- 카메라 udev rule은 물리 포트 기준 — USB 포트 이동 시 rule 수정 필요
