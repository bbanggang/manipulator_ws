# GR00T N1.5 진행 보고서 — FT 3차 반복 진단 (idle→모드고정→vis 학습 중)

> 작성일: 2026-07-20 | 환경구축→v3→v2 변환→FT v1~v4 반복 진단 진행 중 (vis2 학습 대기)
> 프로젝트: SO-ARM101로 4모델(ACT/SmolVLA/π0/GR00T N1.5) 비교. 본 문서는 네 번째 모델
> **GR00T N1.5**의 환경 구축(다수 함정 해결), 파인튜닝-진단-수정 3회 반복의 전 과정을 기록한다.
> **pi0에서 확립한 사전점검·진단 도구가 그대로 재사용되어 반복 주기가 크게 단축**됐다.

---

## 1. 버전 선정과 환경 구축 (5090, `~/gr00t_remote/`)

- **N1.5 선택** (계획 문서 준수, HF 블로그·커뮤니티 레퍼런스 최다). 단 Isaac-GR00T main은
  N1.7 전용(Qwen3-VL 백본)으로 **아키텍처가 달라 N1.5 체크포인트가 로드 불가** → `n1d5`
  브랜치 별도 clone. pi0 1차 실패(버전 불일치)와 같은 함정을 사전 차단.
- **Blackwell 대응**: n1d5 기본 torch 2.5.1은 sm_120 미지원 → **torch 2.8.0+cu128 오버라이드**
  + flash-attn 2.8.3 사전빌드 휠(cp310/torch2.8). venv 설치 검증 완료.
- 기타 함정: pyproject의 aarch64 torchcodec 경로휠(git-LFS 포인터)로 uv 해석 실패 →
  x86_64 전용화. 호스트 ffmpeg 부재 → 변환은 imageio-ffmpeg 정적 바이너리, 학습은
  `--video-backend torchvision_av`(pyav 내장)로 우회.

## 2. 데이터 준비 (LeRobot v3 → GR00T v2)

- GR00T는 LeRobot **v2** 요구 → 공식 `convert_v3_to_v2.py`로 변환 (도구 전용 uv env,
  `--root`는 상위 디렉터리 지정 주의, 로컬 v3 원본은 무손상 유지)
- **modality.json**: `front ← observation.images.top`, `wrist ← wrist` 매핑 + 6-dim
  single_arm/gripper 슬라이싱 — **pi0 #1 실패(카메라 키)를 정식 메커니즘으로 해결**
- 변환본 stats.json의 `count`([1])가 GR00T 차원 슬라이싱에서 IndexError → `fix_stats_for_gr00t.py`로 제거

## 3. 파인튜닝 반복 진단 (T1)

| 버전 | 구성 | loss | 실기 결과 | 진단 |
|---|---|---|---|---|
| v1 `gr00t_t1` | 기본(백본 freeze, projector+DiT 학습), batch32/10k | 0.008 | 홈에서 정지 | **idle 어트랙터**: 시연 초반 평균 2.8s 정지 구간을 충실히 재현. open-loop은 GT 완벽 추종(모델 정상) |
| v2 `gr00t_t1_trim` | + **idle 트리밍**(120s 제거) | 0.010 | 결단력 있는 뻗기, 그러나 **큐브 무관 pan -77 고정** | **모드 고정**: 파지 pan 분포[-77,+27] 중 왼쪽 지배 모드만 재생. 우측 시작 시 파지·운반·릴리즈 수행 → 시작상태가 모드를 바꾸는 우연이지 시각 그라운딩 아님(홈+우측 큐브 → 또 -85) |
| v3 `gr00t_t1_vis` | + `tune_visual`(vision 학습), batch4/10k | **0.041** | 무동작 | **학습부족**: 유효노출 1.6 epoch(v2의 1/8) |
| v4 `gr00t_t1_vis2` | batch4×**grad accum 8**=유효32, tune_visual | (진행 중) | — | v2와 동일 노출로 vis 효과 공정 검증 |

### 3.1 진단 방법론 (재사용 자산)

- **open-loop 평가**(`eval_policy.py`): 학습 데이터 재생으로 모델/배포 문제 분리 — v1이
  GT를 완벽 추종함을 확인해 "모델은 정상, 배포 조건이 문제"로 조기 확정
- **관측 비교 실험**: 라이브 관측 vs 학습 첫프레임을 동일 zmq 서버에 주입 → 이미지 통계·
  state 일치, **둘 다 출발 명령 생성** 확인 (전처리 무결 입증)
- **클라이언트 계측**: rerun(카메라·state·action) + CSV(청크별 state/추론지연, 스텝별 액션)
  + 정책 시점 video.mp4 — "movement인지 진동인지 주차인지"를 수치로 판정
- **캘리브레이션 검증**: 7/10 재캘리브레이션 전후 좌표계 비교 — 신 캘리브레이션이 학습
  좌표계와 일치(구본은 wrist_roll -54° 어긋남 = 혼 슬립을 신본이 흡수). 홈 자세 손배치
  오차가 ±10°에 달해 **`goto_home.py`(학습 평균 자세 ±2° 복귀)를 측정 프로토콜에 도입**

### 3.2 인프라 (원격 추론)

- 서버: 5090 `inference_service.py`(zmq:5555), `gr00t-server.sh {t1|t2|logs|stop}`
- 클라이언트: 공식 `eval_lerobot.py`+`service.py`를 로컬로 가져와 패치
  (lerobot 0.6 모듈명, gr00t 미설치 로컬 대응 optional import, 계측 로깅, atexit 비디오 마무리)
- 측정 러너: `eval_gr00t_t1_trials.sh` — 시행마다 홈 복귀→배치 대기→60초 자동 실행
- 동기식 구조라 추론(~330ms)마다 정지가 끼는 주기 끊김은 구조적 (async 아님)

## 4. T2

- v3→v2 변환·stats 수정 완료, v1 학습 완료(loss 0.0105, 도중 checkpoint-7500 저장 행(hang)
  → 깨진 체크포인트 삭제 후 5000부터 resume으로 복구). **T1 판정 확정 후 동일 처방 적용 예정**

## 5. 현재 상태와 다음

- [진행 중] vis2 학습 (~5.5h) → 완료 시: loss 확인 → 좌/우 큐브 dry-run으로 **시각 그라운딩 판정**
  - 성공 → T1 정식 10회 측정 (5지점, goto_home, 개입 금지)
  - 부족 → co-training(자체 50ep + HF `youliangtan/so101-table-cleanup`) — `dataset_path` List 지원
- 이후: T2 동일 처방 → 측정 → 4모델 종합 비교

## 6. 시뮬레이터 선정 (차기 단계, 4모델 공통)

**Isaac Sim + Isaac Lab (공식 Sim-to-Real SO-101 Workshop + LeIsaac)** 확정:
- SO-101이 Isaac Sim **공식 내장 자산**, NVIDIA가 SO-101+Isaac Lab+GR00T 전 과정을 워크숍으로 제공
- LeIsaac teleop → **LeRobot 포맷 출력** = 4모델 전부 기존 파이프라인 재사용 가능 (결정적 이유)
- 배치: 양 PC Docker 설치(워크숍 공식이 Docker 기반, RTX 5090 테스트됨) — **경량(ACT/SmolVLA)
  → 5070Ti, 대형(π0/GR00T) → 5090** 분업. sim 이미지 양쪽 빌드 진행 중
- 로컬 Docker 설치 중 부수 이슈 해결: claude-desktop apt 저장소 GPG 만료(비활성화),
  systemd 미추적 잔존 sshd의 포트 22 점유로 openssh-server postinst 실패(정리 후 복구)

---

*관련: [model_markdown/04_GR00T_N1.5.md](../model_markdown/04_GR00T_N1.5.md),
[setup/gr00t/](../setup/gr00t/) (modality/변환/트리밍/측정 스크립트), [report/Pi0_report.md](Pi0_report.md)*
