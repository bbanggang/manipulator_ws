# Sim-to-Real 계획: Isaac Sim에서 4모델 학습·추론 → 실기 전이 (SO-101)

> 목표: **시뮬레이터에서 데이터 수집→학습→추론(평가)까지 완결한 뒤, 그 체크포인트를 실기
> SO-ARM101로 전이(sim-to-real)** 하는 실험을 ACT / SmolVLA / π0 / GR00T N1.5 4모델에서 수행.
> 기반: NVIDIA 공식 [Sim-to-Real SO-101 Workshop](https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop)
> (Isaac Sim + Isaac Lab, Docker, RTX 5090 공식 테스트) + [LeIsaac](https://wiki.seeedstudio.com/simulate_soarm101_by_leisaac/) 참고.

## 0. 선정 근거 (요약)

1. SO-101이 Isaac Sim **공식 내장 자산** / 2. NVIDIA 공식 워크숍이 SO-101+Isaac Lab+GR00T
전 과정 제공 / 3. **sim teleop → LeRobot 포맷 출력 = 4모델 기존 학습 파이프라인 그대로 재사용**
/ 4. sim+real co-training 근거(+38%, arXiv:2503.24361)

## 1. 아키텍처 (2 PC 분업, 사용자 확정)

```
[로컬 RTX 5070Ti 16GB]                    [원격 RTX 5090 32GB]
 sim(teleop-docker) + 실기 로봇/리더암      sim(teleop-docker) + 추론 컨테이너
 경량 모델 학습·추론: ACT, SmolVLA          대형 모델 학습·추론: π0, GR00T
 sim teleop 데이터 수집(리더암 입력)         (데이터는 rsync/HF로 동기화)
```

- 실기·리더암·카메라가 로컬에 물려 있으므로 **sim 데이터 수집과 실기 전이 평가는 로컬**,
  대형 모델 학습·서빙은 5090 (기존 π0/GR00T 원격 인프라 재사용)

## 2. 공통 파이프라인 (5단계)

### Phase 0 — 환경 (진행 중)
- [x] 양 PC Docker + NVIDIA Container Toolkit
- [ ] 양 PC `teleop-docker`(Isaac Sim/Lab) 이미지 빌드 — **진행 중**
- [ ] sim 컨테이너 기동 + SO-101 씬 로드 확인 (`list_envs`, `zero_agent`)
- [ ] (필요 시) 5090 추론 컨테이너 `./docker/real/build.sh blackwell`

### Phase 1 — sim 데이터 수집 (로컬)
- 워크숍 `lerobot_agent`(sim teleop): **실물 SO-101 리더암을 sim 입력으로 사용**
  (키보드 대안 있음) → **LeRobot 포맷 데이터셋 출력** → `lerobot_push_dataset`로 Hub 업로드
- 태스크: 워크숍 기본 `Lerobot-So101-Teleop-Vials-To-Rack` (+`-DR` 도메인 랜덤화 변형)
- 수집 규칙은 실기와 동일 철학: 성공으로 끝나는 에피소드만, 위치 다양화, idle 최소화
  (**GR00T idle 어트랙터 교훈** — 녹화 시작 즉시 동작 개시)

### Phase 2 — 4모델 학습
| 모델 | PC | 방법 (기존 스크립트 변형) |
|---|---|---|
| ACT | 5070Ti | `lerobot-train --policy.type=act` from-scratch (setup/act/ 패턴) |
| SmolVLA | 5070Ti | `smolvla_base` FT (setup/smolvla/ 패턴, batch16/20k~) |
| π0 | 5090 | expert-only FT (setup/pi0/remote_5090/pi0-train.sh 패턴) |
| GR00T | 5090 | n1d5 FT — v2 변환+modality.json+트리밍 도구 재사용 (setup/gr00t/) |

- sim 데이터도 LeRobot v3로 나오면 GR00T용은 v3→v2 변환 필요 (도구 완비)
- **실기 실험에서 확정된 처방 선적용**: GR00T는 tune_visual, π0는 expert-only, idle 트리밍

### Phase 3 — sim 내 추론·평가 (closed-loop)
- GR00T: 워크숍 `lerobot_eval` + `-Eval`/`-DR-Eval` 환경으로 **공식 지원**
- ACT/SmolVLA/π0: 워크숍 eval은 GR00T 전제 → **lerobot 정책용 sim 평가 어댑터 제작 필요**
  (Isaac Lab env 관측 → lerobot 정책 입력 브리지; `lerobot_eval` 소스 참고해 작성)
- 지표: sim SR (task당 N회, 시드/초기위치 변화)

### Phase 4 — 실기 전이 (sim-to-real, 로컬)
- sim 학습 체크포인트를 **실기 평가 파이프라인 그대로** 실행
  (ACT/SmolVLA: lerobot-record eval, π0: async 서버, GR00T: zmq 서버 — 모두 기존 스크립트)
- 전이 조건: 실기 소품을 sim 태스크와 일치시킴 (바이알/랙 또는 T1 재현 여부는 미결 — 아래 D1)
- 측정: **zero-shot 전이 SR** (10회) → (선택) **sim+real co-training** 후 재측정
- 비교 기준: 실데이터 단독 학습 결과(기존 측정치) vs sim 단독 vs sim+real

### Phase 5 — 정리
- 4모델 × {real-only, sim-only(전이), sim+real} SR 매트릭스 → 종합 보고서

## 3. ⚠️ 미결정 사항 — 사용자 결정 필요

| # | 결정 항목 | 선택지 | 비고/추천 |
|---|---|---|---|
| **D1** | **sim 태스크** | (a) 워크숍 기본 Vials-To-Rack 그대로 (b) 우리 T1(red cube→black box) 씬 자작 (c) a로 검증 후 b | (a)는 자산·DR·평가 완비지만 실기 소품(바이알·랙) 필요, (b)는 기존 벤치마크와 직접 비교 가능하나 씬 제작 공수. **추천 (c)** |
| **D2** | **GR00T 버전 정합** | (a) sim도 N1.5(n1d5)로 통일 (b) sim은 워크숍 최신(N1.7 계열) 수용 (c) 실기도 N1.7로 승격 | 워크숍 추론 컨테이너는 최신 Isaac-GR00T를 clone(GROOT_REF). (a)는 비교 일관성↑·호환 리스크, (b)는 편하지만 실기 N1.5와 어긋남 |
| **D3** | **teleop 입력장치** | (a) 실물 리더암 (b) 키보드 | (a) 권장 — 데이터 질. 단 리더암이 로컬에 있으므로 수집은 로컬 확정 (5070Ti sim 구동 확인 필요) |
| **D4** | **수집 규모·DR** | 에피소드 수(50ep 기준 유지?), DR 태스크 사용 여부 | DR 사용 권장(전이 성능 핵심). 규모는 실기와 동일 50ep 시작 추천 |
| **D5** | **sim 평가 범위** | (a) 4모델 전부 sim closed-loop (어댑터 제작) (b) GR00T만 sim 평가, 나머지는 sim-train→real-eval | (a)는 공수 있음(어댑터), (b)는 빠르나 "sim 추론" 커버리지 부분적 |
| **D6** | **전이 실험 폭** | zero-shot 전이만 / +co-training까지 | co-training이 이 접근의 본 가치(+38% 근거) — 일정 여유에 따라 |
| **D7** | **일정 순서** | GR00T 실기(vis2→측정) 완결 후 sim 착수 vs 병행 | GPU 경합(5090) 고려. 현재는 GR00T 실기 마무리 우선으로 진행 중 |

## 4. 리스크

| 리스크 | 대응 |
|---|---|
| 5070Ti에서 Isaac Sim 렌더+teleop 성능 | 해상도·물리 스텝 조정, 안 되면 수집도 5090에서(리더암 이동 필요) |
| 워크숍 GROOT_REF와 n1d5 비호환 | D2 결정에 따라 추론 컨테이너 GROOT_REF 오버라이드 시도 |
| sim 카메라 뷰 ↔ 실기 top/wrist 구도 차이 | sim 카메라 포즈를 실기와 유사하게 조정 + DR로 완화 |
| 캘리브레이션 좌표계(실기 전이 시) | goto_home/state 대조 도구 재사용 (GR00T 실기에서 검증됨) |
| ACT/π0/SmolVLA sim 평가 어댑터 공수 | D5에서 범위 결정, GR00T 우선 |

## 5. 설치 기록

- 양 PC clone: `~/Sim-to-Real-SO-101-Workshop`
- sim 이미지: `docker build -t teleop-docker -f docker/sim/Dockerfile .` (양쪽 빌드 진행 중)
- 로컬 함정 해결: claude-desktop apt GPG 만료(비활성화), 잔존 sshd 포트22 점유로
  openssh postinst 실패(정리), docker 그룹 미반영 셸은 `sg docker -c` 우회

## 6. 참고

- 워크숍: https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop
- 교육 문서: https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/index.html
- LeIsaac: https://wiki.seeedstudio.com/simulate_soarm101_by_leisaac/
- 모델/데이터셋 목록: https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/datasets-and-models.html
