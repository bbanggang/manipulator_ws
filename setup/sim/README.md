# 시뮬레이터: Isaac Sim + Isaac Lab (SO-101, 4모델 공통)

sim에서 데이터 수집·학습·평가를 하고 실기(sim-to-real)로 잇는 차기 단계.
**선정: Isaac Sim/Isaac Lab + NVIDIA 공식 Sim-to-Real SO-101 Workshop (+LeIsaac)**.

## 선정 근거 (2026-07-20)

1. **SO-101이 Isaac Sim 공식 내장 자산** (커뮤니티 포팅 아님)
2. **NVIDIA 공식 워크숍**이 SO-101 + Isaac Lab + GR00T의 "sim에서 먼저, 실기로" 전 과정 제공
   — RTX 5090(Blackwell) 공식 테스트, Docker 기반
3. **LeIsaac teleop → LeRobot 포맷 출력** = ACT/SmolVLA/π0/GR00T **4모델 전부 기존 학습
   파이프라인 재사용** (다른 시뮬레이터 대비 결정적 장점)
4. sim+real co-training이 실데이터 단독 대비 평균 +38% (arXiv:2503.24361) — 계획 문서에 기존 명시

차선: MuJoCo menagerie(SO-ARM100)류 — 경량이나 LeRobot 파이프라인·VLA 평가 통합 없음.

## 배치 (2 PC 분업)

| PC | 역할 | 상태 |
|---|---|---|
| 로컬 RTX 5070Ti (16GB) | sim + **경량 모델**(ACT/SmolVLA) 학습·추론 | Docker+Toolkit 설치 완료, sim 이미지 빌드 중 |
| 5090 RTX 5090 (32GB) | sim + **대형 모델**(π0/GR00T) 학습·추론 | sim 이미지 빌드 중 |

## 설치 (양쪽 공통, Docker 기반)

```bash
git clone https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop.git ~/Sim-to-Real-SO-101-Workshop
cd ~/Sim-to-Real-SO-101-Workshop
docker build -t teleop-docker -f docker/sim/Dockerfile .      # sim/teleop 컨테이너
./docker/real/build.sh blackwell                              # 추론(GR00T) 컨테이너 — 필요 시
```

실행(컨테이너 기동)은 워크숍 README의 `docker run` 블록 참조 (X11·/dev·HF 캘리브레이션 마운트 포함).

## 설치 중 만난 함정 (로컬)

- claude-desktop apt 저장소 GPG 만료 → `.list` 비활성화 후 진행 (복구는 키 갱신로)
- systemd 미추적 잔존 sshd가 포트 22 점유 → openssh-server postinst 실패로 apt 마비
  → 잔존 프로세스 kill + `dpkg --configure -a`로 복구
- docker 그룹 미반영 셸은 `sg docker -c "..."` 로 우회

## 다음 단계

1. 양쪽 sim 이미지 빌드 완료 확인 → 컨테이너 기동 + SO-101 씬 로드 테스트
2. sim teleop으로 T1 유사 task 데이터 수집 (LeRobot 포맷)
3. 4모델 sim 데이터 학습 → sim 내 평가 → 실기 전이 실험

## 참고

- 워크숍: https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop
- 교육 문서: https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/index.html
- LeIsaac: https://wiki.seeedstudio.com/simulate_soarm101_by_leisaac/
