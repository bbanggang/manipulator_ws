# 워크숍 저장소 로컬 패치

`~/Sim-to-Real-SO-101-Workshop`은 이 repo 밖(NVIDIA 공식 저장소)이라 수정이
버전관리되지 않는다. 여기에 패치를 보관해 **재클론/재설치 시 재적용**할 수 있게 한다.

적용: `cd ~/Sim-to-Real-SO-101-Workshop && git apply <이 디렉터리>/workshop_local_changes.patch`

## workshop_local_changes.patch (2026-07-21)

3개 파일 수정:

### 1. `utils/lerobot_recorder.py` — VRAM 버그 2건
- **버퍼 해제 오타**: `save_episode`/`cancel_recording`이 `self.rgb_buffer_tensor`(단수)를
  비우는데 실제 GPU 버퍼는 `rgb_buffer_tensors`(복수) → 에피소드마다 6.2GiB 누수.
  다음 에피소드가 해제 전에 새로 할당해 순간 12.4GiB 필요 → 16GB GPU에서 2번째
  에피소드에 OOM. 복수형으로 수정 + `torch.cuda.empty_cache()` 추가.
- **버퍼 용량**: 120초 고정(3600프레임/캠=3.09GiB, 2캠 6.2GiB)을 `SIM_RECORD_SECONDS`
  (기본 60)로 조정 가능하게 — 6.18→3.09GiB.

### 2. `mdp/resets.py` — 관절별 시작자세 무작위화 함수 추가
- `reset_joints_by_offset_per_joint()`: 표준 함수와 달리 관절별로 다른 오프셋 범위 적용.
  교정(recovery) 데이터 수집용 — pan/pitch/elbow는 크게, Jaw(그리퍼)는 0.

### 3. `tasks/so101_env_cfg.py` — 교정 모드 토글
- 환경변수 `SIM_RECOVERY=1`이면 시작 자세를 관절별 무작위화(교정 시연), 아니면 고정(정상 시연).
- 범위: Rotation ±0.35rad, Pitch/Elbow ±0.25, Wrist ±0.15~0.20, Jaw 0.

근거: 실기 GR00T 실패의 근본 원인이 covariate shift(교정 시연 부재)였음
— report/GR00T_report.md §3.2, model_markdown/sim2real/05_SimToReal.md §0.1.1.
