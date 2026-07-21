# 워크숍 저장소 로컬 패치

`~/Sim-to-Real-SO-101-Workshop`은 이 repo 밖(NVIDIA 공식 저장소)이라 수정이
버전관리되지 않는다. 여기에 패치를 보관해 **재클론/재설치 시 재적용**할 수 있게 한다.

적용: `cd ~/Sim-to-Real-SO-101-Workshop && git apply <이 디렉터리>/workshop_local_changes.patch`

## workshop_local_changes.patch (2026-07-21)

2개 파일 수정 (recorder VRAM 버그 + 참고용 reset 함수):

### 1. `utils/lerobot_recorder.py` — VRAM 버그 2건
- **버퍼 해제 오타**: `save_episode`/`cancel_recording`이 `self.rgb_buffer_tensor`(단수)를
  비우는데 실제 GPU 버퍼는 `rgb_buffer_tensors`(복수) → 에피소드마다 6.2GiB 누수.
  다음 에피소드가 해제 전에 새로 할당해 순간 12.4GiB 필요 → 16GB GPU에서 2번째
  에피소드에 OOM. 복수형으로 수정 + `torch.cuda.empty_cache()` 추가.
- **버퍼 용량**: 120초 고정(3600프레임/캠=3.09GiB, 2캠 6.2GiB)을 `SIM_RECORD_SECONDS`
  (기본 60)로 조정 가능하게 — 6.18→3.09GiB.

### 2. `mdp/resets.py` — 관절별 시작자세 무작위화 함수 (참고용, 미사용)
- `reset_joints_by_offset_per_joint()`: 관절별로 다른 오프셋 범위 적용.
- ⚠️ **teleop 수집에는 무효**: teleop이 절대 위치 추종이라 리셋 오프셋이 첫 스텝에
  리더암 자세로 덮어써짐(2026-07-21 실측 확인). `tasks/so101_env_cfg.py`의 토글 시도는
  **원복**했다(패치에 미포함). 교정 데이터는 사람이 리더암으로 수동 생성한다.
  이 함수는 RL/autonomous rollout에서 재사용 가능성이 있어 남겨둠.

근거: 실기 GR00T 실패의 근본 원인이 covariate shift(교정 시연 부재)였음
— report/GR00T_report.md §3.2, model_markdown/sim2real/05_SimToReal.md §0.1.1.
