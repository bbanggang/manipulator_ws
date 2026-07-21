#!/usr/bin/env bash
# GR00T N1.6 sim 학습 — 워크숍 75ep 데이터셋 충실 재현 (옵션 1/A). **5090에서 실행**.
#
# 워크숍 레시피(10-groot 문서) 그대로: base GR00T-N1.6-3B, 20k steps, lr 1e-4,
# wd 1e-5, warmup 0.05, global batch 64, color jitter. 단일 GPU라 global 64를
# grad accum 16(=per-device 4)로 재현 — 유효 배치 동일, 실기 n1d5 메모리 footprint와 일치.
#
# 목적: 아까 사전학습 체크포인트로 나온 sim SR 60%를 **우리 손으로 재현**해 학습 파이프라인
#   전체(데이터→학습→평가)를 검증. 이후 우리 자체 수집 데이터 학습의 대조군.
#
# 데이터: sreetz-nv/so101_teleop_vials_rack_left (75ep, LeRobot v3.0) — N1.6 로더가 v3
#   청크 포맷 네이티브 지원(v2 변환 불필요). meta/modality.json으로 카메라 매핑
#   (front←external_D455, wrist←ego) 추가함.
set -euo pipefail

DATA_DIR="${1:-$HOME/sim_data}"           # /vials75 를 포함하는 상위 디렉터리
OUT="gr00t_vials75_ours"
HF_TOKEN_VAL="$(cat $HOME/.cache/huggingface/token 2>/dev/null || true)"

docker rm -f gr00t-train 2>/dev/null || true

docker run -d --name gr00t-train --rm --gpus all --network host \
  -e HF_TOKEN="$HF_TOKEN_VAL" -e PYTHONUNBUFFERED=1 \
  -v "$DATA_DIR:/data" \
  -v "$HOME/models:/workspace/models" \
  real-robot \
  bash -c "cd /Isaac-GR00T && python3 gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.6-3B \
    --dataset-path /data/vials75 \
    --modality-config-path examples/SO100/so100_config.py \
    --embodiment-tag NEW_EMBODIMENT \
    --num-gpus 1 \
    --output-dir /workspace/models/$OUT \
    --save-steps 5000 --save-total-limit 5 --max-steps 20000 \
    --warmup-ratio 0.05 --weight-decay 1e-5 --learning-rate 1e-4 \
    --global-batch-size 64 --gradient-accumulation-steps 16 \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader-num-workers 4"

echo "학습 시작 (컨테이너 gr00t-train). 로그: docker logs -f gr00t-train"
echo "출력: ~/models/$OUT"
