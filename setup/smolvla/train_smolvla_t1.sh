#!/usr/bin/env bash
# SmolVLA-T1 Few-episode Fine-tuning (Pick-and-Place, 기존 30ep 재사용) — 야간 실행용
# 사용법: ./setup/smolvla/train_smolvla_t1.sh
#         nohup ./setup/smolvla/train_smolvla_t1.sh > ~/manipulator_ws/logs/smolvla_t1.log 2>&1 &
#
# 근거: SmolVLA 논문(arXiv:2506.01844) 프로토콜 — task당 50demo(5위치×10궤적), 우리는
#   기존 ACT용 30ep(3지점 그리드)를 그대로 재사용(§ model_markdown/02_SmolVLA.md 참조).
#   하이퍼파라미터는 SmolVLAConfig 기본값(lr=1e-4 cosine decay, warmup=1000)을 그대로 쓰고,
#   batch_size/steps는 커뮤니티 SO-101 FT 사례(ggando.com: RTX3090 24GB/batch64/20k step)를
#   우리 GPU(5070 Ti 16GB)에 맞춰 batch_size=16으로 축소, steps는 20k 그대로 유지.
# 카메라 키: 데이터셋의 top/wrist를 그대로 사용 — zero-shot과 달리 파인튜닝은 모델이
#   우리 카메라 이름에 새로 적응하므로 camera1/2 리네이밍 불필요.
# VRAM 부족(OOM) 시 --batch_size를 8로 낮출 것.
cd "$(dirname "$0")/../../envs/lerobot" || exit 1

exec uv run lerobot-train \
  --dataset.repo_id=heongyu/so101_t1_pickplace \
  --policy.type=smolvla \
  --policy.pretrained_path=lerobot/smolvla_base \
  --policy.device=cuda \
  --policy.push_to_hub=true \
  --policy.repo_id=heongyu/smolvla_so101_t1 \
  --output_dir=outputs/train/smolvla_t1 \
  --job_name=smolvla_t1 \
  --batch_size=16 \
  --steps=20000 \
  --save_freq=5000 \
  --log_freq=200 \
  --wandb.enable=false
