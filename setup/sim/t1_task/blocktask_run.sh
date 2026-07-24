#!/usr/bin/env bash
# coworker 포크(별도 클론 ~/blocktask_ws)의 Task1 블록 씬을 띄운다. 우리 T1(~/Sim-to-Real-SO-101-Workshop)과
# 완전 분리 — 그 소스만 마운트하고 기존 teleop-docker 이미지를 재사용(이미지·우리 작업 불변).
#
# 사용법: ./blocktask_run.sh view      # zero_agent로 씬 육안 확인 (로봇 0액션)
#         ./blocktask_run.sh record    # 리더암 teleop 녹화 (학습 데이터 수집)
#
# 태스크 ID: Lerobot-So101-Teleop-Vials-To-Rack (coworker가 이 등록을 블록 씬으로 덮어씀)
set -euo pipefail

MODE="${1:-view}"
WORKSHOP="$HOME/blocktask_ws/Sim-to-Real-SO-101-Workshop"
[ -d "$WORKSHOP/source" ] || { echo "❌ $WORKSHOP 없음 — 먼저 클론하세요"; exit 1; }

export DISPLAY="${DISPLAY:-:1}"
xhost +local: >/dev/null 2>&1 || true
docker_run() { if docker ps >/dev/null 2>&1; then eval "$1"; else sg docker -c "$1"; fi; }
docker_run "docker rm -f blocktask" >/dev/null 2>&1 || true

CALIB=".cache/huggingface/lerobot/calibration"
mkdir -p "$WORKSHOP/outputs" "$WORKSHOP/datasets"

if [ "$MODE" = "record" ]; then
  # 리더암 teleop 녹화 (S=에피소드 저장/중지, R=리셋). repo_id는 본인 데이터셋으로 조정.
  INNER="lerobot_agent --task Lerobot-So101-Teleop-Vials-To-Rack-DR --num_envs 1 \
    --port /dev/ttyLEADER --robot_id leader \
    --repo_id heongyu/sim_so101_blocktask \
    --repo_root /workspace/Sim-to-Real-SO-101-Workshop/datasets/sim_so101_blocktask \
    --task_name 'Pick up the block and place it in the box'"
else
  INNER="zero_agent --task Lerobot-So101-Teleop-Vials-To-Rack --num_envs 1"
fi

echo "모드: $MODE | 워크숍: $WORKSHOP"
echo "종료: Ctrl+C 또는 다른 터미널에서 docker rm -f blocktask"

RUN="docker run --name blocktask --rm -it --privileged --gpus all \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e DISPLAY=$DISPLAY --network=host \
  -v /dev:/dev -v /run/udev:/run/udev:ro \
  -v /tmp/.X11-unix:/tmp/.X11-unix -v $HOME/.Xauthority:/root/.Xauthority \
  -v $HOME/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
  -v $HOME/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
  -v $HOME/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
  -v $HOME/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
  -v $HOME/$CALIB:/root/$CALIB \
  -v $WORKSHOP/docker/env:/root/env \
  -v $WORKSHOP/source:/workspace/Sim-to-Real-SO-101-Workshop/source \
  -v $WORKSHOP/outputs:/workspace/Sim-to-Real-SO-101-Workshop/outputs \
  -v $WORKSHOP/datasets:/workspace/Sim-to-Real-SO-101-Workshop/datasets \
  teleop-docker:latest bash -c '$INNER'"

docker_run "$RUN"
