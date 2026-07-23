#!/usr/bin/env bash
# 커스텀 T1(빨간 큐브→검은 박스) 씬 육안 확인 — 5090 모니터(GUI)에서 실행.
# 사용법: ./zero_t1.sh          # base 태스크
#         ./zero_t1.sh DR       # DR 태스크
#         ./zero_t1.sh Eval     # 성공 판정 포함(큐브를 박스에 옮기면 종료)
set -euo pipefail

SUFFIX="${1:-}"
TASK="Lerobot-So101-T1-CubeBox"
[ -n "$SUFFIX" ] && TASK="Lerobot-So101-T1-CubeBox-$SUFFIX"

WORKSHOP="$HOME/Sim-to-Real-SO-101-Workshop"
export DISPLAY="${DISPLAY:-:1}"
xhost +local: >/dev/null 2>&1 || true
docker rm -f t1zero >/dev/null 2>&1 || true

echo "태스크: $TASK  (DISPLAY=$DISPLAY)"
echo "종료: Ctrl+C"

docker run --name t1zero --rm -it --privileged --gpus all \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e DISPLAY="$DISPLAY" --network=host \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$HOME/.Xauthority:/root/.Xauthority" \
  -v "$HOME/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw" \
  -v "$HOME/docker/isaac-sim/cache/ov:/root/.cache/ov:rw" \
  -v "$HOME/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw" \
  -v "$HOME/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw" \
  -v "$WORKSHOP/docker/env:/root/env" \
  -v "$WORKSHOP/source:/workspace/Sim-to-Real-SO-101-Workshop/source" \
  teleop-docker:latest bash -c "zero_agent --task $TASK --num_envs 1"
