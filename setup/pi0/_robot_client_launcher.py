"""robot_client 런처 — so101_follower 등 로봇 config를 draccus 파싱 전에 등록.

LeRobot의 `python -m lerobot.async_inference.robot_client`는 `so_follower` 서브모듈을
import하지 않아 `--robot.type` 선택지가 비어버린다(레지스트리 미등록). lerobot_record.py는
`from lerobot.robots import so_follower`로 명시 등록하는데 robot_client는 빠뜨림.
이 런처가 로봇/카메라 모듈을 먼저 import해 등록한 뒤 draccus-wrapped main을 호출한다.
"""

import lerobot.robots.so_follower  # noqa: F401  — so100/so101_follower 등록
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.async_inference.robot_client import async_client

if __name__ == "__main__":
    async_client()
