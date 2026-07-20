# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This is the new Gr00T policy eval script with so100, so101 robot arm. Based on:
https://github.com/huggingface/lerobot/pull/777

Example command:

```shell

python eval_gr00t_so100.py \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=lil_guy \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: 9, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 15, width: 640, height: 480, fps: 30}}" \
    --policy_host=10.112.209.136 \
    --lang_instruction="Grab markers and place into pen holder."
```


First replay to ensure the robot is working:
```shell
python -m lerobot.replay \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=lil_guy \
    --dataset.repo_id=youliangtan/so100-table-cleanup \
    --dataset.episode=2
```
"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

import draccus
import matplotlib.pyplot as plt
import numpy as np
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    make_robot_from_config,
    so_follower,  # lerobot 0.6.x: so100/so101_follower 통합 모듈 (구버전 이름에서 패치)
)
from lerobot.utils.utils import init_logging, log_say

# NOTE:
# Sometimes we would like to abstract different env, or run this on a separate machine
# User can just move this single python class method gr00t/eval/service.py
# to their code or do the following line below
# sys.path.append(os.path.expanduser("~/Isaac-GR00T/gr00t/eval/"))
from service import ExternalRobotInferenceClient

# from gr00t.eval.service import ExternalRobotInferenceClient

#################################################################################
# rerun 실시간 시각화 + 지연 계측 (RERUN=1 기본, pi0 진단 패턴 재사용)
# - camera/*: front/wrist 영상
# - state/*: 관절 상태, action_sent/*: 실제 전송 액션(클램프 후)
# - timing/infer_ms: 서버 추론+왕복 지연, timing/act_dt_ms: send_action 간격(끊김 진단)
import os as _os

_RERUN = _os.environ.get("RERUN", "1") == "1"
if _RERUN:
    import rerun as rr

    rr.init("gr00t_infer", spawn=True)
_FRAME = {"i": 0, "t_act": None}


def _rr_obs(observation_dict, camera_keys, state_keys):
    if not _RERUN:
        return
    rr.set_time_sequence("frame", _FRAME["i"])
    for k in camera_keys:
        v = observation_dict.get(k)
        if v is not None and getattr(v, "ndim", 0) == 3:
            rr.log(f"camera/{k}", rr.Image(v))
    for k in state_keys:
        if k in observation_dict:
            rr.log(f"state/{k}", rr.Scalars(float(observation_dict[k])))


def _rr_action(sent):
    if not _RERUN:
        return
    import time as _time

    now = _time.perf_counter()
    if _FRAME["t_act"] is not None:
        rr.log("timing/act_dt_ms", rr.Scalars((now - _FRAME["t_act"]) * 1e3))
    _FRAME["t_act"] = now
    rr.set_time_sequence("frame", _FRAME["i"])
    _FRAME["i"] += 1
    for k, v in sent.items():
        rr.log(f"action_sent/{k}", rr.Scalars(float(v)))


def _rr_infer_ms(ms):
    if _RERUN:
        rr.set_time_sequence("frame", _FRAME["i"])
        rr.log("timing/infer_ms", rr.Scalars(ms))


#################################################################################
# 파일 로깅 + 영상 녹화 (LOGDIR 환경변수, 기본 on) — 실행 후 오프라인 분석용
# - chunks.csv: 청크별 [관측 state 6, infer_ms]  → 청크 경계 왕복(진동) 진단
# - actions.csv: 스텝별 전송 액션 6              → 청크 내/간 명령 궤적
# - video.mp4: 정책이 본 front|wrist (청크 경계마다 1프레임)
import csv as _csv
from datetime import datetime as _dt

_LOGDIR = _os.environ.get(
    "LOGDIR",
    _os.path.expanduser(f"~/manipulator_ws/logs/gr00t_infer/{_dt.now():%Y%m%d_%H%M%S}"),
)
_os.makedirs(_LOGDIR, exist_ok=True)
_JOINTS = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
           "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
_chunk_f = open(f"{_LOGDIR}/chunks.csv", "w", buffering=1)
_chunk_w = _csv.writer(_chunk_f)
_chunk_w.writerow(["chunk", "infer_ms"] + [f"state_{j}" for j in _JOINTS])
_act_f = open(f"{_LOGDIR}/actions.csv", "w", buffering=1)
_act_w = _csv.writer(_act_f)
_act_w.writerow(["chunk", "i", f_ := "t_ms"] + [f"sent_{j}" for j in _JOINTS])
_VID = {"w": None}
_CHUNK = {"n": 0, "t0": None}
print(f"[LOG] {_LOGDIR}")

import atexit as _atexit


@_atexit.register
def _finalize_video():
    # Ctrl+C 종료 시에도 mp4 moov atom이 기록되도록 릴리즈 (2026-07-16 깨짐 재발 방지)
    if _VID["w"] is not None:
        _VID["w"].release()
    _chunk_f.close()
    _act_f.close()


def _log_chunk(observation_dict, camera_keys, infer_ms):
    import cv2, time as _t

    if _CHUNK["t0"] is None:
        _CHUNK["t0"] = _t.perf_counter()
    st = [observation_dict.get(j, float("nan")) for j in _JOINTS]
    _chunk_w.writerow([_CHUNK["n"], f"{infer_ms:.0f}"] + [f"{v:.2f}" for v in st])
    imgs = [observation_dict[k] for k in camera_keys if k in observation_dict]
    if imgs:
        import numpy as _np

        frame = _np.concatenate(imgs, axis=1)
        if _VID["w"] is None:
            h, w = frame.shape[:2]
            _VID["w"] = cv2.VideoWriter(f"{_LOGDIR}/video.mp4",
                                        cv2.VideoWriter_fourcc(*"mp4v"), 4, (w, h))
        _VID["w"].write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    _CHUNK["n"] += 1


def _log_action(sent):
    import time as _t

    t_ms = (_t.perf_counter() - (_CHUNK["t0"] or _t.perf_counter())) * 1e3
    _act_w.writerow([_CHUNK["n"] - 1, _FRAME["i"], f"{t_ms:.0f}"]
                    + [f"{float(sent.get(j, float('nan'))):.2f}" for j in _JOINTS])


#################################################################################


class Gr00tRobotInferenceClient:
    """The exact keys used is defined in modality.json

    This currently only supports so100_follower, so101_follower
    modify this code to support other robots with other keys based on modality.json
    """

    def __init__(
        self,
        host="localhost",
        port=5555,
        camera_keys=[],
        robot_state_keys=[],
        show_images=False,
    ):
        self.policy = ExternalRobotInferenceClient(host=host, port=port)
        self.camera_keys = camera_keys
        self.robot_state_keys = robot_state_keys
        self.show_images = show_images
        assert (
            len(robot_state_keys) == 6
        ), f"robot_state_keys should be size 6, but got {len(robot_state_keys)} "
        self.modality_keys = ["single_arm", "gripper"]

    def get_action(self, observation_dict, lang: str):
        # first add the images
        obs_dict = {f"video.{key}": observation_dict[key] for key in self.camera_keys}

        # show images
        if self.show_images:
            view_img(obs_dict)

        # Make all single float value of dict[str, float] state into a single array
        state = np.array([observation_dict[k] for k in self.robot_state_keys])
        obs_dict["state.single_arm"] = state[:5].astype(np.float64)
        obs_dict["state.gripper"] = state[5:6].astype(np.float64)
        obs_dict["annotation.human.task_description"] = lang

        # then add a dummy dimension of np.array([1, ...]) to all the keys (assume history is 1)
        for k in obs_dict:
            if isinstance(obs_dict[k], np.ndarray):
                obs_dict[k] = obs_dict[k][np.newaxis, ...]
            else:
                obs_dict[k] = [obs_dict[k]]

        # get the action chunk via the policy server
        # Example of obs_dict for single camera task:
        # obs_dict = {
        #     "video.front": np.zeros((1, 480, 640, 3), dtype=np.uint8),
        #     "video.wrist": np.zeros((1, 480, 640, 3), dtype=np.uint8),
        #     "state.single_arm": np.zeros((1, 5)),
        #     "state.gripper": np.zeros((1, 1)),
        #     "annotation.human.action.task_description": [self.language_instruction],
        # }
        action_chunk = self.policy.get_action(obs_dict)

        # convert the action chunk to a list of dict[str, float]
        lerobot_actions = []
        action_horizon = action_chunk[f"action.{self.modality_keys[0]}"].shape[0]
        for i in range(action_horizon):
            action_dict = self._convert_to_lerobot_action(action_chunk, i)
            lerobot_actions.append(action_dict)
        return lerobot_actions

    def _convert_to_lerobot_action(
        self, action_chunk: dict[str, np.array], idx: int
    ) -> dict[str, float]:
        """
        This is a magic function that converts the action chunk to a dict[str, float]
        This is because the action chunk is a dict[str, np.array]
        and we want to convert it to a dict[str, float]
        so that we can send it to the robot
        """
        concat_action = np.concatenate(
            [np.atleast_1d(action_chunk[f"action.{key}"][idx]) for key in self.modality_keys],
            axis=0,
        )
        assert len(concat_action) == len(self.robot_state_keys), "this should be size 6"
        # convert the action to dict[str, float]
        action_dict = {key: concat_action[i] for i, key in enumerate(self.robot_state_keys)}
        return action_dict


#################################################################################


def view_img(img, overlay_img=None):
    """
    This is a matplotlib viewer since cv2.imshow can be flaky in lerobot env
    """
    if isinstance(img, dict):
        # stack the images horizontally
        img = np.concatenate([img[k] for k in img], axis=1)

    plt.imshow(img)
    plt.title("Camera View")
    plt.axis("off")
    plt.pause(0.001)  # Non-blocking show
    plt.clf()  # Clear the figure for the next frame


def print_yellow(text):
    print("\033[93m {}\033[00m".format(text))


@dataclass
class EvalConfig:
    robot: RobotConfig  # the robot to use
    policy_host: str = "localhost"  # host of the gr00t server
    policy_port: int = 5555  # port of the gr00t server
    action_horizon: int = 8  # number of actions to execute from the action chunk
    lang_instruction: str = "Grab pens and place into pen holder."
    play_sounds: bool = False  # whether to play sounds
    timeout: int = 60  # timeout in seconds
    show_images: bool = False  # whether to show images


@draccus.wrap()
def eval(cfg: EvalConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    # Step 1: Initialize the robot
    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    # get camera keys from RobotConfig
    camera_keys = list(cfg.robot.cameras.keys())
    print("camera_keys: ", camera_keys)

    log_say("Initializing robot", cfg.play_sounds, blocking=True)

    language_instruction = cfg.lang_instruction

    # NOTE: for so100/so101, this should be:
    # ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos', 'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos']
    robot_state_keys = list(robot._motors_ft.keys())
    print("robot_state_keys: ", robot_state_keys)

    # Step 2: Initialize the policy
    policy = Gr00tRobotInferenceClient(
        host=cfg.policy_host,
        port=cfg.policy_port,
        camera_keys=camera_keys,
        robot_state_keys=robot_state_keys,
    )
    log_say(
        "Initializing policy client with language instruction: " + language_instruction,
        cfg.play_sounds,
        blocking=True,
    )

    # Step 3: Run the Eval Loop
    while True:
        # get the realtime image
        observation_dict = robot.get_observation()
        _rr_obs(observation_dict, camera_keys, robot_state_keys)
        _t0 = time.perf_counter()
        action_chunk = policy.get_action(observation_dict, language_instruction)
        _infer_ms = (time.perf_counter() - _t0) * 1e3
        _rr_infer_ms(_infer_ms)
        _log_chunk(observation_dict, camera_keys, _infer_ms)
        print(f"inference+RTT: {_infer_ms:.0f}ms")

        for i in range(cfg.action_horizon):
            action_dict = action_chunk[i]
            sent = robot.send_action(action_dict)
            sent = sent if isinstance(sent, dict) else action_dict
            _rr_action(sent)
            _log_action(sent)
            time.sleep(0.02)  # Implicitly wait for the action to be executed


if __name__ == "__main__":
    eval()
