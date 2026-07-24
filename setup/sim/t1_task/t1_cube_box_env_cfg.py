# SPDX-License-Identifier: Apache-2.0
# 커스텀 T1 태스크: 빨간 큐브 → 검은 박스 (Pick red cube, place in black box)
#
# 설계 원칙(2026-07-24 재작성): 워크숍 vials_to_rack 태스크의 **검증된 구성**(로봇 위치·mat.usda·
# 라이트박스·물체 높이)을 그대로 계승하고, 아래 5가지만 최소 변경한다.
#   1) 바이알 3개 → 빨간 큐브 1개(프리미티브)   2) 랙 → 검은 박스(tray_black)
#   3) 로봇 보라색   4) 박스를 오른쪽 배치   5) 성공 판정 = 큐브가 박스 안
# mat.usda는 건드리지 않는다(로봇이 제대로 서고 collision·높이가 검증됨). 검은 박스가 어두운 mat에
# 묻히는 것만 **얇은 흰색 시각 커버(collision 없음)**로 해결한다.
# 파지/배치 판정은 vials 함수 재사용(vials=["cube"], rack="box", vertical_threshold=0=방향 무관).
# 리셋만 reset_cube_box 신설(랙 슬롯 로직 불필요).
import os
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.assets import RigidObjectCfg, ArticulationCfg, AssetBaseCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm

from sim_to_real_so101 import assets
from sim_to_real_so101.assets.so101 import S0101_CONTACT_GRASP_CFG
from sim_to_real_so101.mdp import (
    reset_cube_box,
    randomize_sky_light,
    ROBOT_COLORS,
    randomize_robot_color,
    any_vial_grasped,
    vial_placed_on_rack,
    vial_placed_on_rack_termination,
    time_out,
)

from .task_env_cfg import (
    SO101TaskSceneCfg,
    SO101TaskEnvCfg,
    TaskEventCfg,
    TaskObservationsCfg,
)

assets_path = os.path.dirname(os.path.abspath(assets.__file__))

# 물체 스폰 높이 — vials(VIAL_SPAWN_Z=0.05)와 동일. mat.usda 표면(top≈0.035)에 안착.
OBJ_SPAWN_Z = 0.05
CUBE_SIZE = 0.025               # 실기 빨간 큐브(~2.5cm)
MAT_SURF = 0.035                # mat.usda 표면 높이(흰 커버·판정 기준)

# --- 빨간 큐브 (프리미티브, USD 불필요) ---
cube = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Cube",
    spawn=sim_utils.CuboidCfg(
        size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.02),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.05, 0.05)),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.24, 0.03, OBJ_SPAWN_Z)),
)

# --- 검은 오픈탑 박스 (윗면만 열린 직육면체: 바닥+4벽, open_box.usda) ---
box = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Box",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{assets_path}/usd/open_box.usda",
        mass_props=sim_utils.MassPropertiesCfg(mass=0.3),
    ),
    # 로봇 base(-0.05,0) 우측(-y). mat 2배 확장으로 base 옆 배치 가능.
    init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.04, -0.22, MAT_SURF + 0.005)),
)

# 박스 로컬 판정 경계 — open_box는 원점이 캐비티 중심 → ± 대칭. 캐비티 반경 ~0.03 + 여유.
BOX_LOCAL_XY = 0.045
BOX_LOCAL_Z_MAX = 0.05

# 큐브 판정 공통 파라미터 (vials 함수 재사용, vertical_threshold=0 → 방향 무관)
_CUBE_PLACE_PARAMS = dict(
    contact_sensor_cfg=SceneEntityCfg("contact_grasp"),
    vials=["cube"],
    rack_name="box",
    warmup_steps=30,
    grasp_history_window=20,
    force_threshold=2,
    rack_local_x_min=-BOX_LOCAL_XY,
    rack_local_x_max=BOX_LOCAL_XY,
    rack_local_y_min=-BOX_LOCAL_XY,
    rack_local_y_max=BOX_LOCAL_XY,
    rack_local_z_max=BOX_LOCAL_Z_MAX,
    vertical_threshold=0.0,
)


@configclass
class T1CubeBoxSceneCfg(SO101TaskSceneCfg):
    # 로봇: vials와 동일(위치 변경 없음 — 원본 mat.usda 위에 정상 안착). 접촉 센서만 활성화.
    robot: ArticulationCfg = S0101_CONTACT_GRASP_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    cube = cube.replace()
    box = box.replace()

    # mat.usda 2배 확장 (로봇 base 옆까지 커버 → 박스 우측 배치 가능). 표면 높이는 유지(z scale 1).
    mat = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Mat",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{assets_path}/usd/mat.usda",
            scale=(2.0, 2.0, 1.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.22, 0.0, 0.032)),
    )

    # 검은 박스가 어두운 mat에 묻히지 않도록 얇은 흰색 시각 커버(collision 없음 → 물리 불변).
    # mat 표면(0.035) 바로 위에 덮음. 물체는 mat collision(0.035)에 안착. mat 2배에 맞춰 커버도 2배.
    mat_cover = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/MatCover",
        spawn=sim_utils.CuboidCfg(
            size=(0.68, 0.96, 0.002),  # mat 2배(world 0.61×0.914) 덮기
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.9)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.22, 0.0, MAT_SURF + 0.001)),
    )

    # 그리퍼 jaw ↔ 큐브 접촉 센서 (파지 감지)
    contact_grasp = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/jaw",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
    )


@configclass
class T1CubeBoxDRSceneCfg(T1CubeBoxSceneCfg):
    sky_light = AssetBaseCfg(
        prim_path="/World/sky_light",
        spawn=sim_utils.DomeLightCfg(
            intensity=1000.0,
            texture_file=f"{assets_path}/hdri/moon_lab_1k.exr",
            visible_in_primary_ray=False,
            enable_color_temperature=True,
            color_temperature=6500.0,
        ),
    )

    def __post_init__(self) -> None:
        super().__post_init__()


@configclass
class T1CubeBoxEventCfg(TaskEventCfg):
    """큐브·박스 리셋 + 로봇 보라색. (mat 회전 DR은 mat.usda 그대로라 상속 유지)"""

    reset_cube_setup = EventTerm(
        func=reset_cube_box,
        mode="reset",
        params={
            "cube": "cube",
            "box": "box",
            "cube_pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "yaw": (-1.57, 1.57),
            },
            "box_pose_range": {
                "x": (-0.02, 0.02),
                "y": (-0.02, 0.02),
                "yaw": (-0.3, 0.3),
            },
            "fixed_cube_z": OBJ_SPAWN_Z,
        },
    )

    # 로봇 보라색 (ROBOT_COLORS["purple"] — deploy가 팔레트에 추가)
    set_robot_purple = EventTerm(
        func=randomize_robot_color,
        mode="reset",
        params={"color_names": ["purple"]},
    )


@configclass
class T1CubeBoxEventDRCfg(T1CubeBoxEventCfg):
    # DR에서는 색 무작위화 (base의 보라색 고정 해제)
    set_robot_purple = None

    reset_set_robot_visual_material = EventTerm(
        func=randomize_robot_color,
        mode="reset",
        params={"color_names": list(ROBOT_COLORS.keys())},
    )

    reset_sky_light = EventTerm(
        func=randomize_sky_light,
        mode="reset",
        params={
            "exposure_range": (-4.0, 3.0),
            "temperature_range": (2500.0, 9500.0),
            "textures_root": f"{assets_path}/hdri",
            "asset_cfg": SceneEntityCfg("sky_light"),
        },
    )


@configclass
class T1CubeBoxObservationsCfg(TaskObservationsCfg):
    @configclass
    class SubtaskCfg(ObsGroup):
        cube_grasped = ObsTerm(
            func=any_vial_grasped,
            params={
                "contact_sensor_cfg": SceneEntityCfg("contact_grasp"),
                "vials": ["cube"],
                "min_height": MAT_SURF + 0.02,
                "warmup_steps": 30,
                "force_threshold": 2,
            },
        )
        cube_placed = ObsTerm(
            func=vial_placed_on_rack,
            params=dict(_CUBE_PLACE_PARAMS),
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class T1CubeBoxTerminationsCfg:
    time_out = DoneTerm(func=time_out, time_out=True)
    success = DoneTerm(
        func=vial_placed_on_rack_termination,
        time_out=False,
        params=dict(_CUBE_PLACE_PARAMS),
    )


@configclass
class T1CubeBoxEnvCfg(SO101TaskEnvCfg):
    """Base."""
    scene: T1CubeBoxSceneCfg = T1CubeBoxSceneCfg()
    events: T1CubeBoxEventCfg = T1CubeBoxEventCfg()
    observations: T1CubeBoxObservationsCfg = T1CubeBoxObservationsCfg()


@configclass
class T1CubeBoxDREnvCfg(T1CubeBoxEnvCfg):
    """Domain Randomization."""
    scene: T1CubeBoxDRSceneCfg = T1CubeBoxDRSceneCfg()
    events: T1CubeBoxEventDRCfg = T1CubeBoxEventDRCfg()


@configclass
class T1CubeBoxEvalEnvCfg(T1CubeBoxEnvCfg):
    """Eval (성공 종료 판정 포함)."""
    terminations: T1CubeBoxTerminationsCfg = T1CubeBoxTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 450 / 60.0


@configclass
class T1CubeBoxEvalDREnvCfg(T1CubeBoxDREnvCfg):
    """Eval + DR."""
    terminations: T1CubeBoxTerminationsCfg = T1CubeBoxTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 450 / 60.0
