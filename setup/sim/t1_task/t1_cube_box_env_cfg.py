# SPDX-License-Identifier: Apache-2.0
# 커스텀 T1 태스크: 빨간 큐브 → 검은 박스 (Pick red cube, place in black box)
# vials_to_rack_env_cfg.py 를 개조. 재사용:
#   - 파지/배치 판정은 vials 함수를 그대로 사용(any_vial_grasped / vial_placed_on_rack[_termination])
#     · vials=["cube"], rack_name="box", vertical_threshold=0.0(큐브는 방향 무관 → 수직조건 무력화)
#   - 리셋만 신설(reset_cube_box): 박스는 슬롯 prim이 없어 reset_vials_rack 재사용 불가
# 에셋: 빨간 큐브 = 프리미티브 CuboidCfg(USD 불필요) / 검은 박스 = tray_black.usda(tray 재색·축소)
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
    randomize_mat_rotation,
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

MAT_TOP = 0.037              # 작업 표면(테이블) 상단 높이 — 큐브·박스가 이 위에 안착
CUBE_SIZE = 0.025            # 실기 빨간 큐브(~2.5cm)와 일치
CUBE_SPAWN_Z = MAT_TOP + CUBE_SIZE / 2 + 0.006  # ≈0.056, mat 위에서 살짝 떨궈 안착
BOX_SCALE = 0.6             # tray(0.2×0.16) → 0.12×0.096 (rack 풋프린트와 유사)
BOX_SPAWN_Z = MAT_TOP + 0.005  # tray 바닥이 mat 위에 안착

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
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.22, -0.10, CUBE_SPAWN_Z)),
)

# --- 검은 박스 (tray 재색·축소) ---
box = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Box",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{assets_path}/usd/tray_black.usda",
        scale=(BOX_SCALE, BOX_SCALE, 1.0),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.14, 0.03, BOX_SPAWN_Z)),
)

# 박스 로컬 판정 경계 (tray extent 0→(0.2,0.16), 축소 0.6 → 0→(0.12,0.096)). 검증 시 튜닝.
BOX_LOCAL_X_MAX = 0.2 * BOX_SCALE     # 0.12
BOX_LOCAL_Y_MAX = 0.16 * BOX_SCALE    # 0.096
BOX_LOCAL_Z_MAX = 0.06                 # 큐브 중심이 이 아래면 "박스 안"


@configclass
class T1CubeBoxSceneCfg(SO101TaskSceneCfg):
    robot: ArticulationCfg = S0101_CONTACT_GRASP_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    cube = cube.replace()
    box = box.replace()

    # 흰색 작업 테이블. ⚠️ base scene엔 ground plane이 없고 원래 mat.usda(MDL 재질, 재색 불가)가
    # 바닥 콜라이더였음 → 교체 시 collision 필수. AssetBaseCfg 정적 collision은 등록이 불안정해
    # 큐브가 바닥 없이 낙하→경계 밖 삭제→크래시했음. → kinematic RigidObject로 안정적 콜라이더 확보.
    # top=MAT_TOP(0.037)에 큐브·박스 안착. (DR reset_mat_rotation은 대칭 평면이라 무해)
    mat = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Mat",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.42, 0.02),  # top = MAT_TOP
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.9, 0.9, 0.9)  # 흰색에 가깝게
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.22, 0.0, MAT_TOP - 0.01)),
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
    """큐브·박스 리셋 (큐브 위치 다양화, 박스는 소폭)."""

    reset_cube_setup = EventTerm(
        func=reset_cube_box,
        mode="reset",
        params={
            "cube": "cube",
            "box": "box",
            # 큐브 위치 다양화 (실기 5지점 그리드 반영). 검증 시 도달범위 맞춰 조정
            "cube_pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.06, 0.06),
                "yaw": (-1.57, 1.57),
            },
            # 박스는 소폭만
            "box_pose_range": {
                "x": (-0.03, 0.03),
                "y": (-0.02, 0.02),
                "yaw": (-0.3, 0.3),
            },
            "fixed_cube_z": CUBE_SPAWN_Z,
        },
    )


@configclass
class T1CubeBoxEventDRCfg(T1CubeBoxEventCfg):
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

    reset_mat_rotation = EventTerm(
        func=randomize_mat_rotation,
        mode="reset",
        params={
            "yaw_range": (-0.3, 0.3),
            "asset_cfg": SceneEntityCfg("mat"),
        },
    )


# 큐브 판정 공통 파라미터 (vials 함수 재사용, vertical_threshold=0 → 방향 무관)
_CUBE_PLACE_PARAMS = dict(
    contact_sensor_cfg=SceneEntityCfg("contact_grasp"),
    vials=["cube"],
    rack_name="box",
    warmup_steps=30,
    grasp_history_window=20,
    force_threshold=2,
    rack_local_x_min=0.0,
    rack_local_x_max=BOX_LOCAL_X_MAX,
    rack_local_y_min=0.0,
    rack_local_y_max=BOX_LOCAL_Y_MAX,
    rack_local_z_max=BOX_LOCAL_Z_MAX,
    vertical_threshold=0.0,
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
                "min_height": 0.055,
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
