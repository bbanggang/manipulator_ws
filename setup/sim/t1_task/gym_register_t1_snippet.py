

# ==== 커스텀 T1 태스크 (빨간 큐브 → 검은 박스) 등록 ====
gym.register(
    id="Lerobot-So101-T1-CubeBox",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.t1_cube_box_env_cfg:T1CubeBoxEnvCfg",
    },
)

gym.register(
    id="Lerobot-So101-T1-CubeBox-DR",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.t1_cube_box_env_cfg:T1CubeBoxDREnvCfg",
    },
)

gym.register(
    id="Lerobot-So101-T1-CubeBox-Eval",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.t1_cube_box_env_cfg:T1CubeBoxEvalEnvCfg",
    },
)

gym.register(
    id="Lerobot-So101-T1-CubeBox-DR-Eval",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.t1_cube_box_env_cfg:T1CubeBoxEvalDREnvCfg",
    },
)
