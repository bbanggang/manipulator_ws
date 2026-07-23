

def reset_cube_box(
        env,
        env_ids: torch.Tensor,
        cube: str,
        box: str,
        cube_pose_range: dict[str, tuple[float, float]],
        box_pose_range: dict[str, tuple[float, float]],
        fixed_cube_z: float,
):
    """커스텀 T1(빨간 큐브→검은 박스) 리셋.

    reset_vials_rack 은 rack의 슬롯 prim(top_*)에 의존해 박스(트레이)에선 쓸 수 없으므로 신설.
    박스는 소폭 무작위(고정 타깃), 큐브는 넓게 무작위(z 고정). random_asset_pose 재사용.
    """
    # 박스 (소폭 무작위)
    box_obj = env.scene[box]
    random_asset_pose(env, env_ids, box_obj, box_pose_range, {})
    box_obj.write_root_velocity_to_sim(
        torch.zeros((len(env_ids), 6), device=box_obj.device), env_ids=env_ids
    )

    # 큐브 (넓게 무작위, z 고정)
    cube_obj = env.scene[cube]
    default_z = cube_obj.data.default_root_state[env_ids[0], 2].item()
    pos_offset = {"z": fixed_cube_z - default_z}
    pose_range_z_fixed = {**cube_pose_range, "z": (0.0, 0.0)}
    random_asset_pose(env, env_ids, cube_obj, pose_range_z_fixed, pos_offset)
    cube_obj.write_root_velocity_to_sim(
        torch.zeros((len(env_ids), 6), device=cube_obj.device), env_ids=env_ids
    )
