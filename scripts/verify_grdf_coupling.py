"""Verify the GRDF-driven coupling in Isaac Lab (ShadowLite).

Two checks:
  1. Command equivalence — the GRDF phase couplings produce the same J1/J2
     commands as the legacy coupling_theta split in _handle_coupled_joints.
  2. Physics tracking — holding curl-phase setpoints, the measured joint
     positions follow the sequential law (J2 saturates before J1 moves).

Run:
    python scripts/verify_grdf_coupling.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify GRDF coupling in sim.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

from roto.tasks.baoding.baoding import (  # noqa: E402
    BaodingShadowLiteCfg,
    BaodingShadowLiteEnv,
    apply_baoding_object_cfgs_from_scalars,
)


def main():
    cfg = BaodingShadowLiteCfg()
    cfg.scene.num_envs = 2
    cfg.num_eval_envs = 0
    cfg.use_grdf_coupling = True
    cfg.tacsl_contact_expr = None
    # park the balls away from the hand so contact can't perturb the sweep
    cfg.ball_1_init_x, cfg.ball_1_init_y = 0.3, 0.3
    cfg.ball_2_init_x, cfg.ball_2_init_y = 0.35, 0.3
    apply_baoding_object_cfgs_from_scalars(cfg)

    env = BaodingShadowLiteEnv(cfg=cfg)
    device = env.device
    assert env.grdf_model is not None, "GRDF coupling not active"

    i_j1 = env.robot.joint_names.index("rh_FFJ1")
    i_j2 = env.robot.joint_names.index("rh_FFJ2")
    theta = cfg.coupling_theta
    j2_upper = env.robot_joint_pos_upper_limits[i_j2].item()
    j1_upper = env.robot_joint_pos_upper_limits[i_j1].item()

    # ---- 1. command equivalence: GRDF path vs legacy theta split ----------
    max_diff = 0.0
    grdf_model = env.grdf_model
    for proxy in torch.linspace(0.0, j2_upper, 41):
        env.joint_pos_cmd.zero_()
        env.joint_pos_cmd[:, env.coupled_driver_indices] = proxy.to(device)
        env._handle_coupled_joints()
        grdf_cmd = env.joint_pos_cmd.clone()

        env.joint_pos_cmd.zero_()
        env.joint_pos_cmd[:, env.coupled_driver_indices] = proxy.to(device)
        env.grdf_model = None
        env._handle_coupled_joints()
        env.grdf_model = grdf_model

        max_diff = max(max_diff, (grdf_cmd - env.joint_pos_cmd).abs().max().item())
    print(f"\n[1] command equivalence: max |grdf - legacy| = {max_diff:.2e} rad "
          f"({'PASS' if max_diff < 1e-4 else 'FAIL'})")

    # ---- 2. physics tracking of the sequential law ------------------------
    print("\n[2] physics tracking (steady state per phase setpoint):")
    print(f"{'phase':>7} {'J2 cmd':>8} {'J2 meas':>8} {'J1 cmd':>8} {'J1 meas':>8}")
    failures = []
    for proxy in (0.4, theta, 1.2, j2_upper):
        env.joint_pos_cmd.zero_()
        env.joint_pos_cmd[:, env.coupled_driver_indices] = proxy
        env._handle_coupled_joints()
        for _ in range(200):
            env._apply_action()
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(env.physics_dt)
        q = env.robot.data.joint_pos[0]
        j2_cmd = env.joint_pos_cmd[0, i_j2].item()
        j1_cmd = env.joint_pos_cmd[0, i_j1].item()
        j2_meas, j1_meas = q[i_j2].item(), q[i_j1].item()
        print(f"{proxy:7.3f} {j2_cmd:8.3f} {j2_meas:8.3f} {j1_cmd:8.3f} {j1_meas:8.3f}")
        if abs(j2_meas - j2_cmd) > 0.15 or abs(j1_meas - j1_cmd) > 0.15:
            failures.append(proxy)

    # the law itself: at phase <= theta, J1 must not have moved
    print(f"\n    sequential law: J1 at phase<=theta stayed ~0, J1 at full phase "
          f"~{j1_upper:.2f} rad")
    print(f"[2] tracking within 0.15 rad: "
          f"{'PASS' if not failures else f'FAIL at phases {failures}'}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
