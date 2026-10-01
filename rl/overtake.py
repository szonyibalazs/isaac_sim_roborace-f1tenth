"""Train (from scratch) / watch overtaking against a frozen slow lap-policy opponent.
  python rl/overtake.py --opp_checkpoint rl/logs/<run>/model_250.pt --track porto --num_envs 64 --headless --max_iterations 300
  python rl/overtake.py --play --opp_checkpoint <lap model> --checkpoint rl/logs/overtake/<run>/model_299.pt --track porto --num_envs 1   (prints ego lap/overtakes)
"""
import argparse, os, sys, time
from datetime import datetime

from isaaclab.app import AppLauncher
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
from track_util import add_track_arg, check_track

p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=64)
p.add_argument("--max_iterations", type=int, default=300)
p.add_argument("--opp_checkpoint", required=True, help="frozen slow-opponent policy (trained lap model)")
p.add_argument("--checkpoint", default=None, help="ego weights: omit to train from scratch, give an overtake model to resume / --play")
p.add_argument("--opp_speed_scale", type=float, default=0.4)
p.add_argument("--play", action="store_true")
p.add_argument("--steps", type=int, default=0)
add_track_arg(p)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
check_track(args.track)
app = AppLauncher(args).app

import torch
from rsl_rl.runners import OnPolicyRunner
import importlib.metadata as md
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from overtake_env import OvertakeEnv, OvertakeEnvCfg
from rsl_cfg import F1TenthPPORunnerCfg

cfg = OvertakeEnvCfg(); cfg.track = args.track; cfg.scene.num_envs = args.num_envs; cfg.opp_speed_scale = args.opp_speed_scale
agent_cfg = handle_deprecated_rsl_rl_cfg(F1TenthPPORunnerCfg(), md.version('rsl-rl-lib'))
agent_cfg.experiment_name = "overtake"; agent_cfg.max_iterations = args.max_iterations
if args.checkpoint: agent_cfg.algorithm.learning_rate = 3e-4  # resuming: gentle
env = RslRlVecEnvWrapper(OvertakeEnv(cfg), clip_actions=1.0)
u = env.unwrapped
dev = u.device

opp_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
opp_runner.load(args.opp_checkpoint); u.opp_policy = opp_runner.get_inference_policy(device=dev)  # frozen: never updated

log_dir = os.path.join(here, "logs", "overtake", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None if args.play else log_dir, device=agent_cfg.device)
if args.checkpoint: runner.load(args.checkpoint); print("ego loaded", args.checkpoint)
else: assert not args.play, "--play needs --checkpoint"

if args.play:
    policy = runner.get_inference_policy(device=dev)
    obs = env.get_observations(); i = 0; n_pass = 0; dist = t0 = 0.0; lap_len = u.cl_n * u.cl_ds
    while app.is_running() and (not args.steps or i < args.steps):
        with torch.inference_mode():
            obs, rew, dones, _ = env.step(policy(obs)); policy.reset(dones)
        i += 1; dist += u.progress[0].item()
        if u.passed[0]: n_pass += 1; print(f"overtake #{n_pass} (t={i * u.step_dt:.1f}s)")
        if dones[0]: print("env 0 reset (crash/timeout)"); dist, t0 = 0.0, i * u.step_dt
        elif dist >= lap_len: t = i * u.step_dt; print(f"lap time: {t - t0:.2f} s"); dist -= lap_len; t0 = t
else:
    dump_yaml(os.path.join(log_dir, "env.yaml"), cfg)
    t = time.time()
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    print(f"Training time {time.time() - t:.1f}s, logs: {log_dir}")
env.close(); app.close()
