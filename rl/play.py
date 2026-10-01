"""Play a trained policy.  python rl/play.py --checkpoint rl/logs/<run>/model_300.pt [--num_envs 16] [--headless]"""
import argparse, glob, os, sys

from isaaclab.app import AppLauncher
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from track_util import add_track_arg, check_track

p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=16)
p.add_argument("--checkpoint", type=str, default=None, help="default: newest rl/logs/*/model_*.pt")
p.add_argument("--steps", type=int, default=0, help="stop after N steps (0 = until closed)")
add_track_arg(p)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
check_track(args.track)
app = AppLauncher(args).app

import torch
from rsl_rl.runners import OnPolicyRunner
import importlib.metadata as md
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
from f1tenth_env import F1TenthEnv, F1TenthEnvCfg
from rsl_cfg import F1TenthPPORunnerCfg

ckpt = args.checkpoint or sorted(glob.glob(f"{here}/logs/*/model_*.pt"), key=os.path.getmtime)[-1]
env_cfg = F1TenthEnvCfg(); env_cfg.track = args.track; env_cfg.scene.num_envs = args.num_envs
env = RslRlVecEnvWrapper(F1TenthEnv(env_cfg), clip_actions=1.0)
agent_cfg = handle_deprecated_rsl_rl_cfg(F1TenthPPORunnerCfg(), md.version('rsl-rl-lib'))
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(ckpt); print("loaded", ckpt)
policy = runner.get_inference_policy(device=env.unwrapped.device)
obs = env.get_observations(); i = 0
tot = torch.zeros(env.num_envs, device=env.unwrapped.device)
u = env.unwrapped; lap_len = u.cl_n * u.cl_ds; dist = t0 = 0.0  # lap timer for env 0 (sim time)
while app.is_running() and (not args.steps or i < args.steps):
    with torch.inference_mode():
        obs, rew, dones, _ = env.step(policy(obs)); policy.reset(dones)
    tot += rew; i += 1
    dist += u.progress[0].item()
    if dones[0]:
        print("env 0 reset (crash/timeout): lap timer restarted"); dist, t0 = 0.0, i * u.step_dt
    elif dist >= lap_len:
        t = i * u.step_dt; print(f"lap time: {t - t0:.2f} s"); dist -= lap_len; t0 = t
print(f"mean reward/step {tot.mean().item() / max(i, 1):.3f}, total progress mean {env.unwrapped.cl_idx.float().mean().item():.1f}")
env.close(); app.close()
