"""Train F1TenthEnv with rsl_rl PPO.  python rl/train.py --num_envs 256 --headless --max_iterations 300"""
import argparse, os, sys, time
from datetime import datetime

from isaaclab.app import AppLauncher
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from track_util import add_track_arg, check_track

p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=256)
p.add_argument("--max_iterations", type=int, default=None)
p.add_argument("--checkpoint", type=str, default=None, help="resume from this model_*.pt")
p.add_argument("--seed", type=int, default=42)
add_track_arg(p)
AppLauncher.add_app_launcher_args(p)  # adds --headless, --device ...
args = p.parse_args()
check_track(args.track)
app = AppLauncher(args).app

import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab.utils.io import dump_yaml
import importlib.metadata as md
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from f1tenth_env import F1TenthEnv, F1TenthEnvCfg
from rsl_cfg import F1TenthPPORunnerCfg

env_cfg = F1TenthEnvCfg(); env_cfg.track = args.track; env_cfg.scene.num_envs = args.num_envs; env_cfg.seed = args.seed
agent_cfg = handle_deprecated_rsl_rl_cfg(F1TenthPPORunnerCfg(), md.version('rsl-rl-lib')); agent_cfg.seed = args.seed
if args.max_iterations: agent_cfg.max_iterations = args.max_iterations
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

env = RslRlVecEnvWrapper(F1TenthEnv(env_cfg), clip_actions=1.0)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
if args.checkpoint: runner.load(args.checkpoint)
dump_yaml(os.path.join(log_dir, "env.yaml"), env_cfg)
t = time.time()
runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
print(f"Training time {time.time() - t:.1f}s, logs: {log_dir}")
env.close(); app.close()
