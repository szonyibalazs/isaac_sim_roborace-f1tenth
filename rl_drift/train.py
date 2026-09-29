"""Train the drift policy.  python rl_drift/train.py --task circle|eight --num_envs 512 --headless --max_iterations 500
Logs: rl_drift/logs/<task>/<timestamp>/.  Final model is copied to rl_drift/checkpoints/<task>_last.pt"""
import argparse, os, shutil, sys, time
from datetime import datetime

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--task", choices=["circle", "eight"], default="circle")
p.add_argument("--num_envs", type=int, default=512)
p.add_argument("--max_iterations", type=int, default=None)
p.add_argument("--checkpoint", type=str, default=None, help="resume from this model_*.pt")
p.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

import importlib.metadata as md
from rsl_rl.runners import OnPolicyRunner
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
from drift_env import DriftEnv, DriftEnvCfg
from rsl_cfg import DriftPPORunnerCfg

env_cfg = DriftEnvCfg(); env_cfg.task = args.task; env_cfg.scene.num_envs = args.num_envs; env_cfg.seed = args.seed
agent_cfg = handle_deprecated_rsl_rl_cfg(DriftPPORunnerCfg(), md.version("rsl-rl-lib")); agent_cfg.seed = args.seed
agent_cfg.experiment_name = f"drift_{args.task}"
if args.max_iterations: agent_cfg.max_iterations = args.max_iterations
log_dir = os.path.join(here, "logs", args.task, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

env = RslRlVecEnvWrapper(DriftEnv(env_cfg), clip_actions=1.0)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
if args.checkpoint: runner.load(args.checkpoint)
dump_yaml(os.path.join(log_dir, "env.yaml"), env_cfg)
t = time.time()
runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
os.makedirs(os.path.join(here, "checkpoints"), exist_ok=True)
runner.save(os.path.join(log_dir, "model_final.pt"))
shutil.copy(os.path.join(log_dir, "model_final.pt"), os.path.join(here, "checkpoints", f"{args.task}_last.pt"))
print(f"Training time {time.time() - t:.1f}s, logs: {log_dir}")
sys.stdout.flush(); env.close(); app.close()
