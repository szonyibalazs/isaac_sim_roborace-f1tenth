"""Scaling: python rl/bench.py --num_envs 1024 --headless  -> env steps/s (random actions)."""
import argparse, os, sys, time
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); p.add_argument("--num_envs", type=int, default=256); p.add_argument("--steps", type=int, default=200)
AppLauncher.add_app_launcher_args(p); a = p.parse_args(); app = AppLauncher(a).app
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from f1tenth_env import F1TenthEnv, F1TenthEnvCfg
cfg = F1TenthEnvCfg(); cfg.scene.num_envs = a.num_envs
env = F1TenthEnv(cfg); env.reset()
for i in range(a.steps + 20):
    if i == 20: torch.cuda.synchronize(); t = time.time()
    env.step(torch.rand(a.num_envs, 2, device=env.device) * 2 - 1)
torch.cuda.synchronize(); dt = time.time() - t
print(f"BENCH envs={a.num_envs} env_steps/s={a.num_envs * a.steps / dt:.0f} iters/s={a.steps / dt:.1f} vram_MB={torch.cuda.mem_get_info()[1]/2**20 - torch.cuda.mem_get_info()[0]/2**20:.0f}")
env.close(); app.close()
