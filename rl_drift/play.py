"""Play / evaluate.  python rl_drift/play.py --task circle --checkpoint rl_drift/checkpoints/circle.pt [--num_envs 16] [--headless] [--steps 900]
Prints mean |radius error|, mean beta_o, in-band fraction and episode stats (timeout = success)."""
import argparse, glob, os, sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--task", choices=["circle", "eight"], default="circle")
p.add_argument("--num_envs", type=int, default=16)
p.add_argument("--checkpoint", type=str, default=None, help="default: newest rl_drift/logs/<task>/*/model_*.pt")
p.add_argument("--steps", type=int, default=0, help="stop after N policy steps (0 = until closed)")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

import importlib.metadata as md
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
from drift_env import DriftEnv, DriftEnvCfg, DEG
from rsl_cfg import DriftPPORunnerCfg

ckpt = args.checkpoint or sorted(glob.glob(f"{here}/logs/{args.task}/*/model_*.pt"), key=os.path.getmtime)[-1]
env_cfg = DriftEnvCfg(); env_cfg.task = args.task; env_cfg.scene.num_envs = args.num_envs
env = RslRlVecEnvWrapper(DriftEnv(env_cfg), clip_actions=1.0)
u = env.unwrapped
agent_cfg = handle_deprecated_rsl_rl_cfg(DriftPPORunnerCfg(), md.version("rsl-rl-lib"))
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(ckpt); print("loaded", ckpt)
policy = runner.get_inference_policy(device=u.device)
obs = env.get_observations(); i = 0
n = env.num_envs; dev = u.device
ep_len = torch.zeros(n, device=dev)
S = {k: 0.0 for k in ("err", "beta", "band", "speed", "cnt")}
done_ok, done_fail, lens_ok, lens_fail = 0, 0, [], []
c = u.cfg
while app.is_running() and (not args.steps or i < args.steps):
    with torch.inference_mode():
        obs, rew, dones, _ = env.step(policy(obs)); policy.reset(dones)
    i += 1; ep_len += 1
    assert torch.isfinite(obs["policy"]).all() and torch.isfinite(rew).all(), 'non-finite obs/reward'
    fast = u.speed > c.v_min
    S["err"] += u.err.abs().sum().item(); S["beta"] += u.beta_o.sum().item(); S["speed"] += u.speed.sum().item()
    S["band"] += (((u.beta_o >= c.beta_lo) & (u.beta_o <= c.beta_hi)) & fast).float().sum().item(); S["cnt"] += n
    d = dones.bool()
    for j in d.nonzero().squeeze(-1).tolist():
        if u.reset_terminated[j]: done_fail += 1; lens_fail.append(ep_len[j].item() * u.step_dt)
        else: done_ok += 1; lens_ok.append(ep_len[j].item() * u.step_dt)
    ep_len[d] = 0
cnt = max(S["cnt"], 1)
mean = lambda x: sum(x) / len(x) if x else float("nan")
print(f"RESULT task={args.task} steps={i} envs={n} | mean|radius err|={S['err']/cnt:.3f} m | mean beta_o={S['beta']/cnt/DEG:.1f} deg "
      f"| in-band(20-45deg & v>{c.v_min}) frac={S['band']/cnt:.2f} | mean speed={S['speed']/cnt:.2f} m/s "
      f"| episodes: success(timeout)={done_ok} mean_len={mean(lens_ok):.1f}s, terminated={done_fail} mean_len={mean(lens_fail):.1f}s")
sys.stdout.flush(); env.close(); app.close()
