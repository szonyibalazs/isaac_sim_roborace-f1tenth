# rl_drift - point-drift policy (no track)

Isaac Lab `DirectRLEnv` + rsl_rl PPO. The Roboracer car (`../usd/f1tenth.usd`, not modified) learns to drift
around a point on flat ground. Hundreds of parallel cars, 8 m env spacing, no lidar, no walls.

* `--task circle`: hold radius R (1.5 m) around the origin in a sustained slide.
* `--task eight`: figure eight = two touching circles (centres (-a,0), (+a,0), a = R) driven in opposite senses;
  the active circle flips after 0.95 * 2pi of orbit angle (drift direction flips too).

Status: code smoke-tested only (3 iterations, no learning result measured yet).

## Run (always from the repo root)
```bash
source scripts/env_isaac.sh
# train (6 GB GPU: 512 envs is comfortable, 256 also fine; 1000-1500 iterations, ~32 steps/env/iter)
timeout -s KILL 7200 python rl_drift/train.py --task circle --num_envs 512 --headless --max_iterations 1500
timeout -s KILL 7200 python rl_drift/train.py --task eight  --num_envs 512 --headless --max_iterations 2500
# evaluate (prints mean |radius err|, mean beta_o, in-band fraction, episode success/lengths)
python rl_drift/play.py --task circle --checkpoint rl_drift/logs/circle/<run>/model_1499.pt --num_envs 16 --steps 900 --headless
# watch (GUI, no --headless); default checkpoint = newest rl_drift/logs/<task>/*/model_*.pt
python rl_drift/play.py --task eight --num_envs 8
# resume: train.py ... --checkpoint <model_*.pt>
tensorboard --logdir rl_drift/logs
```
Logs: `rl_drift/logs/<task>/<timestamp>/` (git-ignore this dir). `train.py` also copies the final model to
`rl_drift/checkpoints/<task>_last.pt`; copy the best one to `checkpoints/circle.pt` / `eight.pt`.

TensorBoard: `Train/mean_reward` (should climb and plateau), `Train/mean_episode_length` (-> 600 = 20 s, timeouts),
and the custom `Drift/*` curves: `radius_err` (want < 0.2 m), `abs_beta_deg` (want 20-45), `in_band_frac` (want > 0.7),
`ep_len_s`. If `ep_len_s` grows but `in_band_frac` stays ~0 the car circles by grip, not by drifting.

## Env
* Action `[steer, throttle]` in [-1,1]: steer -> +-0.6 rad centre angle (Ackermann per wheel); throttle -> rear wheel
  speed target 0..6 m/s equivalent.
* Obs (13): centre position in body frame (2, /R), radius error, body vx, vy, yaw rate, slip angle beta_o, tangential
  and radial speed, previous action (2), eight: phase (+-1) and loop progress (0 for circle).
* beta = atan2(vy, vx) (body frame); `beta_o = -s*beta` (s = +1 CCW / -1 CW) is positive when the nose points toward
  the centre (countersteer attitude).
* Reset: random point on the orbit, tangential speed 1.5-3 m/s, nose already turned 0-30 deg toward the centre
  (eight: random phase and loop position).
* Termination: |dist - R| > 1.5 m, flipped, speed < 0.3 m/s for 3 s. Episode 20 s.

## Reward (per policy step, 30 Hz) - `DriftEnvCfg`
| term | formula | weight |
|---|---|---|
| radius | `-\|dist - R\|` | 1.5 |
| slip | `exp(-(dist of beta_o to [20,45] deg / 10 deg)^2)`, only if speed > 1 m/s | 1.5 |
| speed | `exp(-(v - 2.8)^2)` | 0.5 |
| tangential | `clip(v_tangential / 2.8, -1, 1)` (right direction around the point) | 1.0 |
| smoothness | `-\|a - a_prev\|^2` | 0.05 |
| terminated | one-off | -10 |

## Car / physics overrides (applied at load in `DriftEnv.__init__` / `_setup_scene`, USD untouched)
| what | stock USD | here | why |
|---|---|---|---|
| rear wheel drive | maxForce 0.5 Nm, damping 1e-3 (~2.4 m/s max) | 2.0 Nm, damping 0.1, vel limit 200 rad/s | traction limit per rear wheel at mu 0.55 is ~0.5 Nm; more torque lets the rear wheels spin past road speed (power oversteer) |
| steering drive | k 0.1, d 0.01, 5 Nm | k 5.0, d 0.2, 5 Nm | stiff enough to track fast countersteer at 30 Hz |
| tyre friction (multiply combine, ground 1.0) | default 0.5 avg | rear 0.55, front 1.0 | rear breaks loose first, front keeps grip for steering; rear lateral limit ~5.4 m/s^2 vs ~6 m/s^2 needed at R 1.5 m, 3 m/s |
| solver | 4 pos iters | 8 pos iters | contact accuracy during sliding |

## Tuning if it does not drift
* Car grips and drives a clean circle (`abs_beta_deg` ~ 5): lower `mu_rear` (0.4-0.5) or raise `v_target` / `max_speed`
  (drift needs lateral accel above the rear grip limit), raise `rew_slip`.
* Car spins out / flips (`ep_len_s` short): raise `mu_rear` (0.65-0.7), lower `v_target`, widen `beta_hi`, lower `max_steer`.
* Learns to stand still / crawl: raise `v_min`, `rew_tangent`; check `stall_time_s`.
* Eight fails only at the flip: lower `eight_loop_angle` (earlier switch), enlarge `eight_offset` (> R gives a crossing
  with more room), or train circle first and resume eight from that checkpoint.
* Slow / noisy learning: more envs (`--num_envs 1024` if VRAM allows), `learning_rate` and `entropy_coef` in `rsl_cfg.py`.
