"""PPO on CPU for the G1 tracking env (rsl-rl defaults: clipped objective, GAE, adaptive learning rate on KL,
running observation normalisation). One reference motion, a wall-clock budget, checkpoints every few minutes.

    uv run python -m humanoid.train --ref runs/<id>/clips/0278/reference.npz --minutes 180

Logs one line per iteration to <out>/log.jsonl: mean reward, episode length, fall rate, key-body error, assist.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from humanoid.track_env import VecTrackEnv


def mlp(i: int, o: int, hidden=(512, 256, 128)) -> nn.Sequential:
    layers, d = [], i
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ELU()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class Normalizer(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n))
        self.register_buffer("var", torch.ones(n))
        self.register_buffer("count", torch.tensor(1e-4))

    def update(self, x: torch.Tensor) -> None:
        b_mean, b_var, b_n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta, tot = b_mean - self.mean, self.count + b_n
        self.mean += delta * b_n / tot
        self.var = (self.var * self.count + b_var * b_n + delta ** 2 * self.count * b_n / tot) / tot
        self.count = tot

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp((x - self.mean) / torch.sqrt(self.var + 1e-2), -5, 5)   # std floor 0.1


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, init_std: float = 0.8):
        super().__init__()
        self.norm = Normalizer(obs_dim)
        self.actor = mlp(obs_dim, act_dim)
        self.critic = mlp(obs_dim, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(np.log(init_std))))

    def dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        return torch.distributions.Normal(self.actor(self.norm(obs)), self.log_std.exp())

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(self.norm(obs)).squeeze(-1)


def train(ref_path: Path, out: Path, minutes: float, workers: int, envs_per_worker: int, steps: int = 32,
          assist_until: float = 0.4) -> None:
    torch.set_num_threads(2)
    ref = np.load(ref_path)
    bat, top = str(ref.get("bat", "none")), str(ref.get("top_hand", "right"))
    env = VecTrackEnv(ref["qpos"], float(ref["fps"]), workers, envs_per_worker, bat, top)
    ac = ActorCritic(env.obs_dim, env.act_dim)
    opt = torch.optim.Adam(ac.parameters(), lr=1e-3)
    lr, gamma, lam, clip, epochs, minibatches, kl_target = 1e-3, 0.99, 0.95, 0.2, 5, 4, 0.01
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({"ref": str(ref_path), "workers": workers, "envs": env.n,
                                                 "steps": steps, "minutes": minutes, "T": env.T, "bat": bat,
                                                 "top_hand": top}, indent=1))
    obs = torch.as_tensor(env.reset())
    t_start, it, last_ckpt = time.time(), 0, time.time()
    ep_lens, falls, ek = [], [], []
    with (out / "log.jsonl").open("a") as log:
        while (time.time() - t_start) / 60 < minutes:
            frac = (time.time() - t_start) / 60 / minutes
            assist = max(0.0, 1.0 - frac / assist_until) if assist_until > 0 else 0.0
            env.set_assist(assist)
            buf = {k: [] for k in ("obs", "act", "logp", "val", "rew", "done", "mu")}
            t0 = time.time()
            with torch.no_grad():
                for _ in range(steps):
                    ac.norm.update(obs)
                    dist = ac.dist(obs)
                    a = dist.sample()
                    nobs, r, d, infos = env.step(a.numpy())
                    buf["obs"].append(obs)
                    buf["act"].append(a)
                    buf["logp"].append(dist.log_prob(a).sum(-1))
                    buf["mu"].append(dist.mean)
                    buf["val"].append(ac.value(obs))
                    buf["rew"].append(torch.as_tensor(r))
                    buf["done"].append(torch.as_tensor(d, dtype=torch.float32))
                    for i in infos:
                        ek.append(i["e_k_cm"])
                        if "ep_len" in i:
                            ep_lens.append(i["ep_len"])
                            falls.append(i["fell"])
                    obs = torch.as_tensor(nobs)
                last_val = ac.value(obs)
            collect = time.time() - t0
            V = torch.stack(buf["val"])
            R, D = torch.stack(buf["rew"]), torch.stack(buf["done"])
            adv = torch.zeros_like(R)
            gae = torch.zeros(env.n)
            for t in reversed(range(steps)):
                nv = last_val if t == steps - 1 else V[t + 1]
                delta = R[t] + gamma * nv * (1 - D[t]) - V[t]
                gae = delta + gamma * lam * (1 - D[t]) * gae
                adv[t] = gae
            ret = adv + V
            O, A, LP, MU = (torch.cat(buf[k]) for k in ("obs", "act", "logp", "mu"))
            SIG = ac.log_std.exp().detach().clone()
            ADV, RET = adv.flatten(), ret.flatten()
            ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
            n = len(O)
            for _ in range(epochs):
                perm = torch.randperm(n)
                for mb in perm.chunk(minibatches):
                    dist = ac.dist(O[mb])
                    logp = dist.log_prob(A[mb]).sum(-1)
                    ratio = torch.exp(logp - LP[mb])
                    pg = -torch.min(ratio * ADV[mb], torch.clamp(ratio, 1 - clip, 1 + clip) * ADV[mb]).mean()
                    vloss = (ac.value(O[mb]) - RET[mb]).pow(2).mean()
                    ent = dist.entropy().sum(-1).mean()
                    loss = pg + 1.0 * vloss - 0.002 * ent
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(ac.parameters(), 1.0)
                    opt.step()
                    with torch.no_grad():   # analytic KL(old || new) between diagonal Gaussians, as rsl-rl
                        mu1, s1 = ac.dist(O[mb]).mean, ac.log_std.exp()
                        kl = (torch.log(s1 / SIG) + (SIG ** 2 + (MU[mb] - mu1) ** 2) / (2 * s1 ** 2) - 0.5).sum(-1).mean().item()
                    lr = max(1e-5, lr / 1.5) if kl > 2 * kl_target else min(1e-2, lr * 1.5) if kl < kl_target / 2 else lr
                    for g in opt.param_groups:
                        g["lr"] = lr
            it += 1
            row = {"iter": it, "min": round((time.time() - t_start) / 60, 2), "reward": round(R.mean().item(), 4),
                   "ep_len": round(float(np.mean(ep_lens[-200:])), 1) if ep_lens else None,
                   "fall_rate": round(float(np.mean(falls[-200:])), 3) if falls else None,
                   "key_err_cm": round(float(np.mean(ek[-2000:])), 2) if ek else None, "assist": round(assist, 3),
                   "lr": lr, "std": round(ac.log_std.exp().mean().item(), 3),
                   "sps": int(n / (time.time() - t0)), "collect_s": round(collect, 2)}
            log.write(json.dumps(row) + "\n")
            log.flush()
            if it % 10 == 1:
                print(row, flush=True)
            if time.time() - last_ckpt > 300:
                torch.save(ac.state_dict(), out / "policy.pt")
                last_ckpt = time.time()
    torch.save(ac.state_dict(), out / "policy.pt")
    env.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--minutes", type=float, default=180)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--envs", type=int, default=32)
    ap.add_argument("--assist-until", type=float, default=0.4)
    a = ap.parse_args()
    ref = Path(a.ref)
    out = Path(a.out) if a.out else ref.parent / "rl"
    train(ref, out, a.minutes, a.workers, a.envs, assist_until=a.assist_until)


if __name__ == "__main__":
    main()
