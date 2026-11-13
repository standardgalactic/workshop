"""Diagnostic and heterogeneity runs for the two-worker simulation.

For each (heterogeneity, local optimizer) configuration and each seed:
  1. per-step parameter averaging reference (H=1, zero communication cost) fixes theta* and the target loss;
  2. plain averaging at H in {50,200,500} on the modeled WAN gives time to target and final loss;
  3. the fixed-point diagnostic D_H(theta*) is estimated from R independent draws of the local data streams,
     starting from theta* with a stated optimizer initialization.
"""
import argparse, copy, json, sys
import torch
import torch.nn as nn

p = argparse.ArgumentParser()
p.add_argument("--hetero", choices=["mild", "strong"], required=True)
p.add_argument("--opt", choices=["adamw", "sgd"], required=True)
p.add_argument("--seeds", type=int, default=2)
p.add_argument("--budget", type=int, default=3000)
p.add_argument("--draws", type=int, default=16)
p.add_argument("--intervals", type=int, nargs="+", default=[50, 200, 500])
p.add_argument("--out", required=True)
args = p.parse_args()
torch.set_num_threads(1)

STEP_TIME, PAYLOAD_MB, BW_MBPS, LAT = 0.25, 2000.0, 100.0, 0.05
D_IN, D_OUT, N_VAL = 32, 8, 4096
SHIFT_SCALE = 0.5 if args.hetero == "mild" else 2.0
PERT = 0.0 if args.hetero == "mild" else 0.5


def make_model():
    return nn.Sequential(nn.Linear(D_IN, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, D_OUT))


def make_task(seed):
    g = torch.Generator().manual_seed(1000 + seed)
    teacher = make_model()
    for q in teacher.parameters():
        q.data = torch.randn(q.shape, generator=g) * (1.0 / (q.shape[-1] ** 0.5) if q.dim() == 2 else 0.1)
    shifts = [torch.randn(D_IN, generator=g) * SHIFT_SCALE for _ in range(2)]
    teachers = []
    for k in range(2):
        t = copy.deepcopy(teacher)
        if PERT > 0:
            for q in t.parameters():
                scale = (1.0 / (q.shape[-1] ** 0.5) if q.dim() == 2 else 0.1) * PERT
                q.data = q.data + torch.randn(q.shape, generator=g) * scale
        teachers.append(t)

    def sample(shard, n, gen):
        x = torch.randn(n, D_IN, generator=gen) + shifts[shard]
        with torch.no_grad():
            y = teachers[shard](x) + 0.05 * torch.randn(n, D_OUT, generator=gen)
        return x, y

    vg = torch.Generator().manual_seed(7 + seed)
    xs, ys = [], []
    for s in (0, 1):
        x, y = sample(s, N_VAL // 2, vg)
        xs.append(x); ys.append(y)
    return sample, (torch.cat(xs), torch.cat(ys))


def make_opt(w):
    if args.opt == "adamw":
        return torch.optim.AdamW(w.parameters(), lr=2e-3, weight_decay=0.01)
    return torch.optim.SGD(w.parameters(), lr=0.05)


def val_loss(model, val):
    with torch.no_grad():
        return nn.functional.mse_loss(model(val[0]), val[1]).item()


def flat(model):
    return torch.cat([q.data.view(-1) for q in model.parameters()])


def load(model, vec):
    i = 0
    for q in model.parameters():
        n = q.numel(); q.data.copy_(vec[i:i + n].view_as(q)); i += n


def local_steps(w, o, sample, k, gen, h):
    for _ in range(h):
        x, y = sample(k, 64, gen)
        o.zero_grad()
        nn.functional.mse_loss(w(x), y).backward()
        o.step()


def run(H, seed, comm_s):
    torch.manual_seed(seed)
    sample, val = make_task(seed)
    gm = make_model()
    workers = [copy.deepcopy(gm) for _ in range(2)]
    opts = [make_opt(w) for w in workers]
    gens = [torch.Generator().manual_seed(seed * 10 + k) for k in range(2)]
    theta = flat(gm).clone()
    t, done = 0.0, 0
    curve = [(0.0, val_loss(gm, val))]
    while done < args.budget:
        h = min(H, args.budget - done)
        deltas = []
        for k in range(2):
            load(workers[k], theta)
            local_steps(workers[k], opts[k], sample, k, gens[k], h)
            deltas.append(flat(workers[k]) - theta)
        theta = theta + (deltas[0] + deltas[1]) / 2
        load(gm, theta)
        done += h
        t += h * STEP_TIME + comm_s
        curve.append((t, val_loss(gm, val)))
    return dict(curve=curve, final=curve[-1][1], theta=theta, opts=opts, sample=sample, val=val)


def time_to(curve, target):
    for t, l in curve:
        if l <= target:
            return t
    return None


def diagnostic(ref, seed, H, init):
    """Estimate E[D_H(theta*)] from independent data-stream draws."""
    sample, val = ref["sample"], ref["val"]
    theta = ref["theta"]
    Ds = []
    shift_norms = []
    for r in range(args.draws):
        gens = [torch.Generator().manual_seed(555000 + seed * 1000 + r * 10 + k) for k in range(2)]
        deltas = []
        for k in range(2):
            w = make_model(); load(w, theta)
            o = make_opt(w)
            if init == "persist":
                o.load_state_dict(copy.deepcopy(ref["opts"][k].state_dict()))
                # state dict keeps param ids; parameters of w are in same order
            local_steps(w, o, sample, k, gens[k], H)
            deltas.append(flat(w) - theta)
            shift_norms.append((flat(w) - theta).norm().item())
        Ds.append((deltas[0] + deltas[1]) / 2)
    D = torch.stack(Ds)
    mean = D.mean(0)
    se = (D.var(0, unbiased=True).sum().sqrt() / (args.draws ** 0.5)).item()
    m = make_model(); load(m, theta)
    l0 = val_loss(m, val)
    load(m, theta + mean)
    l1 = val_loss(m, val)
    return dict(norm_mean=mean.norm().item(), se=se, shift_norm=sum(shift_norms) / len(shift_norms),
                loss_theta=l0, loss_after=l1)


comm = 2 * PAYLOAD_MB * 8 / BW_MBPS + 2 * LAT
rows = []
for seed in range(args.seeds):
    ref = run(1, seed, 0.0)
    target = ref["final"] * 1.10
    print(f"[{args.hetero}/{args.opt}] seed {seed} reference final loss {ref['final']:.4f}", flush=True)
    for H in args.intervals:
        r = run(H, seed, comm)
        row = dict(hetero=args.hetero, opt=args.opt, seed=seed, H=H, ref_final=ref["final"],
                   final=r["final"], t_target=time_to(r["curve"], target))
        inits = ["reset", "persist"] if args.opt == "adamw" else ["reset"]
        for init in inits:
            row[init] = diagnostic(ref, seed, H, init)
        rows.append(row)
        print(json.dumps(row), flush=True)
json.dump(rows, open(args.out, "w"), indent=1)
