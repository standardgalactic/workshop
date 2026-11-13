"""Round-length experiment for two-worker infrequent-synchronization training.

Two workers start from the same checkpoint, train H local AdamW steps on their
own data shard, and send a shift (delta from the round's checkpoint). The root
combines shifts with either plain averaging or Nesterov outer momentum, and
commits the next checkpoint. Compute and network time are modelled, not
measured, so runs are reproducible on any machine:

  round_time = H * step_time + comm_time
  comm_time  = 2 * payload_bytes / bandwidth + 2 * latency   (shift up, checkpoint down)

Both workers always finish every round (no deadlines, no quorum).
"""
import argparse, copy, json
import torch
import torch.nn as nn

p = argparse.ArgumentParser()
p.add_argument("--intervals", type=int, nargs="+", default=[50, 200, 500])
p.add_argument("--budget", type=int, default=4000, help="local steps per worker")
p.add_argument("--step-time", type=float, default=0.25, help="modelled seconds per local step")
p.add_argument("--payload-mb", type=float, default=2000.0, help="modelled shift size in MB")
p.add_argument("--links", default="lan:10000:0.0005,wan:100:0.05",
               help="name:Mbit/s:latency_s, comma separated")
p.add_argument("--seeds", type=int, default=3)
p.add_argument("--inner-lr", type=float, default=2e-3)
p.add_argument("--outer-lr", type=float, default=0.7)
p.add_argument("--outer-momentum", type=float, default=0.9)
p.add_argument("--target-slack", type=float, default=1.10)
p.add_argument("--out", default="round_length_results.json")
args = p.parse_args()

D_IN, D_OUT, N_VAL = 32, 8, 4096
torch.set_num_threads(2)


def make_model():
    return nn.Sequential(nn.Linear(D_IN, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, D_OUT))


def make_task(seed):
    g = torch.Generator().manual_seed(1000 + seed)
    teacher = make_model()
    for q in teacher.parameters():
        q.data = torch.randn(q.shape, generator=g) * (1.0 / (q.shape[-1] ** 0.5) if q.dim() == 2 else 0.1)
    # two shards with different input distributions (mild heterogeneity)
    shifts = [torch.randn(D_IN, generator=g) * 0.5 for _ in range(2)]

    def sample(shard, n, gen):
        x = torch.randn(n, D_IN, generator=gen) + (shifts[shard] if shard is not None else 0)
        with torch.no_grad():
            y = teacher(x) + 0.05 * torch.randn(n, D_OUT, generator=gen)
        return x, y

    vg = torch.Generator().manual_seed(7 + seed)
    xs, ys = [], []
    for s in (0, 1):
        x, y = sample(s, N_VAL // 2, vg)
        xs.append(x); ys.append(y)
    return sample, (torch.cat(xs), torch.cat(ys))


def val_loss(model, val):
    with torch.no_grad():
        return nn.functional.mse_loss(model(val[0]), val[1]).item()


def flat(model):
    return torch.cat([q.data.view(-1) for q in model.parameters()])


def load(model, vec):
    i = 0
    for q in model.parameters():
        n = q.numel(); q.data.copy_(vec[i:i + n].view_as(q)); i += n


def run(H, rule, seed, link, sync_every_step=False):
    torch.manual_seed(seed)
    sample, val = make_task(seed)
    global_model = make_model()
    workers = [copy.deepcopy(global_model) for _ in range(2)]
    opts = [torch.optim.AdamW(w.parameters(), lr=args.inner_lr, weight_decay=0.01) for w in workers]
    gens = [torch.Generator().manual_seed(seed * 10 + k) for k in range(2)]
    theta = flat(global_model).clone()
    velocity = torch.zeros_like(theta)
    _, mbps, lat = link
    comm = 2 * args.payload_mb * 8 / mbps + 2 * lat if H > 0 else 0.0
    t, bytes_moved, wait, steps_done = 0.0, 0.0, 0.0, 0
    curve = [(0.0, val_loss(global_model, val))]
    while steps_done < args.budget:
        h = min(H, args.budget - steps_done)
        deltas = []
        for k in range(2):
            load(workers[k], theta)  # inner optimizer state persists across rounds
            for _ in range(h):
                x, y = sample(k, 64, gens[k])
                opts[k].zero_grad()
                nn.functional.mse_loss(workers[k](x), y).backward()
                opts[k].step()
            deltas.append(flat(workers[k]) - theta)
        mean_delta = (deltas[0] + deltas[1]) / 2
        if rule == "avg":
            theta = theta + mean_delta
        else:  # nesterov outer step on pseudo-gradient g = -mean_delta
            g = -mean_delta
            velocity = args.outer_momentum * velocity + g
            theta = theta - args.outer_lr * (g + args.outer_momentum * velocity)
        load(global_model, theta)
        steps_done += h
        t += h * args.step_time + comm
        wait += comm
        bytes_moved += 2 * 2 * args.payload_mb * 1e6  # two workers, up and down
        curve.append((t, val_loss(global_model, val)))
    return dict(curve=curve, bytes=bytes_moved, wait=wait, total_time=t, final=curve[-1][1])


def time_to(curve, target):
    for t, l in curve:
        if l <= target:
            return t
    return None


links = [(n, float(b), float(l)) for n, b, l in (s.split(":") for s in args.links.split(","))]
results = {}
for link in links:
    for seed in range(args.seeds):
        # reference: synchronize every step (H=1) with averaging, fixes the target per seed
        ref = run(1, "avg", seed, (link[0], 1e12, 0.0))
        target = ref["final"] * args.target_slack
        for rule in ("avg", "nesterov"):
            for H in args.intervals:
                r = run(H, rule, seed, link)
                r["time_to_target"] = time_to(r["curve"], target)
                r["target"] = target
                results.setdefault((link[0], rule, H), []).append(r)

print(f"payload {args.payload_mb:.0f} MB, step {args.step_time}s, budget {args.budget} steps/worker, "
      f"target = {args.target_slack}x synchronous final loss, {args.seeds} seeds\n")
print(f"{'link':<5}{'outer':<10}{'H':>5}{'reached':>9}{'t_target(s)':>13}{'comm_wait(s)':>14}{'GB moved':>10}{'final loss':>12}")
summary = []
for (ln, rule, H), rs in results.items():
    hit = [r["time_to_target"] for r in rs if r["time_to_target"] is not None]
    mean_t = sum(hit) / len(hit) if hit else float("nan")
    # communication wait and bytes up to the target round (approximate with total for unreached)
    wait = sum(r["wait"] for r in rs) / len(rs)
    gb = sum(r["bytes"] for r in rs) / len(rs) / 1e9
    fin = sum(r["final"] for r in rs) / len(rs)
    print(f"{ln:<5}{rule:<10}{H:>5}{len(hit):>6}/{len(rs)}{mean_t:>13.0f}{wait:>14.0f}{gb:>10.1f}{fin:>12.4f}")
    summary.append(dict(link=ln, rule=rule, H=H, reached=len(hit), seeds=len(rs), time_to_target=mean_t,
                        comm_wait_total=wait, gb_total=gb, final_loss=fin))
json.dump(summary, open(args.out, "w"), indent=1)
