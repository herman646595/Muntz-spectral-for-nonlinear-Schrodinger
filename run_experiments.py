import json, time, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from muntz_tfs import *

def sci(x):
    if x == 0: return "0"
    m, e = f"{x:.2e}".split("e"); return f"${m}\\times10^{{{int(e)}}}$"
def write(name, lines):
    lines = list(lines); lines[-1] = lines[-1].rstrip().rstrip("\\").rstrip()
    open(f"tables/{name}.tex", "w").write("\n".join(lines) + "\n")
def rates(es, hs):
    return ["--"] + [f"{np.log(es[i-1]/es[i])/np.log(hs[i]/hs[i-1]):.2f}" if es[i] > 0 and es[i-1] > 0 else "--" for i in range(1, len(es))]
R = {}

# ---------- T_main
rows = []
cases = {1: (0.3, 0.5, 0.7, 0.9), 2: (0.3, 0.5, 0.8, 0.9), 3: (0.5, 0.7, 0.9), 4: (0.3, 0.5, 0.7, 0.9), 5: (0.3, 0.5, 0.7, 0.9)}
for k, bs in cases.items():
    for b in bs:
        r = run_example(k, b)
        R[f"main_{k}_{b}"] = {kk: float(v) for kk, v in r.items() if isinstance(v, (int, float, np.floating))}
        ph = sci(r["phi_T"]) if "phi_T" in r else "--"
        rows.append(f"{k} & {b} & {r['N']} & {r['n']} & {sci(r['Linf'])} & {sci(r['Linf_time'])} & {ph} & {r['its']} & {r['cpu']:.2f} \\\\")
write("main", rows); print("main done", flush=True)

# ---------- K sweeps and L1 comparison (Ex2, Ex3)
sweeps = {2: {0.5: [4, 6, 8, 10, 12, 16, 20, 24], 0.8: [4, 6, 8, 10, 12, 16, 20, 24]},
          3: {0.5: [8, 12, 16, 20, 24, 28, 32, 36], 0.8: [8, 12, 16, 20, 24, 28, 32, 36]}}
rows = []; sw = {}
for k, d in sweeps.items():
    for b, Ks in d.items():
        for K in Ks:
            r = run_example(k, b, K=K)
            sw[(k, b, K)] = r
            rows.append(f"{k} & {b} & {K} & {sci(r['Linf'])} & {sci(r['Linf_time'])} & {r['its']} & {r['cpu']:.2f} \\\\")
write("ksweep", rows); print("ksweep done", flush=True)

# L1 comparison
rows = []; l1 = {}
for k, N, b, Ms in [(2, 20, 0.5, [20, 80, 320]), (2, 20, 0.8, [20, 80, 320]), (3, 16, 0.5, [16, 64, 128]), (3, 16, 0.8, [16, 64, 128])]:
    mk = DEFAULTS[k][0]
    for kind, rr in (("uniform", 1.0), ("graded", (2 - b) / b)):
        es = []; ts = []
        for M in Ms:
            t0 = time.time(); r = solve_l1(mk(), N, M, b, r=rr); dt = time.time() - t0
            es.append((r["Linf"], r["Linf_time"])); ts.append(dt); l1[(k, b, kind, M)] = (r, dt)
        for M, (a, c), dt in zip(Ms, es, ts):
            rows.append(f"{k} & {b} & L1 {kind} & $M={M}$ & {sci(a)} & {sci(c)} & {dt:.2f} \\\\")
    best = min([(sw[(k, b, K)]["Linf_time"], K) for (kk, bb, K) in sw if kk == k and bb == b])
    r = sw[(k, b, best[1])]
    rows.append(f"{k} & {b} & M\\\"untz (this work) & $K={best[1]}$ & {sci(r['Linf'])} & {sci(r['Linf_time'])} & {r['cpu']:.2f} \\\\")
    rows.append("\\midrule")
write("cmp", rows[:-1]); print("cmp done", flush=True)

# L1 comparison, smooth examples
rows = []
for k, N, b, Ms in [(1, 24, 0.5, [40, 160, 640]), (4, 16, 0.5, [16, 64, 256]), (5, 16, 0.5, [16, 64, 256])]:
    mk = DEFAULTS[k][0]
    for M in Ms:
        t0 = time.time(); r = solve_l1(mk(), N, M, b); dt = time.time() - t0
        rows.append(f"{k} & {b} & L1 uniform & $M={M}$ & {sci(r['Linf'])} & {sci(r['Linf_time'])} & {dt:.2f} \\\\")
    r = run_example(k, b)
    rows.append(f"{k} & {b} & M\\\"untz (this work) & $n={r['n']}$ & {sci(r['Linf'])} & {sci(r['Linf_time'])} & {r['cpu']:.2f} \\\\")
    rows.append("\\midrule")
write("cmp_smooth", rows[:-1]); print("cmp_smooth done", flush=True)

# L1 temporal orders (uniform) for Ex1/4/5 and Ex2 graded
rows = []
for k, N, b, Ms in [(1, 24, 0.5, [20, 40, 80, 160, 320]), (4, 16, 0.5, [8, 16, 32, 64, 128]), (5, 16, 0.5, [8, 16, 32, 64, 128])]:
    mk = DEFAULTS[k][0]; es = [solve_l1(mk(), N, M, b)["Linf"] for M in Ms]
    rt = rates(es, [1 / M for M in Ms][::-1][::-1])
    rt = ["--"] + [f"{np.log(es[i-1]/es[i])/np.log(Ms[i]/Ms[i-1]):.2f}" for i in range(1, len(es))]
    for M, e, q in zip(Ms, es, rt): rows.append(f"{k} & {b} & {M} & {sci(e)} & {q} \\\\")
    rows.append("\\midrule")
write("l1orders", rows[:-1]); print("l1orders done", flush=True)

# ---------- spatial sweeps
rows = []; spat = {}
for k, b, K, Ns in [(1, 0.5, 8, [4, 6, 8, 10, 12, 14, 16, 20, 24]), (3, 0.5, 36, [4, 6, 8, 10, 12, 14, 16])]:
    for N in Ns:
        r = run_example(k, b, K=K, N=N, nfine=0)
        spat[(k, N)] = r["Linf"]
        rows.append(f"{k} & {b} & {N} & {sci(r['Linf'])} & {r['cpu']:.2f} \\\\")
write("space", rows); print("space done", flush=True)

# ---------- stability constants
rows = []
for k, b, N, Ks in [(2, 0.5, 20, [8, 16, 24, 32, 40]), (3, 0.5, 12, [8, 16, 24, 32, 40])]:
    for K in Ks:
        r = run_example(k, b, K=K, N=N, nfine=0)
        rows.append(f"{k} & {b} & {K} & {r['stab']:.2e} & {r['cond']:.2e} \\\\")
write("stab", rows); print("stab done", flush=True)

# ---------- pure Muentz basis for Ex1 (t^2 not in lattice)
rows = []; pe = []
for K in [8, 16, 24, 32, 40]:
    r = run_example(1, 0.7, K=K, kind="muntz")
    pe.append((K, r["Linf"], r["Linf_time"]))
    rows.append(f"{K} & {sci(r['Linf'])} & {sci(r['Linf_time'])} \\\\")
write("pure", rows); print("pure done", flush=True)

# ---------- mass decay
def mass_prob(gamma_):
    ic = lambda x: 3 * np.sin(PI * x) + 1.5 * np.sin(3 * PI * x)
    return dict(dim=1, kappa=1.0, gamma=gamma_, exact=lambda x, t, b: ic(x) + 0j, f=lambda x, t, b: 0.0 * x + 0j)
rows = []
for gmm in (+1.0, -1.0):
    for kind, rr in (("uniform", 1.0), ("graded", 3.0)):
        r = solve_l1(mass_prob(gmm), 24, 200, 0.5, r=rr, record_mass=True)
        m = r["mass"]
        rows.append(f"L1 {kind} & ${'+' if gmm>0 else '-'}1$ & {m[0]:.6f} & {m[-1]:.6f} & {np.max(np.diff(m)):.2e} \\\\")
    lam = exponent_set(0.5, "muntz", 24)
    r = solve_muntz(mass_prob(gmm), 24, lam, 0.5, return_fields=True)
    U = r["U"]; w = Space(24).w
    m = np.sqrt(np.array([np.sum(w * np.abs(u) ** 2) for u in U]))
    rows.append(f"M\\\"untz collocation ($K=24$) & ${'+' if gmm>0 else '-'}1$ & {m[0]:.6f} & {m[-1]:.6f} & {np.max(np.diff(m)):.2e} \\\\")
write("mass", rows); print("mass done", flush=True)

# ---------- as-stated discrepancy
rows = []
for k, N, M in [(2, 16, 400), (3, 12, 128)]:
    mk = {2: ex2, 3: ex3}[k]; b = 0.5
    r = solve_l1(mk(False), N, M, b, r=(2 - b) / b)
    r1 = solve_l1(mk(False), N, M, 0.999, r=1.0)
    rows.append(f"{k} & {sci(r['Linf_time'])} & {sci(r1['Linf'])} \\\\")
write("stated", rows); print("stated done", flush=True)

# ---------- figures
plt.rcParams.update({"font.size": 9})
for k, b, Ks_, Ml, N in [(2, 0.5, [4, 6, 8, 10, 12, 16, 20, 24], [10, 20, 40, 80, 160, 320], 20), (3, 0.5, [8, 12, 16, 20, 24, 28, 32, 36], [8, 16, 32, 64, 128], 16)]:
    mk = DEFAULTS[k][0]
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    for kind, rr, mkr in (("L1 uniform", 1.0, "s-"), ("L1 graded", (2 - b) / b, "^-")):
        xs, ys = [], []
        for M in Ml:
            t0 = time.time(); r = solve_l1(mk(), N, M, b, r=rr); xs.append(time.time() - t0); ys.append(r["Linf_time"])
        ax.loglog(xs, ys, mkr, label=kind)
    ax.loglog([sw[(k, b, K)]["cpu"] for K in Ks_], [sw[(k, b, K)]["Linf_time"] for K in Ks_], "o-", label="Müntz collocation")
    ax.set_xlabel("CPU time (s)"); ax.set_ylabel(r"$\max_t\|u-u_h\|_\infty$"); ax.set_title(f"Example {k}, $\\beta={b}$"); ax.grid(alpha=.3, which="both"); ax.legend()
    fig.tight_layout(); fig.savefig(f"figs/work_precision_ex{k}.pdf"); plt.close(fig)

fig, ax = plt.subplots(figsize=(4.6, 3.3))
for k, lbl in ((1, "Example 1 (time exact in basis)"), (3, "Example 3 ($K=36$)")):
    Ns = sorted(N for (kk, N) in spat if kk == k); ax.semilogy(Ns, [spat[(k, N)] for N in Ns], "o-", label=lbl)
ax.set_xlabel("$N$"); ax.set_ylabel(r"$\|u-u_h\|_\infty$ at $T=1$"); ax.grid(alpha=.3, which="both"); ax.legend(); fig.tight_layout(); fig.savefig("figs/space_conv.pdf"); plt.close(fig)

# error in time, Example 2
b = 0.5; fig, ax = plt.subplots(figsize=(4.6, 3.3))
for kind, rr, st in (("L1 uniform, $M=40$", 1.0, "s-"), ("L1 graded, $M=40$", 3.0, "^-")):
    r = solve_l1(ex2(), 20, 40, b, r=rr); ax.semilogy(r["t"][1:], r["err_series"][1:], st, ms=3, label=kind)
lam = exponent_set(b, "muntz", 12); Db, t, E, tf = time_operator(lam, b, 401)
res = solve_muntz(ex2(), 20, lam, b, nfine=401, return_fields=True)
Pn = E @ res["U"].reshape(len(lam), -1); Ef = np.array([ex2()["exact"](*res["x"], ti, b) for ti in tf]).reshape(401, -1)
ax.semilogy(tf[1:], np.abs(Pn - Ef).max(axis=1)[1:], "-", label="Müntz, $K=12$")
ax.set_xlabel("$t$"); ax.set_ylabel(r"$\|u(t)-u_h(t)\|_\infty$"); ax.set_title("Example 2, $\\beta=0.5$"); ax.grid(alpha=.3, which="both"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig("figs/err_in_time_ex2.pdf"); plt.close(fig)
print("all done")
