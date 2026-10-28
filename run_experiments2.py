"""Per-example tables and figures, extra analyses (Taylor/Chebyshev bounds, Jacobian stability, first-step L1 error)."""
import time, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import gamma as Gam
from muntz_tfs import *

def sci(x):
    if x == 0: return "0"
    m, e = f"{x:.2e}".split("e"); return f"${m}\\times10^{{{int(e)}}}$"
def write(name, lines):
    lines = list(lines); lines[-1] = lines[-1].rstrip().rstrip("\\").rstrip()
    open(f"tables/{name}.tex", "w").write("\n".join(lines) + "\n")
plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": .3})
BETAS = {1: (0.3, 0.5, 0.7, 0.9), 2: (0.3, 0.5, 0.8, 0.9), 3: (0.5, 0.7, 0.9), 4: (0.3, 0.5, 0.7, 0.9), 5: (0.3, 0.5, 0.7, 0.9)}
MK = {1: ex1, 2: ex2, 3: ex3, 4: ex4, 5: ex5}
L1M = {1: [20, 40, 80, 160, 320], 2: [20, 40, 80, 160, 320], 3: [8, 16, 32, 64, 128], 4: [8, 16, 32, 64, 128], 5: [8, 16, 32, 64, 128]}
L1N = {1: 24, 2: 20, 3: 16, 4: 16, 5: 16}
MU = {}                                   # Muentz runs
for k in BETAS:
    rows = []
    for b in BETAS[k]:
        r = run_example(k, b); MU[(k, b)] = r
        ph = sci(r["phi_T"]) if "phi_T" in r else "--"
        rows.append(f"{b} & {r['N']} & {r['n']} & {sci(r['Linf'])} & {sci(r['L2'])} & {sci(r['Linf_time'])} & {ph} & {r['its']} & {r['cpu']:.2f} \\\\")
    write(f"e{k}_muntz", rows)
print("muntz tables", flush=True)

# ---------------- L1 baselines (uniform for all, graded for 2,3)
L1 = {}
for k in BETAS:
    betas = BETAS[k] if k in (1, 4, 5) else (0.5, 0.8)
    rows = []
    for b in betas:
        for kind, rr in (("uniform", 1.0), ("graded", (2 - b) / b)):
            if kind == "graded" and k not in (2, 3): continue
            prev = None
            for M in L1M[k]:
                t0 = time.time(); r = solve_l1(MK[k](), L1N[k], M, b, r=rr); dt = time.time() - t0
                L1[(k, b, kind, M)] = (r, dt)
                rt = "--" if prev is None else f"{np.log2(prev / r['Linf']):.2f}"
                rt2 = "--" if prev is None else f"{np.log2(prev2 / r['Linf_time']):.2f}"
                ph = sci(r["phi_T"]) if "phi_T" in r else "--"
                rows.append(f"{b} & {kind} & {M} & {sci(r['Linf'])} & {rt} & {sci(r['Linf_time'])} & {rt2} & {ph} & {dt:.2f} \\\\")
                prev, prev2 = r["Linf"], r["Linf_time"]
    write(f"e{k}_l1", rows)
print("l1 tables", flush=True)

# ---------------- K sweeps Muentz vs polynomial-in-t (Ex2, Ex3)
KS = {2: [4, 8, 12, 16, 20, 24, 32, 40], 3: [8, 12, 16, 20, 24, 32, 40]}
KSW = {}
for k in (2, 3):
    rows = []
    for b in (0.5, 0.8):
        for K in KS[k]:
            rm = run_example(k, b, K=K); rp = run_example(k, b, K=K, kind="poly")
            KSW[(k, b, K)] = (rm, rp)
            rows.append(f"{b} & {K} & {sci(rm['Linf'])} & {sci(rm['Linf_time'])} & {sci(rp['Linf'])} & {sci(rp['Linf_time'])} \\\\")
    write(f"e{k}_ksweep", rows)
print("ksweep", flush=True)

# ---------------- spatial sweeps
for k, b, K, Ns in [(1, 0.5, 8, [4, 6, 8, 10, 12, 14, 16, 20, 24]), (3, 0.5, 36, [4, 6, 8, 10, 12, 14, 16]),
                    (4, 0.5, 8, [2, 3, 4, 5, 6, 8, 10, 12, 16]), (5, 0.5, 8, [4, 6, 8, 10, 12, 16])]:
    rows = []; SP = {}
    for N in Ns:
        r = run_example(k, b, K=K, N=N, nfine=0); SP[N] = r
        ph = sci(r["phi_T"]) if "phi_T" in r else "--"
        rows.append(f"{N} & {sci(r['Linf'])} & {ph} & {r['cpu']:.2f} \\\\")
    write(f"e{k}_space", rows); globals()[f"SP{k}"] = SP
print("space", flush=True)

# ---------------- Taylor tail vs Chebyshev interpolation vs observed (Ex2)
from numpy.polynomial import chebyshev as Ch
rows = []
for b in (0.5, 0.8):
    c = -1.5j
    sfine = np.linspace(0, 1, 2001); exact = mittag_leffler(c * sfine, b)
    for K in (4, 8, 12, 16, 20):
        tail = sum(abs(c) ** k / Gam(k * b + 1) for k in range(K + 1, 400))
        nodes = (1 - np.cos(np.pi * np.arange(K + 1) / K)) / 2
        vals = mittag_leffler(c * nodes, b)
        cr = Ch.chebfit(2 * nodes - 1, vals.real, K); ci = Ch.chebfit(2 * nodes - 1, vals.imag, K)
        interp = np.abs(Ch.chebval(2 * sfine - 1, cr) + 1j * Ch.chebval(2 * sfine - 1, ci) - exact).max()
        obs = KSW[(2, b, K)][0]["Linf_time"] if (2, b, K) in KSW else run_example(2, b, K=K)["Linf_time"]
        rows.append(f"{b} & {K} & {sci(tail)} & {sci(interp)} & {sci(obs)} \\\\")
write("etaylor", rows)

# ---------------- first-step L1 local error for u = t^beta
rows = []
for b in (0.2, 0.3, 0.5, 0.7, 0.8, 0.9):
    exact = Gam(b + 1); l1 = 1 / Gam(2 - b)
    rows.append(f"{b} & {exact:.4f} & {l1:.4f} & {abs(exact-l1)/exact*100:.1f}\\% \\\\")
write("efirst", rows)

# ---------------- full Jacobian stability  sigma_min(I - DG)
def jac_stab(k, b, N, K):
    r = run_example(k, b, K=K, N=N, nfine=0, return_map=True)
    G, u = r["Gmap"], r["uI"]; shp = u.shape
    pack = lambda z: np.concatenate([z.real.ravel(), z.imag.ravel()])
    def unpack(v): h = v.size // 2; return (v[:h] + 1j * v[h:]).reshape(shp)
    x0 = pack(u); n = x0.size; J = np.zeros((n, n)); h = 1e-6
    F = lambda x: x - pack(G(unpack(x)))
    for i in range(n):
        e = np.zeros(n); e[i] = h; J[:, i] = (F(x0 + e) - F(x0 - e)) / (2 * h)
    sv = np.linalg.svd(J, compute_uv=False)
    return 1 / sv[-1], sv[0] / sv[-1], n
rows = []
for k, b, N, Ks in [(2, 0.5, 12, [8, 12, 16, 20]), (3, 0.5, 6, [8, 12, 16, 20])]:
    for K in Ks:
        inv, cond, n = jac_stab(k, b, N, K)
        rows.append(f"{k} & {N} & {K} & {n} & {inv:.3f} & {cond:.2f} \\\\")
write("ejac", rows)
print("jac", flush=True)

# ================== FIGURES (one panel figure per example) ==================
def fieldcontour(ax, XX, F, title, log=False):
    Z = np.log10(F + 1e-17) if log else F
    cs = ax.contourf(XX[0], XX[1], Z, 20); plt.colorbar(cs, ax=ax, fraction=.046); ax.set_title(title); ax.set_xlabel("$x$"); ax.set_ylabel("$y$")

# Ex1
fig, axs = plt.subplots(1, 3, figsize=(11, 3.1))
r = solve_muntz(ex1(), 24, exponent_set(0.5, "mixed", 8), 0.5, return_fields=True)
x = r["x"][0]; Ut = r["U"][-1]; ue = ex1()["exact"](x, 1.0, 0.5)
axs[0].plot(x, ue.real, "k-", label="exact Re"); axs[0].plot(x, Ut.real, "ro", ms=3, label="M\u00fcntz Re")
axs[0].plot(x, ue.imag, "b-", label="exact Im"); axs[0].plot(x, Ut.imag, "gs", ms=3, label="M\u00fcntz Im"); axs[0].set_xlabel("$x$"); axs[0].set_title("(a) $u(x,1)$, $\\beta=0.5$"); axs[0].legend(fontsize=6)
for b in BETAS[1]:
    Ms = L1M[1]; axs[1].loglog(Ms, [L1[(1, b, "uniform", M)][0]["Linf"] for M in Ms], "o-", ms=3, label=f"L1, $\\beta={b}$")
    axs[1].axhline(MU[(1, b)]["Linf"], ls=":", color="gray")
axs[1].set_xlabel("$M$"); axs[1].set_ylabel("error at $T$"); axs[1].set_title("(b) L1 (dotted: M\u00fcntz, $n\\leq 11$)"); axs[1].legend(fontsize=6)
Ns = sorted(SP1); axs[2].semilogy(Ns, [SP1[N]["Linf"] for N in Ns], "o-"); axs[2].set_xlabel("$N$"); axs[2].set_title("(c) spatial convergence"); axs[2].set_ylabel("error at $T$")
fig.tight_layout(); fig.savefig("figs/ex1_panel.pdf"); plt.close(fig)

# Ex2
fig, axs = plt.subplots(1, 3, figsize=(11, 3.1))
tt = np.linspace(0, 1, 400)
for b in (0.3, 0.5, 0.8):
    axs[0].plot(tt, mittag_leffler(-1.5j * tt ** b, b).real, label=f"$\\beta={b}$")
axs[0].plot(tt, np.cos(1.5 * tt), "k--", label="$\\beta=1$"); axs[0].set_xlabel("$t$"); axs[0].set_title("(a) $\\mathrm{Re}\\,E_\\beta(-\\frac{3i}{2}t^\\beta)$"); axs[0].legend(fontsize=6)
for b, st in ((0.5, "-"), (0.8, "--")):
    Ks = KS[2]
    axs[1].semilogy(Ks, [KSW[(2, b, K)][0]["Linf_time"] for K in Ks], "o" + st, label=f"M\u00fcntz $\\beta={b}$")
    axs[1].semilogy(Ks, [KSW[(2, b, K)][1]["Linf_time"] for K in Ks], "s" + st, label=f"poly-in-$t$ $\\beta={b}$")
axs[1].set_xlabel("$K$"); axs[1].set_ylabel("$\\max_t$ error"); axs[1].set_title("(b) time basis comparison"); axs[1].legend(fontsize=6)
b = 0.5
for kind, rr, lab in (("uniform", 1.0, "L1 uniform"), ("graded", 3.0, "L1 graded")):
    q = solve_l1(ex2(), 20, 40, b, r=rr); axs[2].semilogy(q["t"][1:], q["err_series"][1:], "-", label=lab + " $M=40$")
lam = exponent_set(b, "muntz", 12); Db, tn, E, tf = time_operator(lam, b, 401)
res = solve_muntz(ex2(), 20, lam, b, nfine=401, return_fields=True)
Pn = E @ res["U"].reshape(len(lam), -1); Ef = np.array([ex2()["exact"](*res["x"], ti, b) for ti in tf]).reshape(401, -1)
axs[2].semilogy(tf[1:], np.abs(Pn - Ef).max(axis=1)[1:], label="M\u00fcntz $K=12$"); axs[2].set_xlabel("$t$"); axs[2].set_title("(c) error in time, $\\beta=0.5$"); axs[2].legend(fontsize=6)
fig.tight_layout(); fig.savefig("figs/ex2_panel.pdf"); plt.close(fig)

# Ex3
fig, axs = plt.subplots(1, 3, figsize=(11, 3.1))
res = solve_muntz(ex3(), 16, exponent_set(0.5, "muntz", 24), 0.5, return_fields=True, nfine=0)
XX = res["x"]; UT = res["U"][-1]; fieldcontour(axs[0], XX, np.abs(UT), "(a) $|u(x,y,1)|$, $\\beta=0.5$")
fieldcontour(axs[1], XX, np.abs(UT - ex3()["exact"](*XX, 1.0, 0.5)), "(b) $\\log_{10}$ error at $T$ (M\u00fcntz $K=24$)", log=True)
for b, st in ((0.5, "-"), (0.8, "--")):
    Ks = KS[3]
    axs[2].semilogy(Ks, [KSW[(3, b, K)][0]["Linf_time"] for K in Ks], "o" + st, label=f"M\u00fcntz $\\beta={b}$")
    axs[2].semilogy(Ks, [KSW[(3, b, K)][1]["Linf_time"] for K in Ks], "s" + st, label=f"poly-in-$t$ $\\beta={b}$")
axs[2].set_xlabel("$K$"); axs[2].set_title("(c) $\\max_t$ error vs $K$"); axs[2].legend(fontsize=6)
fig.tight_layout(); fig.savefig("figs/ex3_panel.pdf"); plt.close(fig)

# work-precision for Ex2 and Ex3 (comparison with L1)
for k, b, Ml, N in [(2, 0.5, L1M[2], 20), (3, 0.5, L1M[3], 16)]:
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    for kind, mk in (("uniform", "s-"), ("graded", "^-")):
        ax.loglog([L1[(k, b, kind, M)][1] for M in Ml], [L1[(k, b, kind, M)][0]["Linf_time"] for M in Ml], mk, label="L1 " + kind)
    ax.loglog([KSW[(k, b, K)][0]["cpu"] for K in KS[k]], [KSW[(k, b, K)][0]["Linf_time"] for K in KS[k]], "o-", label="M\u00fcntz")
    ax.loglog([KSW[(k, b, K)][1]["cpu"] for K in KS[k]], [KSW[(k, b, K)][1]["Linf_time"] for K in KS[k]], "d-", label="poly-in-$t$")
    ax.set_xlabel("CPU time (s)"); ax.set_ylabel("$\\max_t$ error"); ax.set_title(f"Example {k}, $\\beta={b}$"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"figs/wp_ex{k}.pdf"); plt.close(fig)

# Ex4
fig, axs = plt.subplots(1, 3, figsize=(11, 3.1))
res = solve_muntz(ex4(), 16, exponent_set(0.5, "mixed", 8), 0.5, return_fields=True)
fieldcontour(axs[0], res["x"], np.abs(res["U"][-1]), "(a) $|u(x,y,1)|$, $\\beta=0.5$")
for b in BETAS[4]:
    Ms = L1M[4]; axs[1].loglog(Ms, [L1[(4, b, "uniform", M)][0]["Linf"] for M in Ms], "o-", ms=3, label=f"L1, $\\beta={b}$")
    axs[1].axhline(MU[(4, b)]["Linf"], ls=":", color="gray")
axs[1].set_xlabel("$M$"); axs[1].set_title("(b) L1 (dotted: M\u00fcntz)"); axs[1].legend(fontsize=6)
Ns = sorted(SP4); axs[2].semilogy(Ns, [SP4[N]["Linf"] for N in Ns], "o-"); axs[2].set_xlabel("$N$"); axs[2].set_title("(c) spatial convergence")
fig.tight_layout(); fig.savefig("figs/ex4_panel.pdf"); plt.close(fig)

# Ex5
fig, axs = plt.subplots(1, 4, figsize=(13.5, 3.1))
res = solve_muntz(ex5(), 16, exponent_set(0.5, "mixed", 8), 0.5, return_fields=True)
fieldcontour(axs[0], res["x"], np.abs(res["U"][-1]), "(a) $|u(x,y,1)|$"); fieldcontour(axs[1], res["x"], res["PHI_T"], "(b) $\\Phi(x,y,1)$")
for b in (0.3, 0.5, 0.9):
    Ms = L1M[5]; axs[2].loglog(Ms, [L1[(5, b, "uniform", M)][0]["Linf"] for M in Ms], "o-", ms=3, label=f"$u$, $\\beta={b}$")
    axs[2].loglog(Ms, [L1[(5, b, "uniform", M)][0]["phi_T"] for M in Ms], "s--", ms=3, label=f"$\\Phi$, $\\beta={b}$")
axs[2].set_xlabel("$M$"); axs[2].set_title("(c) L1 errors at $T$"); axs[2].legend(fontsize=5)
Ns = sorted(SP5); axs[3].semilogy(Ns, [SP5[N]["Linf"] for N in Ns], "o-", label="$u$"); axs[3].semilogy(Ns, [SP5[N]["phi_T"] for N in Ns], "s-", label="$\\Phi$")
axs[3].set_xlabel("$N$"); axs[3].set_title("(d) spatial convergence (M\u00fcntz)"); axs[3].legend(fontsize=6)
fig.tight_layout(); fig.savefig("figs/ex5_panel.pdf"); plt.close(fig)
print("figures done")
