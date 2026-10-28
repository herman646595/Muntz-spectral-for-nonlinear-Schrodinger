#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
muntz_tfs.py  --  Muentz space-time spectral method for a general time-fractional nonlinear
Schroedinger system with local and Poisson-type (Hartree) nonlinearities:

    i D_t^beta u + kappa*Lap(u) + V(x) u + gamma |u|^2 u - sigma*Phi[u] u = f,   (Caputo, 0<beta<1)
    -Lap Phi = |u|^2 + g,  Phi = 0 on the boundary   (needed only if sigma != 0)

Each example is a dict with the parameters (kappa, V, gamma, sigma), the exact solution and the
sources; sigma = 0 switches the Poisson coupling off.

Time  : Muentz collocation. u(x,t) = sum_j c_j(x) t^{lambda_j},  lambda_j in a Muentz lattice
        (lambda_j = j*beta, optionally enlarged by integers). D^beta acts exactly on the basis:
        D^beta t^lam = Gamma(lam+1)/Gamma(lam-beta+1) t^(lam-beta).  The nodal operator matrix is
        assembled in high precision (mpmath) because monomial <-> nodal conversion is ill-conditioned.
Space : Legendre-Galerkin with GLL numerical integration (G-NI) = Legendre collocation at the
        interior GLL nodes, Dirichlet data imposed at boundary nodes; tensor product in 2D.
Solver: ONE space-time nonlinear system (no time stepping, no history), Anderson-accelerated
        Picard iteration; the linear part is inverted exactly via a spatial eigen-decomposition
        and small (n x n) time solves.
Baseline: L1 formula (uniform / graded mesh) with the same spatial discretisation (solve_l1).

Usage:  python muntz_tfs.py            # runs the five examples with their default (max-accuracy) settings
"""
import sys, time
import numpy as np
import mpmath as mp
from numpy.polynomial import legendre as Lg
from scipy.special import gamma as Gam, rgamma

PI = np.pi

# ----------------------------------------------------------------------------- space
def gll(N):
    c = np.zeros(N + 1); c[N] = 1.0
    xi = np.concatenate(([-1.0], np.sort(Lg.legroots(Lg.legder(c)).real), [1.0]))
    PN = Lg.legval(xi, c)
    w = 2.0 / (N * (N + 1) * PN ** 2)
    D = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(N + 1):
            if i != j:
                D[i, j] = PN[i] / (PN[j] * (xi[i] - xi[j]))
    D[0, 0] = -N * (N + 1) / 4.0
    D[N, N] = N * (N + 1) / 4.0
    return (xi + 1) / 2, w / 2, 2 * D                      # mapped to [0,1]


class Space:
    def __init__(self, N):
        self.N = N
        self.x, self.w, D = gll(N)
        self.D2 = D @ D
        lam, X = np.linalg.eig(self.D2[1:-1, 1:-1])
        self.lam, self.X = lam.real, X.real
        self.Xi = np.linalg.inv(self.X)


# ----------------------------------------------------------------------------- utilities
_mlc = {}
def mittag_leffler(z, beta, K=400):
    if (beta, K) not in _mlc:
        _mlc[(beta, K)] = rgamma(np.arange(K) * beta + 1.0)
    g = _mlc[(beta, K)]
    out = np.zeros_like(np.asarray(z, dtype=complex))
    for k in range(K - 1, -1, -1):
        out = out * z + g[k]
    return out

def a_ml(c, t, beta):                                       # E_beta(c t^beta)
    return mittag_leffler(c * t ** beta, beta)

def anderson(G, x0, m=8, tol=1e-12, maxit=400):
    shape = x0.shape
    pack = lambda z: np.concatenate([z.real.ravel(), z.imag.ravel()])
    def unpack(v):
        h = v.size // 2
        return (v[:h] + 1j * v[h:]).reshape(shape)
    x = pack(x0); dF, dG = [], []; fp = gp = None
    for k in range(maxit):
        g = pack(G(unpack(x))); f = g - x
        if np.max(np.abs(f)) < tol * max(1.0, np.max(np.abs(g))):
            return unpack(g), k + 1
        if fp is not None:
            dF.append(f - fp); dG.append(g - gp)
            if len(dF) > m: dF.pop(0); dG.pop(0)
        fp, gp = f, g
        if dF:
            gam = np.linalg.lstsq(np.array(dF).T, f, rcond=None)[0]
            x = g - np.array(dG).T @ gam
        else:
            x = g
    raise RuntimeError("Anderson/Picard iteration did not converge")


# ----------------------------------------------------------------------------- Muentz time operator
def exponent_set(beta, kind="muntz", K=12, integers=(1.0, 2.0, 3.0)):
    """kind='muntz': {j*beta}_{j<=K};  kind='mixed': {j*beta}_{j<=K} U integers (lattice j*beta+k)."""
    if kind == "poly":                                      # classical polynomial-in-t basis (lambda_j = j)
        return np.arange(K + 1, dtype=float)
    lam = [j * beta for j in range(K + 1)]
    if kind == "mixed":
        lam += list(integers)
    return np.unique(np.round(lam, 12))

_top = {}
def time_operator(lam, beta, nfine=0, dps=260, nodes="s"):
    """nodal matrices: Dbeta (n+1)x(n+1), E (nfine x (n+1)) interpolation to a fine t-grid, nodes t_i."""
    key = (tuple(lam), beta, nfine, nodes)
    if key in _top: return _top[key]
    mp.mp.dps = dps
    n = len(lam) - 1
    b = mp.mpf(beta)
    s = [(1 - mp.cos(mp.pi * i / n)) / 2 for i in range(n + 1)]
    t = [si ** (1 / b) if si != 0 else mp.mpf(0) for si in s] if nodes == "s" else list(s)
    L = [mp.mpf(float(l)) for l in lam]
    pw = lambda x, e: mp.mpf(1) if e == 0 else (x ** e if x != 0 else mp.mpf(0))
    V = mp.matrix(n + 1, n + 1); A = mp.matrix(n + 1, n + 1)
    g = [mp.mpf(0) if L[j] == 0 else mp.gamma(L[j] + 1) / mp.gamma(L[j] - b + 1) for j in range(n + 1)]
    for i in range(n + 1):
        for j in range(n + 1):
            V[i, j] = pw(t[i], L[j])
            A[i, j] = mp.mpf(0) if L[j] == 0 else g[j] * pw(t[i], L[j] - b)
    Vinv = mp.inverse(V)
    Db = np.array((A * Vinv).tolist(), dtype=float)
    E = None
    if nfine:
        sf = [mp.mpf(i) / (nfine - 1) for i in range(nfine)]
        tf = [x ** (1 / b) if x != 0 else mp.mpf(0) for x in sf] if nodes == "s" else list(sf)
        Vf = mp.matrix(nfine, n + 1)
        for i in range(nfine):
            for j in range(n + 1):
                Vf[i, j] = pw(tf[i], L[j])
        E = np.array((Vf * Vinv).tolist(), dtype=float)
        tf = np.array([float(x) for x in tf])
    else:
        tf = None
    out = (Db, np.array([float(x) for x in t]), E, tf)
    _top[key] = out
    return out


# ----------------------------------------------------------------------------- Muentz solver
def solve_muntz(prob, N, lam, beta, nfine=0, return_fields=False, nodes="s", return_map=False):
    d = prob["dim"]; S = Space(N); x = S.x
    Db, t, E, tf = time_operator(lam, beta, nfine, nodes=nodes)
    n = len(lam) - 1
    if d == 1:
        XX = (x,)
        lap = lambda U: U @ S.D2.T
        sl = (slice(1, None), slice(1, N)); sint = (slice(1, N),)
        tm = lambda A: np.einsum('mp,ip->im', S.Xi, A)
        fm = lambda A: np.einsum('pm,im->ip', S.X, A)
        eig = S.lam; W = S.w
    else:
        XX = np.meshgrid(x, x, indexing="ij")
        lap = lambda U: np.einsum('ap,ipq->iaq', S.D2, U) + np.einsum('bq,ipq->ipb', S.D2, U)
        sl = (slice(1, None), slice(1, N), slice(1, N)); sint = (slice(1, N), slice(1, N))
        tm = lambda A: np.einsum('ap,ipq,bq->iab', S.Xi, A, S.Xi).reshape(A.shape[0], -1)
        fm = lambda A: np.einsum('ap,ipq,bq->iab', S.X, A.reshape(A.shape[0], N - 1, N - 1), S.X)
        eig = (S.lam[:, None] + S.lam[None, :]).ravel(); W = np.outer(S.w, S.w)
    shape = XX[0].shape
    kappa, gam = prob["kappa"], prob["gamma"]
    Vp = prob["V"](*XX) if "V" in prob else 0.0
    Uex = np.array([prob["exact"](*XX, ti, beta) for ti in t])
    F = np.array([prob["f"](*XX, ti, beta) for ti in t])
    bmask = np.ones(shape, bool); bmask[sint] = False
    Ub = np.where(bmask[None], Uex, 0.0)
    U0 = Uex[0]
    rhs0 = (F - kappa * lap(Ub))[sl] - 1j * Db[1:, 0].reshape((n,) + (1,) * d) * U0[sint][None]
    A = 1j * Db[1:, 1:][None] + kappa * eig[:, None, None] * np.eye(n)[None]
    Ainv = np.linalg.inv(A)
    sig = prob.get("sigma", 0.0); poisson = sig != 0.0
    if poisson:
        Gs = np.array([prob["g"](*XX, ti, beta) for ti in t])[sl]
    def phi_of(uint):                                         # Poisson solve on interior nodes
        rho = np.abs(uint) ** 2 + Gs
        return fm(tm(rho) / (-eig))
    def Gmap(uI):
        u = Ub.copy(); u[sl] = uI
        nl = (gam * np.abs(u) ** 2 * u + Vp * u)[sl]
        if poisson: nl = nl - sig * phi_of(uI) * uI
        rh = tm(rhs0 - nl)
        return fm(np.einsum('mij,jm->im', Ainv, rh))
    u_init = np.repeat(U0[sint][None], n, axis=0)
    uI, its = anderson(Gmap, u_init)
    U = Ub.copy(); U[sl] = uI; U[0] = U0
    res = dict(Linf=np.abs(U[-1] - Uex[-1]).max(),
               L2=np.sqrt(np.sum(W * np.abs(U[-1] - Uex[-1]) ** 2)),
               Linf_nodes=np.abs(U - Uex).max(), its=its, n=n,
               stab=np.linalg.norm(Ainv, 2, axis=(1, 2)).max(),
               cond=np.linalg.cond(A).max())
    if nfine:
        P = E @ U.reshape(n + 1, -1)
        Ef = np.array([prob["exact"](*XX, ti, beta).ravel() for ti in tf])
        res["Linf_time"] = np.abs(P - Ef).max()
    if poisson:
        phi_int = phi_of(uI)
        PH = np.zeros((n + 1,) + shape); PH[sl] = phi_int; PH = PH[1:]
        PHex = np.array([prob["exact_phi"](*XX, ti, beta) for ti in t[1:]])
        res["phi_T"] = np.abs(PH[-1] - PHex[-1]).max(); res["PHI_T"] = PH[-1]
        res["phi_nodes"] = np.abs(PH - PHex).max()
    if return_fields: res["U"] = U; res["t"] = t; res["x"] = XX
    if return_map: res["Gmap"] = Gmap; res["uI"] = uI
    return res


# ----------------------------------------------------------------------------- L1 baseline
def l1_coeffs(t, n, beta):
    tk, tk1 = t[1:n + 1], t[0:n]
    return ((t[n] - tk1) ** (1 - beta) - (t[n] - tk) ** (1 - beta)) / (Gam(2 - beta) * (tk - tk1))

def solve_l1(prob, N, M, beta, T=1.0, r=1.0, record_mass=False):
    """Fully implicit L1 scheme on t_k = T (k/M)^r + the same Legendre G-NI space discretisation."""
    d = prob["dim"]; S = Space(N); x = S.x
    if d == 1:
        XX = (x,); W = S.w; lap = lambda U: S.D2 @ U; it = slice(1, N)
        tm = lambda A: S.Xi @ A; fm = lambda A: S.X @ A; eig = S.lam
    else:
        XX = np.meshgrid(x, x, indexing="ij"); W = np.outer(S.w, S.w)
        lap = lambda U: S.D2 @ U + U @ S.D2.T; it = (slice(1, N), slice(1, N))
        tm = lambda A: S.Xi @ A @ S.Xi.T; fm = lambda A: S.X @ A @ S.X.T
        eig = S.lam[:, None] + S.lam[None, :]
    t = T * (np.arange(M + 1) / M) ** r
    shape = XX[0].shape
    U = np.zeros((M + 1,) + shape, complex); dU = np.zeros_like(U)
    U[0] = prob["exact"](*XX, 0.0, beta)
    kappa, gam = prob["kappa"], prob["gamma"]
    Vp = prob["V"](*XX) if "V" in prob else 0.0
    bmask = np.ones(shape, bool); bmask[it] = False
    errmax = 0.0; errs = [0.0]; mass = [np.sqrt(np.sum(W * np.abs(U[0]) ** 2))]
    sig = prob.get("sigma", 0.0); poisson = sig != 0.0
    for n in range(1, M + 1):
        tn = t[n]; c = l1_coeffs(t, n, beta); cnn = c[-1]
        R = cnn * U[n - 1]
        if n > 1: R = R - np.tensordot(c[:-1], dU[1:n], axes=(0, 0))
        Uex = prob["exact"](*XX, tn, beta)
        ub = np.where(bmask, Uex, 0.0)
        F = prob["f"](*XX, tn, beta) + 1j * R
        rhs0 = (F - kappa * lap(ub))[it]
        den = 1j * cnn + kappa * eig
        if poisson: gs = prob["g"](*XX, tn, beta)
        def G(uI):
            u = ub.copy(); u[it] = uI
            nl = gam * np.abs(u) ** 2 * u + Vp * u
            if poisson:
                phi = np.zeros(shape); phi[it] = fm(tm((np.abs(u) ** 2 + gs)[it]) / (-eig))
                nl = nl - sig * phi * u
            return fm(tm(rhs0 - nl[it]) / den)
        uI, _ = anderson(G, U[n - 1][it])
        U[n] = ub; U[n][it] = uI; dU[n] = U[n] - U[n - 1]
        errmax = max(errmax, np.abs(U[n] - Uex).max()); errs.append(np.abs(U[n] - Uex).max())
        mass.append(np.sqrt(np.sum(W * np.abs(U[n]) ** 2)))
    e = np.abs(U[M] - prob["exact"](*XX, T, beta))
    out = dict(Linf=e.max(), Linf_time=errmax, err_series=np.array(errs), t=t)
    if poisson:
        phi = np.zeros(shape); phi[it] = fm(tm((np.abs(U[M]) ** 2 + prob["g"](*XX, T, beta))[it]) / (-eig))
        out["phi_T"] = np.abs(phi - prob["exact_phi"](*XX, T, beta)).max()
    out["U"] = U[M]; out["x"] = XX
    if record_mass: out["mass"] = np.array(mass)
    return out


# ----------------------------------------------------------------------------- the five examples
def ex1():
    def u(x, t, b): return t ** 2 * (np.sin(2 * PI * x) + 1j * np.cos(2 * PI * x))
    def f(x, t, b):
        A = 2 * t ** (2 - b) / Gam(3 - b); B = -4 * PI ** 2 * t ** 2 + t ** 6
        return -A * np.cos(2 * PI * x) + B * np.sin(2 * PI * x) + 1j * (A * np.sin(2 * PI * x) + B * np.cos(2 * PI * x))
    return dict(dim=1, kappa=1.0, gamma=1.0, sigma=0.0, exact=u, f=f)

def ex2(manufactured=True):
    c = -1.5j
    def u(x, t, b): return np.sin(x) * a_ml(c, t, b)
    def f(x, t, b):
        if not manufactured: return 0.0 * x + 0j
        a = a_ml(c, t, b); return np.sin(x) ** 3 * a * (1 - abs(a) ** 2)
    return dict(dim=1, kappa=0.5, gamma=-1.0, V=lambda x: -np.cos(x) ** 2, sigma=0.0, exact=u, f=f)

def ex3(manufactured=True):
    c = -2.5j
    def u(x, y, t, b): return np.sin(x) * np.sin(y) * a_ml(c, t, b)
    def f(x, y, t, b):
        if not manufactured: return 0.0 * x + 0j
        a = a_ml(c, t, b); s2 = (np.sin(x) * np.sin(y)) ** 2
        return u(x, y, t, b) * (0.5 + s2 * (1 - abs(a) ** 2))
    return dict(dim=2, kappa=0.5, gamma=-1.0, V=lambda x, y: -(1 - (np.sin(x) * np.sin(y)) ** 2), sigma=0.0, exact=u, f=f)

def ex4():
    def u(x, y, t, b): return t ** 2 * np.sin(x + y) + 0j
    def f(x, y, t, b):
        return 1j * 2 * t ** (2 - b) / Gam(3 - b) * np.sin(x + y) + t ** 2 * np.sin(x + y) * (2 + abs(t ** 2 * np.sin(x + y)) ** 2)
    return dict(dim=2, kappa=-1.0, gamma=1.0, sigma=0.0, exact=u, f=f)

def ex5():
    s = lambda x, y: np.sin(PI * x) * np.sin(PI * y)
    def u_ex(x, y, t, b): return (1 + t ** 2) * s(x, y) + 0j
    def phi(x, y, t, b): return t ** 2 * s(x, y)
    def f(x, y, t, b):
        S = s(x, y)
        return (1j * 2 * t ** (2 - b) / Gam(3 - b) * S - 2 * PI ** 2 * (1 + t ** 2) * S
                + (1 + t ** 2) ** 3 * S ** 3 - t ** 2 * (1 + t ** 2) * S ** 2)
    def g(x, y, t, b):
        S = s(x, y); return 2 * PI ** 2 * t ** 2 * S - (1 + t ** 2) ** 2 * S ** 2
    return dict(dim=2, kappa=1.0, gamma=1.0, sigma=1.0, exact=u_ex, f=f, g=g, exact_phi=phi)

# default (maximum-accuracy) settings: (problem, N, exponent-set kind, K)
DEFAULTS = {
    1: (ex1, 24, "mixed", 8),
    2: (ex2, 20, "muntz", 24),
    3: (ex3, 16, "muntz", 36),
    4: (ex4, 16, "mixed", 8),
    5: (ex5, 16, "mixed", 8),
}

def run_example(k, beta, nfine=401, **over):
    mk, N, kind, K = DEFAULTS[k]
    N = over.get("N", N); kind = over.get("kind", kind); K = over.get("K", K)
    lam = exponent_set(beta, kind, K)
    nodes = "t" if kind == "poly" else "s"
    _top.pop((tuple(lam), beta, 0, nodes), None)           # force a timed (re)assembly of D_beta
    t0 = time.time(); time_operator(lam, beta, 0, nodes=nodes); setup = time.time() - t0
    if nfine: time_operator(lam, beta, nfine, nodes=nodes)  # fine-grid interpolation: post-processing, not timed
    t0 = time.time(); r = solve_muntz(mk(), N, lam, beta, nfine=nfine, nodes=nodes, return_map=over.get("return_map", False))
    solve = time.time() - t0
    r["cpu"] = setup + solve; r["cpu_setup"] = setup; r["cpu_solve"] = solve; r.update(N=N, K=K, kind=kind)
    return r

if __name__ == "__main__":
    betas = [0.5, 0.8] if len(sys.argv) < 2 else [float(a) for a in sys.argv[1:]]
    print("Example  beta   N   n(time dofs)   Linf(T)      max_t Linf   [Phi err T]   its   CPU(s)")
    for k in range(1, 6):
        for b in betas:
            r = run_example(k, b)
            ph = f"{r['phi_T']:.2e}" if "phi_T" in r else "     -    "
            print(f"  {k}      {b:.1f}  {r['N']:3d}   {r['n']:3d}      {r['Linf']:.3e}   {r['Linf_time']:.3e}   {ph}    {r['its']:3d}   {r['cpu']:.2f}")
