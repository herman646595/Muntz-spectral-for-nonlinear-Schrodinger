import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from muntz_tfs import a_ml
beta, c, K = 0.5, -1.5j, 8
f = lambda t: a_ml(c, t, beta).imag
tt = np.linspace(0, 1, 2001)
def cheb(n): return 0.5*(1-np.cos(np.pi*np.arange(n+1)/n))
# interpolation in t (polynomial in t) and in s=t^beta (Muentz), both degree K, Chebyshev-Lobatto nodes
tn = cheb(K); pt = np.polynomial.chebyshev.Chebyshev.fit(tn, f(tn), K, domain=[0,1])
sn = cheb(K); tsn = sn**(1/beta)
ps = np.polynomial.chebyshev.Chebyshev.fit(sn, f(tsn), K, domain=[0,1])
ut = pt(tt); us = ps(tt**beta); ex = f(tt)
et, es = np.abs(ut-ex).max(), np.abs(us-ex).max()
print("max err poly in t: %.3e, poly in s: %.3e"%(et,es))
fig, ax = plt.subplots(1, 3, figsize=(11, 3.0))
ax[0].plot(tt, ex, 'k', lw=2, label='exact'); ax[0].plot(tt, ut, 'r--', label=r'degree 8 in $t$'); ax[0].plot(tt, us, 'b:', lw=2, label=r'degree 8 in $s=t^\beta$')
ax[0].set_title(r'(a) $\mathrm{Im}\,E_\beta(-1.5\,\mathrm{i}\,t^\beta)$, $\beta=0.5$'); ax[0].set_xlabel('t'); ax[0].legend(fontsize=7)
ss = np.linspace(0, 1, 2001)
ax[1].plot(ss, f(ss**(1/beta)), 'k', lw=2); ax[1].set_title(r'(b) same function against $s=t^\beta$'); ax[1].set_xlabel(r'$s=t^\beta$')
ax[2].semilogy(tt, np.abs(ut-ex)+1e-17, 'r', label='in $t$'); ax[2].semilogy(tt, np.abs(us-ex)+1e-17, 'b', label='in $s$')
ax[2].set_title('(c) interpolation error'); ax[2].set_xlabel('t'); ax[2].legend(fontsize=7)
plt.tight_layout(); plt.savefig('figs/intuition.pdf')
