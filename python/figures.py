"""Figures for the phenomenology section (run after run_all.py)."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from emlib.states import Library
from emlib import hybrids as HY
from emlib.amplitudes import photon_energy

ROOT = os.path.join(os.path.dirname(__file__), '..')
RES = os.path.join(ROOT, 'results'); FIG = os.path.join(ROOT, 'figures')
C1, C2, C3, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#1baf7a', '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.linewidth': 0.6,
                     'legend.frameon': False, 'font.family': 'serif', 'mathtext.fontset': 'cm'})
summ = json.load(open(os.path.join(RES, 'summary.json')))

# ---------------------------------------------------------------- Fig. 1
pulls = pd.read_csv(os.path.join(RES, 'quarkonium_pulls.csv'))
p = pulls[(pulls['nonlocal'] == True) & (pulls['kind'] == 'width')].copy()
p = p[~p['id'].str.contains(r'\(1P\)$') | ~p['id'].str.startswith('Y3S')]   # drop hindered (shown in table)
p['ratio'] = p['data'] / p['pred_kappa1']
p['err'] = p['err_exp'] / p['pred_kappa1']
fig, ax = plt.subplots(figsize=(5.2, 3.4))
y = np.arange(len(p))[::-1]
for fl, col, key in (('c', C1, 'kappa_c'), ('b', C2, 'kappa_b')):
    k, dl, dh, _ = summ[key]
    sel = (p['flavor'] == fl).values
    ax.axvspan((k - dl) ** 2, (k + dh) ** 2, ymin=0, ymax=1, color=col, alpha=0.12, lw=0)
    ax.axvline(k ** 2, color=col, lw=1.0)
    ax.errorbar(p['ratio'][sel], y[sel], xerr=p['err'][sel], fmt='o', ms=4, color=col, elinewidth=1,
                capsize=0, label=(r'$c\bar c$: $\kappa_c^2$' if fl == 'c' else r'$b\bar b$: $\kappa_b^2$'))
ax.axvline(1.0, color=MUTED, lw=0.6, ls=':')
import re
labels = [re.sub(r'chi([cb])(\d)', r'$\\chi_{\1\2}$', s).replace('->', r'$\to\gamma$').replace('psi2S', r'$\psi(2S)$').replace('psi3770', r'$\psi(3770)$')
          .replace('Jpsi', r'$J/\psi$').replace('hc', r'$h_c$').replace('etac', r'$\eta_c$')
          .replace('Y2S', r'$\Upsilon(2S)$').replace('Y3S', r'$\Upsilon(3S)$')
          for s in p['id']]
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7.5)
ax.set_xlabel(r'$\Gamma_{\rm exp}/\Gamma_{\rm LO}(\kappa_Q=1)$')
ax.grid(axis='x', color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.legend(loc='lower right', fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_E1_ratios.pdf')); fig.savefig(os.path.join(FIG, 'fig_E1_ratios.png'), dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- Fig. 2
lib = Library('c'); r = lib.r; dV = HY.load_dV('c')
h = [x for x in HY.HYBRIDS['c'] if x[0] == '1(s/d)1'][0]
st = HY.hybrid_state(lib, h)
comps = {c.l: c.c for c in st.comps}
cPi = (np.sqrt(2) * comps[0] + comps[2]) / np.sqrt(3)
f = lib.quarkonium(1, 0)
E = photon_energy(4.040, 2.98409)
uPi = lib.u(cPi); uf = lib.u(f.comps[0].c)
base = uf * uPi
fig, ax = plt.subplots(figsize=(4.6, 3.0))
ax.axhline(0, color=MUTED, lw=0.6)
ax.plot(r, base, color=MUTED, lw=1.2, ls='--', label=r'$u_{\eta_c}u_\Pi$ ($g_{B\perp}$ only)')
ax.plot(r, base * (1 - E * r / np.pi), color=C1, lw=2, label=r'$\times(1-E_q r/\pi)$  [A]')
ax.plot(r, base * np.sqrt(dV * r / np.pi) * (1 - E / dV), color=C2, lw=2, label=r'$\times\sqrt{\Delta V r/\pi}\,(1-E_q/\Delta V)$  [B]')
ax.set_xlim(0, 12); ax.set_xlabel(r'$r$ (GeV$^{-1}$)'); ax.set_ylabel('radial integrand (GeV)')
ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.legend(fontsize=7.5, loc='upper right')
ax.set_title(r'$\psi(4040)\,[1(s/d)_1]\to\gamma\,\eta_c(1S)$, $E_q=918$ MeV', fontsize=9, color=INK)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_hybrid_integrand.pdf')); fig.savefig(os.path.join(FIG, 'fig_hybrid_integrand.png'), dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- Fig. 3
Qs = np.load(os.path.join(RES, 'quadratic_forms.npy'), allow_pickle=True).item()
ULs = summ['ULs_gamma_etac']
c2 = np.linspace(-1.5, 1.5, 601); c13 = np.linspace(-1.5, 1.5, 601)
C2g, C13g = np.meshgrid(c2, c13)
fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.2), sharey=True)
kc = summ['kappa_c'][0]
for ax, v, ttl in zip(axs, ('A-NL', 'B-NL'), (r'A: $E_1=\pi/r$', r'B: $E_1=\Delta V(r)$')):
    allowed = np.ones_like(C2g, dtype=bool)
    for cand, col in (('psi(4040)', C1), ('psi(4415)', C2)):
        Q = Qs[f'{cand}|{v}']
        G = (C2g ** 2 * Q[1, 1] + 2 * C2g * C13g * (Q[0, 1] + Q[1, 2])
             + C13g ** 2 * (Q[0, 0] + Q[2, 2] + 2 * Q[0, 2]))
        ok = G < ULs[cand]['Gamma_UL_keV']
        allowed &= ok
        ax.contour(C2g, C13g, G, levels=[ULs[cand]['Gamma_UL_keV']], colors=[col], linewidths=1.2)
        ax.plot([], [], color=col, lw=1.2, label=cand.replace('psi', r'$\psi$') + r'$\to\gamma\eta_c$')
    ax.contourf(C2g, C13g, allowed.astype(float), levels=[0.5, 1.5], colors=[C3], alpha=0.25)
    ax.plot(c2, c2, color=MUTED, lw=0.8, ls='--', label=r'EST: $c_1=c_2=c_3$')
    ax.plot([kc], [kc], 'o', color=INK, ms=4, label=r'$c=\kappa_c e e_c$')
    ax.set_title(ttl, fontsize=9, color=INK)
    ax.set_xlabel(r'$c_2/(e e_c)$')
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
axs[0].set_ylabel(r'$\frac{1}{2}(c_1+c_3)/(e e_c)$')
axs[1].legend(fontsize=7, loc='lower right')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_c2_c13_plane.pdf')); fig.savefig(os.path.join(FIG, 'fig_c2_c13_plane.png'), dpi=200)
plt.close(fig)
print('figures written to', FIG)
