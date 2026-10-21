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
lib = Library('c'); r = lib.r; dV = HY.load_dV('c'); dVS = HY.load_dVS('c')
sig = 0.21
h = [x for x in HY.HYBRIDS['c'] if x[0] == '1(s/d)1'][0]
st = HY.hybrid_state(lib, h)
comps = {c.l: c.c for c in st.comps}
cPi = (np.sqrt(2) * comps[0] + comps[2]) / np.sqrt(3)
cSig = (comps[0] - np.sqrt(2) * comps[2]) / np.sqrt(3)
f = lib.quarkonium(1, 0)
E = photon_energy(4.040, 2.98409)
uPi = lib.u(cPi); uSig = lib.u(cSig); uf = lib.u(f.comps[0].c)
base = uf * uPi
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0))
ax = axs[0]
ax.axhline(0, color=MUTED, lw=0.6)
ax.plot(r, base, color=MUTED, lw=1.2, ls='--', label=r'$u_{\eta_c}u_\Pi$ ($g_{B\perp}$ only)')
ax.plot(r, base * (1 - E * r / np.pi), color=C1, lw=2, label=r'$\times(1-E_q r/\pi)$  [A]')
ax.plot(r, base * np.sqrt(dV * r / np.pi) * (1 - E / dV), color=C2, lw=2, label=r'$\times\sqrt{\Delta V r/\pi}\,(1-E_q/\Delta V)$  [B]')
ax.set_xlim(0, 12); ax.set_xlabel(r'$r$ (GeV$^{-1}$)'); ax.set_ylabel('radial integrand (GeV)')
ax.set_title(r'$\Pi_u$: $\sqrt{2/3}\,\mathcal{A}_1$', fontsize=9, color=INK)
ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.legend(fontsize=7, loc='upper right')
ax = axs[1]
w = 2 * np.sqrt(np.pi / sig) / r          # A_0 weight per unit c: (2 sqrt(pi)/sqrt(sigma)) / r
bS = uf * uSig * w
ax.axhline(0, color=MUTED, lw=0.6)
ax.plot(r, bS, color=MUTED, lw=1.2, ls='--', label=r'$u_{\eta_c}u_\Sigma\,\frac{2\sqrt{\pi}}{\sqrt{\sigma}\,r}$ ($g_{B\parallel}$ only)')
ax.plot(r, bS * (1 - E * r / (3 * np.pi)), color=C1, lw=2, label=r'$\times(1-E_qr/3\pi)$  [A, EST]')
ax.plot(r, bS * np.sqrt(dVS * r / (3 * np.pi)) * (1 - E / dVS), color=C2, lw=2,
        label=r'$\times\sqrt{\Delta V_\Sigma r/(3\pi)}\,(1-E_q/\Delta V_\Sigma)$  [B]')
ax.plot(r, uf * uSig * (1 - E * r / (3 * np.pi)), color=C3, lw=1.5, ls='-.', label=r'norm: $g_{B\parallel}\to g_{B\perp}$ [A]')
ax.set_xlim(0, 12); ax.set_xlabel(r'$r$ (GeV$^{-1}$)')
ax.set_title(r'$\Sigma_u^-$: $\sqrt{1/3}\,\mathcal{A}_0$', fontsize=9, color=INK)
ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.legend(fontsize=6.5, loc='upper right')
fig.suptitle(r'$\psi(4040)\,[1(s/d)_1]\to\gamma\,\eta_c(1S)$, $E_q=918$ MeV', fontsize=9, color=INK)
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
        # c = (c13, c2, c13, c4 = c2): Gamma = c^T Q c
        a2 = Q[1, 1] + Q[3, 3] + 2 * Q[1, 3]                          # c2^2
        a13 = Q[0, 0] + Q[2, 2] + 2 * Q[0, 2]                         # c13^2
        a213 = Q[0, 1] + Q[1, 2] + Q[0, 3] + Q[2, 3]                  # c2 c13
        G = C2g ** 2 * a2 + 2 * C2g * C13g * a213 + C13g ** 2 * a13
        ok = G < ULs[cand]['Gamma_UL_keV']
        allowed &= ok
        ax.contour(C2g, C13g, G, levels=[ULs[cand]['Gamma_UL_keV']], colors=[col], linewidths=1.2)
        ax.plot([], [], color=col, lw=1.2, label=cand.replace('psi', r'$\psi$') + r'$\to\gamma\eta_c$')
    ax.contourf(C2g, C13g, allowed.astype(float), levels=[0.5, 1.5], colors=[C3], alpha=0.25)
    ax.plot(c2, c2, color=MUTED, lw=0.8, ls='--', label=r'EST: $c_1=c_2=c_3=c_4$')
    ax.plot([kc], [kc], 'o', color=INK, ms=4, label=r'$c=\kappa_c e e_c$')
    ax.set_title(ttl, fontsize=9, color=INK)
    ax.set_xlabel(r'$c_2/(e e_c)$')
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
axs[0].set_ylabel(r'$\frac{1}{2}(c_1+c_3)/(e e_c)$')
for ax in axs:
    ax.set_xlabel(r'$c_2/(e e_c)=c_4/(e e_c)$')
axs[1].legend(fontsize=7, loc='lower right')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_c2_c13_plane.pdf')); fig.savefig(os.path.join(FIG, 'fig_c2_c13_plane.png'), dpi=200)
plt.close(fig)
# ---------------------------------------------------------------- Fig. 4: Sigma coupling scan
sc = pd.read_csv(os.path.join(RES, 'sigma_scan.csv'))
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.2))
cols = {'J/psi (mixed)': C3, 'psi(2S) (mixed)': C1, 'psi(3770) (mixed)': '#8a5cc2',
        'psi(4040) (mixed)': C2, 'psi(4415) (mixed)': INK,
        'psi(4040) (unmixed)': C2, 'psi(4415) (unmixed)': INK}
lab = lambda s_: s_.replace('J/psi', 'JPSI').replace('psi', r'$\psi$').replace('JPSI', r'$J/\psi$')
for ax, kind in zip(axs, ('unmixed', 'mixed')):
    d = sc[(sc['variant'] == 'B-NL') & sc['state'].str.endswith('(' + kind + ')')]
    for st_, g in d.groupby('state', sort=False):
        ax.plot(g['c4_over_c'], g['Gamma_keV'], color=cols[st_], lw=1.6, label=lab(st_.split(' (')[0]))
    if kind == 'unmixed':
        for cand, col in (('psi(4040)', C2), ('psi(4415)', INK)):
            ax.axhline(ULs[cand]['Gamma_UL_keV'], color=col, lw=0.8, ls=':')
    else:
        ax.axhspan(1.05 - 0.15, 1.05 + 0.15, color=C1, alpha=0.2, lw=0)
        ax.axhline(21.8, color='#8a5cc2', lw=0.8, ls=':')
        ax.axhspan(1.31 - 0.33, 1.31 + 0.33, color=C3, alpha=0.2, lw=0)
    ax.set_yscale('log'); ax.set_xlabel(r'$c_4/c$  ($c_1=c_2=c_3=c=\kappa_c e e_c$)')
    ax.set_title(('unmixed hybrids' if kind == 'unmixed' else 'mixed $1^{--}$ states (mix=true)') + ', variant B (NL)',
                 fontsize=9, color=INK)
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.legend(fontsize=7, loc='lower left' if kind == 'mixed' else 'upper right', ncol=2 if kind == 'mixed' else 1)
axs[0].set_ylabel(r'$\Gamma(V\to\gamma\eta_c(1S))$ (keV)')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_sigma_scan.pdf')); fig.savefig(os.path.join(FIG, 'fig_sigma_scan.png'), dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- Fig. 5: LEFs
rr = np.linspace(0.3, 12, 400)
fig, ax = plt.subplots(figsize=(4.6, 3.0))
ax.plot(rr, 2 / np.sqrt(np.pi * sig) * np.ones_like(rr), color=C1, lw=2, label=r'$g_{B\perp}=2c/\sqrt{\pi\sigma}$')
ax.plot(rr, 4 / (sig * rr), color=C2, lw=2, label=r'$g_{B\parallel}=4c/(\sigma r)$ (EST)')
ax.plot(rr, -3 * np.pi / rr * (-4 / (3 * np.pi * sig)) * np.ones_like(rr), color=C2, lw=0.8, ls=':')
ax.axvline(2 * np.sqrt(np.pi / sig), color=MUTED, lw=0.6, ls='--')
ax.text(2 * np.sqrt(np.pi / sig) + 0.2, 9, r'$r=2\sqrt{\pi/\sigma}$', fontsize=7.5, color=MUTED)
ax.set_ylim(0, 12); ax.set_xlim(0, 12)
ax.set_xlabel(r'$r$ (GeV$^{-1}$)'); ax.set_ylabel(r'LEF / $c$ (GeV$^{-1}$)')
ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True); ax.legend(fontsize=7.5)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_LEFs_B.pdf')); fig.savefig(os.path.join(FIG, 'fig_LEFs_B.png'), dpi=200)
plt.close(fig)
# ---------------------------------------------------------------- Fig. 6: Gamma(E_q) vs lattice
ge = pd.read_csv(os.path.join(RES, 'gamma_vs_Eq.csv'))
kc = summ['kappa_c'][0]
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.7), sharey=True)
sty = {'off': ('-', C1), 'EST': ('-', C2), 'norm': ('--', C3)}
lhd = pd.read_csv(os.path.join(RES, 'lattice_hybrids.csv'))
lhd = lhd[(lhd['variant'] == 'A-NL') & (lhd['sigma_mode'] == 'off')]
for ax, v, ttl in zip(axs, ('A-NL', 'B-NL'), ('variant A (NL)', 'variant B (NL)')):
    for m in ('off', 'EST', 'norm'):
        d = ge[(ge['variant'] == v) & (ge['sigma_mode'] == m)]
        ax.plot(d['E_q'], kc ** 2 * d['Gamma_k1_keV'], ls=sty[m][0], color=sty[m][1], lw=1.6,
                label=r'$\Sigma_u^-$ ' + m)
    mk = {'quenched': ('o', INK, 'quenched LQCD'), 'Nf3': ('*', '#8a5cc2', r'$N_f=2+1$ LQCD (thesis)'),
          'Nf3-F0': ('*', '#c9b3e6', r'$N_f=2+1$, from $F(0)$')}
    for ds, (mm, col, lab_) in mk.items():
        d = lhd[lhd['dataset'] == ds]
        ax.errorbar(d['E_q'], d['lat_keV'], yerr=np.hypot(d['dlat_keV'], 0.3 * d['lat_keV']), fmt=mm, color=col,
                    ms=5 if mm == 'o' else 7, capsize=2, label=lab_)
    js = lhd[(lhd['dataset'] == 'quenched')]
    ax.plot(js['E_q'], js['JS_keV'], 's', color=MUTED, ms=4, label='J&S model')
    ax.plot([0.918], [summ['ULs_gamma_etac']['psi(4040)']['Gamma_UL_keV']], 'v', color=C2, ms=6,
            label=r'BESIII $\psi(4040)$ UL')
    ax.set_yscale('log'); ax.set_ylim(1, 1e5); ax.set_xlim(0.6, 1.5)
    ax.set_xlabel(r'$E_q$ (GeV)'); ax.set_title(ttl, fontsize=9, color=INK)
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
axs[0].set_ylabel(r'$\Gamma(1(s/d)_1\to\gamma\,1S)$ (keV), $\kappa_H=\kappa_c$')
hdl, lbl = axs[0].get_legend_handles_labels()
fig.legend(hdl, lbl, fontsize=7, loc='lower center', ncol=5, frameon=False)
fig.tight_layout(rect=(0, 0.12, 1, 1)); fig.savefig(os.path.join(FIG, 'fig_lattice_hybrids.pdf')); fig.savefig(os.path.join(FIG, 'fig_lattice_hybrids.png'), dpi=200)
plt.close(fig)
print('figures written to', FIG)
