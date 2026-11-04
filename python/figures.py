"""Figures for the phenomenology section (run after run_all.py). EM_Pheno v4: hybrid
couplings normalized with dkappa_c = kappa_c - 1.
Main text (matched gaps pi/r, 3pi/r; single coupling c_i = dkappa_c e e_c): fig_E1_ratios,
fig_hybrid_integrand, fig_LEFs_B, fig_lattice_hybrids.  App. app:potential_gaps: fig_potential_gaps
(matched versus potential gaps)."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from emlib.states import Library, pi_sigma_projections
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
p = pulls[(pulls['nonlocal'] == True) & (pulls['rel'] == False) & (pulls['kind'] == 'width')].copy()
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

def save(fig, name):
    fig.savefig(os.path.join(FIG, name + '.pdf')); fig.savefig(os.path.join(FIG, name + '.png'), dpi=200)
    plt.close(fig)


def style(ax, title=None):
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=9, color=INK)


# ---------------------------------------------------------------- Fig. 2: integrands (matched gaps)
lib = Library('c'); r = lib.r; dV = HY.load_dV('c'); dVS = HY.load_dVS('c')
sig = 0.21
h = [x for x in HY.HYBRIDS['c'] if x[0] == '1(s/d)1'][0]
st = HY.hybrid_state(lib, h)
pr = pi_sigma_projections(st)
cPi, cSig = pr['R_Pi'], pr['R_Sigma']          # Eqs. (eq:lab_RPi), (eq:lab_RSigma), J = 1
f = lib.quarkonium(1, 0)
E = photon_energy(4.040, 2.98409)
uPi = lib.u(cPi); uSig = lib.u(cSig); uf = lib.u(f.comps[0].c)
base = uf * uPi
w = 2 * np.sqrt(np.pi / sig) / r          # A_0 weight per unit c: (2 sqrt(pi)/sqrt(sigma)) / r
bS = uf * uSig * w
integ = {'Pi': {'mag': base, 'matched': base * (1 - E * r / np.pi),
                'potential': base * np.sqrt(dV * r / np.pi) * (1 - E / dV)},
         'Sigma': {'mag': bS, 'matched': bS * (1 - E * r / (3 * np.pi)),
                   'potential': bS * np.sqrt(dVS * r / (3 * np.pi)) * (1 - E / dVS)}}
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0))
for ax, X, lm, lf, ttl in ((axs[0], 'Pi', r'$u_{\eta_c}u_\Pi$ ($g_{B\perp}$ only)', r'equal $c_i$: $\times(1-E_q r/\pi)$',
                            r'$\Pi_u$ channel: $A_1$ ($p_0=\sqrt{2/3}$)'),
                           (axs[1], 'Sigma', r'$u_{\eta_c}u_\Sigma\,\frac{2\sqrt{\pi}}{\sqrt{\sigma}\,r}$ ($g_{B\parallel}$ only)',
                            r'equal $c_i$: $\times(1-E_qr/3\pi)$', r'$\Sigma_u^-$ channel: $A_0$ ($s_0=\sqrt{1/3}$)')):
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.plot(r, integ[X]['mag'], color=MUTED, lw=1.2, ls='--', label=lm)
    ax.plot(r, integ[X]['matched'], color=C1, lw=2, label=lf)
    ax.set_xlim(0, 12); ax.set_xlabel(r'$r$ (GeV$^{-1}$)')
    style(ax, ttl); ax.legend(fontsize=7, loc='upper right')
axs[0].set_ylabel('radial integrand (GeV)')
fig.suptitle(r'$\psi(4040)\,[1(s/d)_1]\to\gamma\,\eta_c(1S)$, $E_q=918$ MeV', fontsize=9, color=INK)
fig.tight_layout(); save(fig, 'fig_hybrid_integrand')

# ---------------------------------------------------------------- common inputs (v4: c_i = dkappa_c e e_c, c4 = c)
ULs = summ['ULs_gamma_etac']
kc = summ['dkappa_c'][0]

# ---------------------------------------------------------------- Fig. 5: LEFs
rr = np.linspace(0.3, 12, 400)
fig, ax = plt.subplots(figsize=(4.6, 3.0))
ax.plot(rr, 2 / np.sqrt(np.pi * sig) * np.ones_like(rr), color=C1, lw=2, label=r'$g_{B\perp}=2c/\sqrt{\pi\sigma}$')
ax.plot(rr, 4 / (sig * rr), color=C2, lw=2, label=r'$|g_{B\parallel}|=4c/(\sigma r)$ ($g_{B\parallel}<0$)')
ax.plot(rr, np.abs(-3 * np.pi / rr * (4 / (3 * np.pi * sig))) * np.ones_like(rr), color=C2, lw=0.8, ls=':')
ax.axvline(2 * np.sqrt(np.pi / sig), color=MUTED, lw=0.6, ls='--')
ax.text(2 * np.sqrt(np.pi / sig) + 0.2, 9, r'$r=2\sqrt{\pi/\sigma}$', fontsize=7.5, color=MUTED)
ax.set_ylim(0, 12); ax.set_xlim(0, 12)
ax.set_xlabel(r'$r$ (GeV$^{-1}$)'); ax.set_ylabel(r'$|$LEF$|$ / $c$ (GeV$^{-1}$)')
style(ax); ax.legend(fontsize=7.5)
fig.tight_layout(); save(fig, 'fig_LEFs_B')

# ---------------------------------------------------------------- Fig. 6: Gamma(E_q) vs lattice
ge = pd.read_csv(os.path.join(RES, 'gamma_vs_Eq.csv'))
lhd = pd.read_csv(os.path.join(RES, 'lattice_hybrids.csv'))
lhd = lhd[lhd['variant'] == 'NL']
hext = pd.read_csv(os.path.join(RES, 'hybrid_external.csv'))
sty = {'Gamma_k1_keV': ('-', INK, r'$\Gamma$ (both channels)'), 'Pi_keV': ('--', C1, r'$\Gamma_{\Pi\Pi}$'),
       'Sigma_keV': ('--', C2, r'$\Gamma_{\Sigma\Sigma}$')}


def lattice_points(ax, labels=True):
    mk = {'quenched': ('o', INK, 'quenched LQCD'), 'Nf3': ('*', '#8a5cc2', r'$N_f=2+1$ LQCD (thesis)'),
          'Nf3-F0': ('*', '#c9b3e6', r'$N_f=2+1$, from $F(0)$')}
    for ds, (mm, col, lab_) in mk.items():
        d = lhd[lhd['dataset'] == ds]
        ax.errorbar(d['E_q'], d['lat_keV'], yerr=np.hypot(d['dlat_keV'], 0.3 * d['lat_keV']), fmt=mm, color=col,
                    ms=5 if mm == 'o' else 7, capsize=2, label=lab_ if labels else None)
    js = lhd[(lhd['dataset'] == 'quenched')]
    ax.plot(js['E_q'], js['JS_keV'], 's', color=MUTED, ms=4, label='J&S model' if labels else None)
    ax.plot([0.918], [ULs['psi(4040)']['Gamma_UL_keV']], 'v', color=C2, ms=6,
            label=r'BESIII $\psi(4040)$ UL' if labels else None)
    # Coulomb-gauge model (1402.5863), M = 4.35 GeV: range of its two normalizations
    for _, r_ in hext[hext['model_lo_keV'].notna()].iterrows():
        ax.plot([r_['E_q']] * 2, [r_['model_lo_keV'], r_['model_hi_keV']], color='#b08a2e', lw=2.2, alpha=0.8,
                label=('GYS model' if labels and r_['channel'].startswith('Y') else None))


fig, ax = plt.subplots(figsize=(5.4, 4.4))
d = ge[ge['variant'] == 'NL']
for k, (ls, col, lab_) in sty.items():
    ax.plot(d['E_q'], kc ** 2 * d[k], ls=ls, color=col, lw=1.6 if k == 'Gamma_k1_keV' else 1.2, label=lab_)
lattice_points(ax)
ax.set_yscale('log'); ax.set_ylim(0.1, 1e4); ax.set_xlim(0.6, 1.5)
ax.set_xlabel(r'$E_q$ (GeV)'); ax.set_ylabel(r'$\Gamma(1(s/d)_1\to\gamma\,1S)$ (keV), $\kappa_H=\delta\kappa_c$')
style(ax)
hdl, lbl = ax.get_legend_handles_labels()
fig.legend(hdl, lbl, fontsize=6.8, loc='lower center', ncol=4, frameon=False)
fig.tight_layout(rect=(0, 0.11, 1, 1)); save(fig, 'fig_lattice_hybrids')

# ---------------------------------------------------------------- App. app:potential_gaps: summary figure
fig, axs = plt.subplots(2, 2, figsize=(8.2, 6.4))
ax = axs[0, 0]
m = (r > 0.8) & (r < 15)
ax.plot(r[m], np.pi / r[m], color=C1, lw=1.6, label=r'$\pi/r$')
ax.plot(r[m], dV[m], color=C1, lw=1.6, ls='--', label=r'$\Delta V=V_{\Pi_u}-V_{\Sigma_g^+}$')
ax.plot(r[m], 3 * np.pi / r[m], color=C2, lw=1.6, label=r'$3\pi/r$')
ax.plot(r[m], dVS[m], color=C2, lw=1.6, ls='--', label=r'$\Delta V_\Sigma=V_{\Sigma_u^-}-V_{\Sigma_g^+}$')
ax.axhline(E, color=MUTED, lw=0.8, ls=':')
ax.text(10.5, E + 0.08, r'$E_q(\psi(4040)\to\gamma\eta_c)$', fontsize=6.5, color=MUTED, ha='center')
ax.set_ylim(0, 4); ax.set_xlim(0, 15); ax.set_xlabel(r'$r$ (GeV$^{-1}$)'); ax.set_ylabel('gap (GeV)')
style(ax, '(a) gaps'); ax.legend(fontsize=6.5, loc='upper right')
for ax, X, ttl in ((axs[0, 1], 'Pi', r'(b) $\Pi_u$ integrand, $\psi(4040)\to\gamma\eta_c$'),
                   (axs[1, 0], 'Sigma', r'(c) $\Sigma_u^-$ integrand, $\psi(4040)\to\gamma\eta_c$')):
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.plot(r, integ[X]['mag'], color=MUTED, lw=1.0, ls=':', label='magnetic only')
    ax.plot(r, integ[X]['matched'], color=C1, lw=1.8, label='matched gap')
    ax.plot(r, integ[X]['potential'], color=C2, lw=1.8, ls='--', label='potential gap')
    ax.set_xlim(0, 12); ax.set_xlabel(r'$r$ (GeV$^{-1}$)')
    style(ax, ttl); ax.legend(fontsize=6.5, loc='upper right')
ax = axs[1, 1]
for v, ls in (('NL', '-'), ('pot-NL', '--')):
    d = ge[ge['variant'] == v]
    nm = 'matched' if v == 'NL' else 'potential'
    ax.plot(d['E_q'], kc ** 2 * d['Gamma_k1_keV'], ls=ls, color=INK, lw=1.6, label=r'$\Gamma$, ' + nm)
    ax.plot(d['E_q'], kc ** 2 * d['Pi_keV'], ls=ls, color=C1, lw=1.0, label=r'$\Gamma_{\Pi\Pi}$, ' + nm)
    ax.plot(d['E_q'], kc ** 2 * d['Sigma_keV'], ls=ls, color=C2, lw=1.0, label=r'$\Gamma_{\Sigma\Sigma}$, ' + nm)
lattice_points(ax, labels=False)
ax.set_yscale('log'); ax.set_ylim(0.01, 1e4); ax.set_xlim(0.6, 1.5); ax.set_xlabel(r'$E_q$ (GeV)')
ax.set_ylabel(r'$\Gamma(1(s/d)_1\to\gamma 1S)$ (keV), $\kappa_H=\delta\kappa_c$')
style(ax, r'(d) $\Gamma(E_q)$ and lattice'); ax.legend(fontsize=5.5, loc='lower left', ncol=2)
fig.tight_layout(); save(fig, 'fig_potential_gaps')
print('figures written to', FIG)
