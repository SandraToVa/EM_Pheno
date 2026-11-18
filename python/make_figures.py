"""Figures of the hybrid analysis (Sec. sec:pheno_H, v12, local form) from results/ (run run_hybrids.py first).

    python3 make_figures.py      (from python/; needs matplotlib)

Written to ../figures and next to sec_pheno.tex (same folder as make_tables.py):
  fig_H_gamma_E.pdf   Gamma(1(s/d)1 -> gamma 1S) as a function of the photon energy, reference fit, both branches,
                      Pi_u and Sigma_u^- channels; lattice (fit inputs), J&S model, BESIII limit at psi(4040)
  fig_H_LEFs.pdf      magnetic LEFs g_B_perp(r), g_B_par(r) of the reference fit (charm, bottom) and the radial functions u_1S,
                      u_Pi = r R_Pi, u_Sigma = r R_Sigma of the lowest charmonium (s/d)1 hybrid
"""
import os, sys, csv, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from emlib.states import Library, hybrid
from emlib.hybrid_data import LATTICE_FIT, JS_MODEL, besiii_gamma_etac_limits
from emlib.e1 import photon_energy, R0

ROOT = os.path.join(os.path.dirname(__file__), '..')
RES, FIG = os.path.join(ROOT, 'results'), os.path.join(ROOT, 'figures')
os.makedirs(FIG, exist_ok=True)
TEX = os.environ.get('EMPHENO_TEX', os.path.join(ROOT, '..', 'TeX file', 'tex_pheno'))   # as make_tables.py
if not os.path.isdir(TEX):
    TEX = os.path.join(ROOT, 'tex')
os.makedirs(TEX, exist_ok=True)
# categorical slots 1-4 of the validated reference palette (dataviz skill), fixed order
BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, INK2 = '#0b0b0b', '#52514e'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'legend.frameon': False, 'lines.linewidth': 2})


def read(name):
    with open(os.path.join(RES, name)) as f:
        return list(csv.DictReader(f))


def save(fig, name):
    for d in (FIG, TEX):
        fig.savefig(os.path.join(d, name), bbox_inches='tight')
    fig.savefig(os.path.join(FIG, name.replace('.pdf', '.png')), dpi=200, bbox_inches='tight')
    print('wrote', name)


summ = json.load(open(os.path.join(RES, 'hybrid_summary.json')))
br, alt = summ['ref_branch'], ('-' if summ['ref_branch'] == '+' else '+')
cur = [r for r in read('hybrid_curves.csv') if r['curve'] == 'Gamma(E)']
E = np.array([float(r['x']) for r in cur])
g = lambda k: np.array([float(r[k]) for r in cur])

# ------------------------------------------------------------------------------------------- Gamma(E_q)
fig, ax = plt.subplots(figsize=(6.4, 4.3))
ax.fill_between(E, g('total_range_lo'), g('total_range_hi'), color=BLUE, alpha=0.15, lw=0)
ax.plot(E, g(f'total_{br}'), color=BLUE, label=f'total, $B_c={summ["fit"][br]["B_c"]:+.3f}$ (reference)')
ax.plot(E, g(f'total_{alt}'), color=BLUE, ls='--', lw=1.5, label=f'total, $B_c={summ["fit"][alt]["B_c"]:+.3f}$')
ax.plot(E, g(f'Sigma_{br}'), color=AQUA, lw=1.5, label=r'$\Gamma_{\Sigma\Sigma}$ ($\Sigma_u^-$ channel)')
ax.plot(E, g(f'Pi_{br}'), color=ORANGE, lw=1.5, label=r'$\Gamma_{\Pi\Pi}$ ($\Pi_u$ channel)')
for d, mk in zip(LATTICE_FIT, ('o', 's')):
    Eq = photon_energy(d['M_i'], d['M_f'])
    lab = r'$Y\to\gamma\eta_c$' if d['S'] == 0 else r'$\eta_{c1}\to\gamma J/\psi$'
    ax.errorbar(Eq, d['value'], yerr=d['err'], fmt=mk, color=INK, ms=7, capsize=3, zorder=5,
                label=f'quenched LQCD, {lab}')
js = [(photon_energy(Mi, {'eta_c(1S)': 2.98409, 'J/psi': 3.0969}[f]), G) for _, S, _, Mi, f, G in JS_MODEL
      if f in ('eta_c(1S)', 'J/psi')]
ax.plot([x for x, _ in js], [y for _, y in js], 'D', mfc='white', mec=INK2, ms=7, mew=1.5, ls='none',
        label='J&S model ($\\Delta S=0$ channels)')
ul = besiii_gamma_etac_limits()['psi(4040)']['Gamma_UL_keV']
Eq = photon_energy(4.040, 2.98409)
ax.errorbar(Eq, ul, yerr=0.35 * ul, uplims=True, fmt='v', color=YELLOW, ms=8, capsize=0, mec=INK2,
            label=r'BESIII limit, $\psi(4040)\to\gamma\eta_c$')
ax.set_yscale('log')
ax.set_xlim(0.6, 1.6)
ax.set_ylim(0.3, 1e4)
ax.set_xlabel(r'photon energy $E_q$ (GeV)')
ax.set_ylabel(r'$\Gamma(1(s/d)_1\to\gamma\,1S)$ (keV)')
ax.grid(axis='y', color='#e4e3df', lw=0.6, which='major')
ax.legend(fontsize=7, loc='upper left', ncol=2)
save(fig, 'fig_H_gamma_E.pdf')

# ------------------------------------------------------------------------------------------- LEFs g_B(r)
kc = [r for r in read('hybrid_curves.csv') if r['curve'] == 'LEF(r)']
rr = np.array([float(r['x']) for r in kc])
E_CH = np.sqrt(4 * np.pi / 137.035999)
k = lambda key: np.array([float(r[key]) for r in kc]) / E_CH          # in units of e (GeV^-1)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.6))
a1.axhline(0, color=INK2, lw=0.6)
a1.plot(rr, k(f'c_B_perp_{br}'), color=BLUE, label=r'$g_{B\perp}$, charm')
a1.plot(rr, k(f'c_B_par_{br}'), color=BLUE, ls='--', lw=1.5, label=r'$g_{B\parallel}$, charm')
a1.plot(rr, k(f'b_B_perp_{br}'), color=ORANGE, label=r'$g_{B\perp}$, bottom')
a1.plot(rr, k(f'b_B_par_{br}'), color=ORANGE, ls='--', lw=1.5, label=r'$g_{B\parallel}$, bottom')
a1.axvline(R0, color=INK2, lw=0.8, ls=':')
a1.text(R0 + 0.1, a1.get_ylim()[0] * 0.9, r'$r_0$', color=INK2)
a1.set_xlabel(r'$r$ (GeV$^{-1}$)')
a1.set_ylabel(r'$g_X(r)/e$ (GeV$^{-1}$)')
a1.set_xlim(0, 8)
a1.legend(fontsize=7.5)
a1.set_title(f'reference fit, $B_c={summ["fit"][br]["B_c"]:+.3f}$', fontsize=9, color=INK2)
lib = Library('c')
Y = hybrid(lib, 'Y', 1, 'plus', 1)
pr = Y.projections()
r = lib.r
a2.axhline(0, color=INK2, lw=0.6)
a2.plot(r, lib.u(lib.state('1S', 1, 0, 0, 0, 2.98).c), color=INK, lw=1.5, label=r'$u_{1S}$')
a2.plot(r, lib.u(pr['R_Pi']), color=ORANGE, label=r'$u_\Pi=rR_\Pi$')
a2.plot(r, lib.u(pr['R_Sigma']), color=AQUA, label=r'$u_\Sigma=rR_\Sigma$')
a2.set_xlim(0, 12)
a2.set_xlabel(r'$r$ (GeV$^{-1}$)')
a2.set_ylabel(r'$u(r)$ (GeV$^{1/2}$)')
a2.legend(fontsize=8)
a2.set_title(r'charmonium $1(s/d)_1$ hybrid and $1S$', fontsize=9, color=INK2)
fig.tight_layout()
save(fig, 'fig_H_LEFs.pdf')
