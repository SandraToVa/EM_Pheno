"""LaTeX tables of the interpolated-g_E1 fits (EM_Pheno v6), from results/interp_*.csv.
Written next to sec_pheno.tex (<EM transitions>/TeX file/tex_pheno, or $EMPHENO_TEX):
  tab_kappa_interp.tex            kappa_Q for the three epsilon options (as tab_kappa_rel)
  tab_E1_interp.tex               observables, n = 2 and n = 7, NL (as tab_E1_rel)
  tab_lattice_chic0_v6.tex        tab_lattice_chic0 + LO with 1/m terms + interpolated fits
  tab_quarkonium_lattice_v6.tex   tab_quarkonium_lattice + interpolated fit columns
The v5 tables are not touched."""
import os, re
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), '..')
RES = os.path.join(ROOT, 'results')
TEX = os.environ.get('EMPHENO_TEX', os.path.join(ROOT, '..', 'TeX file', 'tex_pheno'))
if not os.path.isdir(TEX):
    TEX = os.path.join(ROOT, 'tex')
os.makedirs(TEX, exist_ok=True)
R0 = np.sqrt(np.pi / (6 * 0.21))


def lab(s):
    rep = [('psi2_3823', r'\psi_2(3823)'), ('psi2(3823)', r'\psi_2(3823)'), ('psi2S', r'\psi(2S)'),
           ('psi3770', r'\psi(3770)'), ('Jpsi', r'J/\psi'), ('hc', r'h_c'), ('etac', r'\eta_c'),
           ('Y2S', r'\Upsilon(2S)'), ('Y3S', r'\Upsilon(3S)'), ('Y1S', r'\Upsilon(1S)'),
           ('hb(2P)', r'h_b(2P)'), ('etab1S', r'\eta_b(1S)'), ('etab2S', r'\eta_b(2S)'), ('X(3872)', r'X(3872)')]
    s = re.sub(r'chi([cb])(\d)', r'\\chi_{\1\2}', s)
    for a, b in rep:
        s = s.replace(a, b)
    s = s.replace('->', r'\to\gamma ').replace(': ', r':\ ')
    return '$' + s + '$'


def fmt(x):
    if not np.isfinite(x): return '--'
    if abs(x) >= 100: return f'{x:.0f}'
    if abs(x) >= 10: return f'{x:.1f}'
    if abs(x) >= 1: return f'{x:.2f}'
    if abs(x) >= 0.01: return f'{x:.3f}'
    if abs(x) < 1e-6: return '0'
    m, e = f'{x:.1e}'.split('e')
    return f'${m}\\times10^{{{int(e)}}}$'


def kerr(k, lo, hi):
    return f'${k:+.3f}^{{+{hi:.3f}}}_{{-{lo:.3f}}}$'


def mf(s):
    """<f> column: values outside [-2, 2] (hindered transitions, node cancellations) shown as 'node'"""
    out = []
    for v in str(s).split(' / '):
        x = float(v)
        out.append(f'{x:.2f}' if abs(x) <= 2 else r'\textit{node}')
    return ' / '.join(out)


# ------------------------------------------------------------------ kappa table
fits = pd.read_csv(os.path.join(RES, 'interp_fits.csv'))
main = fits[(fits['shape_kind'] == 'main')]
rows = []
for n in (2, 7):
    for _, r in main[main['n'] == n].iterrows():
        rows.append(f"${r['flavor']}$ & {r['set']} & {'non-local' if r['nonlocal'] else 'local'} & {n} & "
                    f"{int(r['N_meas'])}+{int(r['N_UL'])} & {r['eps_fit']:.3f} & {kerr(r['kappa'], r['k_lo'], r['k_hi'])} & "
                    f"${r['kappa'] / r['eQ']:+.3f}$ & {kerr(r['kappa_eps_v2'], r['k_lo_v2'], r['k_hi_v2'])} & "
                    f"{r['chi2_eps_v2']:.1f}/{int(r['ndf'])} & {kerr(r['kappa_eps0'], r['k_lo_eps0'], r['k_hi_eps0'])} & "
                    f"{r['chi2_eps0']:.0f}/{int(r['ndf'])} \\\\")
    rows.append('\\hline')
with open(os.path.join(TEX, 'tab_kappa_interp.tex'), 'w') as f:
    f.write('\n'.join(rows) + '\n')

# ------------------------------------------------------------------ observables table (NL; n = 2 and n = 7)
p = pd.read_csv(os.path.join(RES, 'interp_pulls.csv'))
p2 = p[p['n'] == 2].reset_index(drop=True)
p7 = p[p['n'] == 7].reset_index(drop=True)


def dat(r):
    return f"{fmt(r['data'])} $\\pm$ {fmt(r['err_exp'])}" if r['kind'] in ('width', 'ratio') else f"$<${fmt(r['data'])}"


rows = []
for (_, a), (_, b) in zip(p2.iterrows(), p7.iterrows()):
    assert a['id'] == b['id']
    rows.append(f"{lab(a['id'])} & {a['set']} & {dat(a)} & {a['delta']} & {mf(a['meanf'])} & {fmt(a['pred_lo_rel'])} & "
                f"{fmt(a['pred_eps0'])} & {a['pull_eps0']:+.2f} & {fmt(a['pred_fit'])} $\\pm$ {fmt(a['err_th'])} & {a['pull']:+.2f} & "
                f"{fmt(b['pred_fit'])} $\\pm$ {fmt(b['err_th'])} & {b['pull']:+.2f} \\\\")
with open(os.path.join(TEX, 'tab_E1_interp.tex'), 'w') as f:
    f.write('\n'.join(rows) + '\n\\hline\n')

# ------------------------------------------------------------------ lattice comparison tables
L = pd.read_csv(os.path.join(RES, 'interp_lattice.csv'))
L2, L7 = L[L['n'] == 2].reset_index(drop=True), L[L['n'] == 7].reset_index(drop=True)
k2 = fits[(fits['n'] == 2) & (fits['r0'].round(3) == round(R0, 3)) & fits['nonlocal'] & (fits['flavor'] == 'c') & (fits['set'] == 'core')].iloc[0]
k7 = fits[(fits['n'] == 7) & (fits['r0'].round(3) == round(R0, 3)) & fits['nonlocal'] & (fits['flavor'] == 'c') & (fits['set'] == 'core')].iloc[0]
c0 = L2[L2['id'] == 'chic0->Jpsi'].iloc[0]; c07 = L7[L7['id'] == 'chic0->Jpsi'].iloc[0]
k5 = np.sqrt(c0['fit_v5_keV'] / c0['LO_NL_keV'])
with open(os.path.join(TEX, 'tab_lattice_chic0_v6.tex'), 'w') as f:
    f.write(f"LO, local, $\\kappa_c=1$ & {c0['LO_L_keV']:.0f} \\\\\n"
            f"LO, non-local, $\\kappa_c=1$ & {c0['LO_NL_keV']:.0f} \\\\\n"
            f"LO + $1/m$, non-local & {c0['LO_rel_keV']:.0f} \\\\\n"
            f"fit, non-local, $\\kappa_c={k5:.3f}$ & {c0['fit_v5_keV']:.0f}$\\pm${c0['dfit_v5_keV']:.0f} \\\\\n"
            f"fit, non-local + $1/m$, $\\kappa_c={c0['kappa_v5_rel']:.3f}$ & {c0['fit_rel_keV']:.0f}$\\pm${c0['dfit_rel_keV']:.0f} \\\\\n"
            f"fit, $g_{{E1}}(r)$ interpolated + $1/m$, $n=2$, $\\kappa_c={k2['kappa']:+.3f}$ & {c0['fit_keV']:.0f}$\\pm${c0['dfit_keV']:.0f} \\\\\n"
            f"fit, $g_{{E1}}(r)$ interpolated + $1/m$, $n=7$, $\\kappa_c={k7['kappa']:+.3f}$ & {c07['fit_keV']:.0f}$\\pm${c07['dfit_keV']:.0f} \\\\\n"
            f"lattice~\\cite{{Delaney:2023fsc}}, experimental masses & 208.6(3) \\\\\n"
            f"lattice~\\cite{{Delaney:2023fsc}}, lattice masses & 247.4(2) \\\\\n"
            f"RPP 2026 & {c0['data_keV']:.0f}$\\pm${c0['ddata_keV']:.0f} \\\\\n\\hline\n")

CITE = {'HSC': r'HSC~\cite{Delaney:2023fsc}', 'ETMC': r'ETMC~\cite{Becirevic:2026gaf}', 'CLQCD': r'CLQCD~\cite{Li:2023zig}'}
LATV = {'chic0->Jpsi': '208.6(0.3)', 'chic1->Jpsi': '327(10)', 'chic2->Jpsi': '368(5)', 'psi2(3823)->chic1': '337(27)'}
rows = []
for (_, a), (_, b) in zip(L2.iterrows(), L7.iterrows()):
    rpp = f"{a['data_keV']:.0f}$\\pm${a['ddata_keV']:.0f}" if np.isfinite(a['data_keV']) else '--'
    rows.append(f"{lab(a['id'])} & {CITE[a['source']]} & {LATV[a['id']]} & {a['LO_L_keV']:.0f} & {a['LO_NL_keV']:.0f} & "
                f"{a['fit_v5_keV']:.0f} & {a['LO_rel_keV']:.0f} & {a['fit_rel_keV']:.0f} & {a['fit_keV']:.0f} & {b['fit_keV']:.0f} & {rpp} & "
                f"{a['dkappa_eff_v5']:+.3f} & {a['kappa_eff']:+.3f} \\\\")
with open(os.path.join(TEX, 'tab_quarkonium_lattice_v6.tex'), 'w') as f:
    f.write('\n'.join(rows) + '\n\\hline\n')
print('tables written to', os.path.abspath(TEX))
