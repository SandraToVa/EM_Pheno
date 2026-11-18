"""LaTeX table bodies of Sec. 5 (quarkonium E1) from results/ (run run_all.py first; no number is typed by hand).

    python3 make_tables.py        (from python/)

Written next to sec_pheno.tex (<EM transitions>/TeX file/tex_pheno, or $EMPHENO_TEX, or EM_Pheno_v11/tex):
  tab_gE1_fit.tex            Table tab:gE1_fit             fits of (A_b, kappa_b)                      (fits.csv)
  tab_E1_widths.tex          Table tab:E1_widths           widths at the reference fit                 (widths.csv)
  tab_J_ratios.tex           Table tab:J_ratios            J dependence of the multiplets              (j_ratios.csv)
  tab_lattice_chic0.tex      Table tab:lattice             chi_c0 -> gamma J/psi, HSC                  (lattice.csv)
  tab_quarkonium_lattice.tex Table tab:quarkonium_lattice  lattice E1 widths                           (lattice.csv)
"""
import os, re, csv, json
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..')
RES = os.path.join(ROOT, 'results')
TEX = os.environ.get('EMPHENO_TEX', os.path.join(ROOT, '..', 'TeX file', 'tex_pheno'))
if not os.path.isdir(TEX):
    TEX = os.path.join(ROOT, 'tex')
os.makedirs(TEX, exist_ok=True)


def read(name):
    with open(os.path.join(RES, name)) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v) if v != '' else np.nan
            except ValueError:
                pass
    return rows


def write(name, lines):
    with open(os.path.join(TEX, name), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('wrote', os.path.join(TEX, name))


def lab(s):
    """observable id -> LaTeX"""
    rep = [('psi2_3823', r'\psi_2(3823)'), ('psi2(3823)', r'\psi_2(3823)'), ('psi2S', r'\psi(2S)'),
           ('psi3770', r'\psi(3770)'), ('Jpsi', r'J/\psi'), ('hc', r'h_c'), ('etac', r'\eta_c'),
           ('Y2S', r'\Upsilon(2S)'), ('Y3S', r'\Upsilon(3S)'), ('Y1S', r'\Upsilon(1S)'),
           ('hb(2P)', r'h_b(2P)'), ('etab1S', r'\eta_b(1S)'), ('etab2S', r'\eta_b(2S)')]
    s = re.sub(r'chi([cb])(\d)', r'\\chi_{\1\2}', s)
    s = re.sub(r'chi([cb])J', r'\\chi_{\1J}', s)
    for a, b in rep:
        s = s.replace(a, b)
    return '$' + s.replace('->', r'\to\gamma ').replace(': ', r':\ ') + '$'


def fmt(x):
    if not np.isfinite(x):
        return '--'
    if abs(x) < 1e-6:
        return '0'
    if abs(x) >= 100:
        return f'{x:.0f}'
    if abs(x) >= 10:
        return f'{x:.1f}'
    if abs(x) >= 1:
        return f'{x:.2f}'
    if abs(x) >= 0.01:
        return f'{x:.3f}'
    m, e = f'{x:.1e}'.split('e')
    return f'${m}\\times10^{{{int(e)}}}$'


def perr(v, lo, hi, d=3):
    """value^{+hi}_{-lo}"""
    v = 0.0 if abs(v) < 0.5 * 10 ** -d else v
    return f"${v:+.{d}f}^{{+{hi:.{d}f}}}_{{-{lo:.{d}f}}}$"


# ------------------------------------------------------------------ Table tab:gE1_fit
rows = []
for set_ in ('core', 'core+ext'):
    for r in read('fits.csv'):
        if r['set'] != set_:
            continue
        nd = int(r['ndf'])
        nm = int(r['N_meas'])
        rows.append(f"{set_} & {'non-local' if r['form'] == 'NL' else 'local'} & {nm}+{int(r['N_UL'])} & "
                    f"{r['chi2_LO_v2']:.1f}/{nm} & {r['chi2_LO1m_v2']:.1f}/{nm} & "
                    f"{perr(r['A_b_v2'], r['A_b_lo_v2'], r['A_b_hi_v2'])} & {perr(r['kappa_b_v2'], r['kappa_b_lo_v2'], r['kappa_b_hi_v2'])} & "
                    f"{r['chi2_v2']:.1f}/{nd} ({r['chi2_c_v2']:.1f}+{r['chi2_b_v2']:.1f}) & "
                    f"{r['chi2_LO_eps0']:.0f}/{nm} & {r['chi2_LO1m_eps0']:.0f}/{nm} & "
                    f"{perr(r['A_b_eps0'], r['A_b_lo_eps0'], r['A_b_hi_eps0'])} & {perr(r['kappa_b_eps0'], r['kappa_b_lo_eps0'], r['kappa_b_hi_eps0'])} & "
                    f"{r['chi2_eps0']:.0f}/{nd} ({r['chi2_c_eps0']:.0f}+{r['chi2_b_eps0']:.0f}) \\\\")
write('tab_gE1_fit.tex', rows + [r'\hline'])

# ------------------------------------------------------------------ Table tab:E1_widths
rows, prev = [], None
for r in read('widths.csv'):
    if prev is not None and r['flavor'] != prev:
        rows.append(r'\hline')
    prev = r['flavor']
    ul = r['kind'].startswith('ul')
    if ul:
        dat = f"$<${fmt(r['data'])}"
    elif np.isfinite(r['err_up']):
        dat = f"${fmt(r['data'])}^{{+{fmt(r['err_up'])}}}_{{-{fmt(r['err_dn'])}}}$"
    else:
        dat = f"{fmt(r['data'])} $\\pm$ {fmt(r['err_exp'])}"
    st = r['set'] if r['fitted'] == 'True' else f"{r['set']}$^\\ast$"
    dl = ' / '.join(f'{float(x):+.3f}' for x in str(r['delta_1m']).split(' / '))
    if r['LO'] < 1e-12:      # E1 forbidden by the spin algebra (psi_2(3823) -> chi_c0 gamma, Delta J = 2)
        rows.append(f"{lab(r['id'])} & {st} & {dat} & 0 & {dl} & 0 & 0 & -- & 0 & -- \\\\")
        continue
    rows.append(f"{lab(r['id'])} & {st} & {dat} & {fmt(r['LO'])} & {dl} & {fmt(r['LO1m'])} & "
                f"{fmt(r['fit_v2'])} $\\pm$ {fmt(r['sth_v2'])} & {r['pull_v2']:+.2f} & {fmt(r['fit_eps0'])} & {r['pull_eps0']:+.2f} \\\\")
write('tab_E1_widths.tex', rows + [r'\hline'])

# ------------------------------------------------------------------ Table tab:J_ratios
names = {'psi2S->chicJ': r'$\psi(2S)\to\gamma\chi_{cJ}$', 'chicJ->Jpsi': r'$\chi_{cJ}\to\gamma J/\psi$',
         'Y2S->chibJ(1P)': r'$\Upsilon(2S)\to\gamma\chi_{bJ}(1P)$', 'Y3S->chibJ(2P)': r'$\Upsilon(3S)\to\gamma\chi_{bJ}(2P)$',
         'Y3S->chibJ(1P)': r'$\Upsilon(3S)\to\gamma\chi_{bJ}(1P)$', 'psi3770->chicJ': r'$\psi(3770)\to\gamma\chi_{cJ}$',
         'chibJ(2P): Y1S/Y2S': r'$\chi_{bJ}(2P)$: $\Upsilon(1S)/\Upsilon(2S)$',
         'psi2(3823): chicJ/chic1': r'$\psi_2(3823)\to\gamma\chi_{cJ}$'}
rows, prev = [], None
for r in read('j_ratios.csv'):
    first = r['multiplet'] != prev
    if first and prev is not None:
        rows.append(r'\hline')
    prev = r['multiplet']
    if r['kind'].startswith('ul'):
        ex = f"$<{r['exp']:.1f}$"
    elif np.isfinite(r.get('exp_err_up', np.nan)):
        ex = f"${r['exp']:.2f}^{{+{r['exp_err_up']:.2f}}}_{{-{r['exp_err_dn']:.2f}}}$"
    else:
        ex = f"{r['exp']:.2f} $\\pm$ {r['exp_err']:.2f}"
    rows.append(f"{names[r['multiplet']] if first else ''} & {int(r['J'])}/{int(r['J_ref'])} & {r['E_MeV']:.0f} & {ex} & "
                f"{r['LO']:.2f} & {r['LO1m']:.2f} & {r['fit']:.2f} \\\\")
write('tab_J_ratios.tex', rows + [r'\hline'])

# ------------------------------------------------------------------ Tables tab:lattice, tab:quarkonium_lattice
summ = json.load(open(os.path.join(RES, 'summary.json')))
L = {r['id']: r for r in read('lattice.csv')}
c0 = L['chic0->Jpsi']
ref = summ['ref_v2']
write('tab_lattice_chic0.tex', [
    f"LO, local & {c0['LO_L']:.0f} \\\\",
    f"LO, non-local & {c0['LO_NL']:.0f} \\\\",
    f"fit, non-local, $g_{{E1}}(r)$ of Eq.~\\eqref{{eq:gE1_interp}} $+\\,1/m$, $\\epsilon=v^2$ & ${c0['fit']:.0f}^{{+{c0['dfit_hi']:.0f}}}_{{-{c0['dfit_lo']:.0f}}}$ \\\\",
    r"\hline",
    f"lattice~\\cite{{Delaney:2023fsc}}, experimental masses & {c0['lat']:.1f}(3) \\\\",
    f"lattice~\\cite{{Delaney:2023fsc}}, lattice masses & {summ['chic0_lattice_masses_keV']:.1f}(2) \\\\",
    f"RPP 2026 & {c0['data']:.0f}$\\pm${c0['ddata']:.0f} \\\\", r"\hline"])
qtex = {'chic0->Jpsi': r'$\chi_{c0}\to\gamma J/\psi$', 'chic1->Jpsi': r'$\chi_{c1}\to\gamma J/\psi$',
        'chic2->Jpsi': r'$\chi_{c2}\to\gamma J/\psi$', 'psi2(3823)->chic1': r'$\psi_2(3823)\to\gamma\chi_{c1}$'}
qsrc = {'HSC': r'HSC~\cite{Delaney:2023fsc}', 'ETMC': r'ETMC~\cite{Becirevic:2026gaf}', 'CLQCD': r'CLQCD~\cite{Li:2023zig}'}
rows = []
for r in L.values():
    dat = '--' if not np.isfinite(r['data']) else f"{r['data']:.0f}$\\pm${r['ddata']:.0f}"
    d = 1 if r['dlat'] < 1 else 0
    rows.append(f"{qtex[r['id']]} & {qsrc[r['source']]} & {r['lat']:.{d}f}({r['dlat'] * 10 ** d:.0f}) & {r['LO_L']:.0f} & {r['LO_NL']:.0f} & "
                f"${r['fit']:.0f}^{{+{r['dfit_hi']:.0f}}}_{{-{r['dfit_lo']:.0f}}}$ & {r['fit_over_lat']:.2f} & {dat} \\\\")
write('tab_quarkonium_lattice.tex', rows + [r'\hline'])
