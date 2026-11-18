"""LaTeX table bodies of the hybrid part of Sec. 5 (v12, local form) from results/ (run run_hybrids.py first; no number is typed by hand).

    python3 make_hybrid_tables.py        (from python/)

Written next to sec_pheno.tex (same folder as make_tables.py):
  tab_H_fit.tex       Table tab:H_fit       fits of B_c and checks                         (hybrid_fit.csv)
  tab_H_lattice.tex   Table tab:H_lattice   lattice and J&S kinematics                     (hybrid_lattice.csv)
  tab_H_pred_c.tex    Table tab:H_pred      S = 0 predictions, charmonium                  (hybrid_predictions.csv)
  tab_H_pred_b.tex    Table tab:H_pred      S = 0 predictions, bottomonium                 (hybrid_predictions.csv)
  tab_H_spin1.tex     Table tab:H_spin1     S = 1 members of 1(s/d)1, 1p1, 1p0              (hybrid_spin1.csv)
  tab_H_besiii.tex    Table tab:H_besiii    psi(4040), psi(4415) -> gamma eta_c vs BESIII  (hybrid_besiii.csv)
"""
import os, csv, json
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


def g(x):
    """a width in keV with 2-3 significant figures (negative values in math mode)"""
    s = _g(x)
    return '$' + s + '$' if s.startswith('-') else s


def _g(x):
    if not np.isfinite(x):
        return '--'
    ax = abs(x)
    if ax < 5e-4:
        return '$<10^{-3}$' if x >= 0 else '$>-10^{-3}$'
    if ax >= 100:
        return f'{x:.0f}'
    if ax >= 10:
        return f'{x:.1f}' if ax < 20 else f'{x:.0f}'
    if ax >= 1:
        return f'{x:.2f}'
    if ax >= 0.1:
        return f'{x:.2f}'
    if ax >= 0.01:
        return f'{x:.3f}'
    return f'{x:.3f}'


TEXNAME = {
    'eta_c(1S)': r'\eta_c(1S)', 'eta_c(2S)': r'\eta_c(2S)', 'eta_c(3S)': r'\eta_c(3S)', 'h_c(1P)': r'h_c(1P)',
    'h_c(2P)': r'h_c(2P)', 'eta_c2(1D)': r'\eta_{c2}(1D)', 'eta_c2(2D)': r'\eta_{c2}(2D)', 'h_c3(1F)': r'h_{c3}(1F)',
    'eta_b(1S)': r'\eta_b(1S)', 'eta_b(2S)': r'\eta_b(2S)', 'eta_b(3S)': r'\eta_b(3S)', 'h_b(1P)': r'h_b(1P)',
    'h_b(2P)': r'h_b(2P)', 'h_b(3P)': r'h_b(3P)', 'eta_b2(1D)': r'\eta_{b2}(1D)', 'eta_b2(2D)': r'\eta_{b2}(2D)',
    'h_b3(1F)': r'h_{b3}(1F)', 'J/psi': r'J/\psi', 'psi(2S)': r'\psi(2S)', 'Upsilon(1S)': r'\Upsilon(1S)',
    'Upsilon(2S)': r'\Upsilon(2S)', 'Upsilon(3S)': r'\Upsilon(3S)',
    'psi(4040)': r'\psi(4040)', 'psi(4415)': r'\psi(4415)', 'psi(4660)': r'\psi(4660)',
    'chi_c1(4140)': r'\chi_{c1}(4140)', 'chi_c0(4500)': r'\chi_{c0}(4500)', 'Upsilon(10753)': r'\Upsilon(10753)',
    'Upsilon(10860)': r'\Upsilon(10860)', 'Upsilon(11020)': r'\Upsilon(11020)',
}
for Q in 'cb':
    for n in (1, 2):
        for J in (0, 1, 2):
            TEXNAME[f'chi_{Q}{J}({n}P)'] = rf'\chi_{{{Q}{J}}}({n}P)'
HYBNAME = {'1(s/d)1': r'$1(s/d)_1$', '2(s/d)1': r'$2(s/d)_1$', '3(s/d)1': r'$3(s/d)_1$', '1p1': r'$1p_1$',
           '1p0': r'$1p_0$', '1(p/f)2': r'$1(p/f)_2$', '1d2': r'$1d_2$'}


def tn(s, star=False):
    if s in ('-', ''):
        return '--'
    return '$' + TEXNAME.get(s, s) + (r'^\ast' if star else '') + '$'


def jpc(s):
    return '$' + s.replace('--', '^{--}').replace('-+', '^{-+}').replace('+-', '^{+-}').replace('++', '^{++}') + '$'


summ = json.load(open(os.path.join(RES, 'hybrid_summary.json')))

# ----------------------------------------------------------------------------------------------- tab:H_fit
CASE = {'ref': r'reference ($\epsilon=0.3$)', 'eps0': r'$\epsilon=0$', 'M-': r'$M_{\rm lat}-1\sigma$',
        'M+': r'$M_{\rm lat}+1\sigma$', 'kappa-': r'$\kappa_c$ (max fit)', 'kappa+': r'$\kappa_c$ (min fit)'}
rows = []
fit = read('hybrid_fit.csv')
for case in CASE:
    rr = sorted([r for r in fit if r['case'] == case], key=lambda r: r['chi2'])
    for i, r in enumerate(rr):
        B = f"${r['B_c']:+.3f}^{{+{r['B_c_hi']:.3f}}}_{{-{r['B_c_lo']:.3f}}}$"
        rows.append(f"{CASE[case] if i == 0 else ''} & ${r['kappa_c']:+.4f}$ & ${r['branch']}$ & {B} & "
                    f"${r['B_b']:+.3f}$ & {r['G_Y->eta_c']:.0f} (${r['pull_Y->eta_c']:+.1f}$) & "
                    f"{r['G_eta_c1->J/psi']:.0f} (${r['pull_eta_c1->J/psi']:+.1f}$) & {r['R_model']:.2f} & "
                    f"{r['chi2']:.1f}/{int(r['ndf'])} \\\\")
    if case in ('ref', 'eps0'):
        rows.append(r'\hline')
write('tab_H_fit.tex', rows + [r'\hline'])

# ------------------------------------------------------------------------------------------- tab:H_lattice
LAB = {'Y->eta_c': r'$Y\to\gamma\eta_c$', 'Y->chi_c0': r'$Y\to\gamma\chi_{c0}$', 'Y->chi_c1': r'$Y\to\gamma\chi_{c1}$',
       'Y->chi_c2': r'$Y\to\gamma\chi_{c2}$', 'H1(0-+)->J/psi': r'$H_1(0^{-+})\to\gamma J/\psi$',
       'H1(0-+)->h_c': r'$H_1(0^{-+})\to\gamma h_c$', 'eta_c1->J/psi': r'$\eta_{c1}\to\gamma J/\psi$',
       'eta_c1->h_c': r'$\eta_{c1}\to\gamma h_c$', 'H1(2-+)->J/psi': r'$H_1(2^{-+})\to\gamma J/\psi$',
       'H1(2-+)->h_c': r'$H_1(2^{-+})\to\gamma h_c$'}
rows = []
for r in read('hybrid_lattice.csv'):
    lat = r['lat_keV']
    if isinstance(lat, float) and np.isfinite(lat):
        lat = f"{lat:.0f}({r['dlat_keV']:.0f})"
    elif isinstance(lat, str) and lat.startswith('<'):
        lat = f'$\\lesssim{lat[1:]}$'
    else:
        lat = '--'
    if np.isfinite(r['E_q']):
        rows.append(f"{LAB[r['channel']]} & {r['M_i']:.3f} & {1e3 * r['E_q']:.0f} & {lat} & {r['JS_keV']:.0f} & "
                    f"{r['G']:.0f} [{r['G_lo']:.0f}, {r['G_hi']:.0f}] & {g(r['Pi'])} & {g(r['Sigma'])} & {g(r['int'])} & "
                    f"{r['G_alt']:.0f} \\\\")
    else:
        rows.append(f"{LAB[r['channel']]} & {r['M_i']:.3f} & -- & {lat} & {r['JS_keV']:.0f} & "
                    r"\multicolumn{5}{c}{0 ($\Delta S=1$)} \\")
write('tab_H_lattice.tex', rows + [r'\hline'])

# ---------------------------------------------------------------------------------------------- tab:H_pred
pred = read('hybrid_predictions.csv')
for fl in ('c', 'b'):
    rows, last = [], None
    for r in pred:
        if r['flavor'] != fl or r['shown'] != 'True':
            continue
        first = r['hybrid'] != last
        if first and last is not None:
            rows.append(r'\hline')
        last = r['hybrid']
        mi = f"{r['M_i']:.3f}" + ('' if r['M_i_src'] == 'exp' else '$^\\ast$')
        br = '--' if not np.isfinite(r['BR']) else (f"{r['BR']:.1e}".replace('e-0', r'\times10^{-').replace('e-', r'\times10^{-') + '}')
        if br != '--':
            br = '$' + br + '$'
        gt = '--' if not np.isfinite(r['Gtot_MeV']) else f"{r['Gtot_MeV']:.0f}"
        head = (f"{HYBNAME[r['hybrid']]} & {jpc(r['JPC'])} & {tn(r['candidate'])} & {mi} & {gt} & "
                f"{100 * r['P_Sigma']:.0f}") if first else ' & & & & & '
        extra = f" & {g(r['G_eQ'])}" if fl == 'b' else ''
        rows.append(f"{head} & {tn(r['final'], r['M_f_src'] != 'exp')} & {1e3 * r['E_q']:.0f} & {g(r['Pi'])} & {g(r['Sigma'])} & {g(r['int'])} & "
                    f"{g(r['G'])} & [{g(r['G_lo'])}, {g(r['G_hi'])}] & {g(r['G_alt'])}{extra} & {br} \\\\")
    write(f'tab_H_pred_{fl}.tex', rows + [r'\hline'])

# --------------------------------------------------------------------------------------------- tab:H_spin1
sp = read('hybrid_spin1.csv')
JPCS1 = {'1(s/d)1': ['0^{-+}', '1^{-+}', '2^{-+}'], '1p1': ['0^{+-}', '1^{+-}', '2^{+-}'], '1p0': [None, '1^{+-}', None]}
FIN = {'c': {'1(s/d)1': ['J/psi', 'psi(2S)'], '1p1': ['chi_c0(1P)', 'chi_c1(1P)', 'chi_c2(1P)'],
             '1p0': ['chi_c0(1P)', 'chi_c1(1P)', 'chi_c2(1P)']},
       'b': {'1(s/d)1': ['Upsilon(1S)', 'Upsilon(2S)'], '1p1': ['chi_b0(1P)', 'chi_b1(1P)', 'chi_b2(1P)'],
             '1p0': ['chi_b0(1P)', 'chi_b1(1P)', 'chi_b2(1P)']}}
rows = []
for fl in ('c', 'b'):
    rows.append(r'\multicolumn{6}{l}{\textit{' + ('charmonium' if fl == 'c' else 'bottomonium') + r'}}\\')
    for h in ('1(s/d)1', '1p1', '1p0'):
        for i, f in enumerate(FIN[fl][h]):
            cells = []
            Eq = np.nan
            for Jcal in (0, 1, 2):
                m = [r for r in sp if r['flavor'] == fl and r['hybrid'] == h and int(r['Jcal']) == Jcal and r['final'] == f]
                if JPCS1[h][Jcal] is None:
                    cells.append('')
                elif m:
                    cells.append(g(m[0]['G']))
                    Eq = m[0]['E_q']
                else:
                    cells.append('0')
            hd = f"{HYBNAME[h]}" if i == 0 else ''
            rows.append(f"{hd} & {tn(f)} & {1e3 * Eq:.0f} & " + ' & '.join(cells) + r' \\')
        rows.append(r'\hline')
write('tab_H_spin1.tex', rows)

# -------------------------------------------------------------------------------------------- tab:H_besiii
rows = []
for r in read('hybrid_besiii.csv'):
    rows.append(f"{tn(r['candidate'])} & {HYBNAME[r['hybrid']]} & {r['sqrt_s']:.3f} & {r['sigma_UL_pb']:.1f} & "
                f"{r['Gamma_UL_keV']:.0f} & {g(r['Pi'])} & {g(r['Sigma'])} & {g(r['int'])} & {r['G']:.0f} "
                f"[{r['G_lo']:.0f}, {r['G_hi']:.0f}] & {r['G_alt']:.0f} & {r['ratio']:.2f} \\\\")
write('tab_H_besiii.tex', rows + [r'\hline'])
