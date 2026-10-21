"""LaTeX tables for the phenomenology section, generated from results/*.csv
(no hand transcription of numbers)."""
import os, json, re
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), '..')
RES = os.path.join(ROOT, 'results')
# tables go next to sec_pheno.tex: <EM transitions>/TeX file/tex_pheno if it
# exists (layout of the EM transitions folder), otherwise EM_Pheno/tex
TEX = os.path.join(ROOT, '..', 'TeX file', 'tex_pheno')
if not os.path.isdir(TEX):
    TEX = os.path.join(ROOT, 'tex')
os.makedirs(TEX, exist_ok=True)
summ = json.load(open(os.path.join(RES, 'summary.json')))


def lab(s):
    rep = [('psi2_3823', r'\psi_2(3823)'), ('psi2(3823)', r'\psi_2(3823)'), ('psi2S', r'\psi(2S)'),
           ('psi3770', r'\psi(3770)'), ('Jpsi', r'J/\psi'), ('hc', r'h_c'), ('etac', r'\eta_c'),
           ('Y2S', r'\Upsilon(2S)'), ('Y3S', r'\Upsilon(3S)'), ('Y1S', r'\Upsilon(1S)'),
           ('hb(2P)', r'h_b(2P)'), ('etab1S', r'\eta_b(1S)'), ('etab2S', r'\eta_b(2S)'),
           ('X(3872)', r'X(3872)')]
    s = re.sub(r'chi([cb])(\d)', r'\\chi_{\1\2}', s)
    for a, b in rep:
        s = s.replace(a, b)
    s = s.replace('->', r'\to\gamma ').replace(': ', r':\ ')
    return '$' + s + '$'


def fmt(x, err=None):
    if err is None:
        if abs(x) >= 100: return f'{x:.0f}'
        if abs(x) >= 10: return f'{x:.1f}'
        if abs(x) >= 1: return f'{x:.2f}'
        if abs(x) >= 0.01: return f'{x:.3f}'
        if abs(x) < 1e-6: return '0'
        m, e = f'{x:.1e}'.split('e')
        return f'${m}\\times10^{{{int(e)}}}$'
    d = max(0, -int(np.floor(np.log10(err))) + 1) if err > 0 else 2
    d = min(d, 4)
    return f'{x:.{d}f}({int(round(err * 10 ** d))})' if err > 0 else f'{x:.{d}f}'


# --------------------------------------------------------- Table: E1 data / fit
pulls = pd.read_csv(os.path.join(RES, 'quarkonium_pulls.csv'))
rows = []
for nl in (True,):
    p = pulls[pulls['nonlocal'] == nl]
    for _, r in p.iterrows():
        kind = r['kind']
        if kind in ('width', 'ratio'):
            dat = f"{fmt(r['data'])} $\\pm$ {fmt(r['err_exp'])}"
        else:
            dat = f"$<${fmt(r['data'])}"
        rows.append(f"{lab(r['id'])} & {r['set']} & {dat} & {fmt(r['pred_kappa1'])} & "
                    f"{fmt(r['pred_fit'])} $\\pm$ {fmt(r['err_th'])} & {r['pull']:+.2f} \\\\")
with open(os.path.join(TEX, 'tab_E1_fit.tex'), 'w') as f:
    f.write('\n'.join(rows) + '\n\\hline\n')

fits = pd.read_csv(os.path.join(RES, 'quarkonium_fits.csv'))
rows = []
for _, r in fits.iterrows():
    rows.append(f"${'c' if r['flavor'] == 'c' else 'b'}$ & {r['set']} & {'non-local' if r['nonlocal'] else 'local'} & "
                f"{int(r['N_meas'])}+{int(r['N_UL'])} & {r['eps_fit']:.3f} & "
                f"${r['kappa']:.3f}^{{+{r['dk_hi']:.3f}}}_{{-{r['dk_lo']:.3f}}}$ & "
                f"${r['kappa_eps_v2']:.3f}^{{+{r['dk_hi_v2']:.3f}}}_{{-{r['dk_lo_v2']:.3f}}}$ & "
                f"{r['chi2_eps_v2']:.1f}/{int(r['ndf'])} \\\\")
with open(os.path.join(TEX, 'tab_kappa.tex'), 'w') as f:
    f.write('\n'.join(rows) + '\n\\hline\n')

# --------------------------------------------------------- Table: hybrid predictions
hp = pd.read_csv(os.path.join(RES, 'hybrid_predictions.csv'))
fin_tex = lambda s: '$' + s.replace('eta_c2', r'\eta_{c2}').replace('eta_b2', r'\eta_{b2}').replace('eta_c', r'\eta_c')\
    .replace('eta_b', r'\eta_b').replace('h_c3', 'h_{c3}').replace('h_b3', 'h_{b3}') + '$'
cand_tex = lambda s: '$' + s.replace('Upsilon', '@U@').replace('psi', r'\psi').replace('@U@', r'\Upsilon').replace('chi_c1', r'\chi_{c1}')\
    .replace('chi_c0', r'\chi_{c0}') + '$' if s != '-' else '--'
rows = []
for fl in ('c', 'b'):
    d = hp[hp['flavor'] == fl]
    last = None
    for _, r in d.iterrows():
        if r['hybrid'] == '1p0':
            continue
        if max(r['A-NL'], r['B-NL'], r['A-L'], r['B-L']) < 0.5:
            continue
        if r['hybrid'] != last:
            m = f"{r['M_i']:.3f}" + ('' if r['M_i_src'] == 'exp' else '$^*$')
            gt = '--' if np.isnan(r['Gtot_MeV']) else f"{r['Gtot_MeV']:.0f}"
            head = (f"${r['hybrid'].replace('(s/d)', '(s/d)_').replace('p1', 'p_1').replace('d2', 'd_2').replace('(p/f)2', '(p/f)_2')}$ & "
                    f"{r['JPC'].replace('--', '$^{--}$').replace('++', '$^{++}$')} & {cand_tex(r['candidate'])} & {m} & {gt}")
            last = r['hybrid']
        else:
            head = ' & & & & '
        br = '--'
        if not np.isnan(r['Gtot_MeV']):
            x = r['B-NL'] / (r['Gtot_MeV'] * 1e3)
            mant, ex = f"{x:.1e}".split('e')
            br = f"${mant}\\times10^{{{int(ex)}}}$"
        rows.append(f"{head} & {fin_tex(r['final'])} & {r['k_MeV']:.0f} & {fmt(r['c2only_A-NL'])} & "
                    f"{fmt(r['A-L'])} & {fmt(r['A-NL'])} & {fmt(r['B-L'])} & {fmt(r['B-NL'])} & {br} \\\\")
    rows.append(r'\hline')
with open(os.path.join(TEX, 'tab_hybrids.tex'), 'w') as f:
    f.write('\n'.join(rows[:-1]) + '\n\\hline\n')

# --------------------------------------------------------- Table: kappa_H constraints
hc = pd.read_csv(os.path.join(RES, 'hybrid_kappa_constraints.csv'))
rows = []
for _, r in hc.iterrows():
    rows.append(f"{cand_tex(r['candidate'])} & {r['variant']} & {fmt(r['Gamma_c2only_k1_keV'])} & "
                f"{fmt(r['Gamma_k1_keV'])} & {fmt(r['Gamma_UL_keV'])} & {r['kappa_H_max']:.2f} & {r['kappa_c2only_max']:.2f} \\\\")
with open(os.path.join(TEX, 'tab_kappaH.tex'), 'w') as f:
    f.write('\n'.join(rows) + '\n\\hline\n')

# --------------------------------------------------------- Table: mixing
mx = pd.read_csv(os.path.join(RES, 'mixing_1mm.csv'))
mtex = lambda s: '$' + s.replace('eta_c', r'\eta_c').replace('chi_c', r'\chi_{c').replace('(1P)', '}(1P)')\
    .replace('(2P)=X(3872)', '}(2P)=X(3872)').replace('(2P)', '}(2P)').replace('}}', '}') + '$'
rows = []
last = None
for _, r in mx.iterrows():
    if r['state'] in ('psi(2S)', 'psi(3770)'):
        continue
    cont = r['content'].replace('psi(3S)', '$\\psi(3S)$').replace('psi(2D)', '$\\psi(2D)$').replace('(s/d)1 hybrid', '$(s/d)_1$ hybrid')
    head = f"{cand_tex(r['state'])} & {cont} & {r['P_Q']:.3f}" if r['state'] != last else ' & & '
    last = r['state']
    rows.append(f"{head} & {mtex(r['final'])} & {r['k_MeV']:.0f} & {fmt(r['A-NL'])} & {fmt(r['B-NL'])} \\\\")
with open(os.path.join(TEX, 'tab_mixing.tex'), 'w') as f:
    f.write('\n'.join(rows) + '\n\\hline\n')
print('tables written to', TEX)
print(open(os.path.join(TEX, 'tab_mixing.tex')).read()[:1500])
