"""LaTeX tables for the phenomenology section, generated from results/*.csv
(no hand transcription of numbers)."""
import os, json, re
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), '..')
RES = os.path.join(ROOT, 'results')
# tables go next to sec_pheno.tex: <EM transitions>/TeX file/tex_pheno if it
# exists (layout of the EM transitions folder), otherwise EM_Pheno/tex
TEX = os.environ.get('EMPHENO_TEX', os.path.join(ROOT, '..', 'TeX file', 'tex_pheno'))
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
hyb_tex = lambda s: '$' + s.replace('(s/d)', '(s/d)_').replace('(p/f)2', '(p/f)_2').replace('p1', 'p_1')\
    .replace('d2', 'd_2').replace('p0', 'p_0') + '$'
SPLIT = ('Pi', 'Sigma', 'int')
def fsplit(r, v):
    """Gamma_PiPi, Gamma_SigmaSigma, Gamma_int, Gamma for variant v (keV); '--' for an absent channel"""
    out = []
    for X in SPLIT:
        x = r[f'{v}_{X}']
        out.append('--' if abs(x) < 1e-6 * max(abs(r[v]), 1e-12) and abs(x) < 1e-9 else fmt(x))
    return out + [fmt(r[v])]
cols = ['A-NL', 'B-NL']
rows = []
for fl in ('c', 'b'):
    d = hp[hp['flavor'] == fl]
    last = None
    for _, r in d.iterrows():
        if max(max(abs(r[c]), abs(r[c + '_Pi']), abs(r[c + '_Sigma'])) for c in cols) < 0.5:
            continue
        if r['hybrid'] != last:
            m = f"{r['M_i']:.3f}" + ('' if r['M_i_src'] == 'exp' else '$^*$')
            gt = '--' if np.isnan(r['Gtot_MeV']) else f"{r['Gtot_MeV']:.0f}"
            head = (f"{hyb_tex(r['hybrid'])} & "
                    f"{r['JPC'].replace('--', '$^{--}$').replace('++', '$^{++}$')} & {cand_tex(r['candidate'])} & {m} & {gt} & "
                    f"{100 * r['P_Sigma']:.0f}")
            last = r['hybrid']
        else:
            head = ' & & & & & '
        br = '--'
        if not np.isnan(r['Gtot_MeV']):
            x = r['B-NL'] / (r['Gtot_MeV'] * 1e3)
            mant, ex = f"{x:.1e}".split('e')
            br = f"${mant}\\times10^{{{int(ex)}}}$"
        rows.append(f"{head} & {fin_tex(r['final'])} & {r['k_MeV']:.0f} & "
                    + ' & '.join(sum((fsplit(r, c) for c in cols), [])) + f" & {br} \\\\")
    rows.append(r'\hline')
with open(os.path.join(TEX, 'tab_hybrids.tex'), 'w') as f:
    f.write('\n'.join(rows[:-1]) + '\n\\hline\n')

# --------------------------------------------------------- Table: kappa_H constraints
hc = pd.read_csv(os.path.join(RES, 'hybrid_kappa_constraints.csv'))
rows = []
for _, g in hc.iterrows():
    rows.append(f"{cand_tex(g['candidate'])} & {g['variant']} & {fmt(g['Gamma_mag_k1_keV'])} & "
                + ' & '.join(fmt(g[k]) for k in ('Gamma_k1_Pi_keV', 'Gamma_k1_Sigma_keV', 'Gamma_k1_int_keV', 'Gamma_k1_keV'))
                + f" & {fmt(g['Gamma_UL_keV'])} & "
                + ' & '.join(f"{g[k]:.2f}" for k in ('kappa_H_max', 'kappa_Pi_max', 'kappa_Sigma_max')) + ' \\\\')
with open(os.path.join(TEX, 'tab_kappaH.tex'), 'w') as f:
    f.write('\n'.join(rows) + '\n\\hline\n')

# --------------------------------------------------------- Table: mixing
mx = pd.read_csv(os.path.join(RES, 'mixing_1mm.csv'))
mtex = lambda s: '$' + s.replace('eta_c', r'\eta_c').replace('chi_c', r'\chi_{c').replace('(1P)', '}(1P)')\
    .replace('(2P)=X(3872)', '}(2P)=X(3872)').replace('(2P)', '}(2P)').replace('}}', '}') + '$'
rows = []
last = None
dash = lambda x, ref: '--' if abs(x) < 1e-6 * max(abs(ref), 1e-12) else fmt(x)
for _, r in mx.iterrows():
    if r['state'] in ('psi(2S)', 'psi(3770)'):
        continue
    cont = r['content'].replace('psi(3S)', '$\\psi(3S)$').replace('psi(2D)', '$\\psi(2D)$').replace('(s/d)1 hybrid', '$(s/d)_1$ hybrid')
    head = f"{cand_tex(r['state'])} & {cont} & {r['P_Q']:.3f}" if r['state'] != last else ' & & '
    last = r['state']
    rows.append(f"{head} & {mtex(r['final'])} & {r['k_MeV']:.0f} & {fmt(r['A-NL'])} & {fmt(r['B-NL'])} & "
                + ' & '.join(dash(r['B-NL_' + X], r['B-NL']) for X in ('E1', 'Pi', 'Sigma', 'int')) + ' \\\\')
with open(os.path.join(TEX, 'tab_mixing.tex'), 'w') as f:
    f.write('\n'.join(rows) + '\n\\hline\n')
# --------------------------------------------------------- Table: hindered (hybrid admixtures)
hd = pd.read_csv(os.path.join(RES, 'mixing_hindered.csv'))
rows = []
for _, r in hd.iterrows():
    tr = r['transition'].replace('psi(2S)', r'$\psi(2S)$').replace('J/psi', r'$J/\psi$').replace('->gamma eta_c(1S)', r'$\to\gamma\eta_c(1S)$')
    rows.append(f"{tr} & {r['variant']} & {fmt(r['path_H_to_Q'])} & {fmt(r['path_Q_to_H'])} & "
                + ' & '.join(fmt(r[k]) for k in ('coherent_Pi', 'coherent_Sigma', 'coherent_int', 'coherent')) + ' \\\\')
with open(os.path.join(TEX, 'tab_hindered.tex'), 'w') as f:
    f.write('\n'.join(rows) + '\n\\hline\n')

# --------------------------------------------------------- Table: lattice comparison
lat = summ['lattice']
rows = []
for r in lat['radii']:
    st = {'eta_c(1S)': r'$\eta_c(1S)$', 'J/psi': r'$J/\psi$', 'eta_c(2S)': r'$\eta_c(2S)$', 'chi_c0(1P)': r'$\chi_{c0}(1P)$'}[r['state']]
    rows.append(f"{st} & {r['nl']} & {r['r_rms_fm']:.3f} & {r['r_ch_ours_fm']:.3f} & {r['r_ch_lat_fm']:.3f}({int(round(r['err_lat_fm'] * 1000))}) & {r['ratio']:.2f} \\\\")
with open(os.path.join(TEX, 'tab_lattice_radii.tex'), 'w') as f:
    f.write('\n'.join(rows) + '\n\\hline\n')
c0 = lat['chic0_Jpsi']
rows = [f"LO, local, $\\kappa_c=1$ & {c0['Gamma_LO_local_keV']:.0f} \\\\",
        f"LO, non-local, $\\kappa_c=1$ & {c0['Gamma_LO_nonlocal_keV']:.0f} \\\\",
        f"fit, non-local, $\\kappa_c={summ['kappa_c'][0]:.3f}$ & {c0['Gamma_fit_keV']:.0f}$\\pm${c0['dGamma_fit_keV']:.0f} \\\\",
        f"lattice~\\cite{{Delaney:2023fsc}}, experimental masses & {c0['Gamma_lattice_keV']:.1f}(3) \\\\",
        f"lattice~\\cite{{Delaney:2023fsc}}, lattice masses & {c0['Gamma_lattice_latmasses_keV']:.1f}(2) \\\\",
        f"RPP 2026 & {c0['Gamma_PDG_keV']:.0f}$\\pm${c0['dGamma_PDG_keV']:.0f} \\\\"]
with open(os.path.join(TEX, 'tab_lattice_chic0.tex'), 'w') as f:
    f.write('\n'.join(rows) + '\n\\hline\n')
# --------------------------------------------------------- Tables: lattice hybrids
lh = pd.read_csv(os.path.join(RES, 'lattice_hybrids.csv'))
chtex = {'Y->eta_c': r'$Y(1^{--})\to\gamma\eta_c$', 'eta_c1->J/psi': r'$\eta_{c1}(1^{-+})\to\gamma J/\psi$',
         'H1(0-+)->J/psi': r'$H_1(0^{-+})\to\gamma J/\psi$', 'H1(2-+)->J/psi': r'$H_1(2^{-+})\to\gamma J/\psi$'}
dstex = {'quenched': r'quenched~\cite{Dudek:2009kk}', 'Nf3': r'$N_f{=}2{+}1$, quoted~\cite{Delaney:2023thesis}',
         'Nf3-F0': r'$N_f{=}2{+}1$, from $F(0)$', 'JS-kin': '--'}
k2 = summ['kappa_c'][0] ** 2
rows = []
for (ch, ds), d in lh.groupby(['channel', 'dataset'], sort=False):
    g = {r['variant']: r for _, r in d.iterrows()}
    r0 = d.iloc[0]
    lat = '--' if np.isnan(r0['lat_keV']) else f"{r0['lat_keV']:g}({r0['dlat_keV']:g})"
    js = '--' if (np.isnan(r0['JS_keV']) or ds in ('Nf3', 'Nf3-F0')) else f"{r0['JS_keV']:.0f}"
    vals = []
    for v in ('A-NL', 'B-NL'):
        vals += [fmt(k2 * g[v][k]) for k in ('Gamma_k1_Pi_keV', 'Gamma_k1_Sigma_keV', 'Gamma_k1_int_keV', 'Gamma_k1_keV')]
    rows.append(f"{chtex[ch]} & {dstex[ds]} & {r0['M_i']:.3f} & {r0['E_q']*1e3:.0f} & {lat} & {js} & " + ' & '.join(vals) + ' \\\\')
with open(os.path.join(TEX, 'tab_lattice_hybrids.tex'), 'w') as f:
    f.write('\n'.join(rows) + '\n\\hline\n')
rows = []
def pick(ch, ds, v):
    return lh[(lh['channel'] == ch) & (lh['dataset'] == ds) & (lh['variant'] == v)].iloc[0]
def R(a, b, k='Gamma_k1_keV'):
    return a[k] / b[k]
for v in ('A-NL', 'A-L', 'B-NL', 'B-L'):
    yq, eq_ = pick('Y->eta_c', 'quenched', v), pick('eta_c1->J/psi', 'quenched', v)
    yu, yf, eu = pick('Y->eta_c', 'Nf3', v), pick('Y->eta_c', 'Nf3-F0', v), pick('eta_c1->J/psi', 'Nf3', v)
    rows.append(f"{v} & {yq['kappaH']:.2f} & {eq_['kappaH']:.2f} & {R(eq_, yq):.2f} & "
                f"{yu['kappaH']:.2f} ({yf['kappaH']:.2f}) & {eu['kappaH']:.2f} & {R(eu, yu):.2f} & "
                f"{R(eu, yu, 'Gamma_k1_Pi_keV'):.2f} & {R(eu, yu, 'Gamma_k1_Sigma_keV'):.2f} \\\\")
with open(os.path.join(TEX, 'tab_lattice_kappaH.tex'), 'w') as f:
    f.write('\n'.join(rows) + '\n\\hline\n')
ls_ = pd.read_csv(os.path.join(RES, 'lattice_channel_solutions.csv'))
dsn = {'quenched': 'quenched', 'Nf3': r'$N_f{=}2{+}1$, quoted', 'Nf3-F0': r'$N_f{=}2{+}1$, $Y$ from $F(0)$'}
rows = []
for (v, ds), d in ls_.groupby(['variant', 'Y_dataset'], sort=False):
    sol = '; '.join(f"$({r['kappa_Pi']:.2f},\\,{r['kappa_Sigma']:+.2f})$" for _, r in d.iterrows())
    rows.append(f"{v} & {dsn[ds]} & {sol} \\\\")
with open(os.path.join(TEX, 'tab_lattice_channels.tex'), 'w') as f:
    f.write('\n'.join(rows) + '\n\\hline\n')
print('tables written to', TEX)
print(open(os.path.join(TEX, 'tab_mixing.tex')).read()[:1500])
