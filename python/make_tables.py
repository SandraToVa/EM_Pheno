"""LaTeX tables for the phenomenology section, generated from results/*.csv
(no hand transcription of numbers). Main text: matched gaps (variants NL, L).
App. app:pheno_mixing: tab_mixing, tab_hindered. App. app:potential_gaps:
tab_potential_gaps (matched versus potential gaps)."""
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
                f"${r['kappa'] - 1:+.3f}^{{+{r['dk_hi']:.3f}}}_{{-{r['dk_lo']:.3f}}}$ & "
                f"${r['kappa_eps_v2'] - 1:+.3f}^{{+{r['dk_hi_v2']:.3f}}}_{{-{r['dk_lo_v2']:.3f}}}$ & "
                f"{r['chi2_eps_v2']:.1f}/{int(r['ndf'])} & "
                f"${r['kappa_eps0'] - 1:+.3f}^{{+{r['dk_hi_eps0']:.3f}}}_{{-{r['dk_lo_eps0']:.3f}}}$ & "
                f"{r['chi2_eps0']:.0f}/{int(r['ndf'])} \\\\")
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
DK = summ['dkappa_c'][0], summ['dkappa_b'][0]
DKF = {'c': summ['dkappa_c'][0], 'b': summ['dkappa_b'][0]}


def at_dk(r, key):
    """width of a hybrid_predictions row at kappa_H = dkappa_Q (keV)"""
    return DKF[r['flavor']] ** 2 * r[key]


def fsplit_dk(r, v):
    """Gamma_PiPi, Gamma_SigmaSigma, Gamma_int, Gamma at kappa_H = dkappa_Q (keV); '--' for an absent channel"""
    out = []
    for X in SPLIT:
        x = r[f'{v}_{X}']
        out.append('--' if abs(x) < 1e-6 * max(abs(r[v]), 1e-12) and abs(x) < 1e-9 else fmt(at_dk(r, f'{v}_{X}')))
    return out + [fmt(at_dk(r, v))]


rows = []
for fl in ('c', 'b'):
    d = hp[hp['flavor'] == fl]
    last = None
    for _, r in d.iterrows():
        gmax = d[d['hybrid'] == r['hybrid']]['NL'].abs().max()
        if max(abs(r['NL']), abs(r['NL_Pi']), abs(r['NL_Sigma'])) < 1e-3 * gmax:   # negligible channels of this state
            continue
        if r['hybrid'] != last:
            m = f"{r['M_i']:.3f}" + ('' if r['M_i_src'] == 'exp' else '$^*$')
            gt = '--' if np.isnan(r['Gtot_MeV']) else f"{r['Gtot_MeV']:.0f}"
            head_ = (f"{hyb_tex(r['hybrid'])} & "
                     f"{r['JPC'].replace('--', '$^{--}$').replace('++', '$^{++}$')} & {cand_tex(r['candidate'])} & {m} & {gt} & "
                     f"{100 * r['P_Sigma']:.0f}")
            last = r['hybrid']
        else:
            head_ = ' & & & & & '
        br = '--'
        if not np.isnan(r['Gtot_MeV']):
            x = at_dk(r, 'NL') / (r['Gtot_MeV'] * 1e3)
            mant, ex = f"{x:.1e}".split('e')
            br = f"${mant}\\times10^{{{int(ex)}}}$"
        rows.append(f"{head_} & {fin_tex(r['final'])} & {r['k_MeV']:.0f} & "
                    + ' & '.join(fsplit_dk(r, 'NL')) + f" & {fmt(at_dk(r, 'L'))} & {br} \\\\")
    rows.append(r'\hline')
with open(os.path.join(TEX, 'tab_hybrids.tex'), 'w') as f:
    f.write('\n'.join(rows[:-1]) + '\n\\hline\n')

# --------------------------------------------------------- Table: BESIII gamma eta_c limits at kappa_H = dkappa_c
hc = pd.read_csv(os.path.join(RES, 'hybrid_kappa_constraints.csv'))
k2 = summ['dkappa_c'][0] ** 2
rows = []
for _, g in hc[hc['variant'].isin(['NL', 'L'])].iterrows():
    G = k2 * g['Gamma_k1_keV']
    rows.append(f"{cand_tex(g['candidate'])} & {g['variant']} & "
                + ' & '.join(fmt(k2 * g[k]) for k in ('Gamma_k1_Pi_keV', 'Gamma_k1_Sigma_keV', 'Gamma_k1_int_keV', 'Gamma_k1_keV'))
                + f" & {fmt(g['Gamma_UL_keV'])} & {G / g['Gamma_UL_keV']:.1f} \\\\")
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
    if r['state'] == 'psi(2S)':
        continue
    cont = r['content'].replace('psi(1D)', '$\\psi(1D)$').replace('psi(3S)', '$\\psi(3S)$').replace('psi(2D)', '$\\psi(2D)$').replace('(s/d)1 hybrid', '$(s/d)_1$ hybrid')
    head = f"{cand_tex(r['state'])} & {cont} & {r['P_Q']:.3f}" if r['state'] != last else ' & & '
    last = r['state']
    rows.append(f"{head} & {mtex(r['final'])} & {r['k_MeV']:.0f} & {fmt(r['NL'])} & "
                + ' & '.join(dash(r['NL_' + X], r['NL']) for X in ('E1', 'Pi', 'Sigma', 'int')) + ' \\\\')
with open(os.path.join(TEX, 'tab_mixing.tex'), 'w') as f:
    f.write('\n'.join(rows) + '\n\\hline\n')
# --------------------------------------------------------- Table: hindered (hybrid admixtures)
hd = pd.read_csv(os.path.join(RES, 'mixing_hindered.csv'))
rows = []
for _, r in hd[hd['variant'] == 'NL'].iterrows():
    tr = r['transition'].replace('psi(2S)', r'$\psi(2S)$').replace('J/psi', r'$J/\psi$').replace('->gamma eta_c(1S)', r'$\to\gamma\eta_c(1S)$')
    rows.append(f"{tr} & {fmt(r['path_H_to_Q'])} & {fmt(r['path_Q_to_H'])} & "
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
k2 = summ['dkappa_c'][0] ** 2          # v4: c_i = dkappa_c e e_c
rows = []
for (ch, ds), d in lh.groupby(['channel', 'dataset'], sort=False):
    g = {r['variant']: r for _, r in d.iterrows()}
    r0 = d.iloc[0]
    lat = '--' if np.isnan(r0['lat_keV']) else f"{r0['lat_keV']:g}({r0['dlat_keV']:g})"
    js = '--' if (np.isnan(r0['JS_keV']) or ds in ('Nf3', 'Nf3-F0')) else f"{r0['JS_keV']:.0f}"
    vals = []
    vals += [fmt(k2 * g['NL'][k]) for k in ('Gamma_k1_Pi_keV', 'Gamma_k1_Sigma_keV', 'Gamma_k1_int_keV', 'Gamma_k1_keV')]
    vals += [fmt(k2 * g['L']['Gamma_k1_keV'])]
    rows.append(f"{chtex[ch]} & {dstex[ds]} & {r0['M_i']:.3f} & {r0['E_q']*1e3:.0f} & {lat} & {js} & " + ' & '.join(vals) + ' \\\\')
rows = []
for (ch, ds), d in lh.groupby(['channel', 'dataset'], sort=False):
    g = {r['variant']: r for _, r in d.iterrows()}
    r0 = d.iloc[0]
    lat = '--' if np.isnan(r0['lat_keV']) else f"{r0['lat_keV']:g}({r0['dlat_keV']:g})"
    js = '--' if (np.isnan(r0['JS_keV']) or ds in ('Nf3', 'Nf3-F0')) else f"{r0['JS_keV']:.0f}"
    vals = [fmt(k2 * g['NL'][k]) for k in ('Gamma_k1_Pi_keV', 'Gamma_k1_Sigma_keV', 'Gamma_k1_int_keV', 'Gamma_k1_keV')]
    vals += [fmt(k2 * g['L']['Gamma_k1_keV'])]
    rows.append(f"{chtex[ch]} & {dstex[ds]} & {r0['M_i']:.3f} & {r0['E_q']*1e3:.0f} & {lat} & {js} & " + ' & '.join(vals) + ' \\\\')
with open(os.path.join(TEX, 'tab_lattice_hybrids.tex'), 'w') as f:
    f.write('\n'.join(rows) + '\n\\hline\n')


def pick(ch, ds, v):
    return lh[(lh['channel'] == ch) & (lh['dataset'] == ds) & (lh['variant'] == v)].iloc[0]


def R(a, b, k='Gamma_k1_keV'):
    return a[k] / b[k]


rows = []
for v in ('NL', 'L'):
    yq, eq_ = pick('Y->eta_c', 'quenched', v), pick('eta_c1->J/psi', 'quenched', v)
    yu, yf, eu = pick('Y->eta_c', 'Nf3', v), pick('Y->eta_c', 'Nf3-F0', v), pick('eta_c1->J/psi', 'Nf3', v)
    rat = lambda x: k2 * x['Gamma_k1_keV'] / x['lat_keV']
    rows.append(f"{v} & {rat(yq):.1f} & {rat(eq_):.1f} & {R(eq_, yq):.2f} & "
                f"{rat(yu):.1f} ({rat(yf):.1f}) & {rat(eu):.1f} & {R(eu, yu):.2f} & "
                f"{R(eu, yu, 'Gamma_k1_Pi_keV'):.2f} & {R(eu, yu, 'Gamma_k1_Sigma_keV'):.2f} \\\\")
with open(os.path.join(TEX, 'tab_lattice_kappaH.tex'), 'w') as f:
    f.write('\n'.join(rows) + '\n\\hline\n')

# --------------------------------------------------------- Quarkonium: other lattice calculations
qe = pd.read_csv(os.path.join(RES, 'quarkonium_external.csv'))
qtex = {'chic0->Jpsi': r'$\chi_{c0}\to\gamma J/\psi$', 'chic1->Jpsi': r'$\chi_{c1}\to\gamma J/\psi$',
        'hc->etac': r'$h_c\to\gamma\eta_c$', 'chic2->Jpsi': r'$\chi_{c2}\to\gamma J/\psi$',
        'psi2(3823)->chic1': r'$\psi_2(3823)\to\gamma\chi_{c1}$'}
qsrc = {'chic0->Jpsi': r'HSC~\cite{Delaney:2023fsc}', 'chic1->Jpsi': r'ETMC~\cite{Becirevic:2026gaf}$^a$',
        'hc->etac': r'ETMC~\cite{Becirevic:2026gaf}$^b$', 'chic2->Jpsi': r'CLQCD~\cite{Li:2023zig}',
        'psi2(3823)->chic1': r'CLQCD~\cite{Li:2023zig}'}
rows = []
for _, r in qe.iterrows():
    dat = '--' if np.isnan(r['data_keV']) else f"{r['data_keV']:.0f}$\\pm${r['ddata_keV']:.0f}"
    rows.append(f"{qtex[r['id']]} & {qsrc[r['id']]} & {r['lat_keV']:g}({r['dlat_keV']:g}) & {r['LO_L_keV']:.0f} & {r['LO_NL_keV']:.0f} & "
                f"{r['fit_NL_keV']:.0f} & {dat} & {r['dkappa_eff']:+.3f} \\\\")
with open(os.path.join(TEX, 'tab_quarkonium_lattice.tex'), 'w') as f:
    f.write('\n'.join(rows) + '\n\\hline\n')

# --------------------------------------------------------- Hybrids: Coulomb-gauge model and eta_c1 lattice
he = pd.read_csv(os.path.join(RES, 'hybrid_external.csv'))
htex = {'Y(1--)->eta_c': r'$Y(1^{--})\to\gamma\eta_c$', 'H1(S=1)->J/psi': r'$(0,1,2)^{-+}\to\gamma J/\psi$',
        'eta_c1->J/psi': r'$\eta_{c1}(1^{-+})\to\gamma J/\psi$'}
rows = []
for _, r in he.iterrows():
    mod = '--' if np.isnan(r['model_lo_keV']) else f"{r['model_lo_keV']:.0f}--{r['model_hi_keV']:.0f}"
    br = '--' if np.isnan(r['Gtot_MeV']) else f"${r['NL'] / (r['Gtot_MeV'] * 1e3) * 1e4:.1f}\\times10^{{-4}}$"
    rows.append(f"{htex[r['channel']]} & {r['M_i']:.3f} & {r['E_q'] * 1e3:.0f} & {mod} & "
                + ' & '.join(fmt(r[k]) for k in ('NL_Pi', 'NL_Sigma', 'NL_int', 'NL', 'L')) + f" & {br} \\\\")
with open(os.path.join(TEX, 'tab_hybrid_models.tex'), 'w') as f:
    f.write('\n'.join(rows) + '\n\\hline\n')

# --------------------------------------------------------- App. potential gaps: matched vs potential (all at dkappa_c)
mx = pd.read_csv(os.path.join(RES, 'mixing_1mm.csv'))


def hp_row(fl, hyb, fin, v):
    r = hp[(hp['flavor'] == fl) & (hp['hybrid'] == hyb) & (hp['final'] == fin)].iloc[0]
    return ' / '.join(fsplit_dk(r, v))


def mixr(state, fin, v):
    r = mx[(mx['state'] == state) & (mx['final'] == fin)].iloc[0]
    return ' / '.join(dash(r[f'{v}_{X}'], r[v]) for X in ('Pi', 'Sigma', 'int')) + ' / ' + fmt(r[v])


rows = []      # the first heading row is in app_pheno.tex (\multicolumn cannot follow \input)
for fl, hyb, fin, lab_ in (('c', '1(s/d)1', 'eta_c(1S)', r'$1(s/d)_1\to\gamma\eta_c(1S)$ [$\psi(4040)$]'),
                           ('c', '2(s/d)1', 'eta_c(1S)', r'$2(s/d)_1\to\gamma\eta_c(1S)$ [$\psi(4415)$]'),
                           ('c', '3(s/d)1', 'eta_c(1S)', r'$3(s/d)_1\to\gamma\eta_c(1S)$ [$\psi(4660)$]'),
                           ('c', '1p1', 'h_c(1P)', r'$1p_1\to\gamma h_c(1P)$ [$\chi_{c1}(4140)$]'),
                           ('c', '1p0', 'h_c(1P)', r'$1p_0\to\gamma h_c(1P)$ [$\chi_{c0}(4500)$]')):
    rows.append(f"{lab_} & {hp_row(fl, hyb, fin, 'NL')} & {hp_row(fl, hyb, fin, 'pot-NL')} \\\\")
rows.append(r'\hline')
rows.append(r'\multicolumn{3}{l}{\emph{BESIII $\gamma\eta_c$ limits, Eq.~\eqref{eq:UL_etac}: $\Gamma/\Gamma_{\rm UL}$}}\\')
for cand in ('psi(4040)', 'psi(4415)'):
    vals = []
    for v in ('NL', 'pot-NL'):
        g = hc[(hc['candidate'] == cand) & (hc['variant'] == v)].iloc[0]
        vals.append(f"{k2 * g['Gamma_k1_keV'] / g['Gamma_UL_keV']:.1f}")
    rows.append(f"{cand_tex(cand)} & " + ' & '.join(vals) + ' \\\\')
rows.append(r'\hline')
rows.append(r'\multicolumn{3}{l}{\emph{lattice, $N_f=2+1$~\cite{Delaney:2023thesis} (quoted widths; lattice $R=1.78\pm0.23$)}}\\')
vals = []
for v in ('NL', 'pot-NL'):
    yu, eu = pick('Y->eta_c', 'Nf3', v), pick('eta_c1->J/psi', 'Nf3', v)
    vals.append(f"{R(eu, yu):.2f} / {k2 * yu['Gamma_k1_keV'] / yu['lat_keV']:.1f} / {k2 * eu['Gamma_k1_keV'] / eu['lat_keV']:.1f}")
rows.append(r"$R$ / $\Gamma(Y)/\Gamma_{\rm lat}$ / $\Gamma(\eta_{c1})/\Gamma_{\rm lat}$ & " + ' & '.join(vals) + ' \\\\')
rows.append(r'\hline')
rows.append(r'\multicolumn{3}{l}{\emph{mixed states (App.~\ref{app:pheno_mixing}), keV: $\Gamma_{\Pi\Pi}$ / $\Gamma_{\Sigma\Sigma}$ / $\Gamma_{\rm int}$ / $\Gamma$}}\\')
for state, fin, lab_ in (('psi(2S)', 'eta_c(1S)', r'$\psi(2S)\to\gamma\eta_c$ (RPP $1.05\pm0.15$)'),
                         ('psi(3770)', 'eta_c(1S)', r'$\psi(3770)\to\gamma\eta_c$ (BESIII $<5.5$)'),
                         ('psi(3770)', 'eta_c(2S)', r'$\psi(3770)\to\gamma\eta_c(2S)$ (BESIII $<29.4$)'),
                         ('psi(4230)', 'eta_c(1S)', r'$\psi(4230)\to\gamma\eta_c$')):
    rows.append(f"{lab_} & {mixr(state, fin, 'NL')} & {mixr(state, fin, 'pot-NL')} \\\\")
with open(os.path.join(TEX, 'tab_potential_gaps.tex'), 'w') as f:
    f.write('\n'.join(rows) + '\n\\hline\n')
# --------------------------------------------------------- Summary of the hybrid comparisons at kappa_H = dkappa_c
def lrow(ch, ds, v):
    x = pick(ch, ds, v)
    return k2 * x['Gamma_k1_keV'], x['lat_keV'], x['dlat_keV']
rows = []
for cand, lab_ in (('psi(4040)', r'$\psi(4040)\to\gamma\eta_c$'), ('psi(4415)', r'$\psi(4415)\to\gamma\eta_c$')):
    gN = hc[(hc['candidate'] == cand) & (hc['variant'] == 'NL')].iloc[0]; gL = hc[(hc['candidate'] == cand) & (hc['variant'] == 'L')].iloc[0]
    rows.append(f"{lab_} & BESIII, Eq.~\\eqref{{eq:UL_etac}} & {fmt(k2 * gN['Gamma_k1_keV'])} & {fmt(k2 * gL['Gamma_k1_keV'])} & $<{fmt(gN['Gamma_UL_keV'])}$ & {k2 * gN['Gamma_k1_keV'] / gN['Gamma_UL_keV']:.1f} \\\\")
for ch, ds, lab_, src in (('Y->eta_c', 'quenched', r'$Y\to\gamma\eta_c$', r'quenched~\cite{Dudek:2009kk}'),
                          ('eta_c1->J/psi', 'quenched', r'$\eta_{c1}\to\gamma J/\psi$', r'quenched~\cite{Dudek:2009kk}'),
                          ('Y->eta_c', 'Nf3', r'$Y\to\gamma\eta_c$', r'$N_f{=}2{+}1$~\cite{Delaney:2023thesis}'),
                          ('Y->eta_c', 'Nf3-F0', r'$Y\to\gamma\eta_c$', r'$N_f{=}2{+}1$, from $F(0)$'),
                          ('eta_c1->J/psi', 'Nf3', r'$\eta_{c1}\to\gamma J/\psi$', r'$N_f{=}2{+}1$~\cite{Delaney:2023thesis}')):
    gN, G, dG = lrow(ch, ds, 'NL'); gL = lrow(ch, ds, 'L')[0]
    rows.append(f"{lab_} & {src} & {fmt(gN)} & {fmt(gL)} & {G:g}({dG:g}) & {gN / G:.1f} \\\\")
yN, eN = pick('Y->eta_c', 'Nf3', 'NL'), pick('eta_c1->J/psi', 'Nf3', 'NL')
yL, eL = pick('Y->eta_c', 'Nf3', 'L'), pick('eta_c1->J/psi', 'Nf3', 'L')
rows.append(f"$R$, Eq.~\\eqref{{eq:R_lattice}} & $N_f{{=}}2{{+}}1$~\\cite{{Delaney:2023thesis}} & {R(eN, yN):.2f} & {R(eL, yL):.2f} & 1.78(23) & {R(eN, yN) / 1.78:.2f} \\\\")
for _, r in he[he['model_lo_keV'].notna()].iterrows():
    lab_ = htex[r['channel']]
    rows.append(f"{lab_} ($M=4.35$~GeV) & GYS model~\\cite{{Guo:2014zva}} & {fmt(r['NL'])} & {fmt(r['L'])} & {r['model_lo_keV']:.0f}--{r['model_hi_keV']:.0f} & "
                f"{r['NL'] / r['model_hi_keV']:.0f}--{r['NL'] / r['model_lo_keV']:.0f} \\\\")
for state, fin, lab_, lim, dlim, src in (('psi(2S)', 'eta_c(1S)', r'$\psi(2S)\to\gamma\eta_c$', 1.05, 0.15, 'RPP 2026'),
                                         ('psi(3770)', 'eta_c(1S)', r'$\psi(3770)\to\gamma\eta_c$', 5.5, None, r'BESIII~\cite{BESIII:2026psi3770}'),
                                         ('psi(3770)', 'eta_c(2S)', r'$\psi(3770)\to\gamma\eta_c(2S)$', 29.4, None, r'BESIII~\cite{BESIII:2026psi3770}')):
    r = mx[(mx['state'] == state) & (mx['final'] == fin)].iloc[0]
    d = f"$<{lim:g}$" if dlim is None else f"{lim:g}({dlim * 100:.0f})"
    rows.append(f"{lab_} (mixing, App.~\\ref{{app:pheno_mixing}}) & {src} & {fmt(r['NL'])} & -- & {d} & {r['NL'] / lim:.1g} \\\\")
rj = hd[(hd['variant'] == 'NL') & hd['transition'].str.startswith('J/psi')].iloc[0]
rows.append(f"$J/\\psi\\to\\gamma\\eta_c$ (mixing, App.~\\ref{{app:pheno_mixing}}) & BESIII~\\cite{{BESIII:2025jpsietac}} & {fmt(rj['coherent'])} & -- & 2.12(18) & {rj['coherent'] / 2.12:.1g} \\\\")
with open(os.path.join(TEX, 'tab_hybrid_summary.tex'), 'w') as f:
    f.write('\n'.join(rows) + '\n\\hline\n')
print('tables written to', TEX)
