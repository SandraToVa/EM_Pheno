"""Runs the full EM-transition phenomenology (EM_Pheno v3: one hybrid vertex
with its Pi_u and Sigma_u^- channels on the same footing, every hybrid width
split as Gamma_PiPi + Gamma_SigmaSigma + Gamma_int) and writes results/ (CSV, JSON).
Tables: make_tables.py; figures: figures.py; size estimates: r_moments.py.

    python3 run_all.py            (from the python/ folder)

Requires the wave-function files data/wf_*.mat produced by
matlab/exportWavefunctions.m.
"""
import os, sys, csv, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from emlib.states import Library
from emlib.data import DATA, MASS
from emlib.quarkonium import Fitter, predict_row
from emlib import hybrids as HY
from emlib import mixing as MX
from emlib.amplitudes import Params, width, photon_energy
from emlib import lattice as LT

ROOT = os.path.join(os.path.dirname(__file__), '..')
RES = os.path.join(ROOT, 'results'); FIG = os.path.join(ROOT, 'figures')
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)


def write_csv(name, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(os.path.join(RES, name), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f'{v:.6g}' if isinstance(v, float) else v) for k, v in r.items()})


summary = {}
libs = {'c': Library('c'), 'b': Library('b')}

# =========================================================== 1. quarkonium E1
print('== quarkonium E1 fit')
fits = []
pull_rows = []
for nonloc in (False, True):
    for fl, sets, tag in (('c', ('core',), 'core'), ('c', ('core', 'ext'), 'core+ext'), ('b', ('core',), 'core')):
        rows = [r for r in DATA if r['flavor'] == fl and r['set'] in sets]
        F = Fitter(libs, rows, nonloc)
        eps = F.eps_for_unit_chi2()
        k, c, dl, dh = F.fit_kappa(eps)
        k0, c0, dl0, dh0 = F.fit_kappa(0.3 if fl == 'c' else 0.1)
        fits.append({'flavor': fl, 'set': tag, 'nonlocal': nonloc, 'N_meas': len(F.meas), 'N_UL': len(F.uls),
                     'eps_fit': eps, 'kappa': k, 'dk_lo': dl, 'dk_hi': dh, 'chi2': c, 'ndf': F.ndf(),
                     'kappa_eps_v2': k0, 'dk_lo_v2': dl0, 'dk_hi_v2': dh0, 'chi2_eps_v2': c0})
        print(f"  {fl} {tag:9s} {'NL' if nonloc else 'L '} eps={eps:.3f} kappa={k:.3f} -{dl:.3f}+{dh:.3f}  "
              f"(eps=v^2: kappa={k0:.3f}, chi2/ndf={c0:.1f}/{F.ndf()})")
        if tag == 'core+ext' or fl == 'b':
            tot, chi, chiul, p, sth, pulls, ulp, mod = F.chi2(k, eps, detail=True)
            for a, kk in enumerate(F.meas):
                r = rows[kk]
                pull_rows.append({'nonlocal': nonloc, 'flavor': fl, 'id': r['id'], 'kind': r['kind'],
                                  'data': r['value'], 'err_exp': float(np.sqrt(F.Cexp[a, a])),
                                  'pred_kappa1': (F.pred1[kk][0] if r['kind'] == 'width' else F.pred1[kk][0] / F.pred1[kk][2]),
                                  'pred_fit': p[a], 'err_th': sth[a], 'pull': pulls[a], 'set': r['set']})
            for a, kk in enumerate(F.uls):
                r = rows[kk]
                pull_rows.append({'nonlocal': nonloc, 'flavor': fl, 'id': r['id'], 'kind': r['kind'],
                                  'data': r['value'], 'err_exp': 0.0,
                                  'pred_kappa1': (F.pred1[kk][0] if r['kind'] == 'ul_width' else F.pred1[kk][0] / F.pred1[kk][2]),
                                  'pred_fit': mod[kk][0], 'err_th': mod[kk][1], 'pull': ulp[a], 'set': r['set']})
write_csv('quarkonium_fits.csv', fits)
write_csv('quarkonium_pulls.csv', pull_rows)
fitd = {(f['flavor'], f['set'], f['nonlocal']): f for f in fits}
kc = fitd[('c', 'core', True)]; kb = fitd[('b', 'core', True)]
summary['kappa_c'] = (kc['kappa'], kc['dk_lo'], kc['dk_hi'], kc['eps_fit'])
summary['kappa_b'] = (kb['kappa'], kb['dk_lo'], kb['dk_hi'], kb['eps_fit'])

# X(3872) test: Z = Gamma_exp / Gamma_E1(chi_c1(2P)) at the fitted kappa
xr = [r for r in DATA if r['id'] == 'X(3872)->Jpsi'][0]
xp = {nl: predict_row(xr, libs, nl, 1.0)[0] for nl in (False, True)}
summary['X3872'] = {'Gamma_exp': xr['value'], 'err': xr['err'],
                    'Gamma_E1_local_k1': xp[False], 'Gamma_E1_nonlocal_k1': xp[True],
                    'Z_local': xr['value'] / (kc['kappa'] ** 2 * xp[False]),
                    'Z_nonlocal': xr['value'] / (kc['kappa'] ** 2 * xp[True])}

# =========================================================== 2. hybrids
print('== hybrids')
hyb_rows = []
for fl in ('c', 'b'):
    kap = summary['kappa_c'][0] if fl == 'c' else summary['kappa_b'][0]
    rows = HY.predictions(fl, kap)
    hyb_rows += rows
write_csv('hybrid_predictions.csv', hyb_rows)

# data constraints on kappa_H from BESIII gamma eta_c
ULs = HY.derived_ul_gamma_etac()
summary['ULs_gamma_etac'] = ULs
libc = libs['c']; gapc = HY.load_gaps('c')
hc = {h[0]: h for h in HY.HYBRIDS['c']}
fin_etac = [f for f in HY.FINALS['c'] if f[0] == 'eta_c(1S)'][0]
cons = []
for cand, hl in (('psi(4040)', '1(s/d)1'), ('psi(4415)', '2(s/d)1')):
    UL = ULs[cand]['Gamma_UL_keV']
    for v in HY.VARIANTS:
        sp, E = HY.gamma_split('c', libc, gapc, hc[hl], fin_etac, v, (1.0,) * 4)
        gB = HY.gamma('c', libc, gapc, hc[hl], fin_etac, v, (0.0, 1.0, 0.0, 1.0))[0]
        b_ = HY.channel_bounds(sp, UL)
        cons.append({'candidate': cand, 'hybrid': hl, 'variant': v, 'Gamma_k1_keV': sp['total'],
                     'Gamma_k1_Pi_keV': sp['Pi'], 'Gamma_k1_Sigma_keV': sp['Sigma'], 'Gamma_k1_int_keV': sp['int'],
                     'Gamma_mag_k1_keV': gB, 'Gamma_UL_keV': UL, **b_,
                     'kappa_mag_max': np.sqrt(UL / gB)})
        print(f"  {cand} -> gamma eta_c {v}: Gamma(k=1)={sp['total']:.1f} keV "
              f"[Pi {sp['Pi']:.1f}, Sigma {sp['Sigma']:.1f}, int {sp['int']:.1f}], UL {UL:.1f} -> kappa_H < {b_['kappa_H_max']:.3f}")
write_csv('hybrid_kappa_constraints.csv', cons)

# quadratic forms in c = (c1, c2, c3, c4)/(e e_c): Gamma = c^T Q c (keV)
planes = {}
for cand, hl in (('psi(4040)', '1(s/d)1'), ('psi(4415)', '2(s/d)1')):
    for v in ('A-NL', 'B-NL', 'A-L', 'B-L'):
        planes[(cand, v)] = HY.quadratic_form('c', libc, gapc, hc[hl], fin_etac, v)
np.save(os.path.join(RES, 'quadratic_forms.npy'), {f'{k[0]}|{k[1]}': v for k, v in planes.items()}, allow_pickle=True)

# channel plane (kappa_Pi, kappa_Sigma): Gamma = kappa_Pi^2 G_Pi + kappa_Sigma^2 G_Sigma + kappa_Pi kappa_Sigma G_int.
# For each ray kappa_Sigma = t kappa_Pi the bound is kappa_Pi^2 (G_Pi + t^2 G_Sigma + t G_int) < UL
pl_rows = []
cmap = {(r_['candidate'], r_['variant']): r_ for r_ in cons}
for v in ('A-NL', 'B-NL', 'A-L', 'B-L'):
    for t in (0.0, 0.25, 0.5, 1.0, 2.0):
        kmax = min(np.sqrt(ULs[cand]['Gamma_UL_keV'] /
                           (cmap[(cand, v)]['Gamma_k1_Pi_keV'] + t * t * cmap[(cand, v)]['Gamma_k1_Sigma_keV']
                            + t * cmap[(cand, v)]['Gamma_k1_int_keV']))
                   for cand in ('psi(4040)', 'psi(4415)'))
        pl_rows.append({'variant': v, 'kSigma_over_kPi': t, 'kappa_Pi_max': kmax})
    # largest |c4| allowed for c1=c2=c3=kappa_c (fitted E1 normalization): c4 is the
    # only coefficient not shared by the two channels
    kc0 = summary['kappa_c'][0]
    best = []
    for cand in ('psi(4040)', 'psi(4415)'):
        Q = planes[(cand, v)]; UL = ULs[cand]['Gamma_UL_keV']
        a = Q[3, 3]; b = 2 * kc0 * Q[3, :3].sum(); c = kc0 ** 2 * Q[:3, :3].sum() - UL
        disc = b * b - 4 * a * c
        best.append(((-b - np.sqrt(disc)) / (2 * a), (-b + np.sqrt(disc)) / (2 * a)) if disc > 0 else (np.nan, np.nan))
    lo = max(x[0] for x in best); hi = min(x[1] for x in best)
    summary.setdefault('c4_range_at_kappa_c', {})[v] = (lo / kc0, hi / kc0)
    pl_rows.append({'variant': v, 'kSigma_over_kPi': 'c4/c range at kappa_c', 'kappa_Pi_max': f'[{lo / kc0:.3f}, {hi / kc0:.3f}]'})
write_csv('hybrid_channel_plane.csv', pl_rows)
for r_ in pl_rows:
    print('   plane', r_)

# spin-1 hybrids (heavy-quark spin 1): explicit Jcal -> J' widths for 1(s/d)1 and 1p1
spin1 = []
for fl in ('c', 'b'):
    lib = libs[fl]; gp = HY.load_gaps(fl)
    kap = summary['kappa_c'][0] if fl == 'c' else summary['kappa_b'][0]
    hh = {h[0]: h for h in HY.HYBRIDS[fl]}
    for hl, finals in (('1(s/d)1', [(1, 0), (2, 0)]), ('1p1', [(1, 1)])):
        h = hh[hl]
        for (nf, lf) in finals:
            fin = [f for f in HY.FINALS[fl] if f[1] == nf and f[2] == lf][0]
            for Jcal in range(abs(h[1] - 1), h[1] + 2):
                for Jf in range(abs(lf - 1), lf + 2):
                    for v in ('A-NL', 'B-NL'):
                        sp, E = HY.gamma_split(fl, lib, gp, h, fin, v, (kap,) * 4, S=1, Jcal=Jcal, Jf=Jf)
                        spin1.append({'flavor': fl, 'hybrid': hl, 'Jcal_i': Jcal, 'final_n_l': f'{nf}{"SPDF"[lf]}',
                                      'J_f': Jf, 'variant': v, 'Gamma_keV': sp['total'], 'Pi_keV': sp['Pi'],
                                      'Sigma_keV': sp['Sigma'], 'int_keV': sp['int']})
write_csv('hybrid_spin1.csv', spin1)

# =========================================================== 3. mixing (charm)
print('== mixing')
kH = summary['kappa_c'][0]
mix_rows = MX.table(summary['kappa_c'][0], kH)
write_csv('mixing_1mm.csv', mix_rows)
# kappa_H limits from the mixed (mix=true) states
mixcons = []
ULmix = {'psi(4040)': ULs['psi(4040)']['Gamma_UL_keV'], 'psi(4415)': ULs['psi(4415)']['Gamma_UL_keV'],
         'psi(3770)': 8e-4 * 27.2e3}      # PDG 2026: B(psi(3770) -> gamma eta_c) < 8e-4, Gamma = 27.2 MeV
for r in mix_rows:
    if r['final'] == 'eta_c(1S)' and r['state'] in ULmix:
        for v in ('A-NL', 'B-NL'):
            mixcons.append({'state': r['state'], 'content': r['content'], 'variant': v,
                            'Gamma_at_kappaH_keV': r[v], 'Pi_keV': r[v + '_Pi'], 'Sigma_keV': r[v + '_Sigma'],
                            'int_keV': r[v + '_int'], 'kappa_H_used': kH, 'Gamma_UL_keV': ULmix[r['state']],
                            'kappa_H_max': kH * np.sqrt(ULmix[r['state']] / r[v])})
write_csv('mixing_kappa_constraints.csv', mixcons)
for r in mixcons:
    print('   mix-constraint', r['state'], r['variant'], f"kappa_H < {r['kappa_H_max']:.2f}")
hind = MX.hindered_psi2S_etac(kH)
write_csv('mixing_hindered.csv', hind)
for r in hind:
    print('  ', r)

# c4 scan: c1 = c2 = c3 = kappa_c, c4 = t kappa_c (g_B_par is the only LEF not fixed by the Pi_u channel)
print('== c4 scan')
scan = MX.c4_scan(kH, np.round(np.linspace(0.0, 1.4, 57), 4))
hcd = {h[0]: h for h in HY.HYBRIDS['c']}
for t in sorted(set(r_['c4_over_c'] for r_ in scan)):
    for cand, hl in (('psi(4040)', '1(s/d)1'), ('psi(4415)', '2(s/d)1')):
        for v in ('A-NL', 'B-NL'):
            g = HY.gamma('c', libc, gapc, hcd[hl], fin_etac, v, (kH, kH, kH, t * kH))[0]
            scan.append({'c4_over_c': t, 'state': cand + ' (unmixed)', 'variant': v, 'Gamma_keV': g})
write_csv('c4_scan.csv', scan)
# c4 window allowed by the mixed-state data: the hybrid-admixture contribution alone must not
# exceed the measured J/psi, psi(2S) -> gamma eta_c widths (+2 sigma; they also receive the M1
# contribution, not included) nor the psi(3770) limit
from emlib.mixing import GAMMA_ETAC_EXP
win = {}
for v in ('A-NL', 'B-NL'):
    ok = []
    for t in sorted(set(r_['c4_over_c'] for r_ in scan)):
        g = {r_['state']: r_['Gamma_keV'] for r_ in scan if r_['c4_over_c'] == t and r_['variant'] == v}
        good = (g['J/psi (mixed)'] < GAMMA_ETAC_EXP['J/psi'][0] + 2 * GAMMA_ETAC_EXP['J/psi'][1] and
                g['psi(2S) (mixed)'] < GAMMA_ETAC_EXP['psi(2S)'][0] + 2 * GAMMA_ETAC_EXP['psi(2S)'][1] and
                g['psi(3770) (mixed)'] < GAMMA_ETAC_EXP['psi(3770)'][1])
        if good:
            ok.append(t)
    win[v] = (min(ok), max(ok)) if ok else None
summary['c4_window_mixed_states'] = win
print('   c4/c window from J/psi, psi(2S), psi(3770) -> gamma eta_c:', win)

# =========================================================== 4. lattice (arXiv:2301.08213)
print('== lattice comparison')
rad = LT.radii_table(libs['c'])
write_csv('lattice_radii.csv', rad)
for r_ in rad:
    print('  ', r_)
row = [r for r in DATA if r['id'] == 'chic0->Jpsi'][0]
gL = predict_row(row, libs, False, 1.0)[0]; gN = predict_row(row, libs, True, 1.0)[0]
kc_, dkl, dkh, _ = summary['kappa_c']
summary['lattice'] = {'chic0_Jpsi': LT.chic0_comparison(gL, gN, kc_, 0.5 * (dkl + dkh), row['value'], row['err']),
                      'beta2': LT.beta2_from_sizes(libs['c']), 'radii': rad}
print('  ', summary['lattice']['chic0_Jpsi'])
print('  ', summary['lattice']['beta2'])

# hybrid channels computed on the lattice (Dudek 2009; Delaney 2023 thesis) and the
# constituent-gluon model of Jafarzade & Swanson (arXiv:2610.22088)
h_sd1 = [h for h in HY.HYBRIDS['c'] if h[0] == '1(s/d)1'][0]
lh = LT.hybrid_lattice_table(libs['c'], HY.load_gaps('c'), h_sd1, ('A-NL', 'A-L', 'B-NL', 'B-L'),
                             summary['kappa_c'][0])
write_csv('lattice_hybrids.csv', lh)
# channel strengths that reproduce both lattice widths of a data set
lsol = LT.channel_solutions(lh)
write_csv('lattice_channel_solutions.csv', lsol)
summary['lattice_channel_solutions'] = lsol
for r_ in lsol:
    print('   lattice channel solution', r_)
Es = np.round(np.arange(0.60, 1.501, 0.0125), 4)
write_csv('gamma_vs_Eq.csv', LT.gamma_vs_Eq(libs['c'], HY.load_gaps('c'), h_sd1, ('A-NL', 'B-NL'), Es))
for r_ in lh:
    if r_['variant'] in ('A-NL', 'B-NL'):
        print(f"   {r_['channel']:15s} {r_['dataset']:8s} {r_['variant']} E={r_['E_q']:.3f} "
              f"G(kc)={r_['Gamma_kappac_keV']:9.1f}  lat={r_['lat_keV']}  kH={r_['kappaH']:.3f}")

with open(os.path.join(RES, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=1, default=float)
print(json.dumps(summary, indent=1, default=float))
