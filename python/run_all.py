"""Runs the full EM-transition phenomenology and writes results/ (CSV, LaTeX
tables) and figures/.

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
libc = libs['c']; dVc = HY.load_dV('c')
hc = {h[0]: h for h in HY.HYBRIDS['c']}
fin_etac = [f for f in HY.FINALS['c'] if f[0] == 'eta_c(1S)'][0]
cons = []
for cand, hl in (('psi(4040)', '1(s/d)1'), ('psi(4415)', '2(s/d)1')):
    for v in HY.VARIANTS:
        g1, E = HY.gamma('c', libc, dVc, hc[hl], fin_etac, v, (1.0, 1.0, 1.0))
        gB = HY.gamma('c', libc, dVc, hc[hl], fin_etac, v, (0.0, 1.0, 0.0))[0]
        kmax = np.sqrt(ULs[cand]['Gamma_UL_keV'] / g1)
        cons.append({'candidate': cand, 'hybrid': hl, 'variant': v, 'Gamma_k1_keV': g1,
                     'Gamma_c2only_k1_keV': gB, 'Gamma_UL_keV': ULs[cand]['Gamma_UL_keV'],
                     'kappa_H_max': kmax, 'kappa_c2only_max': np.sqrt(ULs[cand]['Gamma_UL_keV'] / gB)})
        print(f"  {cand} -> gamma eta_c {v}: Gamma(k=1)={g1:.1f} keV, UL {ULs[cand]['Gamma_UL_keV']:.1f} -> kappa_H < {kmax:.2f}")
write_csv('hybrid_kappa_constraints.csv', cons)

# (c2, c13) plane: quadratic forms
planes = {}
for cand, hl in (('psi(4040)', '1(s/d)1'), ('psi(4415)', '2(s/d)1')):
    for v in ('A-NL', 'B-NL'):
        Q = HY.quadratic_form('c', libc, dVc, hc[hl], fin_etac, v)
        planes[(cand, v)] = Q
np.save(os.path.join(RES, 'quadratic_forms.npy'), {f'{k[0]}|{k[1]}': v for k, v in planes.items()}, allow_pickle=True)

# spin-1 hybrids (heavy-quark spin 1): explicit J -> J' widths for 1(s/d)1 and 1p1
spin1 = []
for fl in ('c', 'b'):
    lib = libs[fl]; dV = HY.load_dV(fl)
    kap = summary['kappa_c'][0] if fl == 'c' else summary['kappa_b'][0]
    hh = {h[0]: h for h in HY.HYBRIDS[fl]}
    for hl, finals in (('1(s/d)1', [(1, 0), (2, 0)]), ('1p1', [(1, 1)])):
        h = hh[hl]
        for (nf, lf) in finals:
            fin = [f for f in HY.FINALS[fl] if f[1] == nf and f[2] == lf][0]
            for Ji in range(abs(h[1] - 1), h[1] + 2):
                for Jf in range(abs(lf - 1), lf + 2):
                    for v in ('A-NL', 'B-NL'):
                        g, E = HY.gamma(fl, lib, dV, h, fin, v, (kap,) * 3, S=1, Ji=Ji, Jf=Jf)
                        spin1.append({'flavor': fl, 'hybrid': hl, 'Jcal_i': Ji, 'final_n_l': f'{nf}{"SPDF"[lf]}',
                                      'J_f': Jf, 'variant': v, 'Gamma_keV': g})
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
                            'Gamma_at_kappaH_keV': r[v], 'kappa_H_used': kH, 'Gamma_UL_keV': ULmix[r['state']],
                            'kappa_H_max': kH * np.sqrt(ULmix[r['state']] / r[v])})
write_csv('mixing_kappa_constraints.csv', mixcons)
for r in mixcons:
    print('   mix-constraint', r['state'], r['variant'], f"kappa_H < {r['kappa_H_max']:.2f}")
hind = MX.hindered_psi2S_etac(kH)
write_csv('mixing_hindered.csv', hind)
for r in hind:
    print('  ', r)

with open(os.path.join(RES, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=1, default=float)
print(json.dumps(summary, indent=1, default=float))
