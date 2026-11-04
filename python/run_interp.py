"""EM_Pheno v6: quarkonium E1 fits with the interpolated LEF of Sec. sec:pheno_E1_interpol,

    g_E1(r) = e r [e_Q + kappa_Q f(r)],   f(r) = (r/r0)^n / (1 + (r/r0)^n),

plus the O(k/m_Q) (1/m) corrections, Delta l term only since v7 (quarkonium.rel_delta), applied to the
whole amplitude at all r. kappa_Q is fitted separately for charm and bottom, with the three
theory-error options of v5: epsilon from chi2/ndf = 1, epsilon = v^2 (0.3 charm, 0.1 bottom),
epsilon = 0. Shapes: n = 2 (baseline) and n = 7, r0 = sqrt(pi/(6 sigma)) = 1.58 GeV^-1;
r0 = 1.0 GeV^-1 and n = 50 (step limit) as checks.

    python3 run_interp.py      (from python/)   ->  results/interp_*.csv, results/interp_summary.json
    python3 make_tables_interp.py               ->  tab_kappa_interp.tex, tab_E1_interp.tex,
                                                    tab_lattice_chic0_v6.tex, tab_quarkonium_lattice_v6.tex
"""
import os, sys, csv, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from emlib.states import Library
from emlib.data import DATA
from emlib.interp import InterpFitter, predict_parts, quad, mean_f, f_hill, R0_EST, HBARC
from emlib.amplitudes import CHARGE

ROOT = os.path.join(os.path.dirname(__file__), '..')
RES = os.path.join(ROOT, 'results')
os.makedirs(RES, exist_ok=True)


def write_csv(name, rows):
    keys = list(rows[0].keys())
    with open(os.path.join(RES, name), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f'{v:.6g}' if isinstance(v, float) else v) for k, v in r.items()})


libs = {'c': Library('c'), 'b': Library('b')}
EPS_V2 = {'c': 0.3, 'b': 0.1}
SETS = (('c', ('core',), 'core'), ('c', ('core', 'ext'), 'core+ext'), ('b', ('core',), 'core'))
SHAPES = [(2, R0_EST, 'main'), (7, R0_EST, 'main'), (2, 1.0, 'check'), (7, 1.0, 'check'), (50, R0_EST, 'check')]
summary = {'r0_GeVinv': R0_EST, 'r0_fm': R0_EST * HBARC, 'shapes': SHAPES}

print('== E1 fits with g_E1 = e r [e_Q + kappa_Q f(r)] and 1/m corrections')
fits, pull_rows, fitters = [], [], {}
for n, r0, kind in SHAPES:
    for nonloc in ((False, True) if kind == 'main' else (True,)):
        for fl, sets, tag in SETS:
            rows = [r for r in DATA if r['flavor'] == fl and r['set'] in sets]
            F = InterpFitter(libs, rows, nonloc, rel=True, n=n, r0=r0)
            eps = F.eps_for_unit_chi2()
            k, c, dl, dh = F.fit_kappa(eps)
            kv, cv, dlv, dhv = F.fit_kappa(EPS_V2[fl])
            kz, cz, dlz, dhz = F.fit_kappa(0.0)
            fits.append({'n': n, 'r0': r0, 'shape_kind': kind, 'flavor': fl, 'eQ': CHARGE[fl], 'set': tag, 'nonlocal': nonloc,
                         'N_meas': len(F.meas), 'N_UL': len(F.uls), 'ndf': F.ndf(),
                         'eps_fit': eps, 'kappa': k, 'k_lo': dl, 'k_hi': dh, 'chi2': c,
                         'kappa_eps_v2': kv, 'k_lo_v2': dlv, 'k_hi_v2': dhv, 'chi2_eps_v2': cv,
                         'kappa_eps0': kz, 'k_lo_eps0': dlz, 'k_hi_eps0': dhz, 'chi2_eps0': cz})
            fitters[(n, r0, nonloc, fl, tag)] = (F, eps, k, kz)
            print(f"  n={n:2d} r0={r0:.2f} {'NL' if nonloc else 'L '} {fl} {tag:9s} eps={eps:.3f} kappa={k:+.4f} -{dl:.4f}+{dh:.4f} "
                  f"(kappa/e_Q={k / CHARGE[fl]:+.3f}) | eps=v^2: {kv:+.4f} chi2 {cv:.1f}/{F.ndf()} | eps=0: {kz:+.4f} chi2 {cz:.1f}/{F.ndf()}")
            # observable-level results for the E1 table: NL, main shapes, charm core+ext and bottom core
            if kind == 'main' and nonloc and (tag == 'core+ext' or fl == 'b'):
                tot, chi, chiul, p, sth, pulls, ulp, mod = F.chi2(k, eps, detail=True)
                tot0, chi0, chiul0, p0, sth0, pulls0, ulp0, mod0 = F.chi2(kz, 0.0, detail=True)
                mod_lo = F.model(0.0, 0.0)
                for kk, r in enumerate(rows):
                    ps = F.parts[kk]
                    isul = r['kind'].startswith('ul')
                    if isul:
                        a = F.uls.index(kk)
                        pr, eth, pl, pr0, pl0, err = mod[kk][0], mod[kk][1], ulp[a], mod0[kk][0], ulp0[a], 0.0
                    else:
                        a = F.meas.index(kk)
                        pr, eth, pl, pr0, pl0, err = p[a], sth[a], pulls[a], p0[a], pulls0[a], float(np.sqrt(F.Cexp[a, a]))
                    pull_rows.append({'n': n, 'r0': r0, 'flavor': fl, 'id': r['id'], 'kind': r['kind'], 'set': r['set'],
                                      'data': r['value'], 'err_exp': err,
                                      'delta': ' / '.join(f"{q['delta']:+.3f}" for q in ps),
                                      'meanf': ' / '.join(f"{mean_f(q):.2f}" for q in ps),
                                      'pred_lo_rel': mod_lo[kk][0], 'pred_eps0': pr0, 'pull_eps0': pl0,
                                      'pred_fit': pr, 'err_th': eth, 'pull': pl})
write_csv('interp_fits.csv', fits)
write_csv('interp_pulls.csv', pull_rows)

# ------------------------------------------------- comparison with lattice QCD (Sec. sec:pheno_comp_Q)
# reference: NL, n = 2 (baseline) and n = 7, charm core fit with epsilon from chi2/ndf = 1
QEXT = [('chic0->Jpsi', 'chic0', 'Jpsi', 208.6, 0.3, 'HSC'), ('chic1->Jpsi', 'chic1', 'Jpsi', 327.0, 10.0, 'ETMC'),
        ('chic2->Jpsi', 'chic2', 'Jpsi', 368.0, 5.0, 'CLQCD'), ('psi2(3823)->chic1', 'psi2_3823', 'chic1', 337.0, 27.0, 'CLQCD')]
from emlib.quarkonium import Fitter, predict_row
# v5 reference (Table tab:quarkonium_lattice): LO without 1/m terms, and the LO fit kappa_c (NL, core)
F5 = Fitter(libs, [r for r in DATA if r['flavor'] == 'c' and r['set'] == 'core'], True, rel=False)
k5, _, dl5, dh5 = F5.fit_kappa(F5.eps_for_unit_chi2())
summary['v5_kappa_c_LO_NL_core'] = (k5, dl5, dh5)
# fit with the 1/m corrections (Delta l only, v7; Table tab:kappa_rel): g_E1 = (1 + dkappa) e e_Q r, f = 1
F5r = Fitter(libs, [r for r in DATA if r['flavor'] == 'c' and r['set'] == 'core'], True, rel=True)
k5r, _, dl5r, dh5r = F5r.fit_kappa(F5r.eps_for_unit_chi2())
summary['kappa_c_rel_dl_NL_core'] = (k5r, dl5r, dh5r)
_dat = {r['id']: r for r in DATA}
lat_rows = []
for n in (2, 7):
    F, eps, kc, _ = fitters[(n, R0_EST, True, 'c', 'core')]
    fr = [f for f in fits if f['n'] == n and f['r0'] == R0_EST and f['nonlocal'] and f['flavor'] == 'c' and f['set'] == 'core'][0]
    lam = kc / CHARGE['c']
    for qid, ini, fi, G, dG, src in QEXT:
        row = {'flavor': 'c', 'kind': 'width', 'init': ini, 'final': fi}
        ps = predict_parts(row, libs, True, True, n, R0_EST)[0]
        c0, c1, c2 = ps['g']
        pred = quad(ps['g'], lam)
        # error from kappa only (symmetrized Delta chi2 = 1)
        dk = 0.5 * (fr['k_lo'] + fr['k_hi'])
        dpred = abs(2 * (c1 + lam * c2)) * dk / abs(CHARGE['c'])
        # kappa that reproduces the lattice width: root of c2 lam^2 + 2 c1 lam + c0 - G = 0 closest to the fit
        disc = c1 ** 2 - c2 * (c0 - G)
        if disc >= 0:
            roots = [(-c1 + s * np.sqrt(disc)) / c2 for s in (+1, -1)]
            lam_eff = min(roots, key=lambda x: abs(x - lam))
            k_eff = lam_eff * CHARGE['c']
        else:
            k_eff = np.nan
        gL5 = predict_row(row, libs, False, 1.0)[0]; gN5 = predict_row(row, libs, True, 1.0)[0]
        d_ = _dat.get(qid)
        lat_rows.append({'n': n, 'id': qid, 'source': src, 'lat_keV': G, 'dlat_keV': dG,
                         'LO_L_keV': gL5, 'LO_NL_keV': gN5, 'fit_v5_keV': k5 ** 2 * gN5, 'dfit_v5_keV': 2 * k5 * gN5 * 0.5 * (dl5 + dh5),
                         'dkappa_eff_v5': np.sqrt(G / gN5) - 1.0,
                         'data_keV': d_['value'] if d_ else np.nan, 'ddata_keV': d_['err'] if d_ else np.nan,
                         'kappa_v5_rel': k5r, 'fit_rel_keV': k5r ** 2 * c0, 'dfit_rel_keV': 2 * k5r * c0 * 0.5 * (dl5r + dh5r),
                         'delta_1m': ps['delta'],
                         'meanf': mean_f(ps), 'LO_rel_keV': c0, 'kappa_c': kc, 'fit_keV': pred, 'dfit_keV': dpred,
                         'kappa_eff': k_eff})
        print(f"  lattice n={n} {qid:18s} lat {G:6.1f}  LO+1/m {c0:6.1f}  fit {pred:6.1f}+-{dpred:4.1f}  kappa_eff {k_eff:+.3f} (fit {kc:+.3f})")
write_csv('interp_lattice.csv', lat_rows)

# f(r) on a grid, for the figure
rr = np.linspace(0.0, 10.0, 401)
write_csv('interp_f_of_r.csv', [{'r_GeVinv': float(x), 'r_fm': float(x * HBARC), 'f_n2': float(f_hill(x, 2)),
                                 'f_n7': float(f_hill(x, 7)), 'f_n50': float(f_hill(x, 50))} for x in rr])

ref = {(f['n'], f['flavor'], f['set'], f['nonlocal'], f['r0']): f for f in fits}
for n in (2, 7):
    for fl, _, tag in SETS:
        f = ref[(n, fl, tag, True, R0_EST)]
        summary[f'kappa_{fl}_{tag}_n{n}'] = (f['kappa'], f['k_lo'], f['k_hi'], f['eps_fit'])
with open(os.path.join(RES, 'interp_summary.json'), 'w') as f:
    json.dump(summary, f, indent=1, default=float)
print(json.dumps(summary, indent=1, default=float))
