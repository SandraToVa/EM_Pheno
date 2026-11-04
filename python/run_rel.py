"""EM_Pheno v7: quarkonium E1 fits g_E1 = (1 + dkappa_Q) e e_Q r with the O(k/m) corrections
reduced to their spin-independent (Delta l) part, since the c_F^em spin term is cancelled by the
1/m^2 c_S^em spin-orbit term (quarkonium.rel_delta, SPIN = False). Same fits as the rel=True part of
run_all.py (v5), which used the EGTKLY delta with the spin term:

    python3 run_rel.py   ->  results/rel_fits.csv, results/rel_pulls.csv, results/rel_summary.json
                             (tables: make_tables_v7.py -> tab_kappa_rel_v7.tex, tab_E1_rel_v7.tex)
"""
import os, sys, csv, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from emlib.states import Library
from emlib.data import DATA
from emlib import quarkonium as QK
from emlib.quarkonium import Fitter

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
SETS = (('c', ('core',), 'core'), ('c', ('core', 'ext'), 'core+ext'), ('b', ('core',), 'core'))
fits, pull_rows, summary = [], [], {}
print('== E1 fits with the 1/m corrections: Delta l term only (v7) and with the c_F spin term (v5, reference)')
for spin in (False, True):
    QK.SPIN = spin
    for nonloc in (False, True):
        for fl, sets, tag in SETS:
            rows = [r for r in DATA if r['flavor'] == fl and r['set'] in sets]
            F = Fitter(libs, rows, nonloc, rel=True)
            eps = F.eps_for_unit_chi2()
            k, c, dl, dh = F.fit_kappa(eps)
            k0, c0, dl0, dh0 = F.fit_kappa(0.3 if fl == 'c' else 0.1)
            kz, cz, dlz, dhz = F.fit_kappa(0.0)
            fits.append({'spin_term': spin, 'flavor': fl, 'set': tag, 'nonlocal': nonloc, 'N_meas': len(F.meas), 'N_UL': len(F.uls),
                         'eps_fit': eps, 'kappa': k, 'dkappa': k - 1.0, 'dk_lo': dl, 'dk_hi': dh, 'chi2': c, 'ndf': F.ndf(),
                         'kappa_eps_v2': k0, 'dk_lo_v2': dl0, 'dk_hi_v2': dh0, 'chi2_eps_v2': c0,
                         'kappa_eps0': kz, 'dk_lo_eps0': dlz, 'dk_hi_eps0': dhz, 'chi2_eps0': cz})
            print(f"  spin={spin!s:5s} {'NL' if nonloc else 'L '} {fl} {tag:9s} eps={eps:.3f} dkappa={k - 1:+.3f} -{dl:.3f}+{dh:.3f} "
                  f"| eps=v^2: {k0 - 1:+.3f} chi2 {c0:.1f}/{F.ndf()} | eps=0: {kz - 1:+.3f} chi2 {cz:.1f}/{F.ndf()}")
            if not spin and nonloc and (tag == 'core+ext' or fl == 'b'):
                tot, chi, chiul, p, sth, pulls, ulp, mod = F.chi2(k, eps, detail=True)
                tot0, chi0, chiul0, p0, sth0, pulls0, ulp0, mod0 = F.chi2(kz, 0.0, detail=True)
                for kk, r in enumerate(rows):
                    isul = r['kind'].startswith('ul')
                    if isul:
                        a = F.uls.index(kk)
                        pr, eth, pl, pr0, pl0, err = mod[kk][0], mod[kk][1], ulp[a], mod0[kk][0], ulp0[a], 0.0
                    else:
                        a = F.meas.index(kk)
                        pr, eth, pl, pr0, pl0, err = p[a], sth[a], pulls[a], p0[a], pulls0[a], float(np.sqrt(F.Cexp[a, a]))
                    pk1 = F.pred1[kk][0] if r['kind'] in ('width', 'ul_width') else F.pred1[kk][0] / F.pred1[kk][2]
                    pull_rows.append({'flavor': fl, 'id': r['id'], 'kind': r['kind'], 'set': r['set'], 'data': r['value'],
                                      'err_exp': err, 'pred_kappa1': pk1, 'pred_eps0': pr0, 'pull_eps0': pl0,
                                      'pred_fit': pr, 'err_th': eth, 'pull': pl})
QK.SPIN = False
write_csv('rel_fits.csv', fits)
write_csv('rel_pulls.csv', pull_rows)
for f in fits:
    summary[f"{'spin' if f['spin_term'] else 'dl'}_{f['flavor']}_{f['set']}_{'NL' if f['nonlocal'] else 'L'}"] = \
        {'eps': f['eps_fit'], 'dkappa': f['dkappa'], 'dk_lo': f['dk_lo'], 'dk_hi': f['dk_hi'],
         'dkappa_eps0': f['kappa_eps0'] - 1, 'chi2_eps0': f['chi2_eps0'], 'ndf': f['ndf']}
with open(os.path.join(RES, 'rel_summary.json'), 'w') as f:
    json.dump(summary, f, indent=1, default=float)
