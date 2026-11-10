"""Runs the full EM-transition phenomenology (EM_Pheno v8) and writes results/ (CSV, JSON).

    python3 run_all.py            all sections (from the python/ folder)
    python3 run_all.py --e1       quarkonium E1 only (sections 1-3)

Sections
  1. quarkonium E1 fits of g_E1 = (1 + dkappa_Q) e e_Q r, one block per correction option
     corr (emlib.quarkonium.rel_delta):
       'LO'   no relativistic correction                          (Tables tab:kappa,    tab:E1)
       '1m'   O(k/m_Q) of EGTKLY: c_F^em spin term + Delta l term  (Tables tab:kappa_1m, tab:E1_1m)
       '1m2'  '1m' + 1/m^2 c_S^em term = Delta l term only          (Tables tab:kappa_1m2, tab:E1_1m2)
     theory-error options: epsilon from chi2/ndf = 1, epsilon = v^2 (LO) or v^3 (1m, 1m2), epsilon = 0.
     The results (pulls, lattice comparison, figure, hybrid normalization) use epsilon = v^2 (v^3)
     and epsilon = 0; the epsilon of chi2/ndf = 1 is only reported in the kappa tables.
     -> quarkonium_fits.csv, quarkonium_pulls.csv (column corr)
  2. interpolated LEF g_E1(r) = e r [e_Q + kappa_Q f(r)], f = Hill function (emlib.interp), with
     the '1m' corrections; n = 2, 7 at r0 = 1.58 GeV^-1, checks r0 = 1.0, n = 50
     -> interp_fits.csv, interp_pulls.csv, interp_f_of_r.csv (Tables tab:kappa_interp, tab:E1_interp)
  2b. (v8) short-distance term: g_E1 = e r [e_Q + (A_Q r + kappa_Q x)/(1 + x)], x = (r/r0)^n,
     (A_Q, kappa_Q) per flavour, with the '1m' corrections, n = 2, 7 (emlib.interp2)
  2c. (v8) flavour-independent fit of charm and bottom together, A_c = -A_b/2, kappa_c = -kappa_b/2
     (A_Q = A/e_Q, kappa_Q = kappa/e_Q; and A_c = A_b, kappa_c = kappa_b as a check)
     -> interp_short_fits.csv, interp_short_pulls.csv (Tables tab:kappa_interp_short, tab:E1_interp_short,
        tab:kappa_interp_flav, tab:E1_interp_flav)
  3. quarkonium E1 widths from lattice QCD versus all the fits above
     -> quarkonium_lattice.csv (Tables tab:lattice, tab:quarkonium_lattice)
  4. hybrids (normalized with the LO dkappa_Q: c_i = dkappa_Q e e_Q), BESIII constraints,
     channel plane, spin-1 hybrids;  5. mixing (charm);  6. hybrid lattice comparison;
  7. external hybrid models.  Variants: 'NL' (non-local) and 'L' (local).
Tables: make_tables.py; figures: figures.py; size estimates: r_moments.py.

Requires the wave-function files data/wf_*.mat produced by matlab/exportWavefunctions.m.
"""
import os, sys, csv, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from emlib.states import Library
from emlib.data import DATA, MASS
from emlib.quarkonium import Fitter, predict_row, rel_delta, CORRECTIONS, EPS_ALT, EPS_ALT_LABEL
from emlib.interp import InterpFitter, predict_parts, quad, mean_f, f_hill, R0_EST, HBARC
from emlib import hybrids as HY
from emlib import mixing as MX
from emlib.amplitudes import Params, width, photon_energy, CHARGE
from emlib import lattice as LT

ROOT = os.path.join(os.path.dirname(__file__), '..')
RES = os.path.join(ROOT, 'results'); FIG = os.path.join(ROOT, 'figures')
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)
E1_ONLY = '--e1' in sys.argv[1:]


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
SETS = (('c', ('core',), 'core'), ('c', ('core', 'ext'), 'core+ext'), ('b', ('core',), 'core'))


def deltas(row, corr):
    """rel_delta of each width of an observable (one value, or two for a ratio)"""
    fins = [row['final']] + ([row['final2']] if 'final2' in row else [])
    return ' / '.join(f"{rel_delta(row['init'], f, photon_energy(MASS[row['init']], MASS[f]), corr):+.4f}" for f in fins)


def pull_table(F, rows, k, eps, kz, extra):
    """observable-level results at the fit (k, eps) and at the epsilon = 0 fit (kz)"""
    out = []
    tot, chi, chiul, p, sth, pulls, ulp, mod = F.chi2(k, eps, detail=True)
    tot0, chi0, chiul0, p0, sth0, pulls0, ulp0, mod0 = F.chi2(kz, 0.0, detail=True)
    for kk, r in enumerate(rows):
        if r['kind'].startswith('ul'):
            a = F.uls.index(kk)
            pr, eth, pl, pr0, pl0, err = mod[kk][0], mod[kk][1], ulp[a], mod0[kk][0], ulp0[a], 0.0
        else:
            a = F.meas.index(kk)
            pr, eth, pl, pr0, pl0, err = p[a], sth[a], pulls[a], p0[a], pulls0[a], float(np.sqrt(F.Cexp[a, a]))
        out.append({**extra, 'id': r['id'], 'kind': r['kind'], 'set': r['set'], 'data': r['value'], 'err_exp': err,
                    'pred_fit': pr, 'err_th': eth, 'pull': pl, 'pred_eps0': pr0, 'pull_eps0': pl0})
    return out


# =========================================================== 1. quarkonium E1, g_E1 = (1 + dkappa_Q) e e_Q r
print('== 1. quarkonium E1 fits: corr = LO, 1m (spin + Delta l), 1m2 (Delta l only)')
fits, pull_rows = [], []
for corr in CORRECTIONS:
    for nonloc in (False, True):
        for fl, sets, tag in SETS:
            rows = [r for r in DATA if r['flavor'] == fl and r['set'] in sets]
            F = Fitter(libs, rows, nonloc, corr)
            eps = F.eps_for_unit_chi2()
            k, c, dl, dh = F.fit_kappa(eps)
            ea = EPS_ALT[corr][fl]
            ka, ca, dla, dha = F.fit_kappa(ea)
            kz, cz, dlz, dhz = F.fit_kappa(0.0)          # experimental errors only (no theory error)
            fits.append({'corr': corr, 'flavor': fl, 'set': tag, 'nonlocal': nonloc, 'N_meas': len(F.meas), 'N_UL': len(F.uls),
                         'ndf': F.ndf(), 'eps_fit': eps, 'kappa': k, 'dkappa': k - 1.0, 'dk_lo': dl, 'dk_hi': dh, 'chi2': c,
                         'eps_alt_label': EPS_ALT_LABEL[corr], 'eps_alt': ea,
                         'kappa_eps_alt': ka, 'dk_lo_alt': dla, 'dk_hi_alt': dha, 'chi2_eps_alt': ca,
                         'kappa_eps0': kz, 'dk_lo_eps0': dlz, 'dk_hi_eps0': dhz, 'chi2_eps0': cz})
            print(f"  {corr:3s} {'NL' if nonloc else 'L '} {fl} {tag:9s} eps={eps:.3f} dkappa={k - 1:+.3f} -{dl:.3f}+{dh:.3f} | "
                  f"eps={EPS_ALT_LABEL[corr]}={ea:.3f}: {ka - 1:+.3f} chi2 {ca:.1f}/{F.ndf()} | eps=0: {kz - 1:+.3f} chi2 {cz:.1f}/{F.ndf()}")
            if tag == 'core+ext' or fl == 'b':
                pr = pull_table(F, rows, ka, ea, kz, {'corr': corr, 'nonlocal': nonloc, 'flavor': fl, 'eps': ea})
                for q, r in zip(pr, rows):
                    q['delta'] = deltas(r, corr)
                    q['pred_kappa1'] = F.pred1[rows.index(r)][0] if r['kind'] in ('width', 'ul_width') else \
                        F.pred1[rows.index(r)][0] / F.pred1[rows.index(r)][2]
                pull_rows += pr
write_csv('quarkonium_fits.csv', fits)
write_csv('quarkonium_pulls.csv', pull_rows)
fitd = {(f['corr'], f['flavor'], f['set'], f['nonlocal']): f for f in fits}
for corr in CORRECTIONS:
    for fl in ('c', 'b'):
        f = fitd[(corr, fl, 'core', True)]
        summary[f'kappa_{fl}_{corr}'] = (f['kappa_eps_alt'], f['dk_lo_alt'], f['dk_hi_alt'], f['eps_alt'])
kc = fitd[('LO', 'c', 'core', True)]; kb = fitd[('LO', 'b', 'core', True)]
# reference values: fits with epsilon = v^2 (LO); the epsilon of chi2/ndf = 1 only measures the goodness of the fit
summary['kappa_c'] = (kc['kappa_eps_alt'], kc['dk_lo_alt'], kc['dk_hi_alt'], kc['eps_alt'])
summary['kappa_b'] = (kb['kappa_eps_alt'], kb['dk_lo_alt'], kb['dk_hi_alt'], kb['eps_alt'])
# hybrid normalization: c_i = dkappa_Q e e_Q, with the LO fit at epsilon = v^2
summary['dkappa_c'] = (kc['kappa_eps_alt'] - 1.0, kc['dk_lo_alt'], kc['dk_hi_alt'])
summary['dkappa_b'] = (kb['kappa_eps_alt'] - 1.0, kb['dk_lo_alt'], kb['dk_hi_alt'])
DK = {'c': summary['dkappa_c'][0], 'b': summary['dkappa_b'][0]}

# X(3872) test: Z = Gamma_exp / Gamma_E1(chi_c1(2P)) at the fitted kappa (LO)
xr = [r for r in DATA if r['id'] == 'X(3872)->Jpsi'][0]
xp = {nl: predict_row(xr, libs, nl, 1.0)[0] for nl in (False, True)}
summary['X3872'] = {'Gamma_exp': xr['value'], 'err': xr['err'],
                    'Gamma_E1_local_k1': xp[False], 'Gamma_E1_nonlocal_k1': xp[True],
                    'Z_local': xr['value'] / (kc['kappa_eps_alt'] ** 2 * xp[False]),
                    'Z_nonlocal': xr['value'] / (kc['kappa_eps_alt'] ** 2 * xp[True])}

# =========================================================== 2. interpolated g_E1(r) with the 1/m corrections
CORR_INTERP = '1m'
SHAPES = [(2, R0_EST, 'main'), (7, R0_EST, 'main'), (2, 1.0, 'check'), (7, 1.0, 'check'), (50, R0_EST, 'check')]
summary['interp'] = {'corr': CORR_INTERP, 'r0_GeVinv': R0_EST, 'r0_fm': R0_EST * HBARC, 'shapes': SHAPES}
print(f'== 2. E1 fits with g_E1 = e r [e_Q + kappa_Q f(r)], corrections {CORR_INTERP}')
ifits, ipulls, ifitters = [], [], {}
for n, r0, kind in SHAPES:
    for nonloc in ((False, True) if kind == 'main' else (True,)):
        for fl, sets, tag in SETS:
            rows = [r for r in DATA if r['flavor'] == fl and r['set'] in sets]
            F = InterpFitter(libs, rows, nonloc, CORR_INTERP, n=n, r0=r0)
            eps = F.eps_for_unit_chi2()
            k, c, dl, dh = F.fit_kappa(eps)
            ea = EPS_ALT[CORR_INTERP][fl]
            ka, ca, dla, dha = F.fit_kappa(ea)
            kz, cz, dlz, dhz = F.fit_kappa(0.0)
            ifits.append({'corr': CORR_INTERP, 'n': n, 'r0': r0, 'shape_kind': kind, 'flavor': fl, 'eQ': CHARGE[fl], 'set': tag,
                          'nonlocal': nonloc, 'N_meas': len(F.meas), 'N_UL': len(F.uls), 'ndf': F.ndf(),
                          'eps_fit': eps, 'kappa': k, 'k_lo': dl, 'k_hi': dh, 'chi2': c,
                          'eps_alt_label': EPS_ALT_LABEL[CORR_INTERP], 'eps_alt': ea,
                          'kappa_eps_alt': ka, 'k_lo_alt': dla, 'k_hi_alt': dha, 'chi2_eps_alt': ca,
                          'kappa_eps0': kz, 'k_lo_eps0': dlz, 'k_hi_eps0': dhz, 'chi2_eps0': cz})
            ifitters[(n, r0, nonloc, fl, tag)] = (F, ea, ka)
            print(f"  n={n:2d} r0={r0:.2f} {'NL' if nonloc else 'L '} {fl} {tag:9s} eps={eps:.3f} kappa={k:+.4f} -{dl:.4f}+{dh:.4f} "
                  f"(kappa/e_Q={k / CHARGE[fl]:+.3f}) | eps={ea:.3f}: {ka:+.4f} chi2 {ca:.1f}/{F.ndf()} | eps=0: {kz:+.4f} chi2 {cz:.1f}/{F.ndf()}")
            if kind == 'main' and nonloc and (tag == 'core+ext' or fl == 'b'):
                pr = pull_table(F, rows, ka, ea, kz, {'n': n, 'r0': r0, 'flavor': fl, 'eps': ea})
                mod_lo = F.model(0.0, 0.0)
                for kk, q in enumerate(pr):
                    ps = F.parts[kk]
                    q['delta'] = ' / '.join(f"{x['delta']:+.4f}" for x in ps)
                    q['meanf'] = ' / '.join(f"{mean_f(x):.2f}" for x in ps)
                    q['pred_lo_corr'] = mod_lo[kk][0]
                ipulls += pr
write_csv('interp_fits.csv', ifits)
write_csv('interp_pulls.csv', ipulls)
rr = np.linspace(0.0, 10.0, 401)
write_csv('interp_f_of_r.csv', [{'r_GeVinv': float(x), 'r_fm': float(x * HBARC), 'f_n2': float(f_hill(x, 2)),
                                 'f_n7': float(f_hill(x, 7)), 'f_n50': float(f_hill(x, 50))} for x in rr])
for f in ifits:
    if f['nonlocal']:
        summary['interp'][f"kappa_{f['flavor']}_{f['set']}_n{f['n']}_r0{f['r0']:.2f}"] = (f['kappa_eps_alt'], f['k_lo_alt'], f['k_hi_alt'], f['eps_alt'])

# =========================================================== 2b. short-distance + interpolated g_E1(r) (v8):
# g_E1 = e r [e_Q + (A_Q r + kappa_Q x)/(1 + x)], x = (r/r0)^n, two parameters per flavour, corrections CORR_INTERP
# =========================================================== 2c. flavour-independent fit: charm and bottom together,
# A_c = RATIO_LINK A_b, kappa_c = RATIO_LINK kappa_b (RATIO_LINK = e_b/e_c = -1/2: A_Q = A/e_Q, kappa_Q = kappa/e_Q
# with flavour-independent A, kappa; +1, equal A_Q and kappa_Q, as a check)
from emlib.interp2 import Interp2Fitter, V3
RATIO_LINK = -0.5
SHAPES2 = (2, 7)


def fit2_row(F, eps_fit, extra):
    """fits of an Interp2Fitter at eps_fit (chi2/ndf = 1), eps = v^3 and eps = 0; reference-flavour parameters"""
    out = dict(extra)
    ev3 = V3[F.ref] if F.mode == 'free' else dict(V3)
    for tag, eps in (('fit', eps_fit), ('v3', ev3), ('eps0', 0.0)):
        p, c = F.fit(eps)
        ((dAl, dAh), (dkl, dkh)), rho = F.profile_errors(p, c, eps)
        out.update({f'A_{tag}': p[0], f'A_lo_{tag}': dAl, f'A_hi_{tag}': dAh, f'kappa_{tag}': p[1],
                    f'k_lo_{tag}': dkl, f'k_hi_{tag}': dkh, f'chi2_{tag}': c})
        if tag == 'v3':
            out['rho_v3'] = rho
        if F.mode == 'linked':
            for fl in ('c', 'b'):            # chi2 of each flavour at the common parameters
                sub = SUB[(F.nonloc, F.n, out['set'], fl)]
                out[f'chi2_{tag}_{fl}'] = sub.chi2(F.params_of(p, fl), eps[fl] if isinstance(eps, dict) else eps)
        out[f'p_{tag}'] = p
    out['ndf'] = F.ndf()
    return out


print('== 2b. g_E1 = e r [e_Q + (A_Q r + kappa_Q x)/(1 + x)]: two-parameter fits per flavour')
s2_fits, s2_pulls, SUB = [], [], {}
for n in SHAPES2:
    for nonloc in (False, True):
        for fl, sets, tag in SETS:
            rows = [r for r in DATA if r['flavor'] == fl and r['set'] in sets]
            F = Interp2Fitter(libs, rows, nonloc, CORR_INTERP, n=n, r0=R0_EST)
            SUB[(nonloc, n, tag, fl)] = F
            if fl == 'b':
                SUB[(nonloc, n, 'core+ext', 'b')] = F
            ef = F.eps_for_unit_chi2()
            o = fit2_row(F, ef, {'mode': 'free', 'n': n, 'r0': R0_EST, 'flavor': fl, 'eQ': CHARGE[fl], 'set': tag,
                                 'nonlocal': nonloc, 'N_meas': len(F.meas), 'N_UL': len(F.uls), 'eps_fit': ef,
                                 'eps_v3': V3[fl]})
            print(f"  n={n} {'NL' if nonloc else 'L '} {fl} {tag:9s} eps={ef:.3f} | v3: A={o['A_v3']:+.4f} kappa={o['kappa_v3']:+.4f} "
                  f"rho={o['rho_v3']:+.3f} chi2 {o['chi2_v3']:.1f}/{o['ndf']} | eps=0: A={o['A_eps0']:+.4f} kappa={o['kappa_eps0']:+.4f} "
                  f"chi2 {o['chi2_eps0']:.1f}/{o['ndf']}")
            if nonloc and (tag == 'core+ext' or fl == 'b'):
                pr = pull_table(F, rows, o['p_v3'], V3[fl], o['p_eps0'], {'mode': 'free', 'n': n, 'flavor': fl, 'eps': V3[fl]})
                mod_lo = F.model((0.0, 0.0), 0.0)
                for kk, q in enumerate(pr):
                    q['delta'] = ' / '.join(f"{x['delta']:+.4f}" for x in F.forms[kk])
                    q['pred_lo_corr'] = mod_lo[kk][0]
                s2_pulls += pr
            s2_fits.append({k: v for k, v in o.items() if not k.startswith('p_')})

print(f'== 2c. flavour-independent fit: A_c = {RATIO_LINK} A_b, kappa_c = {RATIO_LINK} kappa_b')
for ratio in (RATIO_LINK, 1.0):
    for n in SHAPES2:
        for nonloc in (False, True):
            for tag, sets_c in (('core', ('core',)), ('core+ext', ('core', 'ext'))):
                if ratio != RATIO_LINK and not (nonloc and tag == 'core'):
                    continue                  # ratio = +1 (equal end-point charges): one check per shape
                rows = [r for r in DATA if (r['flavor'] == 'c' and r['set'] in sets_c) or (r['flavor'] == 'b' and r['set'] == 'core')]
                F = Interp2Fitter(libs, rows, nonloc, CORR_INTERP, n=n, r0=R0_EST, mode='linked', ratio=ratio)
                ef = F.eps_for_unit_chi2()
                o = fit2_row(F, ef, {'mode': 'linked', 'ratio': ratio, 'n': n, 'r0': R0_EST, 'flavor': 'c+b', 'set': tag,
                                     'nonlocal': nonloc, 'N_meas': len(F.meas), 'N_UL': len(F.uls),
                                     'eps_fit': ef['c'], 'eps_fit_b': ef['b'], 'eps_v3': V3['c'], 'eps_v3_b': V3['b'],
                                     'params_of': 'b'})
                print(f"  ratio={ratio:+.1f} n={n} {'NL' if nonloc else 'L '} {tag:9s} eps={ef['c']:.3f}/{ef['b']:.3f} | v3: A_b={o['A_v3']:+.4f} "
                      f"kappa_b={o['kappa_v3']:+.4f} chi2 {o['chi2_v3']:.1f}/{o['ndf']} (c {o['chi2_v3_c']:.1f}, b {o['chi2_v3_b']:.1f}) | eps=0: "
                      f"A_b={o['A_eps0']:+.4f} kappa_b={o['kappa_eps0']:+.4f} chi2 {o['chi2_eps0']:.1f}/{o['ndf']} "
                      f"(c {o['chi2_eps0_c']:.1f}, b {o['chi2_eps0_b']:.1f})")
                if ratio == RATIO_LINK and nonloc and tag == 'core+ext':
                    pr = pull_table(F, rows, o['p_v3'], dict(V3), o['p_eps0'], {'mode': 'linked', 'n': n, 'flavor': '', 'eps': 'v3'})
                    mod_lo = F.model((0.0, 0.0), 0.0)
                    for kk, q in enumerate(pr):
                        q['flavor'] = rows[kk]['flavor']
                        q['delta'] = ' / '.join(f"{x['delta']:+.4f}" for x in F.forms[kk])
                        q['pred_lo_corr'] = mod_lo[kk][0]
                    s2_pulls += pr
                s2_fits.append({k: v for k, v in o.items() if not k.startswith('p_')})
keys = list(dict.fromkeys(k for f in s2_fits for k in f))
write_csv('interp_short_fits.csv', [{k: f.get(k, np.nan) for k in keys} for f in s2_fits])
write_csv('interp_short_pulls.csv', s2_pulls)

# =========================================================== 3. quarkonium: lattice QCD versus the fits (NL, charm core,
# epsilon = v^2 for LO, v^3 for 1m, 1m2 and the interpolated fits). (id, init, final, lattice width keV, error, source)
print('== 3. quarkonium E1 widths from lattice QCD')
QEXT = [('chic0->Jpsi', 'chic0', 'Jpsi', 208.6, 0.3, 'HSC'), ('chic1->Jpsi', 'chic1', 'Jpsi', 327.0, 10.0, 'ETMC'),
        ('chic2->Jpsi', 'chic2', 'Jpsi', 368.0, 5.0, 'CLQCD'), ('psi2(3823)->chic1', 'psi2_3823', 'chic1', 337.0, 27.0, 'CLQCD')]
# HSC: 208.6(3) keV with experimental masses, 247.4(2) keV with lattice masses (arXiv:2301.08213, Eq. (5.5));
# ETMC (arXiv:2604.15256): Gamma(chi_c1 -> J/psi e+e-) = 2.869(90) keV / their ratio Gamma(ee)/Gamma(gamma) = 8.78(2)e-3;
# CLQCD (arXiv:2309.09662): N_f = 2, m_pi = 350 MeV
_dat = {r['id']: r for r in DATA}
qlat = []
for qid, ini, fi, G, dG, src in QEXT:
    row = {'flavor': 'c', 'kind': 'width', 'init': ini, 'final': fi}
    d_ = _dat.get(qid)
    out = {'id': qid, 'source': src, 'lat_keV': G, 'dlat_keV': dG,
           'data_keV': d_['value'] if d_ else np.nan, 'ddata_keV': d_['err'] if d_ else np.nan,
           'LO_L_keV': predict_row(row, libs, False, 1.0)[0]}
    for corr in CORRECTIONS:
        g1 = predict_row(row, libs, True, 1.0, corr)[0]                  # NL, kappa = 1, with the corrections corr
        f = fitd[(corr, 'c', 'core', True)]
        k, dk = f['kappa_eps_alt'], 0.5 * (f['dk_lo_alt'] + f['dk_hi_alt'])
        out[f'LO_NL_{corr}_keV'] = g1
        out[f'kappa_{corr}'] = k
        out[f'fit_{corr}_keV'] = k ** 2 * g1
        out[f'dfit_{corr}_keV'] = 2 * k * dk * g1                          # error from kappa only
        out[f'dkappa_eff_{corr}'] = np.sqrt(G / g1) - 1.0                  # (Gamma_lat/Gamma(kappa=1))^(1/2) - 1
    out['delta_1m'] = rel_delta(ini, fi, photon_energy(MASS[ini], MASS[fi]), '1m')
    out['delta_1m2'] = rel_delta(ini, fi, photon_energy(MASS[ini], MASS[fi]), '1m2')
    for n in (2, 7):
        F, eps, kq = ifitters[(n, R0_EST, True, 'c', 'core')]
        fr = [x for x in ifits if x['n'] == n and x['r0'] == R0_EST and x['nonlocal'] and x['flavor'] == 'c' and x['set'] == 'core'][0]
        ps = predict_parts(row, libs, True, CORR_INTERP, n, R0_EST)[0]
        c0, c1, c2 = ps['g']
        lam = kq / CHARGE['c']
        dk = 0.5 * (fr['k_lo_alt'] + fr['k_hi_alt'])
        out[f'kappa_interp_n{n}'] = kq
        out[f'interp_n{n}_keV'] = quad(ps['g'], lam)
        out[f'dinterp_n{n}_keV'] = abs(2 * (c1 + lam * c2)) * dk / abs(CHARGE['c'])
        out[f'meanf_n{n}'] = mean_f(ps)
        # end-point charge that reproduces the lattice width: root of c2 lam^2 + 2 c1 lam + c0 - G = 0 closest to the fit
        disc = c1 ** 2 - c2 * (c0 - G)
        if disc >= 0:
            lam_eff = min([(-c1 + s_ * np.sqrt(disc)) / c2 for s_ in (+1, -1)], key=lambda x: abs(x - lam))
            out[f'kappa_eff_interp_n{n}'] = lam_eff * CHARGE['c']
        else:
            out[f'kappa_eff_interp_n{n}'] = np.nan
    qlat.append(out)
    print(f"  {qid:18s} lat {G:6.1f} | LO NL {out['LO_NL_LO_keV']:6.1f}  fit LO {out['fit_LO_keV']:6.1f}  fit 1m {out['fit_1m_keV']:6.1f}  "
          f"fit 1m2 {out['fit_1m2_keV']:6.1f}  interp n2 {out['interp_n2_keV']:6.1f} n7 {out['interp_n7_keV']:6.1f} | "
          f"dkappa_eff {out['dkappa_eff_LO']:+.3f}  kappa_eff(n=2) {out['kappa_eff_interp_n2']:+.3f}")
write_csv('quarkonium_lattice.csv', qlat)
summary['chic0_lattice_masses_keV'] = 247.4

if E1_ONLY:
    with open(os.path.join(RES, 'summary_e1.json'), 'w') as f:
        json.dump(summary, f, indent=1, default=float)
    sys.exit(0)

# =========================================================== 4. hybrids
print('== 4. hybrids')
hyb_rows = []
for fl in ('c', 'b'):
    rows = HY.predictions(fl, 1.0)            # widths at kappa_H = 1; scale with dkappa_Q^2
    dk, dlo, dhi = summary['dkappa_' + fl]
    for r_ in rows:
        r_['dkappa'] = dk; r_['dkappa_lo'] = dk - dlo; r_['dkappa_hi'] = dk + dhi
        for v in HY.VARIANTS:
            r_[f'{v}_at_dk'] = dk ** 2 * r_[v]
    hyb_rows += rows
write_csv('hybrid_predictions.csv', hyb_rows)

# data constraints on kappa_H from BESIII gamma eta_c
ULs = HY.derived_ul_gamma_etac()
summary['ULs_gamma_etac'] = ULs
libc = libs['c']
hc = {h[0]: h for h in HY.HYBRIDS['c']}
fin_etac = [f for f in HY.FINALS['c'] if f[0] == 'eta_c(1S)'][0]
cons = []
for cand, hl in (('psi(4040)', '1(s/d)1'), ('psi(4415)', '2(s/d)1')):
    UL = ULs[cand]['Gamma_UL_keV']
    for v in HY.VARIANTS:
        sp, E = HY.gamma_split('c', libc, hc[hl], fin_etac, v, (1.0,) * 4)
        gB = HY.gamma('c', libc, hc[hl], fin_etac, v, (0.0, 1.0, 0.0, 1.0))[0]
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
    for v in ('NL', 'L'):
        planes[(cand, v)] = HY.quadratic_form('c', libc, hc[hl], fin_etac, v)
np.save(os.path.join(RES, 'quadratic_forms.npy'), {f'{k[0]}|{k[1]}': v for k, v in planes.items()}, allow_pickle=True)

# channel plane (kappa_Pi, kappa_Sigma): Gamma = kappa_Pi^2 G_Pi + kappa_Sigma^2 G_Sigma + kappa_Pi kappa_Sigma G_int.
# For each ray kappa_Sigma = t kappa_Pi the bound is kappa_Pi^2 (G_Pi + t^2 G_Sigma + t G_int) < UL
pl_rows = []
cmap = {(r_['candidate'], r_['variant']): r_ for r_ in cons}
for v in ('NL', 'L'):
    for t in (0.0, 0.25, 0.5, 1.0, 2.0):
        kmax = min(np.sqrt(ULs[cand]['Gamma_UL_keV'] /
                           (cmap[(cand, v)]['Gamma_k1_Pi_keV'] + t * t * cmap[(cand, v)]['Gamma_k1_Sigma_keV']
                            + t * cmap[(cand, v)]['Gamma_k1_int_keV']))
                   for cand in ('psi(4040)', 'psi(4415)'))
        pl_rows.append({'variant': v, 'kSigma_over_kPi': t, 'kappa_Pi_max': kmax})
    # largest |c4| allowed for c1=c2=c3=kappa_c (fitted E1 normalization): c4 is the
    # only coefficient not shared by the two channels
    kc0 = DK['c']                                 # c = dkappa_c e e_c (v4)
    best = []
    for cand in ('psi(4040)', 'psi(4415)'):
        Q = planes[(cand, v)]; UL = ULs[cand]['Gamma_UL_keV']
        a = Q[3, 3]; b = 2 * kc0 * Q[3, :3].sum(); c = kc0 ** 2 * Q[:3, :3].sum() - UL
        disc = b * b - 4 * a * c
        best.append(tuple(sorted(((-b - np.sqrt(disc)) / (2 * a) / kc0, (-b + np.sqrt(disc)) / (2 * a) / kc0)))
                    if disc > 0 else (np.nan, np.nan))
    lo = max(x[0] for x in best); hi = min(x[1] for x in best)
    summary.setdefault('c4_range_at_dkappa_c', {})[v] = {'psi(4040)': best[0], 'psi(4415)': best[1], 'common': (lo, hi)}
    pl_rows.append({'variant': v, 'kSigma_over_kPi': 'c4/c range at dkappa_c', 'kappa_Pi_max': f'[{lo:.3f}, {hi:.3f}]'})
write_csv('hybrid_channel_plane.csv', pl_rows)
for r_ in pl_rows:
    print('   plane', r_)

# spin-1 hybrids (heavy-quark spin 1): explicit Jcal -> J' widths for 1(s/d)1 and 1p1
spin1 = []
for fl in ('c', 'b'):
    lib = libs[fl]
    kap = DK[fl]
    hh = {h[0]: h for h in HY.HYBRIDS[fl]}
    for hl, finals in (('1(s/d)1', [(1, 0), (2, 0)]), ('1p1', [(1, 1)])):
        h = hh[hl]
        for (nf, lf) in finals:
            fin = [f for f in HY.FINALS[fl] if f[1] == nf and f[2] == lf][0]
            for Jcal in range(abs(h[1] - 1), h[1] + 2):
                for Jf in range(abs(lf - 1), lf + 2):
                    for v in ('NL', 'L'):
                        sp, E = HY.gamma_split(fl, lib, h, fin, v, (kap,) * 4, S=1, Jcal=Jcal, Jf=Jf)
                        spin1.append({'flavor': fl, 'hybrid': hl, 'Jcal_i': Jcal, 'final_n_l': f'{nf}{"SPDF"[lf]}',
                                      'J_f': Jf, 'variant': v, 'Gamma_keV': sp['total'], 'Pi_keV': sp['Pi'],
                                      'Sigma_keV': sp['Sigma'], 'int_keV': sp['int']})
write_csv('hybrid_spin1.csv', spin1)

# =========================================================== 5. mixing (charm)
print('== 5. mixing')
kH = DK['c']                                  # hybrid vertex: dkappa_c; E1: kappa_c = 1 + dkappa_c
mix_rows = MX.table(summary['kappa_c'][0], kH)
write_csv('mixing_1mm.csv', mix_rows)
# kappa_H limits from the mixed (mix=true) states
mixcons = []
ULmix = {'psi(4040)': ULs['psi(4040)']['Gamma_UL_keV'], 'psi(4415)': ULs['psi(4415)']['Gamma_UL_keV'],
         'psi(3770)': MX.GAMMA_ETAC_EXP['psi(3770)'][1]}   # BESIII arXiv:2609.33606: < 5.5 keV (90% CL)
for r in mix_rows:
    if r['final'] == 'eta_c(1S)' and r['state'] in ULmix:
        for v in ('NL', 'L'):
            mixcons.append({'state': r['state'], 'content': r['content'], 'variant': v,
                            'Gamma_at_kappaH_keV': r[v], 'Pi_keV': r[v + '_Pi'], 'Sigma_keV': r[v + '_Sigma'],
                            'int_keV': r[v + '_int'], 'kappa_H_used': kH, 'Gamma_UL_keV': ULmix[r['state']],
                            'kappa_H_max': abs(kH) * np.sqrt(ULmix[r['state']] / r[v])})
write_csv('mixing_kappa_constraints.csv', mixcons)
for r in mixcons:
    print('   mix-constraint', r['state'], r['variant'], f"kappa_H < {r['kappa_H_max']:.2f}")
hind = MX.hindered_psi2S_etac(kH)
write_csv('mixing_hindered.csv', hind)
for r in hind:
    print('  ', r)

# c4 scan: c1 = c2 = c3 = kappa_c, c4 = t kappa_c (g_B_par is the only LEF not fixed by the Pi_u channel)
print('== c4 scan')
scan = MX.c4_scan(kH, np.round(np.linspace(-1.4, 1.4, 113), 4))
hcd = {h[0]: h for h in HY.HYBRIDS['c']}
for t in sorted(set(r_['c4_over_c'] for r_ in scan)):
    for cand, hl in (('psi(4040)', '1(s/d)1'), ('psi(4415)', '2(s/d)1')):
        for v in ('NL', 'L'):
            g = HY.gamma('c', libc, hcd[hl], fin_etac, v, (kH, kH, kH, t * kH))[0]
            scan.append({'c4_over_c': t, 'state': cand + ' (unmixed)', 'variant': v, 'Gamma_keV': g})
write_csv('c4_scan.csv', scan)
# c4 window allowed by the mixed-state data: the hybrid-admixture contribution alone must not
# exceed the measured J/psi, psi(2S) -> gamma eta_c widths (+2 sigma; they also receive the M1
# contribution, not included) nor the psi(3770) limit
from emlib.mixing import GAMMA_ETAC_EXP
win = {}
for v in ('NL', 'L'):
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

# =========================================================== 6. lattice (arXiv:2301.08213): sizes and hybrids
print('== 6. lattice comparison')
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
lh = LT.hybrid_lattice_table(libs['c'], h_sd1, ('NL', 'L'),
                             DK['c'])
write_csv('lattice_hybrids.csv', lh)
# channel strengths that reproduce both lattice widths of a data set
lsol = LT.channel_solutions(lh)
write_csv('lattice_channel_solutions.csv', lsol)
summary['lattice_channel_solutions'] = lsol
for r_ in lsol:
    print('   lattice channel solution', r_)
Es = np.round(np.arange(0.60, 1.501, 0.0125), 4)
write_csv('gamma_vs_Eq.csv', LT.gamma_vs_Eq(libs['c'], h_sd1, ('NL', 'L'), Es))
for r_ in lh:
    if r_['variant'] == 'NL':
        print(f"   {r_['channel']:15s} {r_['dataset']:8s} {r_['variant']} E={r_['E_q']:.3f} "
              f"G(kc)={r_['Gamma_kappac_keV']:9.1f}  lat={r_['lat_keV']}  kH={r_['kappaH']:.3f}")

# =========================================================== 7. external hybrid models
print('== 7. external comparisons')
# hybrids: Coulomb-gauge model (Guo, Yepez-Martinez, Szczepaniak, 1402.5863; M = 4.35 GeV, two normalizations)
# and the eta_c1 of the lattice study 2306.12884 (m = 4.329(36) GeV, Gamma_tot = 258+88+150+1 MeV)
hext = []
h1 = [h for h in HY.HYBRIDS['c'] if h[0] == '1(s/d)1'][0]
fin_jpsi = ('J/psi', 1, 0, 3.096900)
for tag, Mi, fin, S, ref, lo, hi, Gtot in (('Y(1--)->eta_c', 4.35, fin_etac, 0, 'GYS 1402.5863', 39.0, 126.0, np.nan),
                                         ('H1(S=1)->J/psi', 4.35, fin_jpsi, 1, 'GYS 1402.5863', 32.0, 116.0, np.nan),
                                         ('eta_c1->J/psi', 4.329, fin_jpsi, 1, 'CLQCD 2306.12884 (mass, total width)', np.nan, np.nan, 497.0)):
    hh = tuple(h1[:6]) + (Mi,) + tuple(h1[7:])
    row = {'channel': tag, 'source': ref, 'M_i': Mi, 'model_lo_keV': lo, 'model_hi_keV': hi, 'Gtot_MeV': Gtot}
    for v in ('NL', 'L'):
        sp, E = HY.gamma_split('c', libc, hh, fin, v, (DK['c'],) * 4, S=S, Jcal=1 if S == 1 else None, Jf=1 if S == 1 else None)
        row['E_q'] = E
        row[v] = sp['total']
        for X in ('Pi', 'Sigma', 'int'):
            row[f'{v}_{X}'] = sp[X]
    hext.append(row)
write_csv('hybrid_external.csv', hext)
for r_ in hext:
    print('  ', r_['channel'], r_['M_i'], f"E={r_['E_q']:.3f} NL={r_['NL']:.1f} L={r_['L']:.1f}")

with open(os.path.join(RES, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=1, default=float)
print(json.dumps(summary, indent=1, default=float))
