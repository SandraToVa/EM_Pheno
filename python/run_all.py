"""Quarkonium E1 phenomenology (EM_Pheno v11): fits of g_E1(r) = e r [e_Q + (A_Q r + kappa_Q x)/(1 + x)],
x = (r/r0)^2, r0 = 1.58 GeV^-1, with the 1/m corrections, flavour independent (A_c = -A_b/2,
kappa_c = -kappa_b/2), theory error eps = v^2 (0.3 charm, 0.1 bottom) or eps = 0.

    python3 run_all.py        (from the python/ folder; then python3 make_tables.py)

Writes to ../results:
  fits.csv      one row per (form L/NL, data set core/core+ext): chi^2 at LO and at LO + 1/m (no parameters),
                and the fits at eps = v^2 and eps = 0 (A_b, kappa_b with the errors from the fits to
                Gamma_exp,max and Gamma_exp,min, chi^2 = chi^2_c + chi^2_b)                                                -> Table tab:gE1_fit
  widths.csv    every observable at the reference fit (NL, core): LO, delta_1m, LO + 1/m, the fit at
                eps = v^2 (with theory error, propagated fit error and pull) and at eps = 0  -> Table tab:E1_widths
  j_ratios.csv  J dependence of the multiplets: reduced widths Gamma_J/(S_J E_J^3) relative to a reference J
                (S_J: spin factor with the 6j symbol), data and model                       -> Table tab:J_ratios
  lattice.csv   lattice QCD E1 widths versus LO (L, NL) and the reference fit         -> Tables tab:lattice, tab:quarkonium_lattice
  summary.json  reference parameters and numbers quoted in the text
Requires data/wf_c_nomix.mat, data/wf_b_nomix.mat (matlab/exportWavefunctions.m).
"""
import os, sys, csv, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from emlib.states import Library
from emlib.data import DATA, MASS, STATES
from emlib.e1 import R0, N_INTERP, HBARC, photon_energy, g_E1
from emlib.angular import spin_factor
from emlib.fit import Fit, Observable, EPS_V2, flavour_params

RES = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RES, exist_ok=True)
ZERO = np.zeros(2)                                      # A = kappa = 0: LO (corr=False) or LO + 1/m
DATASETS = {'core': {'c': ('core',), 'b': ('core',)}, 'core+ext': {'c': ('core', 'ext'), 'b': ('core',)}}
REF = (True, 'core')                                    # reference fit: non-local form, core data


def write_csv(name, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(RES, name), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f'{v:.6g}' if isinstance(v, float) else v) for k, v in r.items()})


libs = {'c': Library('c'), 'b': Library('b')}
summary = {'r0_GeVinv': R0, 'r0_fm': R0 * HBARC, 'n': N_INTERP, 'eps_v2': EPS_V2}

# ======================================================================== 1. fits
print(f'== 1. flavour-independent fits, n = {N_INTERP}, r0 = {R0:.3f} GeV^-1, with the 1/m corrections')
fits, FITS = [], {}
for nonloc in (False, True):
    for dset, sets in DATASETS.items():
        F = Fit(DATA, libs, nonloc, sets)
        out = {'form': 'NL' if nonloc else 'L', 'set': dset, 'N_meas': len(F.meas), 'N_UL': len(F.uls), 'ndf': F.ndf}
        for et, eps in (('v2', EPS_V2), ('eps0', 0.0)):
            for tag, corr in (('LO', False), ('LO1m', True)):         # no free parameter: ndf = N_meas
                cp = F.chi2_parts(ZERO, eps, corr)
                out[f'chi2_{tag}_{et}'] = cp['c'] + cp['b']
            R = F.run(eps)
            (dAl, dAh), (dkl, dkh) = R['errors']
            out.update({f'A_b_{et}': R['p'][0], f'A_b_lo_{et}': dAl, f'A_b_hi_{et}': dAh,
                        f'kappa_b_{et}': R['p'][1], f'kappa_b_lo_{et}': dkl, f'kappa_b_hi_{et}': dkh,
                        f'A_b_max_{et}': R['p_max'][0], f'kappa_b_max_{et}': R['p_max'][1], f'chi2_max_{et}': R['chi2_max'],
                        f'A_b_min_{et}': R['p_min'][0], f'kappa_b_min_{et}': R['p_min'][1], f'chi2_min_{et}': R['chi2_min'],
                        f'chi2_{et}': R['chi2'],
                        f'chi2_c_{et}': R['chi2_parts']['c'], f'chi2_b_{et}': R['chi2_parts']['b']})
            FITS[(nonloc, dset, et)] = (F, R)
        fits.append(out)
        print(f"  {out['form']:2s} {dset:9s} N={out['N_meas']}+{out['N_UL']} | eps=v2: chi2 LO {out['chi2_LO_v2']:.1f}, "
              f"LO+1/m {out['chi2_LO1m_v2']:.1f}, fit A_b={out['A_b_v2']:+.4f} kappa_b={out['kappa_b_v2']:+.4f} "
              f"[max {out['A_b_max_v2']:+.4f},{out['kappa_b_max_v2']:+.4f}; min {out['A_b_min_v2']:+.4f},{out['kappa_b_min_v2']:+.4f}] chi2 {out['chi2_v2']:.1f}/{F.ndf} ({out['chi2_c_v2']:.1f}+{out['chi2_b_v2']:.1f}) | "
              f"eps=0: chi2 LO {out['chi2_LO_eps0']:.0f}, LO+1/m {out['chi2_LO1m_eps0']:.0f}, fit A_b={out['A_b_eps0']:+.4f} "
              f"kappa_b={out['kappa_b_eps0']:+.4f} [max {out['A_b_max_eps0']:+.4f},{out['kappa_b_max_eps0']:+.4f}; min {out['A_b_min_eps0']:+.4f},{out['kappa_b_min_eps0']:+.4f}] chi2 {out['chi2_eps0']:.1f}/{F.ndf} "
              f"({out['chi2_c_eps0']:.1f}+{out['chi2_b_eps0']:.1f})")
write_csv('fits.csv', fits)

F, RV = FITS[REF + ('v2',)]
_, R0F = FITS[REF + ('eps0',)]
pV, p0 = RV['p'], R0F['p']
for et, R in (('v2', RV), ('eps0', R0F)):
    Ac, kc = flavour_params(R['p'], 'c')
    summary[f'ref_{et}'] = {'A_b': R['p'][0], 'kappa_b': R['p'][1], 'A_c': Ac, 'kappa_c': kc, 'errors_b': R['errors'],
                            'p_max_b': R['p_max'], 'p_min_b': R['p_min'], 'chi2_max': R['chi2_max'],
                            'chi2_min': R['chi2_min'], 'chi2': R['chi2'], 'ndf': F.ndf, 'chi2_c': R['chi2_parts']['c'],
                            'chi2_b': R['chi2_parts']['b']}
# g_E1(r)/(e e_Q r) of the reference fit at a few distances
summary['gE1_over_LO'] = {fl: {f'{r:g}': float(g_E1(r, fl, *flavour_params(pV, fl)) / g_E1(r, fl, 0.0, 0.0))
                               for r in (0.5, 1.0, 1.58, 2.0, 3.0, 4.0, 6.0)} for fl in ('c', 'b')}

# ======================================================================== 2. widths at the reference fit
print('== 2. widths at the reference fit (NL, core)')
FL = FITS[(False, 'core', 'v2')][0]                     # local form, for the LO (L) column
pul_v, pul_0 = F.pulls(pV, EPS_V2), F.pulls(p0, 0.0)
widths = []
for k, (row, o) in enumerate(zip(DATA, F.obs)):
    d = ' / '.join(f"{q['delta']:+.4f}" for q in o.parts)
    w = {'id': row['id'], 'kind': row['kind'], 'set': row['set'], 'flavor': row['flavor'], 'fitted': F.fitted[k],
         'data': row['value'], 'err_exp': pul_v[k][3], 'err_up': row.get('err_up', np.nan), 'err_dn': row.get('err_dn', np.nan),
         'LO_L': FL.obs[k].predict(ZERO, 0.0, corr=False)[0], 'LO': o.predict(ZERO, 0.0, corr=False)[0], 'delta_1m': d,
         'LO1m': o.predict(ZERO, 0.0)[0],
         'fit_v2': pul_v[k][0], 'sth_v2': pul_v[k][1], 'dfit_lo_v2': Fit.prediction_range(o, RV)[0], 'dfit_hi_v2': Fit.prediction_range(o, RV)[1], 'pull_v2': pul_v[k][2],
         'fit_eps0': pul_0[k][0], 'pull_eps0': pul_0[k][2]}
    widths.append(w)
    print(f"  {row['id']:24s} {'fit ' if F.fitted[k] else 'pred'} data {row['value']:9.4g} | LO {w['LO']:9.4g} LO+1/m {w['LO1m']:9.4g} "
          f"| v2 {w['fit_v2']:9.4g} +- {w['sth_v2']:8.3g} pull {w['pull_v2']:+.2f} | eps0 {w['fit_eps0']:9.4g} pull {w['pull_eps0']:+.2f}")
write_csv('widths.csv', widths)

# ======================================================================== 3. J dependence of the multiplets
# At LO the radial integral does not depend on J: Gamma_J = (4/3) alpha e_Q^2 E_J^3 S_J |I|^2 (local form), with the
# spin factor S_J = (2 Jf + 1) max(l, l') {Ji Jf 1; lf li 1}^2. The reduced width Gamma_J/(S_J E_J^3) is the same for
# all J of a multiplet; we give it relative to a reference J (data and model). For the chi_bJ(2P) ratios
# Gamma(-> Y(1S))/Gamma(-> Y(2S)) the spin factors cancel and the reduced ratio is R_J (E_2S/E_1S)^3.
print('== 3. J dependence of the multiplets (reduced widths relative to the reference J)')
byid = {r['id']: k for k, r in enumerate(DATA)}


def factor(init, final):
    _, _, li, S, Ji = STATES[init]
    _, _, lf, _, Jf = STATES[final]
    E = photon_energy(MASS[init], MASS[final])
    return spin_factor(li, S, Ji, lf, Jf) * E ** 3, E


MULTIPLETS = [  # (name, [(J, data id)], reference J, same total width (only the BR errors enter))
    ('psi2S->chicJ', [(J, f'psi2S->chic{J}') for J in (0, 1, 2)], 2, True),
    ('chicJ->Jpsi', [(J, f'chic{J}->Jpsi') for J in (0, 1, 2)], 2, False),
    ('Y2S->chibJ(1P)', [(J, f'Y2S->chib{J}(1P)') for J in (0, 1, 2)], 2, True),
    ('Y3S->chibJ(2P)', [(J, f'Y3S->chib{J}(2P)') for J in (0, 1, 2)], 2, True),
    ('Y3S->chibJ(1P)', [(J, f'Y3S->chib{J}(1P)') for J in (0, 1, 2)], 2, True),
    ('psi3770->chicJ', [(J, f'psi3770->chic{J}') for J in (0, 1, 2)], 1, True),
    ('chibJ(2P): Y1S/Y2S', [(J, f'chib{J}(2P): Y1S/Y2S') for J in (0, 1, 2)], 2, False),
    ('psi2(3823): chicJ/chic1', [(2, 'psi2(3823): chic2/chic1')], 1, False),
]   # psi_2(3823) -> chi_c0 gamma (3D2 -> 3P0, Delta J = 2) vanishes for E1: its spin factor is zero


def reduced(k):
    """reduced factor of row k: S_J E_J^3 (width) or the ratio of the factors (ratio); and the photon energy"""
    row = DATA[k]
    f1, E1 = factor(row['init'], row['final'])
    if 'final2' in row:
        f2, _ = factor(row['init'], row['final2'])
        return f1 / f2, E1
    return f1, E1


jrows = []
for name, members, Jref, same in MULTIPLETS:
    if name.startswith('psi2(3823)'):            # data are already ratios to the J = 1 width
        kref, base = None, {'exp': 1.0, 'LO': 1.0, 'LO1m': 1.0, 'fit': 1.0}
    else:
        kref = byid[dict(members)[Jref]]
        fref = reduced(kref)[0]
        base = {'exp': DATA[kref]['value'] / fref, 'LO': widths[kref]['LO'] / fref,
                'LO1m': widths[kref]['LO1m'] / fref, 'fit': widths[kref]['fit_v2'] / fref}
    for J, rid in members:
        if J == Jref:
            continue
        k = byid[rid]
        row, w = DATA[k], widths[k]
        fJ, E = reduced(k)
        out = {'multiplet': name, 'J': J, 'J_ref': Jref, 'id': rid, 'E_MeV': 1e3 * E, 'kind': row['kind']}
        for t, val in (('LO', w['LO']), ('LO1m', w['LO1m']), ('fit', w['fit_v2'])):
            out[t] = val / fJ / base[t]
        out['exp'] = row['value'] / fJ / base['exp']
        # experimental error of the reduced ratio (relative errors in quadrature; for a common total width only
        # the branching-fraction errors enter)
        def rel(kk):
            r_ = DATA[kk]
            e = r_['err'] if same else np.hypot(r_['err'], r_.get('err_corr', 0.0))
            return e / r_['value']
        if row['kind'].startswith('ul'):             # 'exp' is then the 90% CL upper limit
            out['exp_err'] = np.nan
        elif 'err_up' in row:
            out['exp_err_up'] = out['exp'] * row['err_up'] / row['value']
            out['exp_err_dn'] = out['exp'] * row['err_dn'] / row['value']
            out['exp_err'] = 0.5 * (out['exp_err_up'] + out['exp_err_dn'])
        else:
            out['exp_err'] = out['exp'] * np.hypot(rel(k), rel(kref) if kref is not None else 0.0)
        jrows.append(out)
        print(f"  {name:24s} J={J} (ref {Jref}) E={1e3 * E:6.1f} MeV | exp {out['exp']:.3f} +- {out['exp_err']:.3f} | "
              f"LO(NL) {out['LO']:.3f}  LO+1/m {out['LO1m']:.3f}  fit {out['fit']:.3f}")
write_csv('j_ratios.csv', jrows)

# ======================================================================== 4. lattice QCD
print('== 4. lattice QCD E1 widths versus LO and the reference fit')
QLAT = [('chic0->Jpsi', 'chic0', 'Jpsi', 208.6, 0.3, 'HSC'), ('chic1->Jpsi', 'chic1', 'Jpsi', 327.0, 10.0, 'ETMC'),
        ('chic2->Jpsi', 'chic2', 'Jpsi', 368.0, 5.0, 'CLQCD'), ('psi2(3823)->chic1', 'psi2_3823', 'chic1', 337.0, 27.0, 'CLQCD')]
# HSC (arXiv:2301.08213, Eq. (5.5)): 208.6(3) keV with experimental masses, 247.4(2) keV with lattice masses.
# ETMC (arXiv:2604.15256): Gamma(chi_c1 -> J/psi e+e-) = 2.869(90) keV and Gamma(ee)/Gamma(gamma) = 8.78(2)e-3.
# CLQCD (arXiv:2309.09662): N_f = 2, m_pi = 350 MeV.
lat = []
for qid, ini, fin, G, dG, src in QLAT:
    row = {'flavor': 'c', 'kind': 'width', 'init': ini, 'final': fin}
    oN, oL = Observable(row, libs, True), Observable(row, libs, False)
    k = byid.get(qid)
    out = {'id': qid, 'source': src, 'lat': G, 'dlat': dG,
           'data': DATA[k]['value'] if k is not None else np.nan, 'ddata': widths[k]['err_exp'] if k is not None else np.nan,
           'LO_L': oL.predict(ZERO, 0.0, corr=False)[0], 'LO_NL': oN.predict(ZERO, 0.0, corr=False)[0],
           'LO1m_NL': oN.predict(ZERO, 0.0)[0], 'fit': oN.predict(pV, 0.0)[0]}
    out['dfit_lo'], out['dfit_hi'] = Fit.prediction_range(oN, RV)
    out['fit_over_lat'] = out['fit'] / G
    lat.append(out)
    print(f"  {qid:18s} {src:5s} lat {G:6.1f}({dG:g}) | LO L {out['LO_L']:6.1f} NL {out['LO_NL']:6.1f} LO+1/m {out['LO1m_NL']:6.1f} "
          f"| fit {out['fit']:6.1f} +{out['dfit_hi']:4.1f} -{out['dfit_lo']:4.1f} (fit/lat {out['fit_over_lat']:.3f}) | RPP {out['data']:.1f}")
write_csv('lattice.csv', lat)
summary['chic0_lattice_masses_keV'] = 247.4
# J dependence of the lattice chi_cJ -> gamma J/psi widths (three different lattice set-ups), as in j_ratios.csv
red = {J: lat[J]['lat'] / factor(f'chic{J}', 'Jpsi')[0] for J in (0, 1, 2)}
summary['lattice_J_ratios_chicJ_Jpsi'] = {f'{J}/2': red[J] / red[2] for J in (0, 1)}
print('  lattice chi_cJ -> gamma J/psi, reduced widths relative to J = 2:', summary['lattice_J_ratios_chicJ_Jpsi'])

with open(os.path.join(RES, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=1, default=lambda x: x.tolist() if hasattr(x, 'tolist') else float(x))
print('done')
