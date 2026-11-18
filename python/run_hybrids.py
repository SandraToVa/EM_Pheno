"""Hybrid -> quarkonium + photon (EM_Pheno v12, local form): fit of the short-distance coefficient B_Q of the
interpolated hybrid LEFs to the quenched lattice widths of Dudek:2009kk, with kappa_Q fixed by the quarkonium fit,
and predictions.

    python3 run_all.py        (quarkonium; writes ../results/summary.json, read here)
    python3 run_hybrids.py    (about 10 s)

LEFs, Eq. (eq:gH_interp): g_B(r) = e [B_Q r + kappa_Q ghat_B(r) x]/(1 + x) for g_B_perp, g_B_par;
g_E(r) = e kappa_Q ghat_E(r) x/(1 + x) for g_E_par, g_E_perp, g_E^Sigma; ghat_X the EST shapes of Table tab:HQ_lef;
x = (r/r0)^2. kappa_Q of the reference quarkonium fit (L form is used here for the hybrids; kappa_Q is that of the
reference NL quarkonium fit), B_c fitted, B_b = -2 B_c. Theory error sigma_th = 2 eps Gamma at the model width
(eps = 0.3), as in the quarkonium fit; errors from the fits to Gamma_lat +- sigma_lat (as Eq. (eq:exp_range)).

Writes to ../results:
  hybrid_fit.csv          fits of B_c: reference and checks (eps = 0, lattice masses +-1 sigma, kappa_c at the ends of
                          its quarkonium interval), both minima (branch + / -)                 -> Table tab:H_fit
  hybrid_lattice.csv      lattice and J&S kinematics: our widths with their channel split, lattice, J&S
                                                                                              -> Table tab:H_lattice
  hybrid_predictions.csv  S = 0 hybrids -> S = 0 quarkonia, charm and bottom, with split, error range, other branch,
                          B_Q proportional to e_Q (bottom), BR                                -> Tables tab:H_pred_c/b
  hybrid_spin1.csv        S = 1 members (Jcal) -> S = 1 quarkonia (J')                        -> Table tab:H_spin1
  hybrid_besiii.csv       psi(4040), psi(4415) -> gamma eta_c against the limits from BESIII  -> Table tab:H_besiii
  hybrid_curves.csv       Gamma(E_q) of 1(s/d)1 -> gamma 1S (total and channels) and the LEFs g_X(r) -> figures
  hybrid_summary.json     numbers quoted in the text
"""
import os, sys, csv, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from emlib.states import Library, hybrid
from emlib.e1 import photon_energy, R0, CHARGE
from emlib.hybrid import HybridTransition, lefs, flavour_B
from emlib.hybrid_fit import HybridFit
from emlib.hybrid_data import (LATTICE_FIT, LATTICE_UL, HYBRIDS, FINALS_S0, FINALS_S1, JS_MODEL, EPS_H, GYS,
                               ETAC1_LATTICE, LATTICE_CONVENTIONAL, besiii_gamma_etac_limits)

RES = os.path.join(os.path.dirname(__file__), '..', 'results')


def write_csv(name, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(RES, name), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f'{v:.6g}' if isinstance(v, float) else v) for k, v in r.items()})


with open(os.path.join(RES, 'summary.json')) as f:
    QSUM = json.load(f)
KAPPA = {'c': QSUM['ref_v2']['kappa_c'], 'b': QSUM['ref_v2']['kappa_b']}
dkb_lo, dkb_hi = QSUM['ref_v2']['errors_b'][1]                 # range of kappa_b over the shifted quarkonium fits
KAPPA_C_RANGE = (KAPPA['c'] - dkb_hi / 2.0, KAPPA['c'] + dkb_lo / 2.0)  # kappa_c = -kappa_b/2
libs = {'c': Library('c'), 'b': Library('b')}
summary = {'kappa_c': KAPPA['c'], 'kappa_b': KAPPA['b'], 'kappa_c_range': KAPPA_C_RANGE, 'eps_H': EPS_H,
           'r0': R0, 'form': 'L'}
print(f"kappa_c = {KAPPA['c']:+.5f} [{KAPPA_C_RANGE[0]:+.4f}, {KAPPA_C_RANGE[1]:+.4f}], kappa_b = {KAPPA['b']:+.5f}")

# ============================================================================================ 1. fit of B_c
print('== 1. fit of B_c to the quenched lattice widths (Dudek:2009kk), local form')
fit_rows, REF = [], {}
CASES = [('ref', EPS_H, 0.0, KAPPA['c']), ('eps0', 0.0, 0.0, KAPPA['c']),
         ('M-', EPS_H, -1.0, KAPPA['c']), ('M+', EPS_H, +1.0, KAPPA['c']),
         ('kappa-', EPS_H, 0.0, KAPPA_C_RANGE[0]), ('kappa+', EPS_H, 0.0, KAPPA_C_RANGE[1])]
for tag, eps, dM, kc in CASES:
    F = HybridFit(libs['c'], kc, eps=eps, dM=dM)
    for m in F.run():
        mod = F.model(m['B'])
        out = {'case': tag, 'eps': eps, 'dM_sigma': dM, 'kappa_c': kc, 'branch': m['branch'],
               'B_c': m['B'], 'B_c_lo': m['err'][0], 'B_c_hi': m['err'][1], 'B_c_max': m['B_max'], 'B_c_min': m['B_min'],
               'B_b': flavour_B(m['B'], 'b'), 'chi2': m['chi2'], 'chi2_max': m['chi2_max'], 'chi2_min': m['chi2_min'],
               'ndf': len(LATTICE_FIT) - 1}
        for d, (G, s) in zip(LATTICE_FIT, mod):
            out[f"G_{d['id']}"] = G
            out[f"sth_{d['id']}"] = s
            out[f"pull_{d['id']}"] = (d['value'] - G) / np.hypot(d['err'], s)
        out['R_model'] = mod[1][0] / mod[0][0]
        fit_rows.append(out)
        if tag == 'ref':
            REF[m['branch']] = m
        print(f"  {tag:7s} eps={eps:.1f} dM={dM:+.0f} kappa_c={kc:+.4f} [{m['branch']}] B_c = {m['B']:+.4f} "
              f"-{m['err'][0]:.4f} +{m['err'][1]:.4f} (max {m['B_max']:+.4f}, min {m['B_min']:+.4f})  "
              f"chi2 = {m['chi2']:.2f} | Y {mod[0][0]:6.1f} +- {mod[0][1]:5.1f}  eta_c1 {mod[1][0]:6.1f} +- "
              f"{mod[1][1]:5.1f}  R {out['R_model']:.2f}")
write_csv('hybrid_fit.csv', fit_rows)
R_LAT = LATTICE_FIT[1]['value'] / LATTICE_FIT[0]['value']
dR_LAT = R_LAT * np.hypot(LATTICE_FIT[0]['err'] / LATTICE_FIT[0]['value'], LATTICE_FIT[1]['err'] / LATTICE_FIT[1]['value'])
ref_rows = [r for r in fit_rows if r['case'] == 'ref']
KEYS = ('B_c', 'B_c_lo', 'B_c_hi', 'B_c_max', 'B_c_min', 'B_b', 'chi2', 'chi2_max', 'chi2_min', 'R_model',
        'G_Y->eta_c', 'G_eta_c1->J/psi', 'sth_Y->eta_c', 'sth_eta_c1->J/psi', 'pull_Y->eta_c', 'pull_eta_c1->J/psi')
summary['fit'] = {r['branch']: {k: r[k] for k in KEYS} for r in ref_rows}
summary['R_lattice'] = (R_LAT, dR_LAT)
BREF = min(ref_rows, key=lambda r: r['chi2'])['branch']       # reference branch: lowest chi^2
BALT = '-' if BREF == '+' else '+'
summary['ref_branch'] = BREF
B_SET = {br: (summary['fit'][br]['B_c'], summary['fit'][br]['B_c_max'], summary['fit'][br]['B_c_min'])
         for br in summary['fit']}                             # central, max and min fits of each branch


def params(fl, B_c):
    """(B_Q, kappa_Q) of flavour fl at a given B_c"""
    return flavour_B(B_c, fl), KAPPA[fl]


print(f"  reference: branch {BREF}, B_c = {B_SET[BREF][0]:+.4f} (max fit {B_SET[BREF][1]:+.4f}, min fit "
      f"{B_SET[BREF][2]:+.4f}), B_b = {flavour_B(B_SET[BREF][0], 'b'):+.4f}")
print(f"  lattice R = Gamma(eta_c1)/Gamma(Y) = {R_LAT:.2f} +- {dR_LAT:.2f}")
# LEFs at a few distances (units of e: g/e), reference branch
summary['lefs'] = {fl: {X: {f'{r:g}': float(v / np.sqrt(4 * np.pi / 137.035999))
                            for r, v in zip((0.5, 1.0, 1.58, 2.0, 3.0, 5.0),
                                            lefs(np.array([0.5, 1.0, 1.58, 2.0, 3.0, 5.0]), *params(fl, B_SET[BREF][0]))[X])}
                        for X in ('B_perp', 'B_par')} for fl in ('c', 'b')}
summary['lattice_conventional_ratios'] = {k: v[0] / v[2] for k, v in LATTICE_CONVENTIONAL.items()}


def widths_with_errors(T, fl):
    """split at the reference fit, range (min, max) of the total width over the central and the two shifted fits
    of the reference branch, and the width at the central fit of the other branch"""
    vals = [T.width(params(fl, b)) for b in B_SET[BREF]]
    alt = T.width(params(fl, B_SET[BALT][0])) if BALT in B_SET else np.nan
    return T.split(params(fl, B_SET[BREF][0])), min(vals), max(vals), alt


# ============================================================================ 2. lattice and J&S kinematics
print('== 2. lattice and J&S kinematics')
lat_rows = []
hc = {h[0]: h for h in HYBRIDS['c']}
FS1 = {f[0]: f for f in FINALS_S1['c']}
FS0 = {f[0]: f for f in FINALS_S0['c']}
for name, S, Jcal, Mi, fin, js in JS_MODEL:
    h = hc['1(s/d)1']
    if S == 0:
        f = FS0.get(fin)
        final = libs['c'].state(fin, f[1], f[2], 0, f[2], f[3]) if f else None
    else:
        f = FS1.get(fin)
        final = libs['c'].state(fin, f[1], f[2], 1, f[3], f[4]) if f else None
    row = {'channel': name, 'S': S, 'Jcal': Jcal, 'M_i': Mi, 'JS_keV': js}
    lat = {d['id']: d for d in LATTICE_FIT}.get(name)
    row['lat_keV'] = lat['value'] if lat else (f"<{LATTICE_UL['UL']:g}" if name == LATTICE_UL['id'] else '')
    row['dlat_keV'] = lat['err'] if lat else ''
    if final is None:                         # Delta S = 1: zero at our order
        row.update({'M_f': np.nan, 'E_q': np.nan, 'G': 0.0, 'note': 'Delta S = 1: zero at our order'})
        lat_rows.append(row)
        print(f"  {name:18s} Delta S = 1 -> 0          J&S {js:5.1f}")
        continue
    st = hybrid(libs['c'], name, h[1], h[2], h[3], S=S, Jcal=Jcal)
    E = photon_energy(Mi, final.mass)
    T = HybridTransition(final, st, E, libs['c'].r)
    sp, lo, hi, alt = widths_with_errors(T, 'c')
    row.update({'M_f': final.mass, 'E_q': E, 'G': sp['total'], 'G_lo': lo, 'G_hi': hi, 'G_alt': alt,
                'Pi': sp['Pi'], 'Sigma': sp['Sigma'], 'int': sp['int']})
    lat_rows.append(row)
    print(f"  {name:18s} E_q = {E * 1e3:6.1f} MeV  {row['G']:6.1f} [{lo:.1f}, {hi:.1f}] (Pi {row['Pi']:.1f}, "
          f"Sigma {row['Sigma']:.1f}, int {row['int']:.1f}) alt {alt:6.1f} | lat {row['lat_keV']} | J&S {js:5.1f}")
write_csv('hybrid_lattice.csv', lat_rows)

# eta_c1 at the N_f = 2 lattice mass (Shi:2023sdy) and the GYS mass (4.35 GeV)
st = hybrid(libs['c'], 'eta_c1', 1, 'plus', 1, S=1, Jcal=1)
jpsi = libs['c'].state('J/psi', 1, 0, 1, 1, 3.096900)
T = HybridTransition(jpsi, st, photon_energy(ETAC1_LATTICE['M'], jpsi.mass), libs['c'].r)
sp, lo, hi, alt = widths_with_errors(T, 'c')
summary['etac1_Shi'] = {'M': ETAC1_LATTICE['M'], 'E_q': T.E, 'Gamma': sp['total'], 'range': (lo, hi), 'alt': alt,
                        'BR': sp['total'] * 1e-3 / ETAC1_LATTICE['Gamma_tot_MeV']}
for lab, S, fin in (('Y->eta_c', 0, libs['c'].state('eta_c', 1, 0, 0, 0, 2.98409)), ('S1->J/psi', 1, jpsi)):
    st = hybrid(libs['c'], lab, 1, 'plus', 1, S=S, Jcal=1)
    T = HybridTransition(fin, st, photon_energy(GYS['M'], fin.mass), libs['c'].r)
    sp, lo, hi, alt = widths_with_errors(T, 'c')
    summary.setdefault('GYS_kin', {})[lab] = {'Gamma': sp['total'], 'range': (lo, hi), 'GYS': GYS[lab]}
print('  eta_c1 at the Shi:2023sdy mass:', summary['etac1_Shi'])
print('  GYS kinematics:', summary['GYS_kin'])

# ============================================================== 3. predictions: S = 0 hybrids -> S = 0 quarkonia
print('== 3. predictions (S = 0)')
pred = []
for fl in ('c', 'b'):
    lib = libs[fl]
    for h in HYBRIDS[fl]:
        st = hybrid(lib, h[0], h[1], h[2], h[3])
        pr = st.projections()
        rows_h = []
        for f in FINALS_S0[fl]:
            if f[3] >= h[7]:
                continue
            final = lib.state(f[0], f[1], f[2], 0, f[2], f[3])
            E = photon_energy(h[7], f[3])
            T = HybridTransition(final, st, E, lib.r)
            if np.max(np.abs(T.form())) < 1e-9:              # forbidden (parity, rank)
                continue
            sp, lo, hi, alt = widths_with_errors(T, fl)
            rows_h.append({'flavor': fl, 'hybrid': h[0], 'JPC': h[4], 'JPC_S1': h[5], 'candidate': h[6], 'M_i': h[7],
                           'M_i_src': h[8], 'Gtot_MeV': h[9], 'P_Sigma': pr['P_Sigma'], 'final': f[0], 'M_f': f[3],
                           'M_f_src': f[4], 'E_q': E, 'G': sp['total'], 'G_lo': lo, 'G_hi': hi,
                           'Pi': sp['Pi'], 'Sigma': sp['Sigma'], 'int': sp['int'], 'G_alt': alt,
                           # flavour scaling B_Q proportional to e_Q instead of 1/e_Q (B_b = -B_c/2), kappa_Q as fitted
                           'G_eQ': T.width((B_SET[BREF][0] * CHARGE[fl] / CHARGE['c'], KAPPA[fl])),
                           'BR': sp['total'] * 1e-3 / h[9] if h[9] else np.nan})
        gmax = max((r['G'] for r in rows_h), default=0.0)
        for r in rows_h:
            r['shown'] = r['G'] >= 1e-3 * gmax
            print(f"  {fl} {r['hybrid']:8s} {r['candidate']:15s} -> {r['final']:11s} E={1e3 * r['E_q']:6.1f}  "
                  f"{r['G']:9.3g} [{r['G_lo']:.3g}, {r['G_hi']:.3g}] (Pi {r['Pi']:.3g} Sig {r['Sigma']:.3g} "
                  f"int {r['int']:.3g})  alt {r['G_alt']:9.3g}  eQ {r['G_eQ']:9.3g}")
        pred += rows_h
write_csv('hybrid_predictions.csv', pred)

# =================================================================== 4. S = 1 members -> S = 1 quarkonia
print('== 4. S = 1 hybrids')
spin1 = []
for fl in ('c', 'b'):
    lib = libs[fl]
    for h in HYBRIDS[fl]:
        J = h[1]
        for Jcal in range(abs(J - 1), J + 2):
            st = hybrid(lib, h[0], J, h[2], h[3], S=1, Jcal=Jcal)
            for f in FINALS_S1[fl]:
                if f[4] >= h[7]:
                    continue
                final = lib.state(f[0], f[1], f[2], 1, f[3], f[4])
                E = photon_energy(h[7], f[4])
                T = HybridTransition(final, st, E, lib.r)
                if np.max(np.abs(T.form())) < 1e-9:
                    continue
                sp, lo, hi, alt = widths_with_errors(T, fl)
                spin1.append({'flavor': fl, 'hybrid': h[0], 'Jcal': Jcal, 'M_i': h[7], 'final': f[0], 'n_f': f[1],
                              'l_f': f[2], 'J_f': f[3], 'M_f': f[4], 'E_q': E, 'G': sp['total'], 'G_lo': lo,
                              'G_hi': hi, 'Pi': sp['Pi'], 'Sigma': sp['Sigma'], 'int': sp['int'], 'G_alt': alt})
write_csv('hybrid_spin1.csv', spin1)
print(f'  {len(spin1)} channels')

# ============================================================================ 5. BESIII gamma eta_c limits
print('== 5. BESIII limits')
ULs = besiii_gamma_etac_limits()
bes = []
for cand, hl in (('psi(4040)', '1(s/d)1'), ('psi(4415)', '2(s/d)1')):
    r = [p for p in pred if p['flavor'] == 'c' and p['candidate'] == cand and p['final'] == 'eta_c(1S)'][0]
    out = {'candidate': cand, 'hybrid': hl, **ULs[cand], 'G': r['G'], 'G_lo': r['G_lo'], 'G_hi': r['G_hi'],
           'Pi': r['Pi'], 'Sigma': r['Sigma'], 'int': r['int'], 'G_alt': r['G_alt']}
    out['ratio'] = r['G'] / ULs[cand]['Gamma_UL_keV']
    bes.append(out)
    print(f"  {cand}: {r['G']:.1f} [{r['G_lo']:.1f}, {r['G_hi']:.1f}] keV  vs UL {ULs[cand]['Gamma_UL_keV']:.1f} keV")
write_csv('hybrid_besiii.csv', bes)
summary['besiii'] = bes

# ========================================================================================== 6. curves
print('== 6. curves')
curves = []
lib = libs['c']
Y = hybrid(lib, 'Y', 1, 'plus', 1)
etac = lib.state('eta_c', 1, 0, 0, 0, 2.98409)
for E in np.linspace(0.6, 1.6, 51):
    T = HybridTransition(etac, Y, E, lib.r)
    row = {'curve': 'Gamma(E)', 'x': E}
    for br in B_SET:
        sp = T.split(params('c', B_SET[br][0]))
        row.update({f'total_{br}': sp['total'], f'Pi_{br}': sp['Pi'], f'Sigma_{br}': sp['Sigma'], f'int_{br}': sp['int']})
    vals = [T.width(params('c', b)) for b in B_SET[BREF]]
    row['total_range_lo'], row['total_range_hi'] = min(vals), max(vals)
    curves.append(row)
for r in np.linspace(0.05, 8.0, 160):
    row = {'curve': 'LEF(r)', 'x': r}
    for fl in ('c', 'b'):
        for br in B_SET:
            g = lefs(np.array([r]), *params(fl, B_SET[br][0]))
            for X in ('B_perp', 'B_par'):
                row[f'{fl}_{X}_{br}'] = float(g[X][0])
    curves.append(row)
write_csv('hybrid_curves.csv', curves)

with open(os.path.join(RES, 'hybrid_summary.json'), 'w') as f:
    json.dump(summary, f, indent=1, default=lambda x: x.tolist() if hasattr(x, 'tolist') else float(x))
print('done')
