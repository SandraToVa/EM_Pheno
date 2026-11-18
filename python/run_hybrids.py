"""Hybrid -> quarkonium + photon (EM_Pheno v10): fit of the short-distance slope B_Q of the hybrid vertex to the
quenched lattice widths of Dudek:2009kk, with kappa_Q fixed by the quarkonium fit, and predictions.

    python3 run_all.py        (quarkonium; writes ../results/summary.json, read here)
    python3 run_hybrids.py    (about 1 min)

Hybrid vertex, Eq. (eq:kappaH): c1 = c2 = c3 = c4 = e kappa_H(r), kappa_H = (B_Q r + kappa_Q x)/(1 + x), x = (r/r0)^2,
kappa_Q of the reference quarkonium fit (NL, core, eps = v^2), B_c fitted, B_b = -2 B_c.

Writes to ../results:
  hybrid_fit.csv          fits of B_c: form (NL, L), theory error (t0, model, eps0), the two minima (branch + / -),
                          and the checks (lattice masses +-1 sigma, kappa_c +-1 sigma)          -> Table tab:H_fit
  hybrid_lattice.csv      lattice and J&S kinematics: our widths with their channel split, lattice, J&S
                                                                                              -> Table tab:H_lattice
  hybrid_predictions.csv  S = 0 hybrids -> S = 0 quarkonia, charm and bottom, NL (split) and L, fit-error range,
                          other branch, BR                                                    -> Table tab:H_pred
  hybrid_spin1.csv        S = 1 members (Jcal) -> S = 1 quarkonia (J'), NL                    -> Table tab:H_spin1
  hybrid_besiii.csv       psi(4040), psi(4415) -> gamma eta_c against the limits from BESIII  -> Table tab:H_besiii
  hybrid_curves.csv       Gamma(E_q) of 1(s/d)1 -> gamma 1S (total and channels) and kappa_H(r)  -> figures
  hybrid_summary.json     numbers quoted in the text
"""
import os, sys, csv, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from emlib.states import Library, hybrid
from emlib.e1 import photon_energy, R0, CHARGE
from emlib.hybrid import HybridTransition, kappa_H, flavour_B
from emlib.hybrid_fit import HybridFit, lattice_transitions
from emlib.hybrid_data import (LATTICE_FIT, LATTICE_UL, HYBRIDS, FINALS_S0, FINALS_S1, JS_MODEL, EPS_H, GYS,
                               ETAC1_LATTICE, FLUX_TUBE, LATTICE_CONVENTIONAL, besiii_gamma_etac_limits)

RES = os.path.join(os.path.dirname(__file__), '..', 'results')
REF_FORM = True                      # reference: non-local form


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
dkb_lo, dkb_hi = QSUM['ref_v2']['errors_b'][1]                 # Delta chi^2 = 1 errors of kappa_b
KAPPA_C_RANGE = (KAPPA['c'] - dkb_hi / 2.0, KAPPA['c'] + dkb_lo / 2.0)  # kappa_c = -kappa_b/2
libs = {'c': Library('c'), 'b': Library('b')}
summary = {'kappa_c': KAPPA['c'], 'kappa_b': KAPPA['b'], 'kappa_c_range': KAPPA_C_RANGE, 'eps_H': EPS_H,
           'r0': R0}
print(f"kappa_c = {KAPPA['c']:+.5f} [{KAPPA_C_RANGE[0]:+.4f}, {KAPPA_C_RANGE[1]:+.4f}], kappa_b = {KAPPA['b']:+.5f}")

# ============================================================================================ 1. fit of B_c
print('== 1. fit of B_c to the quenched lattice widths (Dudek:2009kk)')
fit_rows, REF = [], {}
CASES = [('NL', True, 't0', EPS_H, 0.0, KAPPA['c'], 'ref'), ('L', False, 't0', EPS_H, 0.0, KAPPA['c'], 'form'),
         ('NL', True, 'model', EPS_H, 0.0, KAPPA['c'], 'theory'), ('NL', True, 't0', 0.0, 0.0, KAPPA['c'], 'eps0'),
         ('NL', True, 't0', EPS_H, -1.0, KAPPA['c'], 'M-'), ('NL', True, 't0', EPS_H, +1.0, KAPPA['c'], 'M+'),
         ('NL', True, 't0', EPS_H, 0.0, KAPPA_C_RANGE[0], 'kappa-'), ('NL', True, 't0', EPS_H, 0.0, KAPPA_C_RANGE[1], 'kappa+')]
for form, nl, th, eps, dM, kc, tag in CASES:
    F = HybridFit(libs['c'], kc, nonloc=nl, eps=eps, dM=dM, theory=th)
    mins = F.run()
    for m in mins:
        branch = '+' if m['B'] > 0 else '-'
        F.B_t0 = m['B'] if (th == 't0' and eps > 0) else None
        mod = F.model(m['B'])
        out = {'case': tag, 'form': form, 'theory': th, 'eps': eps, 'dM_sigma': dM, 'kappa_c': kc, 'branch': branch,
               'B_c': m['B'], 'B_c_lo': m['err'][0], 'B_c_hi': m['err'][1], 'B_b': flavour_B(m['B'], 'b'),
               'chi2': m['chi2'], 'ndf': len(LATTICE_FIT) - 1}
        for d, (G, s) in zip(LATTICE_FIT, mod):
            out[f"G_{d['id']}"] = G
            out[f"sth_{d['id']}"] = s
            out[f"pull_{d['id']}"] = (d['value'] - G) / np.hypot(d['err'], s)
        out['R_model'] = mod[1][0] / mod[0][0]
        fit_rows.append(out)
        if tag == 'ref' or (tag == 'form' and branch == '+'):
            REF[(form, branch)] = m
        print(f"  {tag:7s} {form:2s} {th:5s} eps={eps:.1f} dM={dM:+.0f} kappa_c={kc:+.4f} [{branch}] B_c = {m['B']:+.4f} "
              f"-{m['err'][0] or np.nan:.4f} +{m['err'][1] or np.nan:.4f}  chi2 = {m['chi2']:.2f} | "
              f"Y {mod[0][0]:6.1f} +- {mod[0][1]:5.1f}  eta_c1 {mod[1][0]:6.1f} +- {mod[1][1]:5.1f}  R {out['R_model']:.2f}")
        F.B_t0 = None
write_csv('hybrid_fit.csv', fit_rows)
R_LAT = LATTICE_FIT[1]['value'] / LATTICE_FIT[0]['value']
dR_LAT = R_LAT * np.hypot(LATTICE_FIT[0]['err'] / LATTICE_FIT[0]['value'], LATTICE_FIT[1]['err'] / LATTICE_FIT[1]['value'])
ref_rows = [r for r in fit_rows if r['case'] == 'ref']
summary['fit'] = {r['branch']: {k: r[k] for k in ('B_c', 'B_c_lo', 'B_c_hi', 'B_b', 'chi2', 'R_model',
                                                  'G_Y->eta_c', 'G_eta_c1->J/psi', 'sth_Y->eta_c', 'sth_eta_c1->J/psi',
                                                  'pull_Y->eta_c', 'pull_eta_c1->J/psi')} for r in ref_rows}
summary['R_lattice'] = (R_LAT, dR_LAT)
BREF = min(ref_rows, key=lambda r: r['chi2'])['branch']       # reference branch: lowest chi^2
BALT = '-' if BREF == '+' else '+'
summary['ref_branch'] = BREF


def params(fl, branch=None, B_c=None):
    """(B_Q, kappa_Q) of flavour fl at the reference fit (branch) or at a given B_c"""
    if B_c is None:
        B_c = summary['fit'][branch or BREF]['B_c']
    return flavour_B(B_c, fl), KAPPA[fl]


B_REF = summary['fit'][BREF]['B_c']
B_RANGE = (B_REF - summary['fit'][BREF]['B_c_lo'], B_REF + summary['fit'][BREF]['B_c_hi'])
print(f"  reference: branch {BREF}, B_c = {B_REF:+.4f} [{B_RANGE[0]:+.4f}, {B_RANGE[1]:+.4f}], B_b = {flavour_B(B_REF, 'b'):+.4f}")
print(f"  lattice R = Gamma(eta_c1)/Gamma(Y) = {R_LAT:.2f} +- {dR_LAT:.2f}")
# kappa_H(r) at a few distances
summary['kappa_H'] = {fl: {br: {f'{r:g}': float(kappa_H(r, *params(fl, br))) for r in (0.5, 1.0, 1.58, 2.0, 3.0, 5.0)}
                           for br in ('+', '-')} for fl in ('c', 'b')}
# lattice systematics from the conventional transitions of the same calculation
summary['lattice_conventional_ratios'] = {k: v[0] / v[2] for k, v in LATTICE_CONVENTIONAL.items()}


def widths_with_errors(T, fl):
    """total width at the reference, its range over the Delta chi^2 = 1 interval of B_c (min, max), the other
    branch, and the reference split"""
    vals = [T.width(params(fl, B_c=b)) for b in np.linspace(B_RANGE[0], B_RANGE[1], 21)]
    sp = T.split(params(fl))
    return sp, min(vals), max(vals), T.width(params(fl, BALT))


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
    if final is None:                         # Delta S = 1 (S = 0 -> h_c is listed with S = 0 finals; here S changes)
        row.update({'M_f': np.nan, 'E_q': np.nan, 'NL': 0.0, 'note': 'Delta S = 1: zero at our order'})
        lat_rows.append(row)
        print(f"  {name:18s} Delta S = 1 -> 0          J&S {js:5.1f}")
        continue
    st = hybrid(libs['c'], name, h[1], h[2], h[3], S=S, Jcal=Jcal)
    E = photon_energy(Mi, final.mass)
    for form, nl in (('NL', True), ('L', False)):
        T = HybridTransition(final, st, E, libs['c'].r, nl)
        sp, lo, hi, alt = widths_with_errors(T, 'c')
        row.update({'M_f': final.mass, 'E_q': E, form: sp['total'], f'{form}_lo': lo, f'{form}_hi': hi, f'{form}_alt': alt})
        if nl:
            row.update({'Pi': sp['Pi'], 'Sigma': sp['Sigma'], 'int': sp['int']})
    lat_rows.append(row)
    print(f"  {name:18s} E_q = {E * 1e3:6.1f} MeV  NL {row['NL']:6.1f} [{row['NL_lo']:.1f}, {row['NL_hi']:.1f}] "
          f"(Pi {row['Pi']:.1f}, Sigma {row['Sigma']:.1f}, int {row['int']:.1f}) L {row['L']:6.1f} alt {row['NL_alt']:6.1f}"
          f" | lat {row['lat_keV']} | J&S {js:5.1f}")
write_csv('hybrid_lattice.csv', lat_rows)
summary['lattice_kin'] = {r['channel']: r.get('NL', 0.0) for r in lat_rows}

# eta_c1 at the N_f = 2 lattice mass (Shi:2023sdy)
st = hybrid(libs['c'], 'eta_c1', 1, 'plus', 1, S=1, Jcal=1)
jpsi = libs['c'].state('J/psi', 1, 0, 1, 1, 3.096900)
E = photon_energy(ETAC1_LATTICE['M'], jpsi.mass)
T = HybridTransition(jpsi, st, E, libs['c'].r, True)
sp, lo, hi, alt = widths_with_errors(T, 'c')
summary['etac1_Shi'] = {'M': ETAC1_LATTICE['M'], 'E_q': E, 'Gamma': sp['total'], 'range': (lo, hi), 'alt': alt,
                        'BR': sp['total'] * 1e-3 / ETAC1_LATTICE['Gamma_tot_MeV']}
# GYS mass (4.35 GeV): Y -> eta_c and S = 1 -> J/psi
for lab, S, fin in (('Y->eta_c', 0, libs['c'].state('eta_c', 1, 0, 0, 0, 2.98409)), ('S1->J/psi', 1, jpsi)):
    st = hybrid(libs['c'], lab, 1, 'plus', 1, S=S, Jcal=1)
    T = HybridTransition(fin, st, photon_energy(GYS['M'], fin.mass), libs['c'].r, True)
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
            TN = HybridTransition(final, st, E, lib.r, True)
            if np.max(np.abs(TN.form())) < 1e-9:              # forbidden (parity, rank)
                continue
            TL = HybridTransition(final, st, E, lib.r, False)
            sp, lo, hi, alt = widths_with_errors(TN, fl)
            v = params(fl)
            rows_h.append({'flavor': fl, 'hybrid': h[0], 'JPC': h[4], 'JPC_S1': h[5], 'candidate': h[6], 'M_i': h[7],
                           'M_i_src': h[8], 'Gtot_MeV': h[9], 'P_Sigma': pr['P_Sigma'], 'final': f[0], 'M_f': f[3],
                           'M_f_src': f[4], 'E_q': E, 'NL': sp['total'], 'NL_lo': lo, 'NL_hi': hi,
                           'Pi': sp['Pi'], 'Sigma': sp['Sigma'], 'int': sp['int'], 'L': TL.width(v), 'NL_alt': alt,
                           'Gbar': TN.width_abs(v),
                           # flavour scaling B_Q proportional to e_Q instead of 1/e_Q (B_b = -B_c/2), kappa_Q as fitted
                           'NL_eQ': TN.width((B_REF * CHARGE[fl] / CHARGE['c'], KAPPA[fl])),
                           'BR': sp['total'] * 1e-3 / h[9] if h[9] else np.nan})
        gmax = max((r['NL'] for r in rows_h), default=0.0)
        for r in rows_h:
            r['shown'] = r['NL'] >= 1e-3 * gmax
            print(f"  {fl} {r['hybrid']:8s} {r['candidate']:15s} -> {r['final']:11s} E={1e3 * r['E_q']:6.1f}  "
                  f"NL {r['NL']:9.3g} [{r['NL_lo']:.3g}, {r['NL_hi']:.3g}] (Pi {r['Pi']:.3g} Sig {r['Sigma']:.3g} "
                  f"int {r['int']:.3g})  L {r['L']:9.3g}  alt {r['NL_alt']:9.3g}")
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
                T = HybridTransition(final, st, E, lib.r, True)
                if np.max(np.abs(T.form())) < 1e-9:
                    continue
                sp, lo, hi, alt = widths_with_errors(T, fl)
                spin1.append({'flavor': fl, 'hybrid': h[0], 'Jcal': Jcal, 'M_i': h[7], 'final': f[0], 'n_f': f[1],
                              'l_f': f[2], 'J_f': f[3], 'M_f': f[4], 'E_q': E, 'NL': sp['total'], 'NL_lo': lo,
                              'NL_hi': hi, 'Pi': sp['Pi'], 'Sigma': sp['Sigma'], 'int': sp['int'], 'NL_alt': alt})
write_csv('hybrid_spin1.csv', spin1)
print(f'  {len(spin1)} channels')

# ============================================================================ 5. BESIII gamma eta_c limits
print('== 5. BESIII limits')
ULs = besiii_gamma_etac_limits()
bes = []
for cand, hl in (('psi(4040)', '1(s/d)1'), ('psi(4415)', '2(s/d)1')):
    r = [p for p in pred if p['flavor'] == 'c' and p['candidate'] == cand and p['final'] == 'eta_c(1S)'][0]
    out = {'candidate': cand, 'hybrid': hl, **ULs[cand], 'NL': r['NL'], 'NL_lo': r['NL_lo'], 'NL_hi': r['NL_hi'],
           'Pi': r['Pi'], 'Sigma': r['Sigma'], 'int': r['int'], 'L': r['L'], 'NL_alt': r['NL_alt']}
    out['ratio'] = r['NL'] / ULs[cand]['Gamma_UL_keV']
    bes.append(out)
    print(f"  {cand}: NL {r['NL']:.1f} [{r['NL_lo']:.1f}, {r['NL_hi']:.1f}] keV  vs UL {ULs[cand]['Gamma_UL_keV']:.1f} keV")
write_csv('hybrid_besiii.csv', bes)
summary['besiii'] = bes

# ========================================================================================== 6. curves
print('== 6. curves')
curves = []
lib = libs['c']
Y = hybrid(lib, 'Y', 1, 'plus', 1)
etac = lib.state('eta_c', 1, 0, 0, 0, 2.98409)
for E in np.linspace(0.6, 1.6, 51):
    T = HybridTransition(etac, Y, E, lib.r, True)
    row = {'curve': 'Gamma(E)', 'x': E}
    for br in ('+', '-'):
        sp = T.split(params('c', br))
        row.update({f'total_{br}': sp['total'], f'Pi_{br}': sp['Pi'], f'Sigma_{br}': sp['Sigma'], f'int_{br}': sp['int']})
    row['total_Brange_lo'] = min(T.width(params('c', B_c=b)) for b in np.linspace(*B_RANGE, 11))
    row['total_Brange_hi'] = max(T.width(params('c', B_c=b)) for b in np.linspace(*B_RANGE, 11))
    curves.append(row)
for r in np.linspace(0.05, 8.0, 160):
    row = {'curve': 'kappa_H(r)', 'x': r}
    for fl in ('c', 'b'):
        for br in ('+', '-'):
            row[f'{fl}_{br}'] = float(kappa_H(r, *params(fl, br)))
    curves.append(row)
write_csv('hybrid_curves.csv', curves)

with open(os.path.join(RES, 'hybrid_summary.json'), 'w') as f:
    json.dump(summary, f, indent=1, default=lambda x: x.tolist() if hasattr(x, 'tolist') else float(x))
print('done')
