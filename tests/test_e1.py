"""Consistency tests of the E1 code (EM_Pheno v11), run from the tests/ folder:  python3 test_e1.py
  1  Clebsch-Gordan and 6j coefficients (against sympy)
  2  LO local widths of every transition of the data set against the closed formula
     Gamma = (4/3) alpha e_Q^2 E^3 S_if |int dr u_f r u_i|^2, S_if = (2Jf+1) max(li,lf) {Ji Jf 1; lf li S}^2
  3  spin sum rule: sum over Jf of the spin-1 widths = spin-0 width
  4  non-local widths against the multipole expansion of j0(q_z r/2) (Eq. (eq:Gamma_E1_NL))
  5  quadratic form: v^T G v = width computed directly with g_E1(r) of Eq. (eq:gE1_interp), for both flavours
  6  1/m correction delta_1m: spin term only for spin-triplet S <-> P, Delta l term
  7  flavour-independent parameters: A_c = -A_b/2, kappa_c = -kappa_b/2
  8  J dependence: at LO (local) the reduced width Gamma/(S_if E^3) is the same for all J of a multiplet
  9  theory error sigma_th = 2 eps Gamma (v11), also for ratios (2 sqrt(2) eps R)
 10  data sets: the hindered Upsilon(3S) -> gamma chi_bJ(1P) are not fitted (set ext, never in the fit sets)
 11  error method (v11): the shifted data are d +- sigma_exp (err_up/err_dn, total-width error included),
     upper limits unchanged; a fit to unshifted data reproduces the central chi^2; the max/min fits are minima
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
import numpy as np
from sympy import S as Sy
from sympy.physics.quantum.cg import CG
from sympy.physics.wigner import wigner_6j
from scipy.special import eval_legendre, spherical_jn
from emlib.angular import cg, sixj, spin_factor
from emlib.states import Library
from emlib.data import DATA, STATES
from emlib.e1 import (ALPHA_EM, E_CHARGE, CHARGE, GEV2KEV, M_QUARK, photon_energy, amplitudes, width_form, g_E1,
                      delta_1m, theory_sigma)
from emlib.fit import build_state, flavour_params, Observable, Fit, EPS_V2

ok = True


def check(name, a, b, tol):
    global ok
    rel = abs(a - b) / max(abs(b), 1e-300)
    flag = rel < tol or (b == 0 and abs(a) < 1e-12)
    ok &= flag
    print(f"{'PASS' if flag else 'FAIL'} {name}: {a:.10e} vs {b:.10e} (rel {rel:.1e})")


def width_direct(fin, ini, E, weight, r, nonloc):
    M = amplitudes(fin, ini, E, weight, r, nonloc)
    return E / (2 * np.pi) / (2 * ini.J + 1) * sum(abs(v) ** 2 for v in M.values()) * GEV2KEV


libs = {'c': Library('c'), 'b': Library('b')}

# 1. CG and 6j
for args in [(1, 0, 1, 1, 2, 1), (2, -1, 1, 1, 1, 0), (3, 2, 2, -1, 3, 1), (1, 1, 1, -1, 0, 0), (2, 0, 2, 0, 2, 0)]:
    check(f"CG{args}", cg(*args), float(CG(*[Sy(x) for x in args]).doit()), 1e-12)
for args in [(1, 2, 1, 1, 0, 1), (0, 1, 1, 1, 1, 1), (2, 1, 1, 1, 2, 1), (1, 1, 1, 2, 1, 1), (3, 2, 1, 1, 2, 2)]:
    check(f"6j{args}", sixj(*args), float(wigner_6j(*args)), 1e-12)

# 2. LO local widths against the closed formula, every transition of DATA
done = set()
for row in DATA:
    lib = libs[row['flavor']]
    ini = build_state(lib, row['init'])
    for lab in [row['final']] + ([row['final2']] if 'final2' in row else []):
        if (row['init'], lab) in done:
            continue
        done.add((row['init'], lab))
        fin = build_state(lib, lab)
        E = photon_energy(ini.mass, fin.mass)
        G = width_form(fin, ini, E, lib.r, False)[0, 0]
        I = np.sum(fin.c * ini.c * lib.r)
        Gf = 4.0 / 3.0 * ALPHA_EM * CHARGE[row['flavor']] ** 2 * E ** 3 * spin_factor(ini.l, ini.S, ini.J, fin.l, fin.J) * I ** 2 * GEV2KEV
        check(f"LO local {row['init']} -> {lab}", G, Gf, 1e-8)

# 3. spin sum rule
lib = libs['c']
for (ni, li, nf, lf) in [(2, 0, 1, 1), (1, 1, 1, 0), (1, 2, 1, 1)]:
    E = 0.3
    s0 = width_direct(lib.state('f', nf, lf, 0, lf, 0), lib.state('i', ni, li, 0, li, 0), E, E_CHARGE * lib.r, lib.r, True)
    for Ji in range(abs(li - 1), li + 2):
        s = sum(width_direct(lib.state('f', nf, lf, 1, Jf, 0), lib.state('i', ni, li, 1, Ji, 0), E, E_CHARGE * lib.r, lib.r, True)
                for Jf in range(abs(lf - 1), lf + 2))
        check(f"sum rule {ni}{'SPD'[li]} J={Ji} -> {nf}{'SPD'[lf]}", s, s0, 1e-8)

# 4. non-local widths against the multipole formula, Gamma = e^2 E^3/pi sum_K C(l K l';00)^2/(2K+1) J_K^2
xg, wg = np.polynomial.legendre.leggauss(160)
proj = lambda x, L: 0.5 * np.sum(wg[None, :] * spherical_jn(0, np.outer(x, xg) / 2) * eval_legendre(L, xg)[None, :], axis=1)
r = lib.r
for (ni, li, nf, lf, E) in [(2, 0, 1, 1, 0.13), (2, 1, 1, 0, 0.7), (2, 2, 1, 1, 0.6), (1, 3, 1, 2, 0.5)]:
    ini, fin = lib.state('i', ni, li, 0, li, 0), lib.state('f', nf, lf, 0, lf, 0)
    Gf = 0.0
    for K in range(1, 14, 2):
        FK = sum((2 * L + 1) * proj(E * r, L) * cg(1, 0, L, 0, K, 0) * cg(1, 1, L, 0, K, 1) for L in range(0, 15, 2))
        Gf += E_CHARGE ** 2 * E ** 3 / np.pi * cg(li, 0, K, 0, lf, 0) ** 2 / (2 * K + 1) * np.sum(fin.c * ini.c * r * FK) ** 2
    check(f"NL {ni}{'SPDF'[li]} -> {nf}{'SPDF'[lf]} multipole formula", width_direct(fin, ini, E, E_CHARGE * r, r, True) / GEV2KEV, Gf, 1e-8)

# 5. quadratic form against the direct width with g_E1(r) (both flavours, L and NL; g_E1 includes e e_Q)
for rid in ('chic1->Jpsi', 'psi2S->chic2', 'Y3S->chib1(1P)', 'Y2S->chib0(1P)'):
    row = [x for x in DATA if x['id'] == rid][0]
    fl, lib = row['flavor'], libs[row['flavor']]
    ini, fin = build_state(lib, row['init']), build_state(lib, row['final'])
    E = photon_energy(ini.mass, fin.mass)
    for nl in (False, True):
        G = width_form(fin, ini, E, lib.r, nl)
        for A, k in ((0.07, -0.05), (-0.12, 0.2)):
            v = np.array([1.0, A, k])
            check(f"quadratic form {rid} A={A} kappa={k} nonlocal={nl}", v @ G @ v,
                  width_direct(fin, ini, E, g_E1(lib.r, fl, A, k), lib.r, nl), 1e-10)

# 6. delta_1m
for rid, s in (('chic0->Jpsi', -1), ('chic1->Jpsi', -1), ('chic2->Jpsi', -1), ('psi2S->chic0', 1), ('Y2S->chib2(1P)', 1)):
    row = [x for x in DATA if x['id'] == rid][0]
    lib = libs[row['flavor']]
    ini, fin = build_state(lib, row['init']), build_state(lib, row['final'])
    E = photon_energy(ini.mass, fin.mass)
    JP = ini.J if ini.l == 1 else fin.J
    dl = ini.l * (ini.l + 1) - fin.l * (fin.l + 1)
    check(f"delta_1m {rid}", delta_1m(ini, fin, E), E / (8 * M_QUARK[row['flavor']]) * (s * (JP * (JP + 1) - 4) - dl / 3), 1e-12)
ini, fin = build_state(libs['c'], 'hc'), build_state(libs['c'], 'etac1S')
E = photon_energy(ini.mass, fin.mass)
check("delta_1m h_c -> eta_c (spin singlet: Delta l only)", delta_1m(ini, fin, E), -E / (8 * M_QUARK['c']) * 2 / 3, 1e-12)
ini, fin = build_state(libs['c'], 'psi3770'), build_state(libs['c'], 'chic1')
E = photon_energy(ini.mass, fin.mass)
check("delta_1m psi(3770) -> chi_c1 (D -> P: Delta l only)", delta_1m(ini, fin, E), -E / (8 * M_QUARK['c']) * 4 / 3, 1e-12)

# 7. flavour-independent parameters
Ac, kc = flavour_params((0.03, -0.1), 'c')
check("A_c = -A_b/2", Ac, -0.015, 1e-12)
check("kappa_c = -kappa_b/2", kc, 0.05, 1e-12)
check("A_Q e_Q flavour independent", Ac * CHARGE['c'], 0.03 * CHARGE['b'], 1e-12)

# 8. J dependence at LO (local): reduced widths equal within a multiplet
for init, fins in (('psi2S', ('chic0', 'chic1', 'chic2')), ('Y3S', ('chib0_2P', 'chib1_2P', 'chib2_2P'))):
    red = []
    for fl_ in fins:
        o = Observable({'flavor': STATES[init][0], 'kind': 'width', 'init': init, 'final': fl_}, libs, False)
        _, _, li, S, Ji = STATES[init]
        _, _, lf, _, Jf = STATES[fl_]
        red.append(o.predict((0.0, 0.0), 0.0, corr=False)[0] / (spin_factor(li, S, Ji, lf, Jf) * o.parts[0]['E'] ** 3))
    for x in red[:-1]:
        check(f"reduced LO width equal for all J, {init} -> {fins}", x, red[-1], 1e-10)
for Ji in (0, 1, 2):
    o = Observable({'flavor': 'c', 'kind': 'width', 'init': f'chic{Ji}', 'final': 'Jpsi'}, libs, False)
    check(f"P -> S spin factor 1/3 for J={Ji}", spin_factor(1, 1, Ji, 0, 1), 1.0 / 3.0, 1e-12)

# 9. theory error (v11): sigma_th = 2 eps Gamma
check("theory error at eps = 0", theory_sigma(10.0, 0.0), 0.0, 1e-15)
check("theory error = 2 eps Gamma", theory_sigma(10.0, 0.3), 6.0, 1e-15)
o = Observable([x for x in DATA if x['id'] == 'chib1(2P): Y1S/Y2S'][0], libs, True)
R, sR = o.predict((0.0, 0.0), 0.1)
check("ratio theory error = 2 sqrt(2) eps R", sR, 2 * np.sqrt(2) * 0.1 * R, 1e-12)
o = Observable([x for x in DATA if x['id'] == 'psi2S->chic1'][0], libs, True)
G, sG = o.predict((0.01, -0.02), {'c': 0.3, 'b': 0.1})
check("width theory error with eps_c", sG, 0.6 * G, 1e-12)

# 10. data sets
for J in (0, 1, 2):
    row = [x for x in DATA if x['id'] == f'Y3S->chib{J}(1P)'][0]
    check(f"Y3S -> chib{J}(1P) in set ext (not fitted)", float(row['set'] == 'ext'), 1.0, 1e-15)
F = Fit(DATA, libs, True, {'c': ('core',), 'b': ('core',)})
check("core: 7 charm + 10 bottom measurements", len(F.meas), 17, 1e-15)
check("core: no hindered transition fitted", float(any('(1P)' in DATA[k]['id'] and DATA[k]['init'] == 'Y3S' for k in F.meas)), 0.0, 1e-15)

# 11. error method
p0, c0 = F.minimize(EPS_V2)
check("chi2 with the data vector = chi2 of the rows", F.chi2(p0, EPS_V2), c0, 1e-12)
for sign in (+1, -1):
    Fs = F.exp_shifted(sign)
    for k in F.meas:
        row = DATA[k]
        e = (row['err_up'] if sign > 0 else row['err_dn']) if 'err_up' in row else np.hypot(row['err'], row.get('err_corr', 0.0))
        check(f"shift {'+' if sign > 0 else '-'} {row['id']}", Fs.data[k], row['value'] + sign * e, 1e-12)
    Fe = Fit(DATA, libs, True, {'c': ('core', 'ext'), 'b': ('core',)}).exp_shifted(sign)
    for k in Fe.uls:
        check(f"upper limit unchanged {DATA[k]['id']}", Fe.data[k], DATA[k]['value'], 1e-15)
    check("original data unchanged by the shifted copy", F.data[F.meas[0]], DATA[F.meas[0]]['value'], 1e-15)
    ps, cs = Fs.minimize(EPS_V2)
    for i in (0, 1):
        for h in (1e-3, -1e-3):
            q = np.array(ps); q[i] += h
            check(f"shifted fit ({sign:+d}) is a minimum in parameter {i}, step {h:+g}", float(Fs.chi2(q, EPS_V2) >= cs), 1.0, 1e-15)

print('ALL PASS' if ok else 'SOME TESTS FAILED')
sys.exit(0 if ok else 1)
