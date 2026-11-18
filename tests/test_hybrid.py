"""Consistency tests of the hybrid code (EM_Pheno v12, local form), run from the tests/ folder:  python3 test_hybrid.py
  1  local widths against the closed formula (eq:lab_Gamma) with D_l' = p_l' I[G1^Pi R_Pi] + s_l' I[G1^Sigma R_Sigma],
     Q_l' = p2_l' I[G2 R_Pi] (Eqs. eq:GPiSigma, eq:lab_DQ), for every kind of hybrid (p0, p1, (s/d)1, d2, (p/f)2) and
     final l' = 0..3, with five DIFFERENT arbitrary LEFs g_X(r) (so that the quadrupole and every term are tested);
     total and channel split Gamma_PiPi, Gamma_SigmaSigma, Gamma_int
  2  selection rules: p1, d2 radiate only through Pi_u, p0 only through Sigma_u^-; parity-forbidden channels vanish;
     the interpolated LEFs have G2 = 0 (no quadrupole), so d2 -> 1S vanishes
  3  projections: P_Pi + P_Sigma = 1, P_Sigma = 0 (p1, d2), = 1 (p0)
  4  heavy-quark spin: for each Jcal of the S = 1 hybrid, sum over J' of the S = 1 widths = S = 0 width; for final
     S waves each Jcal -> J' = 1 width equals the S = 0 one
  5  the non-local form is not defined for the interpolated LEFs: NotImplementedError
  6  quadratic form: v^T H v = width computed directly with the LEFs of Eq. (eq:gH_interp); split adds up to the total
  7  regression: pure EST LEFs with constant c = e e_c reproduce the v8/v10 local widths of Y -> eta_c
  8  LEFs, Eq. (eq:gH_interp): g_B_perp -> e B r at r -> 0, g_B_par -> e (B - 4 kappa/(sigma r0^2)) r (the kappa term
     of g_B_par is also O(r)); -> e kappa ghat_B at r -> infinity;
     electric LEFs = e kappa ghat_E x/(1+x); B_b = -2 B_c
  9  fit: chi^2 stationary at the minima; shifted fits on the same branch; sigma_th = 2 eps Gamma
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
import numpy as np
from emlib.states import Library, hybrid, s_l, p_l
from emlib.e1 import E_CHARGE, CHARGE, SIGMA, GEV2KEV, R0, photon_energy
from emlib.hybrid import (HybridTransition, width_lefs, lefs, est_lefs, lef_shapes, flavour_B, LEFS)
from emlib.hybrid_fit import HybridFit
from emlib.angular import cg

ok = True


def check(name, a, b, tol, atol=1e-12):
    global ok
    rel = abs(a - b) / max(abs(b), 1e-300)
    flag = rel < tol or abs(a - b) < atol
    ok &= bool(flag)
    print(f"{'PASS' if flag else 'FAIL'} {name}: {a:.10e} vs {b:.10e} (rel {rel:.1e})")


def p2(J, l):
    """p^(2)_l of Eq. (eq:lab_p_values)"""
    return np.sqrt(2 * (2 * l + 1) / (2 * J + 1)) * cg(l, 0, 2, 1, J, 1)


def closed_local(final, hyb, E, g):
    """Eq. (eq:lab_Gamma) with the combinations of Eq. (eq:GPiSigma) built from the LEFs g -> keV, split"""
    J, lf = hyb.J, final.l
    cP = sum(p_l(J, l) * cc for l, cc in hyb.comps if (l - lf) % 2 == 0) if hyb.comps else 0
    cS = hyb.projections()['R_Sigma']
    I = lambda cR, f: np.sum(final.c * cR * f)                     # int dr r^2 R_f R f(r) = sum c_f c_R f(r_i)
    G1P = g['B_perp'] + E / 2 * (g['E_par'] - g['E_perp'])
    G1S = g['B_par'] + E * g['E_Sigma']
    G2 = E / 2 * (g['E_par'] + g['E_perp'])
    pl = p_l(J, lf) if abs(lf - J) <= 1 else 0.0
    sl = s_l(J, lf) if abs(lf - J) <= 1 else 0.0
    q = p2(J, lf) if abs(lf - J) <= 2 else 0.0
    IP, IS, IQ = I(cP, G1P), I(cS, G1S), I(cP, G2)
    pref = E ** 3 / np.pi * GEV2KEV
    PP = pref * (pl ** 2 * IP ** 2 / 3 + q ** 2 * IQ ** 2 / 5)
    SS = pref * sl ** 2 * IS ** 2 / 3
    INT = pref * 2 * pl * sl * IP * IS / 3
    return {'Pi': PP, 'Sigma': SS, 'int': INT, 'total': PP + SS + INT}


lib = Library('c')
r = lib.r
# five different smooth LEFs (GeV units)
gfun = {X: E_CHARGE * (0.05 + 0.03 * np.sin(0.7 * r + k) * np.exp(-0.05 * r) + 0.01 * k * r / (1 + r))
        for k, X in enumerate(LEFS)}

# 1. closed formula
print('== 1. local widths vs closed formula (Eqs. eq:lab_Gamma, eq:GPiSigma, eq:lab_DQ), arbitrary LEFs')
KINDS = [(0, 'sigma', 1), (1, 'minus', 1), (1, 'plus', 1), (1, 'plus', 2), (2, 'minus', 1), (2, 'plus', 1)]
for J, kind, n in KINDS:
    h = hybrid(lib, f'{kind}{J}', J, kind, n)
    for lf in range(0, 4):
        fin = lib.state('f', 1, lf, 0, lf, 3.0)
        E = 0.9
        w = width_lefs(fin, h, E, r, gfun)
        cl = closed_local(fin, h, E, gfun)
        if cl['total'] < 1e-12 and w['total'] < 1e-10:
            check(f"J={J} {kind} n={n} -> l'={lf}: forbidden, both zero", w['total'], 0.0, 1e-6, atol=1e-10)
            continue
        for X in ('total', 'Pi', 'Sigma', 'int'):
            check(f"J={J} {kind} n={n} -> l'={lf} {X}", w[X], cl[X], 1e-6, atol=1e-9 * cl['total'])

# 2. selection rules
print('== 2. selection rules')
g0 = lefs(r, 0.1, -0.03)
fin0, fin1 = (lib.state('f', 1, l, 0, l, 3.0) for l in (0, 1))
p1 = hybrid(lib, 'p1', 1, 'minus', 1)
p0 = hybrid(lib, 'p0', 0, 'sigma', 1)
sd1 = hybrid(lib, 'sd1', 1, 'plus', 1)
d2 = hybrid(lib, 'd2', 2, 'minus', 1)
w = width_lefs(fin1, p1, 0.6, r, g0)
check("p1 -> h_c: no Sigma channel", w['Sigma'], 0.0, 1e-6, atol=1e-20)
w = width_lefs(fin1, p0, 0.6, r, g0)
check("p0 -> h_c: no Pi channel", w['Pi'], 0.0, 1e-6, atol=1e-20)
w = width_lefs(fin1, sd1, 0.6, r, g0)
check("(s/d)1 -> P wave: parity forbidden", w['total'], 0.0, 1e-6, atol=1e-20)
w = width_lefs(fin0, d2, 1.0, r, g0)
check("d2 -> 1S with the interpolated LEFs: no quadrupole (G2 = 0)", w['total'], 0.0, 1e-6, atol=1e-20)
gq = dict(g0)
gq['E_par'] = 0.5 * g0['E_par']
w = width_lefs(fin0, d2, 1.0, r, gq)
check("d2 -> 1S with G2 != 0: quadrupole only = closed formula", w['total'], closed_local(fin0, d2, 1.0, gq)['total'], 1e-6)

# 3. projections
print('== 3. Pi/Sigma projections')
for J, kind, n in KINDS:
    pr = hybrid(lib, 'x', J, kind, n).projections()
    check(f"J={J} {kind} n={n}: P_Pi + P_Sigma = 1", pr['P_Pi'] + pr['P_Sigma'], 1.0, 1e-10)
check("p1: P_Sigma = 0", p1.projections()['P_Sigma'], 0.0, 1e-6, atol=1e-14)
check("p0: P_Sigma = 1", p0.projections()['P_Sigma'], 1.0, 1e-10)

# 4. heavy-quark spin
print('== 4. heavy-quark spin (S = 1)')
v = (-0.19, -0.023)
for J, kind, n in [(1, 'plus', 1), (1, 'minus', 1), (2, 'plus', 1), (0, 'sigma', 1)]:
    for lf in range(0, 3):
        h0 = hybrid(lib, 'h', J, kind, n)
        f0 = lib.state('f', 1, lf, 0, lf, 3.0)
        G0 = HybridTransition(f0, h0, 0.8, r).width(v)
        if G0 < 1e-10:
            continue
        for Jcal in range(abs(J - 1), J + 2):
            h1 = hybrid(lib, 'h', J, kind, n, S=1, Jcal=Jcal)
            tot = sum(HybridTransition(lib.state('f', 1, lf, 1, Jf, 3.0), h1, 0.8, r).width(v)
                      for Jf in range(abs(lf - 1), lf + 2))
            check(f"J={J} {kind} Jcal={Jcal} -> l'={lf}: sum_J' Gamma(S=1) = Gamma(S=0)", tot, G0, 1e-8)
            if lf == 0:
                G1 = HybridTransition(lib.state('f', 1, 0, 1, 1, 3.0), h1, 0.8, r).width(v)
                check(f"J={J} {kind} Jcal={Jcal} -> 3S1 = S=0 width", G1, G0, 1e-8)

# 5. non-local form not defined
print('== 5. non-local form')
try:
    HybridTransition(fin0, sd1, 0.9, r, nonloc=True)
    ok = False
    print('FAIL non-local form should raise NotImplementedError')
except NotImplementedError:
    print('PASS non-local form raises NotImplementedError')

# 6. quadratic form
print('== 6. quadratic form in (B_Q, kappa_Q)')
for v in [(-0.19, -0.023), (0.12, -0.023), (0.3, 0.1)]:
    for h, f in ((sd1, fin0), (p0, fin1), (hybrid(lib, 'pf2', 2, 'plus', 1), fin1), (p1, fin1)):
        T = HybridTransition(f, h, 0.85, r)
        d = width_lefs(f, h, 0.85, r, lefs(r, *v))
        sp = T.split(v)
        check(f"{h.name} -> l'={f.l} v={v}: v^T H v", T.width(v), d['total'], 1e-10)
        check(f"   split Pi + Sigma + int = total", sp['Pi'] + sp['Sigma'] + sp['int'], sp['total'], 1e-12)
        check(f"   Sigma part", sp['Sigma'], d['Sigma'], 1e-10, atol=1e-14)

# 7. regression against v8/v10 (local form, c = e e_c)
print('== 7. regression: v8/v10, pure EST LEFs with c = e e_c, quenched kinematics, local form')
etac = lib.state('eta_c', 1, 0, 0, 0, 2.98409)
E = photon_energy(4.4, 2.98409)
w = width_lefs(etac, sd1, E, r, est_lefs(r, CHARGE['c']))
for X, ref in {'total': 29329.1, 'Pi': 66.6212, 'Sigma': 32191.4, 'int': -2928.91}.items():
    check(f"Y -> eta_c {X} (v8 lattice_hybrids.csv, L)", w[X], ref, 2e-5)

# 8. LEFs
print('== 8. interpolated LEFs')
B, k = 0.2, -0.03
rs = np.array([1e-4])
g = lefs(rs, B, k)
check("g_B_perp -> e B r at r -> 0", g['B_perp'][0] / (E_CHARGE * 1e-4), B, 1e-3)
# g_B_par: ghat_B_par x = -4 r/(sigma r0^2) is also O(r), so its slope at r -> 0 is B - 4 kappa/(sigma r0^2), not B
check("g_B_par -> e (B - 4 kappa/(sigma r0^2)) r at r -> 0", g['B_par'][0] / (E_CHARGE * 1e-4), B - 4 * k / (SIGMA * R0 ** 2), 1e-3)
gh = lef_shapes(np.array([1e4]))
gl = lefs(np.array([1e4]), B, k)
for X in LEFS:
    lim = E_CHARGE * k * gh[X][0]
    check(f"g_{X} -> e kappa ghat at r -> infinity", gl[X][0], lim + (E_CHARGE * B * R0 ** 2 / 1e4 if X in ('B_perp', 'B_par') else 0.0),
          2e-3, atol=1e-12)
x = (r / R0) ** 2
ghr = lef_shapes(r)
for X in ('E_par', 'E_perp', 'E_Sigma'):
    check(f"g_{X} = e kappa ghat x/(1+x) (no B term)", float(np.max(np.abs(lefs(r, B, k)[X] - E_CHARGE * k * ghr[X] * x / (1 + x)))), 0.0, 1e-6, atol=1e-15)
check("G2 = (E/2)(g_E_par + g_E_perp) = 0", float(np.max(np.abs(lefs(r, B, k)['E_par'] + lefs(r, B, k)['E_perp']))), 0.0, 1e-6, atol=1e-15)
check("B_b = -2 B_c", flavour_B(0.05, 'b'), -0.1, 1e-12)
check("B_c(B_c) = B_c", flavour_B(0.05, 'c'), 0.05, 1e-12)

# 9. fit
print('== 9. fit')
from scipy.optimize import minimize_scalar
F = HybridFit(lib, -0.0231)
G, s = F.model(-0.19)[0]
check("sigma_th = 2 eps Gamma", s, 2 * 0.3 * G, 1e-12)
for m in F.run():
    h = 1e-6
    d = (F.chi2(m['B'] + h) - F.chi2(m['B'] - h)) / (2 * h)
    check(f"minimum B = {m['B']:+.4f}: d chi2/dB = 0", d, 0.0, 1e-6, atol=1e-3)
    ok_b = np.sign(m['B_max']) == np.sign(m['B']) and np.sign(m['B_min']) == np.sign(m['B'])
    ok &= bool(ok_b)
    print(f"{'PASS' if ok_b else 'FAIL'} branch {m['branch']}: shifted fits on the same branch "
          f"({m['B_max']:+.4f}, {m['B_min']:+.4f})")
    for sgn, Bs in ((+1, m['B_max']), (-1, m['B_min'])):
        Fs = F.shifted(sgn)
        d = (Fs.chi2(Bs + h) - Fs.chi2(Bs - h)) / (2 * h)
        check(f"   shifted fit ({'+' if sgn > 0 else '-'}sigma): d chi2/dB = 0", d, 0.0, 1e-6, atol=1e-3)

print('ALL PASS' if ok else 'SOME TESTS FAILED')
sys.exit(0 if ok else 1)
