"""Consistency tests of the hybrid code (EM_Pheno v10), run from the tests/ folder:  python3 test_hybrid.py
  1  local widths against the closed formula (eq:lab_Gamma_c)-(eq:lab_A012) with r-dependent couplings c_i(r)
     moved inside the radial integrals, for every kind of hybrid (p0, p1, (s/d)1, d2, (p/f)2) and all final l',
     with four DIFFERENT arbitrary couplings (so that the quadrupole A_2 and the c_i-structure are tested);
     channel split Gamma_PiPi, Gamma_SigmaSigma and Gamma_int = -(4E^3/pi^2 sigma)(2/3) p s A_1 A_0 (Eq. eq:Gamma_split)
  2  selection rules: p1, d2 radiate only through Pi_u, p0 only through Sigma_u^-; parity-forbidden channels vanish;
     equal couplings give no quadrupole (A_2 = 0), so d2 -> 1S vanishes in the local form
  3  projections: P_Pi + P_Sigma = 1, P_Sigma = 0 (p1, d2), = 1 (p0)
  4  heavy-quark spin: for each Jcal of the S = 1 hybrid, sum over J' of the S = 1 widths = S = 0 width; for final
     S waves each Jcal -> J' = 1 width equals the S = 0 one
  5  non-local -> local for E_q -> 0 (ratio -> 1)
  6  quadratic form: v^T H v = width with kappa_H(r) computed directly; split adds up to the total
  7  regression: constant c = e e_c reproduces the v8 widths of Y -> eta_c (quenched kinematics)
  8  kappa_H(r): limits B r (r -> 0) and kappa (r -> infinity); B_b = -2 B_c
  9  theory error: Gamma_bar >= Gamma, = Gamma when the radial integrand has a fixed sign
 10  fit: chi^2 stationary at the minima; the t0 iteration is a fixed point
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
import numpy as np
from emlib.states import Library, hybrid, s_l, p_l
from emlib.e1 import E_CHARGE, CHARGE, SIGMA, GEV2KEV, R0, photon_energy
from emlib.hybrid import HybridTransition, width_general, kappa_H, flavour_B
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


def closed_local(final, hyb, E, r, c):
    """Eqs. (eq:lab_Gamma_c), (eq:lab_A012) with c_i(r) inside the integrals (c in units of e) -> keV, split"""
    J, lf = hyb.J, final.l
    pr = hyb.projections()
    # R_Pi of Eq. (eq:lab_RPi): only the components with (-1)^l = (-1)^l'
    cP = sum(p_l(J, l) * cc for l, cc in hyb.comps if (l - lf) % 2 == 0) if hyb.comps else 0
    cS = pr['R_Sigma']
    I = lambda cR, f: np.sum(final.c * cR * f)                     # int dr u_f u_R f(r)
    c1, c2, c3, c4 = c
    A1 = I(cP, c2) - E / (2 * np.pi) * I(cP, r * (c1 + c3))
    A2 = E / (2 * np.pi) * I(cP, r * (c1 - c3))
    A0 = 2 * np.sqrt(np.pi / SIGMA) * (I(cS, c4 / r) - E / (3 * np.pi) * I(cS, c3))
    pl = p_l(J, lf) if abs(lf - J) <= 1 else 0.0
    sl = s_l(J, lf) if abs(lf - J) <= 1 else 0.0
    q = p2(J, lf) if abs(lf - J) <= 2 else 0.0
    pref = 4 * E ** 3 / (np.pi ** 2 * SIGMA) * E_CHARGE ** 2 * GEV2KEV
    PP = pref * (pl ** 2 * A1 ** 2 / 3 + q ** 2 * A2 ** 2 / 5)
    SS = pref * sl ** 2 * A0 ** 2 / 3
    INT = -pref * 2 * pl * sl * A1 * A0 / 3
    return {'Pi': PP, 'Sigma': SS, 'int': INT, 'total': PP + SS + INT}


lib = Library('c')
r = lib.r
rng = np.random.default_rng(1)
# four different smooth couplings c_i(r) (units of e)
cfun = [0.05 + 0.03 * np.sin(0.7 * r + k) * np.exp(-0.05 * r) + 0.01 * k * r / (1 + r) for k in range(4)]

# 1. closed formula
print('== 1. local widths vs closed formula (Eqs. eq:lab_Gamma_c, eq:lab_A012), arbitrary c_i(r)')
KINDS = [(0, 'sigma', 1), (1, 'minus', 1), (1, 'plus', 1), (1, 'plus', 2), (2, 'minus', 1), (2, 'plus', 1)]
for J, kind, n in KINDS:
    h = hybrid(lib, f'{kind}{J}', J, kind, n)
    for lf in range(0, 4):
        fin = lib.state('f', 1, lf, 0, lf, 3.0)
        E = 0.9
        w = width_general(fin, h, E, r, cfun, nonloc=False)
        cl = closed_local(fin, h, E, r, cfun)
        if cl['total'] < 1e-12 and w['total'] < 1e-10:
            check(f"J={J} {kind} n={n} -> l'={lf}: forbidden, both zero", w['total'], 0.0, 1e-6, atol=1e-10)
            continue
        for X in ('total', 'Pi', 'Sigma', 'int'):
            check(f"J={J} {kind} n={n} -> l'={lf} {X}", w[X], cl[X], 1e-6, atol=1e-9 * cl['total'])

# 2. selection rules
print('== 2. selection rules')
one = np.full(len(r), 0.1)
fin0, fin1, fin2 = (lib.state('f', 1, l, 0, l, 3.0) for l in (0, 1, 2))
p1 = hybrid(lib, 'p1', 1, 'minus', 1)
p0 = hybrid(lib, 'p0', 0, 'sigma', 1)
sd1 = hybrid(lib, 'sd1', 1, 'plus', 1)
d2 = hybrid(lib, 'd2', 2, 'minus', 1)
for nl in (False, True):
    w = width_general(fin1, p1, 0.6, r, (one,) * 4, nl)
    check(f"p1 -> h_c: no Sigma channel (nonloc={nl})", w['Sigma'], 0.0, 1e-6, atol=1e-20)
    w = width_general(fin1, p0, 0.6, r, (one,) * 4, nl)
    check(f"p0 -> h_c: no Pi channel (nonloc={nl})", w['Pi'], 0.0, 1e-6, atol=1e-20)
    w = width_general(fin1, sd1, 0.6, r, (one,) * 4, nl)
    check(f"(s/d)1 -> P wave: parity forbidden (nonloc={nl})", w['total'], 0.0, 1e-6, atol=1e-20)
w = width_general(fin0, d2, 1.0, r, (one,) * 4, False)
check("d2 -> 1S, equal couplings, local: no quadrupole (A_2 = 0)", w['total'], 0.0, 1e-6, atol=1e-20)
w = width_general(fin0, d2, 1.0, r, (one, one, 0.5 * one, one), False)
check("d2 -> 1S, c1 != c3, local: quadrupole only = closed formula",
      w['total'], closed_local(fin0, d2, 1.0, r, (one, one, 0.5 * one, one))['total'], 1e-6)

# 3. projections
print('== 3. Pi/Sigma projections')
for J, kind, n in KINDS:
    pr = hybrid(lib, 'x', J, kind, n).projections()
    check(f"J={J} {kind} n={n}: P_Pi + P_Sigma = 1", pr['P_Pi'] + pr['P_Sigma'], 1.0, 1e-10)
check("p1: P_Sigma = 0", p1.projections()['P_Sigma'], 0.0, 1e-6, atol=1e-14)
check("p0: P_Sigma = 1", p0.projections()['P_Sigma'], 1.0, 1e-10)

# 4. heavy-quark spin
print('== 4. heavy-quark spin (S = 1)')
v = (0.08, -0.02)
for J, kind, n in [(1, 'plus', 1), (1, 'minus', 1), (2, 'plus', 1), (0, 'sigma', 1)]:
    for lf in range(0, 3):
        h0 = hybrid(lib, 'h', J, kind, n)
        f0 = lib.state('f', 1, lf, 0, lf, 3.0)
        G0 = HybridTransition(f0, h0, 0.8, r, True).width(v)
        if G0 < 1e-10:
            continue
        for Jcal in range(abs(J - 1), J + 2):
            h1 = hybrid(lib, 'h', J, kind, n, S=1, Jcal=Jcal)
            tot = sum(HybridTransition(lib.state('f', 1, lf, 1, Jf, 3.0), h1, 0.8, r, True).width(v)
                      for Jf in range(abs(lf - 1), lf + 2))
            check(f"J={J} {kind} Jcal={Jcal} -> l'={lf}: sum_J' Gamma(S=1) = Gamma(S=0)", tot, G0, 1e-8)
            if lf == 0:
                G1 = HybridTransition(lib.state('f', 1, 0, 1, 1, 3.0), h1, 0.8, r, True).width(v)
                check(f"J={J} {kind} Jcal={Jcal} -> 3S1 = S=0 width", G1, G0, 1e-8)

# 5. NL -> L
print('== 5. non-local -> local at small E_q')
for h, f in ((sd1, fin0), (p0, fin1), (p1, fin1)):
    a = width_general(f, h, 1e-3, r, cfun, True)['total']
    b = width_general(f, h, 1e-3, r, cfun, False)['total']
    check(f"{h.name} -> l'={f.l}: NL/L at E = 1 MeV", a / b, 1.0, 1e-4)

# 6. quadratic form
print('== 6. quadratic form in (B_Q, kappa_Q)')
for v in [(0.08, -0.022), (-0.05, -0.022), (0.3, 0.1)]:
    for h, f in ((sd1, fin0), (p0, fin1), (hybrid(lib, 'pf2', 2, 'plus', 1), fin1)):
        for nl in (True, False):
            T = HybridTransition(f, h, 0.85, r, nl)
            k = kappa_H(r, *v)
            d = width_general(f, h, 0.85, r, (k, k, k, k), nl)
            sp = T.split(v)
            check(f"{h.name} -> l'={f.l} v={v} nonloc={nl}: v^T H v", T.width(v), d['total'], 1e-10)
            check(f"   split Pi + Sigma + int = total", sp['Pi'] + sp['Sigma'] + sp['int'], sp['total'], 1e-12)
            check(f"   Sigma part", sp['Sigma'], d['Sigma'], 1e-10, atol=1e-14)

# 7. regression against v8
print('== 7. regression: v8, c = e e_c, quenched kinematics')
etac = lib.state('eta_c', 1, 0, 0, 0, 2.98409)
E = photon_energy(4.4, 2.98409)
c = np.full(len(r), CHARGE['c'])
for nl, ref in ((True, {'total': 27529.5, 'Pi': 131.581, 'Sigma': 31467.6, 'int': -4069.66}),
                (False, {'total': 29329.1, 'Pi': 66.6212, 'Sigma': 32191.4, 'int': -2928.91})):
    w = width_general(etac, sd1, E, r, (c,) * 4, nl)
    for X in ref:
        check(f"Y -> eta_c nonloc={nl} {X} (v8 hybrid_external/lattice_hybrids.csv)", w[X], ref[X], 2e-5)

# 8. kappa_H
print('== 8. kappa_H(r)')
check("kappa_H -> B r at small r", kappa_H(1e-4, 0.1, -0.02) / 1e-4, 0.1, 1e-4)
check("kappa_H -> kappa at large r", kappa_H(1e4, 0.1, -0.02), -0.02 + 0.1 * R0 ** 2 / 1e4, 1e-6)
check("kappa_H(r0) = (B r0 + kappa)/2", kappa_H(R0, 0.1, -0.02), (0.1 * R0 - 0.02) / 2, 1e-12)
check("B_b = -2 B_c", flavour_B(0.05, 'b'), -0.1, 1e-12)
check("B_c(B_c) = B_c", flavour_B(0.05, 'c'), 0.05, 1e-12)

# 9. theory error
print('== 9. Gamma_bar')
T = HybridTransition(fin0, sd1, 0.92, r, True)
check("Gamma_bar = Gamma for B, kappa of the same sign (no node of kappa_H)", T.width_abs((-0.05, -0.02)),
      T.width((-0.05, -0.02)), 1e-3)
ok &= T.width_abs((0.02, -0.03)) >= T.width((0.02, -0.03))
print(f"{'PASS' if T.width_abs((0.02, -0.03)) >= T.width((0.02, -0.03)) else 'FAIL'} Gamma_bar >= Gamma with a node "
      f"({T.width_abs((0.02, -0.03)):.3f} >= {T.width((0.02, -0.03)):.3f})")

# 10. fit
print('== 10. fit')
F = HybridFit(lib, -0.0219, nonloc=True, theory='t0')
for m in F.run():
    F.B_t0 = m['B']
    h = 1e-5
    d = (F.chi2(m['B'] + h) - F.chi2(m['B'] - h)) / (2 * h)
    check(f"t0 minimum B = {m['B']:+.4f}: d chi2/dB = 0", d, 0.0, 1e-6, atol=1e-3)
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(F.chi2, bounds=(m['B'] - 0.05, m['B'] + 0.05), method='bounded', options={'xatol': 1e-10})
    check(f"t0 fixed point B = {m['B']:+.4f}", res.x, m['B'], 1e-5)
    F.B_t0 = None

print('ALL PASS' if ok else 'SOME TESTS FAILED')
sys.exit(0 if ok else 1)
