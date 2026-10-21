"""Consistency tests of the generic amplitude engine against the closed
formulas of the notes (Eqs. (106), (140), v9) and standard results."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
import numpy as np
from sympy.physics.quantum.cg import CG
from sympy.physics.wigner import wigner_6j
from sympy import S as Sy
from emlib import Library, Params, width, cg, E_CHARGE, CHARGE

ok = True
def check(name, a, b, tol=1e-6):
    global ok
    rel = abs(a - b) / max(abs(b), 1e-300)
    flag = rel < tol or (b == 0 and abs(a) < 1e-25)   # exact zeros (selection rules)
    ok &= flag
    print(f"{'PASS' if flag else 'FAIL'} {name}: {a:.6e} vs {b:.6e} (rel {rel:.1e})")

# 1. Clebsch-Gordan
for args in [(1, 0, 1, 1, 2, 1), (2, -1, 1, 1, 1, 0), (3, 2, 2, -1, 3, 1), (1, 1, 1, -1, 0, 0), (2, 0, 2, 0, 2, 0)]:
    check(f"CG{args}", cg(*args), float(CG(*[Sy(x) for x in args]).doit()), 1e-12)

lib = Library('c', mix=False)
r = lib.r
P = Params('c')
c = P.eeQ
E = 0.3

def I(cf, ci, k=1):
    return np.sum(cf * ci * r ** k)

# 2. spin-0 E1 vs Eq. (106)
for (nf, lf, ni, li) in [(1, 0, 1, 1), (2, 0, 1, 1), (1, 1, 2, 0), (1, 1, 1, 2), (1, 2, 2, 1), (2, 1, 3, 2)]:
    f = lib.quarkonium(nf, lf); i = lib.quarkonium(ni, li)
    G = width(f, i, E, P, r)
    Gf = c ** 2 / (3 * np.pi) * E ** 3 * max(li, lf) / (2 * li + 1) * I(f.comps[0].c, i.comps[0].c) ** 2
    check(f"E1 spin0 {ni}{'SPD'[li]}->{nf}{'SPD'[lf]}", G, Gf, 1e-8)

# 3. spin-1 E1 vs standard 6j formula, and sum rule
for (nf, lf, ni, li) in [(1, 1, 2, 0), (1, 0, 1, 1), (1, 1, 1, 2)]:
    tot = {}
    for Ji in range(abs(li - 1), li + 2):
        s = 0.0
        for Jf in range(abs(lf - 1), lf + 2):
            f = lib.quarkonium(nf, lf, S=1, J=Jf); i = lib.quarkonium(ni, li, S=1, J=Ji)
            G = width(f, i, E, P, r)
            sixj = float(wigner_6j(Ji, li, 1, lf, Jf, 1))
            Gf = c ** 2 / (3 * np.pi) * E ** 3 * (2 * Jf + 1) * max(li, lf) * sixj ** 2 * I(f.comps[0].c, i.comps[0].c) ** 2
            check(f"E1 spin1 {ni}{'SPD'[li]}{Ji}->{nf}{'SPD'[lf]}{Jf}", G, Gf, 1e-8)
            s += G
        tot[Ji] = s
    f0 = lib.quarkonium(nf, lf); i0 = lib.quarkonium(ni, li)
    G0 = width(f0, i0, E, P, r)
    for Ji, s in tot.items():
        check(f"  sum rule J_i={Ji}", s, G0, 1e-8)

# 4. hybrid -> quarkonium (spin 0) vs compact Eq. (140), with R_Pi^eff
sig = P.sigma
def wK(K, J, lp):
    return cg(J, -1, K, 1, lp, 0) ** 2
for (Jc, kind, n) in [(1, 'plus', 1), (1, 'minus', 1), (1, 'plus', 2), (2, 'plus', 1), (2, 'minus', 1)]:
    k = lib.hybrid_index(Jc, kind, n)
    H = lib.hybrid(Jc, k)
    xi = -1 if kind == 'minus' else +1
    eta = -xi * (-1) ** Jc
    # effective Pi radial coefficients
    comps = {cc.l: cc.c for cc in H.comps}
    if xi == -1:
        cPi = comps[Jc]
    else:
        cPi = (np.sqrt(Jc + 1) * comps[Jc - 1] + np.sqrt(Jc) * comps[Jc + 1]) / np.sqrt(2 * Jc + 1)
    for kh in [(1, 1, 1), (0.7, 1.3, -0.4)]:
        Ph = Params('c', kH=kh)
        c1, c2, c3 = [x * Ph.eeQ for x in kh]
        for lf in range(0, 4):
            if (-1) ** lf != eta or abs(lf - Jc) > 2:
                continue
            for nf in (1, 2):
                f = lib.quarkonium(nf, lf)
                Eq = 0.8
                G = width(f, H, Eq, Ph, r)
                I0 = np.sum(f.comps[0].c * cPi); I1 = np.sum(f.comps[0].c * cPi * r)
                A1 = c2 * I0 - Eq / (2 * np.pi) * (c1 + c3) * I1
                A2 = Eq / (2 * np.pi) * (c1 - c3) * I1
                Gf = 8 * Eq ** 3 / (np.pi ** 2 * sig) * (wK(1, Jc, lf) / 3 * A1 ** 2 + wK(2, Jc, lf) / 5 * A2 ** 2)
                check(f"HQ Jc={Jc} {kind}{n} -> {nf}{'SPDF'[lf]} c={kh}", G, Gf, 1e-7)

# 5. non-local -> local for small photon energy
Pl = Params('c', nonloc=False); Pn = Params('c', nonloc=True)
f = lib.quarkonium(1, 1); i = lib.quarkonium(2, 0)
for Es in (0.3, 0.03, 0.003):
    print(f"  E={Es}: nonlocal/local E1 = {width(f, i, Es, Pn, r) / width(f, i, Es, Pl, r):.6f}")
H = lib.hybrid(1, lib.hybrid_index(1, 'plus', 1)); f = lib.quarkonium(1, 0)
for Es in (0.9, 0.09, 0.009):
    print(f"  E={Es}: nonlocal/local HQ = {width(f, H, Es, Params('c', nonloc=True, kH=(0, 1, 0)), r) / width(f, H, Es, Params('c', kH=(0, 1, 0)), r):.6f}")

# 6. spin-1 hybrid sum rule: sum_Jf Gamma(Jc,S=1,J -> lf,S=1,Jf) = spin-0 width
k = lib.hybrid_index(1, 'plus', 1)
H0 = lib.hybrid(1, k); f0 = lib.quarkonium(1, 0)
G0 = width(f0, H0, 0.9, P, r)
for Ji in (0, 1, 2):
    Hs = lib.hybrid(1, k, S=1, J=Ji)
    s = sum(width(lib.quarkonium(1, 0, S=1, J=1), Hs, 0.9, P, r) for _ in [0])
    check(f"spin-1 (s/d)1 J={Ji} -> 1S(J=1) equals spin-0 width", s, G0, 1e-8)

# 7. 'lattice' gap variant with dV = pi/r reproduces the 'string' variant
k = lib.hybrid_index(1, 'plus', 1); H = lib.hybrid(1, k); f = lib.quarkonium(1, 0)
for nl in (False, True):
    Ps = Params('c', nonloc=nl); Pl = Params('c', nonloc=nl, gap='lattice', dV=np.pi / r)
    check(f"gap lattice(dV=pi/r) == string, nonlocal={nl}", width(f, H, 0.9, Pl, r), width(f, H, 0.9, Ps, r), 1e-10)

# 8. non-local widths vs the closed multipole formulas of the notes
#    (quarkonium: Gamma = c1^2 E^3/pi sum_K C(lKl',00)^2/(2K+1) J_K^2;
#     hybrids:    Gamma = 8E^3/(pi^2 sigma) sum_K w_K/(2K+1) |A_K|^2)
from scipy.special import eval_legendre, spherical_jn
xg, wg = np.polynomial.legendre.leggauss(160)
def proj(fun, x, L):
    return 0.5 * np.sum(wg[None, :] * fun(np.outer(x, xg)) * eval_legendre(L, xg)[None, :], axis=1)
fQ = lambda y: spherical_jn(0, y / 2)
fH = lambda y: np.pi ** 2 * np.cos(y / 2) / (np.pi ** 2 - y ** 2)
LM = 14
Pn = Params('c', nonloc=True)
for (ni, li, nf, lf, Eq) in [(2, 0, 1, 1, 0.13), (2, 1, 1, 0, 0.7), (2, 2, 1, 1, 0.6), (1, 3, 1, 2, 0.5)]:
    i = lib.quarkonium(ni, li); f = lib.quarkonium(nf, lf); x = Eq * r; Gf = 0
    for K in range(1, LM, 2):
        FK = sum((2 * L + 1) * proj(fQ, x, L) * cg(1, 0, L, 0, K, 0) * cg(1, 1, L, 0, K, 1) for L in range(0, LM + 1, 2))
        Gf += Pn.eeQ ** 2 * Eq ** 3 / np.pi * cg(li, 0, K, 0, lf, 0) ** 2 / (2 * K + 1) * np.sum(f.comps[0].c * i.comps[0].c * r * FK) ** 2
    check(f"NL E1 {ni}{'SPDF'[li]}->{nf}{'SPDF'[lf]} multipole formula", width(f, i, Eq, Pn, r), Gf, 1e-8)
for (Jc, kind, n, lf) in [(1, 'plus', 1, 2), (1, 'minus', 1, 3), (2, 'minus', 1, 0), (2, 'plus', 1, 3)]:
    k = lib.hybrid_index(Jc, kind, n); H = lib.hybrid(Jc, k); xi = -1 if kind == 'minus' else 1
    comps = {cc.l: cc.c for cc in H.comps}
    cPi = comps[Jc] if xi == -1 else (np.sqrt(Jc + 1) * comps[Jc - 1] + np.sqrt(Jc) * comps[Jc + 1]) / np.sqrt(2 * Jc + 1)
    kh = (0.7, 1.3, -0.4); Ph = Params('c', kH=kh, nonloc=True); c1, c2, c3 = [v * Ph.eeQ for v in kh]
    f = lib.quarkonium(1, lf); Eq = 0.8; x = Eq * r
    a = c2 - Eq * r / (2 * np.pi) * (c1 + c3); b = Eq * r / (2 * np.pi) * (c1 - c3); Gf = 0
    for K in range(1, LM):
        GK = sum((2 * L + 1) * proj(fH, x, L) * (a * cg(1, 1, L, 0, K, 1) ** 2 + b * cg(2, 1, L, 0, K, 1) ** 2) for L in range(0, LM + 1, 2))
        Gf += 8 * Eq ** 3 / (np.pi ** 2 * Ph.sigma) * wK(K, Jc, lf) / (2 * K + 1) * np.sum(f.comps[0].c * cPi * GK) ** 2
    check(f"NL HQ Jc={Jc} {kind}{n} -> 1{'SPDF'[lf]} multipole formula", width(f, H, Eq, Ph, r), Gf, 1e-8)

# 9. spin-1 hybrids: Gamma(J -> J') = (2J'+1)(2Jc+1) sum_K {l' J' 1; J Jc K}^2 Gamma_K
for (Jc, kind, lf) in [(1, 'minus', 1), (1, 'plus', 2), (2, 'minus', 2)]:
    k = lib.hybrid_index(Jc, kind, 1); xi = -1 if kind == 'minus' else 1
    H0 = lib.hybrid(Jc, k); comps = {cc.l: cc.c for cc in H0.comps}
    cPi = comps[Jc] if xi == -1 else (np.sqrt(Jc + 1) * comps[Jc - 1] + np.sqrt(Jc) * comps[Jc + 1]) / np.sqrt(2 * Jc + 1)
    kh = (0.7, 1.3, -0.4); Ph = Params('c', kH=kh); c1, c2, c3 = [v * Ph.eeQ for v in kh]
    f0 = lib.quarkonium(1, lf); Eq = 0.8
    I0 = np.sum(f0.comps[0].c * cPi); I1 = np.sum(f0.comps[0].c * cPi * r)
    A1 = c2 * I0 - Eq / (2 * np.pi) * (c1 + c3) * I1; A2 = Eq / (2 * np.pi) * (c1 - c3) * I1
    GK = {1: 8 * Eq ** 3 / (np.pi ** 2 * sig) * wK(1, Jc, lf) / 3 * A1 ** 2,
          2: 8 * Eq ** 3 / (np.pi ** 2 * sig) * wK(2, Jc, lf) / 5 * A2 ** 2}
    for Ji in range(abs(Jc - 1), Jc + 2):
        for Jf in range(abs(lf - 1), lf + 2):
            G = width(lib.quarkonium(1, lf, S=1, J=Jf), lib.hybrid(Jc, k, S=1, J=Ji), Eq, Ph, r)
            Gf = sum((2 * Jf + 1) * (2 * Jc + 1) * float(wigner_6j(lf, Jf, 1, Ji, Jc, K)) ** 2 * GK[K] for K in (1, 2))
            check(f"spin-1 HQ Jc={Jc} {kind} J={Ji} -> 1{'SPDF'[lf]} J'={Jf} (6j, K=1,2)", G, Gf, 1e-8)

print("ALL PASS" if ok else "SOME FAILURES")
