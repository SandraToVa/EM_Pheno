"""Consistency tests of the amplitude engine (EM_Pheno v3) against the closed
formulas of the notes and standard results.
  1-3   Clebsch-Gordan, quarkonium E1 (spin 0 and 1)
  4     hybrid widths vs the lab-basis formula Eq. (eq:lab_Gamma_c), channel by
        channel: Gamma_PiPi, Gamma_SigmaSigma, Gamma_int, for every hybrid type
        and general c1..c4 (R_Pi, R_Sigma of Eqs. (eq:lab_RPi), (eq:lab_RSigma))
  5-7   local limit of the non-local forms, spin-1 sum rule, gap variant B -> A
  8     non-local widths vs the multipole formulas (E1, Pi channel)
  9     spin-1 hybrids (Pi channel), 6j recoupling
  10    selection rules of the two channels: l = J states have no Sigma channel,
        J = 0 states no Pi channel
  11-13 p0 (pure Sigma channel): local and non-local closed formulas, FT of rho_Sigma
  14    spin-1 hybrids with both channels"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
import numpy as np
from sympy.physics.quantum.cg import CG
from sympy.physics.wigner import wigner_6j
from sympy import S as Sy
from emlib import Library, Params, width, cg, E_CHARGE, CHARGE
from emlib.amplitudes import width_split
from emlib.states import pi_sigma_projections

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

# 4. hybrid -> quarkonium (spin 0) vs the lab-basis formula, channel by channel:
#    Gamma = 4E^3/(pi^2 sigma) [ (p A1 + s A0)^2/3 + (p2)^2 A2^2/5 ]          (eq:lab_Gamma_c)
#    Gamma_PiPi = 4E^3/(pi^2 sigma)[p^2 A1^2/3 + (p2)^2 A2^2/5], Gamma_SigmaSigma = 4E^3/(pi^2 sigma) s^2 A0^2/3,
#    Gamma_int = 4E^3/(pi^2 sigma) 2 p s A1 A0/3
sig = P.sigma
def pK(K, J, l):
    """p^(K)_l of Eq. (eq:lab_p)"""
    return np.sqrt(2 * (2 * l + 1) / (2 * J + 1)) * cg(l, 0, K, 1, J, 1) if l >= 0 else 0.0
def wK(K, J, lp):
    """(p^(K)_l')^2/2 = C(J -1 K 1|l' 0)^2 (weights of the multipole formulas)"""
    return cg(J, -1, K, 1, lp, 0) ** 2
def s_l(J, l):
    """s_l of Eq. (eq:lab_rY)"""
    return np.sqrt((2 * l + 1) / (2 * J + 1)) * cg(l, 0, 1, 0, J, 0) if l >= 0 else 0.0
def RPi_RSigma(H, lf):
    pr = pi_sigma_projections(H)
    RPi = pr['R_Pi_lJ'] if (lf - pr['J']) % 2 == 0 else pr['R_Pi']
    return RPi, pr['R_Sigma']
def lab_split(H, J, f, lf, Eq, Ph):
    c1, c2, c3, c4 = [x * Ph.eeQ for x in Ph.kH]
    cPi, cSig = RPi_RSigma(H, lf); cf = f.comps[0].c
    A1 = c2 * np.sum(cf * cPi) - Eq / (2 * np.pi) * (c1 + c3) * np.sum(cf * cPi * r)
    A2 = Eq / (2 * np.pi) * (c1 - c3) * np.sum(cf * cPi * r)
    A0 = 2 * np.sqrt(np.pi / sig) * (c4 * np.sum(cf * cSig / r) - c3 * Eq / (3 * np.pi) * np.sum(cf * cSig))
    p, s, p2 = pK(1, J, lf), s_l(J, lf), pK(2, J, lf)
    n = 4 * Eq ** 3 / (np.pi ** 2 * sig)
    out = {'Pi': n * ((p * A1) ** 2 / 3 + (p2 * A2) ** 2 / 5), 'Sigma': n * (s * A0) ** 2 / 3,
           'int': n * 2 * p * s * A1 * A0 / 3}
    out['total'] = out['Pi'] + out['Sigma'] + out['int']
    return out
HTYPES = [(0, 'sigma', 1), (1, 'plus', 1), (1, 'plus', 2), (1, 'minus', 1), (2, 'plus', 1), (2, 'minus', 1)]
for (J, kind, n) in HTYPES:
    k = lib.hybrid_index(J, kind, n); H = lib.hybrid(J, k)
    for kh in [(1, 1, 1, 1), (0.7, 1.3, -0.4, 0.55)]:
        Ph = Params('c', kH=kh)
        for lf in range(0, 5):
            for nf in (1, 2):
                f = lib.quarkonium(nf, lf) if lf < 4 else None
                if f is None:
                    continue
                Eq = 0.8
                sp = width_split(f, H, Eq, Ph, r); ref = lab_split(H, J, f, lf, Eq, Ph)
                if ref['total'] < 1e-20 and sp['total'] < 1e-20:
                    continue              # parity forbidden: both are round-off
                for X in ('total', 'Pi', 'Sigma', 'int'):
                    if max(abs(ref[X]), abs(sp[X])) < 1e-13 * max(ref['total'], 1e-20):
                        continue          # forbidden (parity) or channel absent: checked in test 10
                    check(f"HQ {X:5s} J={J} {kind}{n} -> {nf}{'SPDF'[lf]} c={kh}", sp[X], ref[X], 1e-7)

# 5. non-local -> local for small photon energy
Pl = Params('c', nonloc=False); Pn = Params('c', nonloc=True)
f = lib.quarkonium(1, 1); i = lib.quarkonium(2, 0)
for Es in (0.3, 0.03, 0.003):
    print(f"  E={Es}: nonlocal/local E1 = {width(f, i, Es, Pn, r) / width(f, i, Es, Pl, r):.6f}")
H = lib.hybrid(1, lib.hybrid_index(1, 'plus', 1)); f = lib.quarkonium(1, 0)
for Es in (0.9, 0.09, 0.009):
    print(f"  E={Es}: nonlocal/local HQ = {width(f, H, Es, Params('c', nonloc=True, kH=(0, 1, 0)), r) / width(f, H, Es, Params('c', kH=(0, 1, 0)), r):.6f}")

# 6. spin-1 hybrid sum rule: Gamma(J, S=1, Jcal -> 1S, S=1, J'=1) = spin-0 width for every Jcal
k = lib.hybrid_index(1, 'plus', 1)
H0 = lib.hybrid(1, k); f0 = lib.quarkonium(1, 0)
G0 = width(f0, H0, 0.9, P, r)
for Ji in (0, 1, 2):
    Hs = lib.hybrid(1, k, S=1, Jcal=Ji)
    s = sum(width(lib.quarkonium(1, 0, S=1, J=1), Hs, 0.9, P, r) for _ in [0])
    check(f"spin-1 (s/d)1 Jcal={Ji} -> 1S(J=1) equals spin-0 width", s, G0, 1e-8)

# 7. 'lattice' gap variant with dV = pi/r reproduces the 'string' variant
k = lib.hybrid_index(1, 'plus', 1); H = lib.hybrid(1, k); f = lib.quarkonium(1, 0)
for nl in (False, True):
    Ps = Params('c', nonloc=nl); Pl = Params('c', nonloc=nl, gap='lattice', dV=np.pi / r, dVS=3 * np.pi / r)
    check(f"gap lattice(dV=pi/r, dVS=3pi/r) == string, nonlocal={nl}", width(f, H, 0.9, Pl, r), width(f, H, 0.9, Ps, r), 1e-10)

# 8. non-local widths vs the closed multipole formulas of the notes (hybrids: Pi channel)
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
for (J, kind, n, lf) in [(1, 'plus', 1, 2), (1, 'minus', 1, 3), (2, 'minus', 1, 0), (2, 'plus', 1, 3)]:
    k = lib.hybrid_index(J, kind, n); H = lib.hybrid(J, k); xi = -1 if kind == 'minus' else 1
    comps = {cc.l: cc.c for cc in H.comps}
    cPi = comps[J] if xi == -1 else (np.sqrt(J + 1) * comps[J - 1] + np.sqrt(J) * comps[J + 1]) / np.sqrt(2 * J + 1)
    kh = (0.7, 1.3, -0.4, 0.55); Ph = Params('c', kH=kh, nonloc=True); c1, c2, c3 = [v * Ph.eeQ for v in kh[:3]]
    f = lib.quarkonium(1, lf); Eq = 0.8; x = Eq * r
    a = c2 - Eq * r / (2 * np.pi) * (c1 + c3); b = Eq * r / (2 * np.pi) * (c1 - c3); Gf = 0
    for K in range(1, LM):
        GK = sum((2 * L + 1) * proj(fH, x, L) * (a * cg(1, 1, L, 0, K, 1) ** 2 + b * cg(2, 1, L, 0, K, 1) ** 2) for L in range(0, LM + 1, 2))
        Gf += 8 * Eq ** 3 / (np.pi ** 2 * Ph.sigma) * wK(K, J, lf) / (2 * K + 1) * np.sum(f.comps[0].c * cPi * GK) ** 2
    check(f"NL HQ Pi channel J={J} {kind}{n} -> 1{'SPDF'[lf]} multipole formula", width_split(f, H, Eq, Ph, r)['Pi'], Gf, 1e-8)

# 9. spin-1 hybrids, Pi channel: Gamma_PiPi(Jcal -> J') = (2J'+1)(2J+1) sum_K {l' J' 1; Jcal J K}^2 Gamma_K
for (J, kind, lf) in [(1, 'minus', 1), (1, 'plus', 2), (2, 'minus', 2)]:
    k = lib.hybrid_index(J, kind, 1); xi = -1 if kind == 'minus' else 1
    H0 = lib.hybrid(J, k); comps = {cc.l: cc.c for cc in H0.comps}
    cPi = comps[J] if xi == -1 else (np.sqrt(J + 1) * comps[J - 1] + np.sqrt(J) * comps[J + 1]) / np.sqrt(2 * J + 1)
    kh = (0.7, 1.3, -0.4, 0.55); Ph = Params('c', kH=kh); c1, c2, c3 = [v * Ph.eeQ for v in kh[:3]]
    f0 = lib.quarkonium(1, lf); Eq = 0.8
    I0 = np.sum(f0.comps[0].c * cPi); I1 = np.sum(f0.comps[0].c * cPi * r)
    A1 = c2 * I0 - Eq / (2 * np.pi) * (c1 + c3) * I1; A2 = Eq / (2 * np.pi) * (c1 - c3) * I1
    GK = {1: 8 * Eq ** 3 / (np.pi ** 2 * sig) * wK(1, J, lf) / 3 * A1 ** 2,
          2: 8 * Eq ** 3 / (np.pi ** 2 * sig) * wK(2, J, lf) / 5 * A2 ** 2}
    for Ji in range(abs(J - 1), J + 2):
        for Jf in range(abs(lf - 1), lf + 2):
            G = width_split(lib.quarkonium(1, lf, S=1, J=Jf), lib.hybrid(J, k, S=1, Jcal=Ji), Eq, Ph, r)['Pi']
            Gf = sum((2 * Jf + 1) * (2 * J + 1) * float(wigner_6j(lf, Jf, 1, Ji, J, K)) ** 2 * GK[K] for K in (1, 2))
            check(f"spin-1 HQ Pi J={J} {kind} Jcal={Ji} -> 1{'SPDF'[lf]} J'={Jf} (6j, K=1,2)", G, Gf, 1e-8)

# ---------------------------------------------------------------- the two channels
def pi_sigma(H, J, xi):
    pr = pi_sigma_projections(H)
    return (pr['R_Pi_lJ'] if xi == -1 else pr['R_Pi']), pr['R_Sigma']

def ps(J, lp):
    return pK(1, J, lp), s_l(J, lp)

# 10. selection rules of the two channels
for (J, kind, n) in HTYPES:
    k = lib.hybrid_index(J, kind, n); H = lib.hybrid(J, k); pr = pi_sigma_projections(H)
    print(f"  {J} {kind}{n}: P_Pi = {pr['P_Pi']:.4f}, P_Sigma = {pr['P_Sigma']:.4f}")
    check(f"norm P_Pi + P_Sigma = 1, J={J} {kind}{n}", pr['P_Pi'] + pr['P_Sigma'], 1.0, 1e-10)
    for lf in range(0, 4):
        sp = width_split(lib.quarkonium(1, lf), H, 0.8, Params('c', kH=(0.7, 1.3, -0.4, 0.55)), r)
        if sp['total'] < 1e-20:          # parity forbidden
            continue
        if kind == 'minus':
            check(f"l=J state has no Sigma channel: J={J} -> 1{'SPDF'[lf]}", (sp['Sigma'] + abs(sp['int'])) / sp['total'] * 1e-13, 0.0)
        if kind == 'sigma':
            check(f"J=0 state has no Pi channel: -> 1{'SPDF'[lf]}", (sp['Pi'] + abs(sp['int'])) / sp['total'] * 1e-13, 0.0)

# 11. p0 (pure Sigma channel): Gamma = 4 E^3/(3 pi^2 sigma) |A0|^2
k = lib.hybrid_index(0, 'sigma', 1); H = lib.hybrid(0, k); _, cSig = pi_sigma(H, 0, 1)
f = lib.quarkonium(1, 1); cf = f.comps[0].c; Ph = Params('c', kH=(1, 1, 1, 1)); Eq = 0.95
A0 = 2 * np.sqrt(np.pi / sig) * Ph.eeQ * (np.sum(cf * cSig / r) - Eq / (3 * np.pi) * np.sum(cf * cSig))
check("p0 -> h_c(1P) closed formula", width(f, H, Eq, Ph, r), 4 * Eq ** 3 / (3 * np.pi ** 2 * sig) * A0 ** 2, 1e-7)

# 12. non-local Sigma vertex, p0 -> 1P: only rank K = 1 survives (effective S -> P);
#     A0 -> sum c_f c_S [w4 c4 F1[f] + w2 c2 F1[f - F3] + w3 c3 F1[(9 f - F3)/8]]
fH3 = lambda y: 9 * np.pi ** 2 * np.cos(y / 2) / (9 * np.pi ** 2 - y ** 2)
def F1of(fun, x):
    return sum((2 * L + 1) * proj(fun, x, L) * cg(1, 0, L, 0, 1, 0) * cg(1, 1, L, 0, 1, 1) for L in range(0, LM + 1, 2))
for kh in [(0, 0, 0, 1), (0, 1, 0, 0), (0, 0, 1, 0), (0.7, 1.3, -0.4, 0.55)]:
    Ph = Params('c', kH=kh, nonloc=True); c1, c2, c3, c4 = [x * Ph.eeQ for x in kh]
    x = Eq * r; sq = np.sqrt(np.pi / sig)
    A0 = np.sum(cf * cSig * (2 * sq / r * c4 * F1of(fH, x) + 0.25 * sq / r * c2 * F1of(lambda y: fH(y) - fH3(y), x)
                            - 2 * sq / r * Eq * r / (3 * np.pi) * c3 * F1of(lambda y: (9 * fH(y) - fH3(y)) / 8, x)))
    check(f"NL p0 -> h_c(1P) multipole formula c={kh}", width(f, H, Eq, Ph, r), 4 * Eq ** 3 / (3 * np.pi ** 2 * sig) * A0 ** 2, 1e-8)

# 13. FT of the Sigma densities (eq:rhoSigma) vs the closed form factors used in the code
from scipy.integrate import quad
for (rr, q) in [(3.0, 0.4), (6.0, 1.1), (2.0, 4.0)]:
    Q = q * rr
    for (c2, c3, c4) in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
        rhoB = lambda z: np.pi / (4 * sig * rr ** 2) * ((8 * c4 + c2) * np.cos(np.pi * z / rr) + 3 * c2 * np.cos(3 * np.pi * z / rr))
        rhoE = lambda z: -c3 / (4 * sig * rr) * (3 * np.cos(np.pi * z / rr) + np.cos(3 * np.pi * z / rr))
        num = quad(lambda z: (rhoB(z) + q * rhoE(z)) * np.cos(q * z), -rr / 2, rr / 2)[0]
        cl = (4 * c4 / (sig * rr) * fH(Q) + c2 / (2 * sig * rr) * (fH(Q) - fH3(Q))
              - 4 * c3 / (3 * np.pi * sig) * q * (9 * fH(Q) - fH3(Q)) / 8)
        check(f"FT rho_Sigma r={rr} q={q} (c2,c3,c4)={(c2, c3, c4)}", num, cl, 1e-9)

# 14. spin-1 hybrids with both channels: 6j recoupling of the rank-1 (Pi + Sigma) and rank-2 widths
for (J, n, lf) in [(1, 1, 0), (1, 1, 2), (2, 1, 1)]:
    k = lib.hybrid_index(J, 'plus', n); H0 = lib.hybrid(J, k); cPi, cSig = pi_sigma(H0, J, 1)
    kh = (0.7, 1.3, -0.4, 0.55); Ph = Params('c', kH=kh); c1, c2, c3, c4 = [v * Ph.eeQ for v in kh]
    f0 = lib.quarkonium(1, lf); cf = f0.comps[0].c; Eq = 0.8
    I0 = np.sum(cf * cPi); I1 = np.sum(cf * cPi * r)
    A1 = c2 * I0 - Eq / (2 * np.pi) * (c1 + c3) * I1; A2 = Eq / (2 * np.pi) * (c1 - c3) * I1
    A0 = 2 * np.sqrt(np.pi / sig) * (c4 * np.sum(cf * cSig / r) - c3 * Eq / (3 * np.pi) * np.sum(cf * cSig))
    u, v = ps(J, lf)
    GK = {1: 8 * Eq ** 3 / (np.pi ** 2 * sig) * (u * A1 + v * A0) ** 2 / 6,
          2: 8 * Eq ** 3 / (np.pi ** 2 * sig) * wK(2, J, lf) / 5 * A2 ** 2}
    for Ji in range(abs(J - 1), J + 2):
        for Jf in range(abs(lf - 1), lf + 2):
            G = width(lib.quarkonium(1, lf, S=1, J=Jf), lib.hybrid(J, k, S=1, Jcal=Ji), Eq, Ph, r)
            Gf = sum((2 * Jf + 1) * (2 * J + 1) * float(wigner_6j(lf, Jf, 1, Ji, J, K)) ** 2 * GK[K] for K in (1, 2))
            check(f"spin-1 + Sigma J={J} plus Jcal={Ji} -> 1{'SPDF'[lf]} J'={Jf}", G, Gf, 1e-8)

print("ALL PASS" if ok else "SOME FAILURES")
