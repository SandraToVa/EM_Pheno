"""Quarkonium E1 transitions: predictions for the data set and global fit of
kappa_Q = c/(e e_Q)  [g_E1(r) = kappa_Q e e_Q r]  with a theory-error model.
"""
import numpy as np
from scipy.optimize import minimize_scalar, brentq
from .states import Library
from .amplitudes import Params, width, photon_energy, theory_sigma
from .data import DATA, MASS, STATES, covariance

GEV2KEV = 1e6
Z90 = 1.6449   # half-normal 90% quantile


def build_state(lib, label):
    fl, n, l, S, J = STATES[label]
    st = lib.quarkonium(n, l, S=S, J=J, mass=MASS[label], name=label)
    return st


# heavy-quark masses of the QQbarSpectrum wave functions (GeV), used in the 1/m relativistic terms
M_QUARK = {'c': 1.496, 'b': 4.88}


# Relativistic corrections of the E1 amplitude, A -> A (1 + delta) (option corr):
#   'LO'   no correction, delta = 0.
#   '1m'   O(k/m_Q) corrections of EGTKLY: c_F^em spin-orbit term + spin-independent Delta l term.
#   '1m2'  '1m' + the 1/m^2 c_S^em spin-orbit operator (Brambilla, Pietrulewicz, Vairo 2012, Eq. (86)).
#          With the LO Schroedinger equation, Eq. (119), (I2' + 2 I1)/(m^2 I3) = +-k_LO/(2m), so the
#          c_S^em term cancels the c_F^em spin term (their Eq. (122)) and only the Delta l term is left.
CORRECTIONS = ('LO', '1m', '1m2')
# second theory-error option of each fit (the first is epsilon from chi2/ndf = 1, the third epsilon = 0):
# epsilon = v^2 at LO, epsilon = v^3 once the O(v^2) corrections are included (v^2 = 0.3 charm, 0.1 bottom)
EPS_ALT = {'LO': {'c': 0.3, 'b': 0.1}, '1m': {'c': 0.3 ** 1.5, 'b': 0.1 ** 1.5}, '1m2': {'c': 0.3 ** 1.5, 'b': 0.1 ** 1.5}}
EPS_ALT_LABEL = {'LO': 'v2', '1m': 'v3', '1m2': 'v3'}


def rel_delta(init, final, E, corr='1m2'):
    """relative correction delta of the E1 amplitude (A -> A (1 + delta)) for option corr:
        delta = k/(8 m_Q) * ( s [J_P (J_P + 1) - 4]  -  Delta_l / 3 ),
    Delta_l = l_i (l_i + 1) - l_f (l_f + 1), k = photon energy.
    Delta l term (corr = '1m', '1m2'): for P -> S (S -> P) it is the -k/6m (+k/6m) of Eqs. (111)-(114)
    of Brambilla, Pietrulewicz, Vairo (2012) in the width (their Eqs. (80), (81) and the factor
    (1 - k/M) of the LO width, non-relativistic normalization here); it is also the Delta l term of
    EGTKLY (Eichten et al. 1980, App. E of the notes), used as well for D -> P.
    Spin term (corr = '1m' only): EGTKLY = BPV Eq. (85) with c_F^em = 1; s = -1 if the P wave is the
    initial state, +1 if final; triplet S <-> P only. Cancelled by the c_S^em term for corr = '1m2'."""
    if corr == 'LO':
        return 0.0
    if corr not in CORRECTIONS:
        raise ValueError(corr)
    fl, ni, li, Si, Ji = STATES[init]
    _, nf, lf, Sf, Jf = STATES[final]
    dl = li * (li + 1) - lf * (lf + 1)
    sp = 0.0
    if corr == '1m' and Si == 1 and {li, lf} == {0, 1}:
        JP, sgn = (Ji, -1.0) if li == 1 else (Jf, +1.0)
        sp = sgn * (JP * (JP + 1) - 4)
    return E / (8.0 * M_QUARK[fl]) * (sp - dl / 3.0)


def predict_row(row, libs, nonloc=False, kappa=1.0, corr='LO'):
    """Returns (pred, pred_abs) at kappa (pred_abs: no-cancellation width used
    for the theory error).  Widths in keV; ratios dimensionless (and the
    corresponding 'abs' ratio pair).  Each width is multiplied by (1 + delta)^2,
    delta = rel_delta(..., corr) (delta = 0 for corr = 'LO')."""
    lib = libs[row['flavor']]
    P = Params(row['flavor'], kappa_Q=kappa, nonloc=nonloc)
    i = build_state(lib, row['init'])
    def G(flab):
        f = build_state(lib, flab)
        E = photon_energy(MASS[row['init']], MASS[flab])
        g = width(f, i, E, P, lib.r) * GEV2KEV
        ga = width(f, i, E, P, lib.r, absval=True) * GEV2KEV
        fac = (1.0 + rel_delta(row['init'], flab, E, corr)) ** 2
        return g * fac, ga * fac
    if row['kind'] in ('width', 'ul_width'):
        return G(row['final'])
    g1, a1 = G(row['final']); g2, a2 = G(row['final2'])
    return (g1, a1, g2, a2)


class Fitter:
    """fit of kappa_Q = 1 + dkappa_Q in g_E1 = kappa_Q e e_Q r, with corrections corr ('LO', '1m', '1m2')"""

    def __init__(self, libs, rows, nonloc=False, corr='LO'):
        self.rows = rows
        self.nonloc = nonloc
        self.corr = corr
        self.pred1 = [predict_row(r, libs, nonloc, 1.0, corr) for r in rows]   # at kappa = 1
        self.meas = [k for k, r in enumerate(rows) if r['kind'] in ('width', 'ratio')]
        self.uls = [k for k, r in enumerate(rows) if r['kind'] in ('ul_width', 'ul_ratio')]
        self.Cexp = covariance([rows[k] for k in self.meas])

    def model(self, kappa, eps):
        """predictions and theory sigma for every row"""
        out = []
        for row, p in zip(self.rows, self.pred1):
            if row['kind'] in ('width', 'ul_width'):
                g, ga = p
                G = kappa ** 2 * g; Ga = kappa ** 2 * ga
                out.append((G, theory_sigma(G, Ga, eps)))
            else:
                g1, a1, g2, a2 = p
                R = g1 / g2
                s1 = theory_sigma(g1, a1, eps) / g1 if g1 > 0 else 0.0
                s2 = theory_sigma(g2, a2, eps) / g2
                out.append((R, R * np.hypot(s1, s2)))
        return out

    def chi2(self, kappa, eps, detail=False):
        mod = self.model(kappa, eps)
        d = np.array([self.rows[k]['value'] for k in self.meas])
        p = np.array([mod[k][0] for k in self.meas])
        sth = np.array([mod[k][1] for k in self.meas])
        C = self.Cexp.copy()
        # asymmetric errors
        for a, k in enumerate(self.meas):
            row = self.rows[k]
            if 'err_up' in row:
                e = row['err_up'] if p[a] > d[a] else row['err_dn']
                C[a, a] = e ** 2
        C += np.diag(sth ** 2)
        res = d - p
        chi = float(res @ np.linalg.solve(C, res))
        pulls = res / np.sqrt(np.diag(C))
        chi_ul = 0.0
        ulpulls = []
        for k in self.uls:
            row = self.rows[k]
            s = np.hypot(row['value'] / Z90, mod[k][1])
            c = (mod[k][0] / s) ** 2 if mod[k][0] > 0 else 0.0
            chi_ul += c
            ulpulls.append(np.sqrt(c))
        if detail:
            return chi + chi_ul, chi, chi_ul, p, sth, pulls, ulpulls, mod
        return chi + chi_ul

    def fit_kappa(self, eps):
        res = minimize_scalar(lambda k: self.chi2(k, eps), bounds=(0.2, 2.0), method='bounded',
                              options={'xatol': 1e-6})
        k0, c0 = res.x, res.fun
        f = lambda k: self.chi2(k, eps) - c0 - 1.0
        lo = brentq(f, 0.05, k0) if f(0.05) > 0 else np.nan
        hi = brentq(f, k0, 3.0) if f(3.0) > 0 else np.nan
        return k0, c0, k0 - lo, hi - k0

    def ndf(self):
        return len(self.meas) - 1

    def eps_for_unit_chi2(self):
        g = lambda e: self.fit_kappa(e)[1] / self.ndf() - 1.0
        if g(1e-4) <= 0:
            return 0.0
        return brentq(g, 1e-4, 2.0)
