"""Quarkonium E1 with an interpolated LEF (EM_Pheno v6, Sec. sec:pheno_E1_interpol of the notes):

    g_E1(r) = e r [ e_Q + kappa_Q f(r) ] = e e_Q r [ 1 + lambda f(r) ],   lambda = kappa_Q / e_Q,

f(r) = (r/r0)^n / (1 + (r/r0)^n)  (f(0) = 0, f(r >> r0) = 1). kappa_Q is the effective charge
added at the string end points at long distances (light-quark screening plus whatever
long-distance correction it absorbs), in units of e. It is fitted separately for c and b.

The amplitude is linear in lambda, A = A_0 + lambda A_f, with A_0 the LO dipole amplitude
(g = e e_Q r) and A_f the same with g = e e_Q r f(r). Hence every width is a quadratic form
    Gamma(lambda) = G00 + 2 lambda G0f + lambda^2 Gff,
computed once (spin sums included) and reused in the fit. <f> = G0f/G00 is the average of f
over the transition (Gamma ~ G00 (1 + 2 lambda <f>) at first order). The O(k/m_Q) corrections
of EGTKLY (quarkonium.rel_delta) multiply the whole amplitude, A -> A (1 + delta), at all r.
The theory error uses the same quadratic form for the no-cancellation (|u_f u_i|) widths.
Setting f = 1 reproduces quarkonium.Fitter exactly with kappa_old = 1 + kappa_Q/e_Q.
"""
import numpy as np
from scipy.optimize import minimize_scalar, brentq
from .amplitudes import Params, amplitudes, photon_energy, theory_sigma, CHARGE
from .data import MASS, covariance
from .quarkonium import build_state, rel_delta, GEV2KEV, Z90

HBARC = 0.1973269804                       # GeV fm
SIGMA = 0.21                               # GeV^2, string tension of the potentials
R0_EST = np.sqrt(np.pi / (6.0 * SIGMA))    # GeV^-1: sigma r^2 = 2 pi |N - 1/12| for N = 0 (= 1.58 GeV^-1 = 0.31 fm)


def f_hill(r, n=2, r0=R0_EST):
    """interpolating function f(r) = (r/r0)^n / (1 + (r/r0)^n); n = None means f = 1 (no interpolation)"""
    r = np.asarray(r, dtype=float)
    if n is None:
        return np.ones_like(r)
    x = (r / r0) ** n
    return x / (1.0 + x)


def _parts(final, initial, E, flavor, nonloc, fr, r, absval):
    """(G00, G0f, Gff) in keV for one transition"""
    P0 = Params(flavor, kappa_Q=1.0, nonloc=nonloc)
    Pf = Params(flavor, kappa_Q=1.0, nonloc=nonloc)
    Pf.fE1 = fr
    M0 = amplitudes(final, initial, E, P0, r, absval)
    Mf = amplitudes(final, initial, E, Pf, r, absval)
    norm = E / (2.0 * np.pi) / (2 * initial.J + 1) * GEV2KEV
    G00 = norm * sum(abs(v) ** 2 for v in M0.values())
    Gff = norm * sum(abs(v) ** 2 for v in Mf.values())
    G0f = norm * sum((M0[k] * np.conj(Mf[k])).real for k in M0)
    return np.array([G00, G0f, Gff])


def predict_parts(row, libs, nonloc=False, rel=False, n=2, r0=R0_EST):
    """quadratic-form coefficients of an observable: list (one per width, two for a ratio) of
    dicts {'g': (G00, G0f, Gff), 'ga': same for |u_f u_i|}, keV; rel=True includes (1+delta)^2."""
    lib = libs[row['flavor']]
    fr = f_hill(lib.r, n, r0)
    i = build_state(lib, row['init'])
    out = []
    for flab in [row['final']] + ([row['final2']] if 'final2' in row else []):
        f = build_state(lib, flab)
        E = photon_energy(MASS[row['init']], MASS[flab])
        g = _parts(f, i, E, row['flavor'], nonloc, fr, lib.r, False)
        ga = _parts(f, i, E, row['flavor'], nonloc, fr, lib.r, True)
        if rel:
            fac = (1.0 + rel_delta(row['init'], flab, E)) ** 2
            g, ga = g * fac, ga * fac
        out.append({'g': g, 'ga': ga, 'delta': rel_delta(row['init'], flab, E)})
    return out


def quad(c, lam):
    return c[0] + 2.0 * lam * c[1] + lam ** 2 * c[2]


def mean_f(part):
    """<f> = G0f/G00 of one width"""
    return part['g'][1] / part['g'][0]


class InterpFitter:
    """One-parameter fit of kappa_Q in g_E1 = e r [e_Q + kappa_Q f(r)], same chi^2 and
    theory-error model as quarkonium.Fitter."""

    def __init__(self, libs, rows, nonloc=False, rel=True, n=2, r0=R0_EST):
        self.rows, self.nonloc, self.rel, self.n, self.r0 = rows, nonloc, rel, n, r0
        self.eQ = CHARGE[rows[0]['flavor']]
        self.parts = [predict_parts(r, libs, nonloc, rel, n, r0) for r in rows]
        self.meas = [k for k, r in enumerate(rows) if r['kind'] in ('width', 'ratio')]
        self.uls = [k for k, r in enumerate(rows) if r['kind'] in ('ul_width', 'ul_ratio')]
        self.Cexp = covariance([rows[k] for k in self.meas])
        # scan range: lambda = kappa/e_Q in [-0.8, 1.0], as kappa_old in [0.2, 2.0] of Fitter
        lo, hi = sorted((-0.8 * self.eQ, 1.0 * self.eQ))
        self.krange = (lo, hi)

    def model(self, kappa, eps):
        lam = kappa / self.eQ
        out = []
        for row, ps in zip(self.rows, self.parts):
            if row['kind'] in ('width', 'ul_width'):
                G, Ga = quad(ps[0]['g'], lam), quad(ps[0]['ga'], lam)
                out.append((G, theory_sigma(G, Ga, eps)))
            else:
                g1, a1 = quad(ps[0]['g'], lam), quad(ps[0]['ga'], lam)
                g2, a2 = quad(ps[1]['g'], lam), quad(ps[1]['ga'], lam)
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
        for a, k in enumerate(self.meas):
            row = self.rows[k]
            if 'err_up' in row:
                e = row['err_up'] if p[a] > d[a] else row['err_dn']
                C[a, a] = e ** 2
        C += np.diag(sth ** 2)
        res = d - p
        chi = float(res @ np.linalg.solve(C, res))
        pulls = res / np.sqrt(np.diag(C))
        chi_ul, ulpulls = 0.0, []
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
        """kappa_Q (global minimum on a grid, then refined), chi2_min, and Delta chi2 = 1 errors (lo, hi)"""
        lo, hi = self.krange
        ks = np.linspace(lo, hi, 721)
        cs = np.array([self.chi2(k, eps) for k in ks])
        j = int(np.argmin(cs))
        a, b = ks[max(j - 1, 0)], ks[min(j + 1, len(ks) - 1)]
        res = minimize_scalar(lambda k: self.chi2(k, eps), bounds=(a, b), method='bounded', options={'xatol': 1e-7})
        k0, c0 = res.x, res.fun
        f = lambda k: self.chi2(k, eps) - c0 - 1.0
        elo = k0 - brentq(f, lo, k0) if f(lo) > 0 else np.nan
        ehi = brentq(f, k0, hi) - k0 if f(hi) > 0 else np.nan
        return k0, c0, elo, ehi

    def ndf(self):
        return len(self.meas) - 1

    def eps_for_unit_chi2(self):
        g = lambda e: self.fit_kappa(e)[1] / self.ndf() - 1.0
        if g(1e-4) <= 0:
            return 0.0
        return brentq(g, 1e-4, 2.0, xtol=1e-5)
