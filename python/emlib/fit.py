"""Observables and the flavour-independent fit of g_E1(r) (Sec. sec:pheno_E1 of the notes).

Parameters: p = (A_b, kappa_b) (A in GeV, kappa in units of e). The charmonium constants follow from the
flavour independence of A = A_Q e_Q and kappa = kappa_Q e_Q (the constants depend on the heavy quark only
through its charge):
    A_c = (e_b/e_c) A_b = -A_b/2,     kappa_c = (e_b/e_c) kappa_b = -kappa_b/2.
Every width includes the 1/m factor (1 + delta_1m)^2 unless corr=False (LO).

chi^2 (Sec. sec:pheno_inputs): full experimental covariance (common total widths), asymmetric errors,
90% CL upper limits as half-normal likelihoods, chi2_UL = (Gamma_th/s)^2, s^2 = (UL/1.645)^2 + sigma_th^2,
and the theory error sigma_th of Eq. (eq:sigma_th) added in quadrature; eps = EPS_V2 (eps_c = v_c^2 = 0.3 for
the charmonium observables, eps_b = v_b^2 = 0.1 for the bottomonium ones) or 0 (experimental errors only).

The fit is restricted to the physical region of both flavours, -0.8 <= kappa_Q/e_Q <= 1 and
|A_Q/e_Q| <= 2 GeV: outside it the correction reverses the sign of the LO amplitude in the region of the
overlaps and the chi^2 (which depends on the square of the amplitude) has spurious mirror minima.
"""
import numpy as np
from scipy.optimize import minimize, minimize_scalar, brentq
from .e1 import CHARGE, photon_energy, width_form, delta_1m, theory_sigma
from .data import MASS, STATES, covariance

EPS_V2 = {'c': 0.3, 'b': 0.1}
Z90 = 1.6449                     # 90% quantile of the half-normal distribution
KAPPA_RANGE = (-0.8, 1.0)        # range of kappa_Q/e_Q
A_MAX = 2.0                      # GeV, maximum of |A_Q/e_Q|


def build_state(lib, label):
    fl, n, l, S, J = STATES[label]
    return lib.state(label, n, l, S, J, MASS[label])


def flavour_params(p, fl):
    """(A_Q, kappa_Q) of flavour fl from p = (A_b, kappa_b)"""
    k = CHARGE['b'] / CHARGE[fl]
    return k * p[0], k * p[1]


def inside(p):
    for fl in ('c', 'b'):
        A, kap = flavour_params(p, fl)
        if not (KAPPA_RANGE[0] <= kap / CHARGE[fl] <= KAPPA_RANGE[1] and abs(A / CHARGE[fl]) <= A_MAX):
            return False
    return True


class Observable:
    """one row of DATA: for each width (two for a ratio) the photon energy, delta_1m and the quadratic forms
    G, Ga (|u_f u_i|) of emlib.e1.width_form, in the L or NL form"""

    def __init__(self, row, libs, nonloc):
        self.row, self.flavor, self.nonloc = row, row['flavor'], nonloc
        lib = libs[self.flavor]
        ini = build_state(lib, row['init'])
        self.parts = []
        for lab in [row['final']] + ([row['final2']] if 'final2' in row else []):
            fin = build_state(lib, lab)
            E = photon_energy(ini.mass, fin.mass)
            self.parts.append({'final': lab, 'E': E, 'delta': delta_1m(ini, fin, E),
                               'G': width_form(fin, ini, E, lib.r, nonloc),
                               'Ga': width_form(fin, ini, E, lib.r, nonloc, absval=True)})
        self.is_ratio = row['kind'] in ('ratio', 'ul_ratio')
        self.is_ul = row['kind'].startswith('ul')

    def widths(self, p, corr=True):
        """[(Gamma, Gamma_abs)] (keV) of each width at p = (A_b, kappa_b)"""
        A, kap = flavour_params(p, self.flavor)
        v = np.array([1.0, A, kap])
        out = []
        for q in self.parts:
            fac = (1.0 + q['delta']) ** 2 if corr else 1.0
            out.append((fac * v @ q['G'] @ v, fac * v @ q['Ga'] @ v))
        return out

    def predict(self, p, eps, corr=True):
        """(prediction, theory error): width (keV) or ratio; eps: number or {'c':, 'b':}"""
        e = eps[self.flavor] if isinstance(eps, dict) else eps
        w = self.widths(p, corr)
        if not self.is_ratio:
            G, Ga = w[0]
            return G, theory_sigma(G, Ga, e)
        (g1, a1), (g2, a2) = w
        R = g1 / g2
        s1 = theory_sigma(g1, a1, e) / g1 if g1 > 0 else 0.0
        return R, R * np.hypot(s1, theory_sigma(g2, a2, e) / g2)


class Fit:
    """flavour-independent fit of p = (A_b, kappa_b) to the rows of DATA with set in fit_sets[flavor]"""

    def __init__(self, rows, libs, nonloc, fit_sets):
        self.rows, self.nonloc = rows, nonloc
        self.obs = [Observable(r, libs, nonloc) for r in rows]
        self.fitted = [r['set'] in fit_sets[r['flavor']] for r in rows]
        self.meas = [k for k, r in enumerate(rows) if self.fitted[k] and not r['kind'].startswith('ul')]
        self.uls = [k for k, r in enumerate(rows) if self.fitted[k] and r['kind'].startswith('ul')]
        self.Cexp = covariance([rows[k] for k in self.meas])
        self.ndf = len(self.meas) - 2

    def model(self, p, eps, corr=True):
        return [o.predict(p, eps, corr) for o in self.obs]

    def chi2_parts(self, p, eps, corr=True):
        """{'c': chi2_c, 'b': chi2_b} (the covariance is block diagonal in flavour)"""
        mod = self.model(p, eps, corr)
        out = {}
        for fl in ('c', 'b'):
            idx = [a for a, k in enumerate(self.meas) if self.rows[k]['flavor'] == fl]
            d = np.array([self.rows[self.meas[a]]['value'] for a in idx])
            pr = np.array([mod[self.meas[a]][0] for a in idx])
            C = self.Cexp[np.ix_(idx, idx)].copy()
            for b, a in enumerate(idx):
                row = self.rows[self.meas[a]]
                if 'err_up' in row:
                    C[b, b] = (row['err_up'] if pr[b] > d[b] else row['err_dn']) ** 2
                C[b, b] += mod[self.meas[a]][1] ** 2
            res = d - pr
            c = float(res @ np.linalg.solve(C, res))
            for k in self.uls:
                if self.rows[k]['flavor'] == fl:
                    s = np.hypot(self.rows[k]['value'] / Z90, mod[k][1])
                    c += (mod[k][0] / s) ** 2
            out[fl] = c
        return out

    def chi2(self, p, eps, corr=True):
        if not inside(p):
            return 1e9
        return sum(self.chi2_parts(p, eps, corr).values())

    # -------------------------------------------------------------- minimum, errors
    def minimize(self, eps):
        """global minimum: grid over the physical region, then Nelder-Mead from the 4 lowest local minima
        of the grid. Returns (p, chi2_min)."""
        e = CHARGE['b']
        As = np.linspace(-A_MAX, A_MAX, 81) * e
        ks = np.linspace(KAPPA_RANGE[0], KAPPA_RANGE[1], 73) * e
        C = np.array([[self.chi2((A, k), eps) for k in ks] for A in As])
        starts = []
        for i in range(len(As)):
            for j in range(len(ks)):
                if C[i, j] <= C[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2].min():
                    starts.append((C[i, j], As[i], ks[j]))
        best = None
        for _, A0, k0 in sorted(starts)[:4]:
            res = minimize(lambda q: self.chi2(q, eps), x0=[A0, k0], method='Nelder-Mead',
                           options={'xatol': 1e-7, 'fatol': 1e-9, 'maxiter': 4000})
            if best is None or res.fun < best.fun:
                best = res
        return np.array(best.x), float(best.fun)

    def covariance(self, p0, eps, h=1e-4):
        """covariance of (A_b, kappa_b) from the Hessian of chi2/2 at the minimum"""
        H = np.zeros((2, 2))
        for a in range(2):
            for b in range(2):
                def c(da, db):
                    q = np.array(p0, dtype=float); q[a] += da; q[b] += db
                    return self.chi2(q, eps)
                H[a, b] = (c(h, h) - c(h, -h) - c(-h, h) + c(-h, -h)) / (8.0 * h * h)
        return np.linalg.inv(H)

    def profile_errors(self, p0, c0, eps):
        """Delta chi2 = 1 errors (lo, hi) of each parameter, minimizing over the other one"""
        errs = []
        for i in (0, 1):
            j = 1 - i
            def prof(x):
                def f(y):
                    q = [0.0, 0.0]; q[i] = x; q[j] = y
                    return self.chi2(q, eps)
                return minimize_scalar(f, bracket=(p0[j] - 0.02, p0[j] + 0.02)).fun - c0 - 1.0
            out = []
            for sgn in (-1, +1):
                step, x, val = 0.01, p0[i], np.nan
                for _ in range(200):
                    x2 = x + sgn * step
                    if prof(x2) > 0:
                        val = abs(brentq(prof, min(x, x2), max(x, x2), xtol=1e-7) - p0[i])
                        break
                    x, step = x2, step * 1.3
                out.append(val)
            errs.append(tuple(out))
        return errs

    def run(self, eps):
        """minimum, profile errors, covariance, correlation and chi2 per flavour at eps"""
        p, c = self.minimize(eps)
        cov = self.covariance(p, eps)
        return {'p': p, 'chi2': c, 'errors': self.profile_errors(p, c, eps), 'cov': cov,
                'rho': cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]), 'chi2_parts': self.chi2_parts(p, eps)}

    # -------------------------------------------------------------- predictions
    @staticmethod
    def prediction_error(obs, p, cov, h=1e-5):
        """error of the prediction of an Observable propagated from the covariance of (A_b, kappa_b)"""
        g = np.zeros(2)
        for a in range(2):
            dp = np.zeros(2); dp[a] = h
            g[a] = (obs.predict(p + dp, 0.0)[0] - obs.predict(p - dp, 0.0)[0]) / (2 * h)
        return float(np.sqrt(g @ cov @ g))

    def pulls(self, p, eps):
        """for every row: prediction, theory error and pull (data - prediction)/sigma, with sigma^2 the
        experimental variance (asymmetric errors: the side of the prediction) plus sigma_th^2; for upper
        limits the pull is sqrt(chi2_UL)"""
        out = []
        for k, (o, row) in enumerate(zip(self.obs, self.rows)):
            pr, sth = o.predict(p, eps)
            if o.is_ul:
                s = np.hypot(row['value'] / Z90, sth)
                out.append((pr, sth, pr / s, 0.0))
                continue
            e = np.hypot(row['err'], row.get('err_corr', 0.0))
            if 'err_up' in row:
                e = row['err_up'] if pr > row['value'] else row['err_dn']
            out.append((pr, sth, (row['value'] - pr) / np.hypot(e, sth), e))
        return out
