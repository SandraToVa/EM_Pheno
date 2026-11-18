"""Observables and the flavour-independent fit of g_E1(r) (Sec. sec:pheno_E1 of the notes).

Parameters: p = (A_b, kappa_b) (A in GeV, kappa in units of e). The charmonium constants follow from the
flavour independence of A = A_Q e_Q and kappa = kappa_Q e_Q (the constants depend on the heavy quark only
through its charge):
    A_c = (e_b/e_c) A_b = -A_b/2,     kappa_c = (e_b/e_c) kappa_b = -kappa_b/2.
Every width includes the 1/m factor (1 + delta_1m)^2 unless corr=False (LO).

chi^2 (Sec. sec:pheno_inputs), Eq. (eq:chi2):
    chi^2 = (d - Gamma)^T C^-1 (d - Gamma) + chi2_UL,     C = C_exp + diag(sigma_th^2),
with C_exp the full experimental covariance (widths that share a total width are correlated through it;
asymmetric errors: the side of the prediction), 90% CL upper limits as half-normal likelihoods,
chi2_UL = (Gamma_th/s)^2, s^2 = (UL/1.645)^2 + sigma_th^2, and the theory error sigma_th = 2 eps Gamma of
Eq. (eq:sigma_th); eps = EPS_V2 (eps_c = v_c^2 = 0.3 for the charmonium observables, eps_b = v_b^2 = 0.1 for the
bottomonium ones) or 0 (experimental errors only).

Errors of the parameters (v11, Eq. (eq:exp_range)): the fit is repeated with every measured width and ratio
moved to the upper end of its experimental interval, d + sigma_exp (asymmetric errors: d + err_up), and to the
lower end, d - sigma_exp (d - err_dn), with the same C and the same upper limits. The interval of each parameter
is the range spanned by the central, "max" and "min" fits; the errors of the predictions are the range of the
prediction over the same three fits. (Delta chi^2 = 1 errors are no longer computed.)

The fit is restricted to the physical region of both flavours, -0.8 <= kappa_Q/e_Q <= 1 and
|A_Q/e_Q| <= 2 GeV: outside it the correction reverses the sign of the LO amplitude in the region of the
overlaps and the chi^2 (which depends on the square of the amplitude) has spurious mirror minima.
"""
import numpy as np
import copy
from scipy.optimize import minimize
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
    """one row of DATA: for each width (two for a ratio) the photon energy, delta_1m and the quadratic form
    G of emlib.e1.width_form, in the L or NL form"""

    def __init__(self, row, libs, nonloc):
        self.row, self.flavor, self.nonloc = row, row['flavor'], nonloc
        lib = libs[self.flavor]
        ini = build_state(lib, row['init'])
        self.parts = []
        for lab in [row['final']] + ([row['final2']] if 'final2' in row else []):
            fin = build_state(lib, lab)
            E = photon_energy(ini.mass, fin.mass)
            self.parts.append({'final': lab, 'E': E, 'delta': delta_1m(ini, fin, E),
                               'G': width_form(fin, ini, E, lib.r, nonloc)})
        self.is_ratio = row['kind'] in ('ratio', 'ul_ratio')
        self.is_ul = row['kind'].startswith('ul')

    def widths(self, p, corr=True):
        """[Gamma] (keV) of each width at p = (A_b, kappa_b)"""
        A, kap = flavour_params(p, self.flavor)
        v = np.array([1.0, A, kap])
        return [((1.0 + q['delta']) ** 2 if corr else 1.0) * v @ q['G'] @ v for q in self.parts]

    def predict(self, p, eps, corr=True):
        """(prediction, theory error): width (keV) or ratio; eps: number or {'c':, 'b':}.
        sigma_th = 2 eps Gamma for a width; for a ratio the two relative errors 2 eps are added in quadrature."""
        e = eps[self.flavor] if isinstance(eps, dict) else eps
        w = self.widths(p, corr)
        if not self.is_ratio:
            return w[0], theory_sigma(w[0], e)
        g1, g2 = w
        R = g1 / g2
        s1 = theory_sigma(g1, e) / g1 if g1 > 0 else 0.0
        return R, R * np.hypot(s1, theory_sigma(g2, e) / g2)


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
        self.data = np.array([r['value'] for r in rows], dtype=float)      # data vector (shifted in exp_shifted)

    def model(self, p, eps, corr=True):
        return [o.predict(p, eps, corr) for o in self.obs]

    def chi2_parts(self, p, eps, corr=True):
        """{'c': chi2_c, 'b': chi2_b} (the covariance is block diagonal in flavour)"""
        mod = self.model(p, eps, corr)
        out = {}
        for fl in ('c', 'b'):
            idx = [a for a, k in enumerate(self.meas) if self.rows[k]['flavor'] == fl]
            d = np.array([self.data[self.meas[a]] for a in idx])
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

    @staticmethod
    def sigma_exp(row, sign):
        """experimental error of a measured row on the side sign (+1 upper, -1 lower): err_up / err_dn where
        quoted, otherwise the total error (err and the total-width error err_corr in quadrature)"""
        if 'err_up' in row:
            return row['err_up'] if sign > 0 else row['err_dn']
        return float(np.hypot(row['err'], row.get('err_corr', 0.0)))

    def exp_shifted(self, sign):
        """copy of the fit with every measured width and ratio moved to d + sign sigma_exp (Gamma_exp,max for
        sign = +1, Gamma_exp,min for sign = -1); upper limits, C_exp and the theory error are unchanged"""
        F = copy.copy(self)
        F.data = self.data.copy()
        for k in self.meas:
            F.data[k] = self.data[k] + sign * self.sigma_exp(self.rows[k], sign)
        return F

    def run(self, eps):
        """central fit and the fits to Gamma_exp,max and Gamma_exp,min (Eq. (eq:exp_range)). errors[i] = (lo, hi)
        of parameter i: distances from the central value to the lowest and highest of the three fitted values"""
        p, c = self.minimize(eps)
        pmax, cmax = self.exp_shifted(+1).minimize(eps)
        pmin, cmin = self.exp_shifted(-1).minimize(eps)
        allp = np.array([p, pmax, pmin])
        errors = [(float(p[i] - allp[:, i].min()), float(allp[:, i].max() - p[i])) for i in (0, 1)]
        return {'p': p, 'chi2': c, 'p_max': pmax, 'chi2_max': cmax, 'p_min': pmin, 'chi2_min': cmin,
                'errors': errors, 'chi2_parts': self.chi2_parts(p, eps)}

    # -------------------------------------------------------------- predictions
    @staticmethod
    def prediction_range(obs, R):
        """(lo, hi) error of the prediction of an Observable: range of its value over the central, max and min
        fits of R = Fit.run(...)"""
        v = [obs.predict(q, 0.0)[0] for q in (R['p'], R['p_max'], R['p_min'])]
        return v[0] - min(v), max(v) - v[0]

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
            e = self.sigma_exp(row, +1 if pr > row['value'] else -1)
            out.append((pr, sth, (row['value'] - pr) / np.hypot(e, sth), e))
        return out
