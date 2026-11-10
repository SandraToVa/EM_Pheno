"""Quarkonium E1 with the interpolated LEF including a short-distance term (EM_Pheno v8,
paragraphs "Short distances interpolation" and "Flavor independent fit" of the notes):

    g_E1(r) = e r [ e_Q + (A_Q r + kappa_Q x) / (1 + x) ],   x = (r/r0)^n,
            = e e_Q r [ 1 + alpha_Q r (1 - f(r)) + lambda_Q f(r) ],
    alpha_Q = A_Q / e_Q  (GeV),   lambda_Q = kappa_Q / e_Q,   f(r) = x / (1 + x)  (as in interp.py).

At short distances (r << r0) the correction is A_Q r (linear in r); at long distances (r >> r0) it is
the end-point charge kappa_Q. A_Q in GeV (r in GeV^-1), kappa_Q in units of e.
The amplitude is linear in (alpha_Q, lambda_Q): A = A_0 + alpha_Q A_s + lambda_Q A_f, with A_s (A_f)
the dipole amplitude with r -> r * r (1 - f) (r -> r f) in the radial weight. Every width is the
quadratic form
    Gamma = (1 + delta)^2 v^T G v,   v = (1, alpha_Q, lambda_Q),   G 3x3 (spin sums included),
computed once; the |u_f u_i| version gives the theory error as in quarkonium.Fitter. delta: the
relativistic correction of quarkonium.rel_delta (default corr = '1m', as in interp.py).

Two fits (same chi^2, upper limits and theory-error model as quarkonium.Fitter / interp.InterpFitter):
  * mode 'free'   one flavour, parameters (A_Q, kappa_Q);
  * mode 'linked' charm and bottom together, parameters (A_b, kappa_b), with
                  A_c = ratio A_b, kappa_c = ratio kappa_b. Default ratio = e_b/e_c = -1/2:
                  A_Q = A/e_Q and kappa_Q = kappa/e_Q with flavour-independent A, kappa (the
                  constants depend on the heavy quark only through its charge), so that the
                  charmonium constants are -1/2 times the bottomonium ones.
For the epsilon of chi^2/ndf = 1 in the linked fit the two flavours keep the ratio of their
v^3 (eps_Q = s v_Q^3, common s); epsilon can also be a number or a dict {'c': .., 'b': ..}.
Checks: with A_Q = 0 the forms reduce to those of interp.py (tests/test_engine.py, test 7).
"""
import numpy as np
from scipy.optimize import minimize, minimize_scalar, brentq
from .amplitudes import Params, amplitudes, photon_energy, theory_sigma, CHARGE
from .data import MASS, covariance
from .quarkonium import build_state, rel_delta, GEV2KEV, Z90
from .interp import f_hill, R0_EST

V3 = {'c': 0.3 ** 1.5, 'b': 0.1 ** 1.5}
LAM_RANGE = (-0.8, 1.0)      # lambda_Q = kappa_Q/e_Q, as the scan range of interp.InterpFitter
ALPHA_MAX = 2.0               # |A_Q/e_Q| in GeV


def _forms(final, initial, E, flavor, nonloc, profiles, r, absval):
    """3x3 matrix G_ij = norm sum Re(M_i M_j^*) (keV) for the radial profiles (1, r (1 - f), f)"""
    Ms = []
    for prof in profiles:
        P = Params(flavor, kappa_Q=1.0, nonloc=nonloc)
        P.fE1 = prof
        Ms.append(amplitudes(final, initial, E, P, r, absval))
    norm = E / (2.0 * np.pi) / (2 * initial.J + 1) * GEV2KEV
    G = np.zeros((3, 3))
    for i in range(3):
        for j in range(i, 3):
            G[i, j] = G[j, i] = norm * sum((Ms[i][k] * np.conj(Ms[j][k])).real for k in Ms[0])
    return G


def predict_forms(row, libs, nonloc=False, corr='1m', n=2, r0=R0_EST):
    """list (one per width, two for a ratio) of {'G', 'Ga' (3x3, keV, with (1+delta)^2), 'delta'}"""
    lib = libs[row['flavor']]
    fr = f_hill(lib.r, n, r0)
    profiles = (None, lib.r * (1.0 - fr), fr)          # weights r, r * r (1 - f), r * f
    i = build_state(lib, row['init'])
    out = []
    for flab in [row['final']] + ([row['final2']] if 'final2' in row else []):
        f = build_state(lib, flab)
        E = photon_energy(MASS[row['init']], MASS[flab])
        d = rel_delta(row['init'], flab, E, corr)
        fac = (1.0 + d) ** 2
        out.append({'G': fac * _forms(f, i, E, row['flavor'], nonloc, profiles, lib.r, False),
                    'Ga': fac * _forms(f, i, E, row['flavor'], nonloc, profiles, lib.r, True), 'delta': d})
    return out


def qform(G, alpha, lam):
    v = np.array([1.0, alpha, lam])
    return float(v @ G @ v)


class Interp2Fitter:
    """Fit of (A, kappa) in g_E1 = e r [e_Q + (A_Q r + kappa_Q x)/(1 + x)].
    rows: E1 rows of one flavour (mode 'free') or of both (mode 'linked')."""

    def __init__(self, libs, rows, nonloc=False, corr='1m', n=2, r0=R0_EST, mode='free', ratio=CHARGE['b'] / CHARGE['c']):
        self.rows, self.nonloc, self.corr, self.n, self.r0 = rows, nonloc, corr, n, r0
        self.mode, self.ratio = mode, ratio
        self.flavors = sorted(set(r['flavor'] for r in rows))
        if mode == 'free' and len(self.flavors) != 1:
            raise ValueError("mode 'free' needs rows of one flavour")
        self.ref = self.flavors[0] if mode == 'free' else 'b'
        self.forms = [predict_forms(r, libs, nonloc, corr, n, r0) for r in rows]
        self.meas = [k for k, r in enumerate(rows) if r['kind'] in ('width', 'ratio')]
        self.uls = [k for k, r in enumerate(rows) if r['kind'] in ('ul_width', 'ul_ratio')]
        self.Cexp = covariance([rows[k] for k in self.meas])

    # parameters p = (A, kappa) of the reference flavour -> (alpha_Q, lambda_Q) of flavour Q
    def coeffs(self, p, fl):
        A, k = p
        if self.mode == 'linked' and fl != self.ref:
            A, k = self.ratio * A, self.ratio * k
        return A / CHARGE[fl], k / CHARGE[fl]

    def params_of(self, p, fl):
        """(A_Q, kappa_Q) of flavour fl"""
        a, l = self.coeffs(p, fl)
        return a * CHARGE[fl], l * CHARGE[fl]

    def _eps(self, eps, fl):
        return eps[fl] if isinstance(eps, dict) else eps

    def model(self, p, eps):
        out = []
        for row, ps in zip(self.rows, self.forms):
            fl = row['flavor']
            a, l = self.coeffs(p, fl)
            e = self._eps(eps, fl)
            if row['kind'] in ('width', 'ul_width'):
                G, Ga = qform(ps[0]['G'], a, l), qform(ps[0]['Ga'], a, l)
                out.append((G, theory_sigma(G, Ga, e)))
            else:
                g1, a1 = qform(ps[0]['G'], a, l), qform(ps[0]['Ga'], a, l)
                g2, a2 = qform(ps[1]['G'], a, l), qform(ps[1]['Ga'], a, l)
                R = g1 / g2
                s1 = theory_sigma(g1, a1, e) / g1 if g1 > 0 else 0.0
                s2 = theory_sigma(g2, a2, e) / g2
                out.append((R, R * np.hypot(s1, s2)))
        return out

    def inside(self, p):
        """physical region (as the scan range of interp.InterpFitter): for every flavour
        -0.8 <= lambda_Q = kappa_Q/e_Q <= 1 and |alpha_Q| = |A_Q/e_Q| <= 2 GeV. Outside it the
        correction can exceed the LO amplitude and flip its sign (Gamma depends on A^2 only), which
        gives spurious mirror minima."""
        for fl in self.flavors:
            a, l = self.coeffs(p, fl)
            if not (LAM_RANGE[0] <= l <= LAM_RANGE[1] and abs(a) <= ALPHA_MAX):
                return False
        return True

    def chi2(self, p, eps, detail=False):
        if not detail and not self.inside(p):
            return 1e9
        mod = self.model(p, eps)
        d = np.array([self.rows[k]['value'] for k in self.meas])
        pr = np.array([mod[k][0] for k in self.meas])
        sth = np.array([mod[k][1] for k in self.meas])
        C = self.Cexp.copy()
        for a, k in enumerate(self.meas):
            row = self.rows[k]
            if 'err_up' in row:
                e = row['err_up'] if pr[a] > d[a] else row['err_dn']
                C[a, a] = e ** 2
        C += np.diag(sth ** 2)
        res = d - pr
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
            return chi + chi_ul, chi, chi_ul, pr, sth, pulls, ulpulls, mod
        return chi + chi_ul

    def ndf(self):
        return len(self.meas) - 2

    # grid of the reference-flavour parameters: alpha in [-2, 2] GeV, lambda in [-0.8, 1] (A = alpha e, kappa = lambda e)
    def _grid(self):
        e = CHARGE[self.ref]
        return (np.linspace(-ALPHA_MAX, ALPHA_MAX, 81) * e, np.linspace(LAM_RANGE[0], LAM_RANGE[1], 73) * e)

    def fit(self, eps):
        """global minimum (grid, then Nelder-Mead from the lowest grid minima): p = (A, kappa) of the
        reference flavour, chi2_min"""
        As, ks = self._grid()
        C = np.array([[self.chi2((A, k), eps) for k in ks] for A in As])
        # local minima of the grid (the chi2 can have a flat valley with more than one minimum):
        # Nelder-Mead from the 4 lowest ones, keep the best
        starts = []
        for i in range(len(As)):
            for j in range(len(ks)):
                nb = C[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2]
                if C[i, j] <= nb.min():
                    starts.append((C[i, j], As[i], ks[j]))
        best = None
        for _, A0, k0 in sorted(starts)[:4]:
            res = minimize(lambda q: self.chi2(q, eps), x0=[A0, k0], method='Nelder-Mead',
                           options={'xatol': 1e-7, 'fatol': 1e-9, 'maxiter': 4000})
            if best is None or res.fun < best.fun:
                best = res
        return np.array(best.x), float(best.fun)

    def profile_errors(self, p0, c0, eps):
        """Delta chi2 = 1 errors of each parameter, minimizing over the other (profile); and the
        correlation from the numerical Hessian"""
        errs = []
        for i in (0, 1):
            j = 1 - i
            def prof(x):
                def f(y):
                    q = [0.0, 0.0]; q[i] = x; q[j] = y
                    return self.chi2(q, eps)
                y0 = p0[j]
                r = minimize_scalar(f, bracket=(y0 - 0.02, y0 + 0.02))
                return r.fun - c0 - 1.0
            out = []
            for sgn in (-1, +1):
                step, x = 0.01, p0[i]
                val = np.nan
                for _ in range(200):
                    x2 = x + sgn * step
                    if prof(x2) > 0:
                        val = abs(brentq(prof, min(x, x2), max(x, x2), xtol=1e-7) - p0[i])
                        break
                    x, step = x2, step * 1.3
                out.append(val)
            errs.append(tuple(out))           # (lo, hi)
        # correlation from the Hessian of chi2/2
        h = 1e-4
        H = np.zeros((2, 2))
        for a in range(2):
            for b in range(2):
                def cc(da, db):
                    q = np.array(p0, dtype=float); q[a] += da; q[b] += db
                    return self.chi2(q, eps)
                H[a, b] = (cc(h, h) - cc(h, -h) - cc(-h, h) + cc(-h, -h)) / (4 * h * h) / 2.0
        try:
            Cov = np.linalg.inv(H)
            rho = Cov[0, 1] / np.sqrt(Cov[0, 0] * Cov[1, 1])
        except np.linalg.LinAlgError:
            rho = np.nan
        return errs, float(rho)

    def eps_for_unit_chi2(self):
        """'free': one epsilon; 'linked': eps_Q = s v_Q^3 with a common s. Returns eps (number or dict)"""
        def mk(s):
            return s if self.mode == 'free' else {fl: s * V3[fl] for fl in self.flavors}
        g = lambda s: self.fit(mk(s))[1] / self.ndf() - 1.0
        lo, hi = (1e-4, 2.0) if self.mode == 'free' else (1e-3, 20.0)
        if g(lo) <= 0:
            return mk(0.0)
        return mk(brentq(g, lo, hi, xtol=1e-5))
