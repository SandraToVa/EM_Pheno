"""Fit of the short-distance slope B_Q of the hybrid vertex, Eq. (eq:kappaH), to the lattice hybrid widths.

kappa_Q is fixed to the value of the reference quarkonium fit (results/summary.json of run_all.py); A_Q plays no
role. The fit parameter is B_c (GeV); B_b = -2 B_c (Eq. (eq:flavour_link_H)).

chi^2(B_c) = sum_i (Gamma_lat,i - Gamma_i)^2 / (sigma_lat,i^2 + sigma_th,i^2),
with sigma_lat the (statistical) lattice error and sigma_th the theory error of Eq. (eq:sigma_th),
sigma_th^2 = 4 eps^2 Gamma Gamma_bar + 2 eps^4 Gamma_bar^2, eps = v_c^2 = 0.3, Gamma_bar the width with the
absolute value of the radial integrand (HybridTransition.width_abs). Two ways of using it:
  theory='t0' (reference): sigma_th is evaluated at a fixed reference value B_t0 and kept fixed in the
      minimization; B_t0 is then replaced by the minimum and the procedure iterated to convergence (the "t0"
      prescription of fits with multiplicative errors). sigma_th stays the error of the model width, but chi^2
      cannot be lowered by moving the prediction to where its own error is larger;
  theory='model': sigma_th at the model width at every B, as in the quarkonium fit. With two data points whose
      ratio the model cannot reproduce, chi^2 then decreases by raising both widths (the error grows like the
      prediction, sigma_th -> 2 eps Gamma) and the minimum is biased upwards; it is kept as a check.
eps = 0 (lattice errors only) is also computed.

Gamma(B) is quadratic, so chi^2 has in general two minima, one on each side of the zero of the amplitude
(B_Q of the same or of the opposite sign of kappa_Q, i.e. kappa_H(r) with or without a node). Both are returned.
"""
import numpy as np
from scipy.optimize import minimize_scalar, brentq
from .e1 import photon_energy
from .states import hybrid
from .hybrid import HybridTransition
from .hybrid_data import LATTICE_FIT, HYBRIDS, EPS_H


def lattice_transitions(lib, nonloc, data=LATTICE_FIT, dM=0.0):
    """HybridTransition of each lattice datum at its kinematics (hybrid mass M_i + dM * dM_i)"""
    hy = {h[0]: h for h in HYBRIDS['c']}
    out = []
    for d in data:
        h = hy[d['hybrid']]
        st = hybrid(lib, d['id'], h[1], h[2], h[3], S=d['S'], Jcal=d['Jcal'])
        fin = lib.state(d['final'], 1, 0, d['S'], d['S'], d['M_f'])        # 1S: eta_c (S=0) or J/psi (S=1)
        E = photon_energy(d['M_i'] + dM * d['dM_i'], d['M_f'])
        out.append(HybridTransition(fin, st, E, lib.r, nonloc))
    return out


class HybridFit:
    def __init__(self, lib_c, kappa_c, nonloc=True, eps=EPS_H, data=LATTICE_FIT, dM=0.0, theory='t0'):
        self.kappa, self.eps, self.data, self.theory = kappa_c, eps, data, theory
        self.B_t0 = None                         # reference point of the theory error (theory = 't0')
        self.T = lattice_transitions(lib_c, nonloc, data, dM)

    def model(self, B):
        """[(Gamma, sigma_th)] (keV) of each datum at B_c = B"""
        v = (B, self.kappa)
        out = []
        vt = v if (self.theory == 'model' or self.B_t0 is None) else (self.B_t0, self.kappa)
        e = self.eps
        for T in self.T:
            G, Gt, Gat = T.width(v), T.width(vt), T.width_abs(vt)
            out.append((G, np.sqrt(max(4 * e ** 2 * Gt * Gat + 2 * e ** 4 * Gat ** 2, 0.0))))
        return out

    def chi2_terms(self, B):
        return [(d['value'] - G) ** 2 / (d['err'] ** 2 + s ** 2) for d, (G, s) in zip(self.data, self.model(B))]

    def chi2(self, B):
        return float(sum(self.chi2_terms(B)))

    def run(self, Bmax=1.0, n=2001, maxit=100, tol=1e-7):
        """the minima of chi^2; for theory = 't0' each minimum is iterated with B_t0 -> B_min to convergence,
        starting from the minima of theory = 'model'. Returns the list of minima (sorted by chi^2), each with
        B_t0 = B at convergence, and the errors at fixed B_t0"""
        if self.theory != 't0' or self.eps == 0.0:
            self.B_t0 = None
            return self.minima(Bmax, n)
        self.theory = 'model'
        start = self.minima(Bmax, n)
        self.theory = 't0'
        out = []
        for m in start:
            B = m['B']
            for _ in range(maxit):
                self.B_t0 = B
                lo, hi = B - 0.1, B + 0.1
                res = minimize_scalar(self.chi2, bounds=(lo, hi), method='bounded', options={'xatol': 1e-10})
                if abs(res.x - B) < tol:
                    B = float(res.x)
                    break
                B = float(res.x)
            self.B_t0 = B
            out.append({'B': B, 'chi2': self.chi2(B), 'err': self._errors(B, self.chi2(B), Bmax)})
        self.B_t0 = None
        return sorted(out, key=lambda m: m['chi2'])

    def minima(self, Bmax=1.0, n=2001):
        """local minima of chi^2 on [-Bmax, Bmax] (grid, then bounded Brent), sorted by chi^2:
        [{'B', 'chi2', 'err': (lo, hi)}]"""
        grid = np.linspace(-Bmax, Bmax, n)
        c = np.array([self.chi2(b) for b in grid])
        out = []
        for i in range(1, n - 1):
            if c[i] <= c[i - 1] and c[i] <= c[i + 1]:
                res = minimize_scalar(self.chi2, bounds=(grid[i - 1], grid[i + 1]), method='bounded',
                                      options={'xatol': 1e-8})
                out.append({'B': float(res.x), 'chi2': float(res.fun)})
        for m in out:
            m['err'] = self._errors(m['B'], m['chi2'], Bmax)
        return sorted(out, key=lambda m: m['chi2'])

    def _errors(self, B0, c0, Bmax):
        """Delta chi^2 = 1 interval around the minimum B0 (None on a side where it is not reached before the
        next maximum of chi^2)"""
        f = lambda b: self.chi2(b) - c0 - 1.0
        errs = []
        for sgn in (-1, +1):
            b, prev, step, found = B0, c0, 1e-3, None
            while abs(b) < Bmax:
                b_new = b + sgn * step
                cur = self.chi2(b_new)
                if cur - c0 >= 1.0:
                    found = brentq(f, min(b, b_new), max(b, b_new))
                    break
                if cur < prev:               # passed a maximum: the interval is open on this side
                    break
                b, prev = b_new, cur
            errs.append(None if found is None else abs(found - B0))
        return tuple(errs)
