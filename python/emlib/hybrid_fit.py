"""Fit of the short-distance coefficient B_Q of the hybrid LEFs, Eq. (eq:gH_interp), to the lattice hybrid widths
(v12, local form).

kappa_Q is fixed to the value of the reference quarkonium fit (results/summary.json of run_all.py); A_Q plays no role.
The fit parameter is B_c (dimensionless); B_b = -2 B_c (Eq. (eq:flavour_link_H)).

chi^2(B_c) = sum_i (Gamma_lat,i - Gamma_i)^2 / (sigma_lat,i^2 + sigma_th,i^2),
with sigma_lat the (statistical) lattice error and sigma_th = 2 eps Gamma_i the theory error of Eq. (eq:sigma_th),
evaluated at the model width at the current B_c, exactly as in the quarkonium fit (eps = v_c^2 = 0.3).

Errors (as Eq. (eq:exp_range) for quarkonium): the fit is repeated with both lattice widths moved together to
Gamma_lat + sigma_lat and to Gamma_lat - sigma_lat; the interval of B_c, and of every prediction, is the range spanned
by the central and the two shifted fits.

Gamma(B) is quadratic, so chi^2 has in general two minima, one on each side of the zero of the amplitude; they are
followed separately (branch + / -: sign of B_c), and each shifted fit is taken on the same branch as the central one.
"""
import copy
import numpy as np
from scipy.optimize import minimize_scalar
from .e1 import photon_energy, theory_sigma
from .states import hybrid
from .hybrid import HybridTransition
from .hybrid_data import LATTICE_FIT, HYBRIDS, EPS_H


def lattice_transitions(lib, data=LATTICE_FIT, dM=0.0):
    """HybridTransition (local form) of each lattice datum at its kinematics (hybrid mass M_i + dM * dM_i)"""
    hy = {h[0]: h for h in HYBRIDS['c']}
    out = []
    for d in data:
        h = hy[d['hybrid']]
        st = hybrid(lib, d['id'], h[1], h[2], h[3], S=d['S'], Jcal=d['Jcal'])
        fin = lib.state(d['final'], 1, 0, d['S'], d['S'], d['M_f'])        # 1S: eta_c (S=0) or J/psi (S=1)
        E = photon_energy(d['M_i'] + dM * d['dM_i'], d['M_f'])
        out.append(HybridTransition(fin, st, E, lib.r))
    return out


class HybridFit:
    def __init__(self, lib_c, kappa_c, eps=EPS_H, data=LATTICE_FIT, dM=0.0):
        self.kappa, self.eps, self.data = kappa_c, eps, data
        self.T = lattice_transitions(lib_c, data, dM)
        self.values = np.array([d['value'] for d in data], dtype=float)   # shifted in shifted()
        self.errors = np.array([d['err'] for d in data], dtype=float)

    def model(self, B):
        """[(Gamma, sigma_th)] (keV) of each datum at B_c = B"""
        return [(G, theory_sigma(G, self.eps)) for G in (T.width((B, self.kappa)) for T in self.T)]

    def chi2_terms(self, B):
        return [(v - G) ** 2 / (e ** 2 + s ** 2) for v, e, (G, s) in zip(self.values, self.errors, self.model(B))]

    def chi2(self, B):
        return float(sum(self.chi2_terms(B)))

    def shifted(self, sign):
        """copy of the fit with both lattice widths moved to Gamma_lat + sign sigma_lat"""
        F = copy.copy(self)
        F.values = self.values + sign * self.errors
        return F

    def minima(self, Bmax=5.0, n=4001):
        """local minima of chi^2 on [-Bmax, Bmax] (grid, then bounded Brent), sorted by chi^2: [{'B', 'chi2'}]"""
        grid = np.linspace(-Bmax, Bmax, n)
        c = np.array([self.chi2(b) for b in grid])
        out = []
        for i in range(1, n - 1):
            if c[i] <= c[i - 1] and c[i] <= c[i + 1]:
                res = minimize_scalar(self.chi2, bounds=(grid[i - 1], grid[i + 1]), method='bounded',
                                      options={'xatol': 1e-10})
                out.append({'B': float(res.x), 'chi2': float(res.fun)})
        return sorted(out, key=lambda m: m['chi2'])

    def run(self):
        """for each branch (sign of B_c): central minimum and the minima of the fits to Gamma_lat +- sigma_lat on the
        same branch; errors (lo, hi) = distances from the central B_c to the lowest and highest of the three"""
        cen = self.minima()
        up, dn = self.shifted(+1).minima(), self.shifted(-1).minima()
        out = []
        for m in cen:
            sgn = np.sign(m['B'])
            if any(np.sign(o['B']) == sgn for o in out):
                continue                                   # one minimum per branch (the lowest)
            pick = lambda ms: min((x for x in ms if np.sign(x['B']) == sgn), key=lambda x: abs(x['B'] - m['B']))
            mu, md = pick(up), pick(dn)
            Bs = [m['B'], mu['B'], md['B']]
            out.append({'branch': '+' if sgn > 0 else '-', 'B': m['B'], 'chi2': m['chi2'],
                        'B_max': mu['B'], 'chi2_max': mu['chi2'], 'B_min': md['B'], 'chi2_min': md['chi2'],
                        'err': (m['B'] - min(Bs), max(Bs) - m['B'])})
        return out
