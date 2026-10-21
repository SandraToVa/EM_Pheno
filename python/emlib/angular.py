"""Angular-momentum tools: Clebsch-Gordan coefficients, spherical harmonics,
spherical basis vectors and the angular quadrature grid.

Conventions (same as the notes EM_transitions.pdf):
  * Condon-Shortley phase for Y_l^m.
  * Spherical unit vectors chi_{+1} = -(1, i, 0)/sqrt2, chi_{-1} = (1, -i, 0)/sqrt2,
    chi_0 = (0, 0, 1)   [Eq. (112)].
  * Photon along z: eps*_h(E z) = -h/sqrt2 (1, -i h, 0) = chi_h^*   [Eq. (128)].
  * C(j1 j2 J; m1 m2) = <j1 m1 j2 m2 | J m1+m2>.
"""
from functools import lru_cache
from math import factorial, sqrt
import numpy as np
from scipy.special import sph_harm_y


@lru_cache(maxsize=None)
def cg(j1, m1, j2, m2, J, M):
    """Clebsch-Gordan <j1 m1 j2 m2|J M> (integer arguments), Racah formula."""
    if m1 + m2 != M:
        return 0.0
    if abs(m1) > j1 or abs(m2) > j2 or abs(M) > J:
        return 0.0
    if J < abs(j1 - j2) or J > j1 + j2:
        return 0.0
    pref = sqrt((2 * J + 1) * factorial(J + j1 - j2) * factorial(J - j1 + j2)
                * factorial(j1 + j2 - J) / factorial(j1 + j2 + J + 1))
    pref *= sqrt(factorial(J + M) * factorial(J - M) * factorial(j1 - m1)
                 * factorial(j1 + m1) * factorial(j2 - m2) * factorial(j2 + m2))
    s = 0.0
    for k in range(0, j1 + j2 - J + 1):
        d = [k, j1 + j2 - J - k, j1 - m1 - k, j2 + m2 - k,
             J - j2 + m1 + k, J - j1 - m2 + k]
        if min(d) < 0:
            continue
        den = 1
        for x in d:
            den *= factorial(x)
        s += (-1) ** k / den
    return pref * s


def sixj_check():  # placeholder to keep API symmetric (6j not needed: we recouple explicitly)
    return None


CHI = {+1: -np.array([1.0, 1.0j, 0.0]) / np.sqrt(2.0),
       -1: np.array([1.0, -1.0j, 0.0]) / np.sqrt(2.0),
       0: np.array([0.0, 0.0, 1.0], dtype=complex)}


def eps_star(h):
    """Complex conjugate polarization vector of a photon moving along +z."""
    return -h / np.sqrt(2.0) * np.array([1.0, -1.0j * h, 0.0])


class AngularGrid:
    """Gauss-Legendre (cos theta) x uniform (phi) product grid on the sphere."""

    def __init__(self, ntheta=48, nphi=64):
        x, w = np.polynomial.legendre.leggauss(ntheta)
        self.costh = x
        th = np.arccos(x)
        ph = np.arange(nphi) * 2.0 * np.pi / nphi
        self.TH, self.PH = np.meshgrid(th, ph, indexing='ij')
        self.W = np.outer(w, np.full(nphi, 2.0 * np.pi / nphi))
        self.COS = np.cos(self.TH)
        self.rhat = np.stack([np.sin(self.TH) * np.cos(self.PH),
                              np.sin(self.TH) * np.sin(self.PH),
                              np.cos(self.TH)], axis=-1)
        self._ycache = {}

    def Y(self, l, m):
        key = (l, m)
        if key not in self._ycache:
            if abs(m) > l:
                self._ycache[key] = np.zeros(self.TH.shape, dtype=complex)
            else:
                self._ycache[key] = sph_harm_y(l, m, self.TH, self.PH)
        return self._ycache[key]

    def vecY(self, l, Jc, M):
        """Vector spherical harmonic sum_mu C(l 1 Jc; M-mu, mu) Y_l^{M-mu} chi_mu
        (angular part of the hybrid wave function, Eq. (112))."""
        out = np.zeros(self.TH.shape + (3,), dtype=complex)
        for mu in (-1, 0, 1):
            c = cg(l, M - mu, 1, mu, Jc, M)
            if c != 0.0:
                out += c * self.Y(l, M - mu)[..., None] * CHI[mu]
        return out

    def integrate(self, f):
        return np.sum(self.W * f)
