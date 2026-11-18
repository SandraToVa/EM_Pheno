"""Angular-momentum tools: Clebsch-Gordan and 6j coefficients, and the angular quadrature grid.

Conventions (as in the notes):
  * Condon-Shortley phase for Y_l^m.
  * Photon along +z with helicity h: eps*_h = -h/sqrt2 (1, -i h, 0).
  * cg(j1, m1, j2, m2, J, M) = <j1 m1 j2 m2 | J M>  (integer arguments).
"""
from functools import lru_cache
from math import factorial, sqrt
import numpy as np
from scipy.special import sph_harm_y


@lru_cache(maxsize=None)
def cg(j1, m1, j2, m2, J, M):
    """Clebsch-Gordan <j1 m1 j2 m2|J M> (integer arguments), Racah formula."""
    if m1 + m2 != M or abs(m1) > j1 or abs(m2) > j2 or abs(M) > J or not abs(j1 - j2) <= J <= j1 + j2:
        return 0.0
    pref = sqrt((2 * J + 1) * factorial(J + j1 - j2) * factorial(J - j1 + j2)
                * factorial(j1 + j2 - J) / factorial(j1 + j2 + J + 1))
    pref *= sqrt(factorial(J + M) * factorial(J - M) * factorial(j1 - m1)
                 * factorial(j1 + m1) * factorial(j2 - m2) * factorial(j2 + m2))
    s = 0.0
    for k in range(0, j1 + j2 - J + 1):
        d = [k, j1 + j2 - J - k, j1 - m1 - k, j2 + m2 - k, J - j2 + m1 + k, J - j1 - m2 + k]
        if min(d) < 0:
            continue
        den = 1
        for x in d:
            den *= factorial(x)
        s += (-1) ** k / den
    return pref * s


def _tri(a, b, c):
    """triangle coefficient Delta(abc), or 0 if the triangle condition fails"""
    if not abs(a - b) <= c <= a + b:
        return 0.0
    return sqrt(factorial(a + b - c) * factorial(a - b + c) * factorial(-a + b + c) / factorial(a + b + c + 1))


@lru_cache(maxsize=None)
def sixj(j1, j2, j3, j4, j5, j6):
    """Wigner 6j symbol {j1 j2 j3; j4 j5 j6} (integer arguments), Racah formula."""
    t = _tri(j1, j2, j3) * _tri(j1, j5, j6) * _tri(j4, j2, j6) * _tri(j4, j5, j3)
    if t == 0.0:
        return 0.0
    a = (j1 + j2 + j3, j1 + j5 + j6, j4 + j2 + j6, j4 + j5 + j3)
    b = (j1 + j2 + j4 + j5, j2 + j3 + j5 + j6, j3 + j1 + j6 + j4)
    s = 0.0
    for k in range(max(a), min(b) + 1):
        den = 1
        for x in a:
            den *= factorial(k - x)
        for x in b:
            den *= factorial(x - k)
        s += (-1) ** k * factorial(k + 1) / den
    return t * s


def spin_factor(li, Si, Ji, lf, Jf):
    """Angular (spin-algebra) factor of an E1 width, Gamma = (4/3) alpha e_Q^2 E^3 S_if |I|^2:
    S_if = (2 Jf + 1) max(li, lf) {Ji Jf 1; lf li S}^2  (S = 1),   max(li, lf)/(2 li + 1)  (S = 0)."""
    if Si == 0:
        return max(li, lf) / (2 * li + 1)
    return (2 * Jf + 1) * max(li, lf) * sixj(Ji, Jf, 1, lf, li, Si) ** 2


def eps_star(h):
    """complex-conjugate polarization vector of a photon moving along +z"""
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
        self.rhat = np.stack([np.sin(self.TH) * np.cos(self.PH), np.sin(self.TH) * np.sin(self.PH),
                              np.cos(self.TH)], axis=-1)
        self._ycache = {}

    def Y(self, l, m):
        """spherical harmonic Y_l^m on the grid"""
        if (l, m) not in self._ycache:
            self._ycache[(l, m)] = (np.zeros(self.TH.shape, dtype=complex) if abs(m) > l
                                    else sph_harm_y(l, m, self.TH, self.PH))
        return self._ycache[(l, m)]
