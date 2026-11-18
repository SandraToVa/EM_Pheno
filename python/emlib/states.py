"""Quarkonium states built from the QQbarSpectrum wave functions (exported by matlab/exportWavefunctions.m).

A state |n (l S) J M> has one radial function, given by its Lagrange-mesh coefficients c (length N):
    u(r_i) = c_i / sqrt(h lambda_i),     int_0^inf dr f(r) u_a(r) u_b(r) = sum_i c_a,i c_b,i f(r_i),
with u = r R. At LO the heavy-quark spin is a spectator: the radial functions depend on n and l only
(families Ql0..Ql3 of the .mat files, computed with S = 0), and S, J only enter through the angular
coupling (emlib.e1).

Hybrid states (v10). The families Hj0, Hj1, Hj2 of the .mat files (fields Hjc0..Hjc2 in older exports) are the solutions of the
coupled Pi_u--Sigma_u^- equations with J = l (x) 1_gluon = 0, 1, 2 and heavy-quark spin S = 0 (no
quarkonium-hybrid mixing). Channel order of each family (exportWavefunctions.m of v8):
    Hj0: [l = 1]                 (p0, pure Sigma_u^-)
    Hj1: [l = 1, l = 2, l = 0]   (p1: l = J; (s/d)1: l = J -+ 1)
    Hj2: [l = 2, l = 3, l = 1]   (d2: l = J; (p/f)2: l = J -+ 1)
A parity eigenstate has either the l = J component only ('minus': p1, d2) or the l = J -+ 1 ones only ('plus':
(s/d)1, (p/f)2); J = 0 has only l = 1 ('sigma': p0). The vector wave function is
    psi_{nJM}(r) = sum_l R^l(r) Y^l_{JM}(rhat),   Eq. (eq:psi_hybrid_chi),
and, at LO, a hybrid with heavy-quark spin S = 1 has the same radial functions coupled as |(J S) Jcal Mcal>.
"""
import os
import numpy as np
from scipy.io import loadmat

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')


class State:
    """quarkonium state n (l S) J of flavour 'c' or 'b', with radial mesh coefficients c and physical mass (GeV)"""

    def __init__(self, name, flavor, n, l, S, J, c, mass):
        self.name, self.flavor, self.n, self.l, self.S, self.J = name, flavor, n, l, S, J
        self.c, self.mass = np.asarray(c), mass

    def __repr__(self):
        return f"State({self.name}: n={self.n}, l={self.l}, S={self.S}, J={self.J})"


class Library:
    """LO wave functions of one flavour: mesh r (GeV^-1) and the coefficients of the families Ql<l>."""

    def __init__(self, flavor, data_dir=DATA_DIR):
        d = loadmat(os.path.join(data_dir, f'wf_{flavor}_nomix.mat'))
        self.flavor = flavor
        self.r = np.asarray(d['r']).ravel()
        self.h, self.lam = float(np.asarray(d['h']).ravel()[0]), np.asarray(d['lambda']).ravel()
        self._C = {l: np.asarray(d[f'Ql{l}_C'])[0] for l in range(4)}    # N x nstates
        self.energies = {l: np.asarray(d[f'Ql{l}_E']).ravel() for l in range(4)}
        self.hybrids = _load_hybrids(d) if ('Hj1_C' in d or 'Hjc1_C' in d) else {}        # J -> (E, C[channel, mesh, state])

    def u(self, c):
        """reduced radial function u = r R at the mesh points from the mesh coefficients c"""
        return np.asarray(c) / np.sqrt(self.h * self.lam)

    def state(self, name, n, l, S, J, mass):
        """n-th (1-based) radial state of orbital l, with heavy-quark spin S and total angular momentum J"""
        return State(name, self.flavor, n, l, S, J, self._C[l][:, n - 1], mass)


HYBRID_CHANNELS = {0: (1,), 1: (1, 2, 0), 2: (2, 3, 1)}       # orbital l of each channel of family Hj<J>


def s_l(J, l):
    """s_l of Eq. (eq:lab_s): hat r . Y^l_{JM} = s_l Y_J^M"""
    return {J - 1: np.sqrt(J / (2 * J + 1)), J: 0.0, J + 1: -np.sqrt((J + 1) / (2 * J + 1))}[l]


def p_l(J, l):
    """p_l of Eq. (eq:lab_p_values)"""
    return {J - 1: np.sqrt((J + 1) / (2 * J + 1)), J: -1.0, J + 1: np.sqrt(J / (2 * J + 1))}[l]


class HybridState:
    """hybrid n of kind 'plus' ((s/d)1, (p/f)2), 'minus' (p1, d2) or 'sigma' (p0) with J = l (x) 1_gluon,
    heavy-quark spin S and total angular momentum Jcal (Jcal = J for S = 0); comps = [(l, c)] with c the mesh
    coefficients of R^l (sum over l of sum c^2 = 1); mass: the mass used for the photon energy (GeV)"""

    def __init__(self, name, flavor, J, kind, n, comps, energy, S=0, Jcal=None, mass=None):
        self.name, self.flavor, self.J, self.kind, self.n = name, flavor, J, kind, n
        self.comps, self.energy, self.S = comps, energy, S
        self.Jcal = J if Jcal is None else Jcal
        self.mass = mass

    def projections(self):
        """R_Sigma = sum_l s_l R^l (Eq. eq:lab_RSigma) and R_Pi = sum_l p_l R^l (Eq. eq:lab_RPi; the l of one
        parity only are present), as mesh coefficients, and P_Sigma = int r^2 R_Sigma^2, P_Pi = 1 - P_Sigma"""
        cS = sum(s_l(self.J, l) * c for l, c in self.comps)
        cP = sum(p_l(self.J, l) * c for l, c in self.comps)
        return {'R_Sigma': cS, 'R_Pi': cP, 'P_Sigma': float(np.sum(cS ** 2)), 'P_Pi': float(np.sum(cP ** 2))}

    def __repr__(self):
        return f"HybridState({self.name}: J={self.J} {self.kind} n={self.n}, S={self.S}, Jcal={self.Jcal})"


def _load_hybrids(d):
    """families Hj0..Hj2 (named Hjc0..Hjc2 by the exports of v8 and earlier)"""
    out = {}
    for J in (0, 1, 2):
        f = f'Hj{J}' if f'Hj{J}_C' in d else f'Hjc{J}'
        E = np.asarray(d[f + '_E']).ravel()
        C = np.asarray(d[f + '_C'])
        if C.ndim == 2:                                    # single channel saved as N x nstates
            C = C[None, :, :]
        out[J] = (E, C)
    return out


def _kind(J, w):
    """kind of an eigenstate of family Hj<J> from its channel weights w"""
    if J == 0:
        return 'sigma'
    return 'minus' if w[0] > 0.5 else 'plus'


def hybrid_index(E, C, J, kind, n):
    """index (0-based, energy order) of the n-th (1-based) hybrid of a given kind in family Hj<J>"""
    cnt = 0
    for k in range(len(E)):
        if _kind(J, np.sum(C[:, :, k] ** 2, axis=1)) == kind:
            cnt += 1
            if cnt == n:
                return k
    raise IndexError(f'hybrid J={J} {kind} n={n} not found')


def hybrid(lib, name, J, kind, n, S=0, Jcal=None, mass=None):
    """the n-th hybrid of a kind in family Hj<J> of Library lib; only the channels of its parity are kept"""
    E, C = lib.hybrids[J]
    k = hybrid_index(E, C, J, kind, n)
    comps = []
    for ich, l in enumerate(HYBRID_CHANNELS[J]):
        keep = (kind == 'sigma') or (kind == 'minus' and l == J) or (kind == 'plus' and l != J)
        if keep:
            comps.append((l, C[ich, :, k].copy()))
    return HybridState(name, lib.flavor, J, kind, n, comps, float(E[k]), S, Jcal, mass)
