"""Quarkonium states built from the QQbarSpectrum wave functions (exported by matlab/exportWavefunctions.m).

A state |n (l S) J M> has one radial function, given by its Lagrange-mesh coefficients c (length N):
    u(r_i) = c_i / sqrt(h lambda_i),     int_0^inf dr f(r) u_a(r) u_b(r) = sum_i c_a,i c_b,i f(r_i),
with u = r R. At LO the heavy-quark spin is a spectator: the radial functions depend on n and l only
(families Ql0..Ql3 of the .mat files, computed with S = 0), and S, J only enter through the angular
coupling (emlib.e1).
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
        self._C = {l: np.asarray(d[f'Ql{l}_C'])[0] for l in range(4)}    # N x nstates

    def state(self, name, n, l, S, J, mass):
        """n-th (1-based) radial state of orbital l, with heavy-quark spin S and total angular momentum J"""
        return State(name, self.flavor, n, l, S, J, self._C[l][:, n - 1], mass)
