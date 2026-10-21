"""Bound states built from the QQbarSpectrum wave functions (exported by
matlab/exportWavefunctions.m).

A State is a superposition of components. Each component is
    kind  : 'Q' (quarkonium, gluonic Sigma_g^+) or 'H' (hybrid, 1+- gluonic vector)
    l     : orbital angular momentum of the heavy pair
    Lc    : angular momentum that couples with the heavy-quark spin S to J
            (Lc = l for 'Q', Lc = J = l (x) 1_gluon for 'H', the J of the S=0 hybrid)
    S     : heavy-quark spin (0 or 1)
    c     : Lagrange-mesh coefficients (length N), u(r_i) = c_i/sqrt(h lambda_i);
            int f u_a u_b dr = sum_i c_a,i c_b,i f(r_i).
The angular state of a component is |(Lc S) J M>, with
    'Q': sum_s C(l S J; M-s, s) Y_l^{M-s} chi^S_s
    'H': sum_s C(Jc S J; M-s, s) [sum_mu C(l 1 Jc; .., mu) Y_l chi_mu] chi^S_s
i.e. the same coupling order (orbital first) as in Eq. (112) of the notes and
as in the QQbarSpectrum potential matrices.
"""
import os
import numpy as np
from scipy.io import loadmat

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

# channel content of the exported families: list of (kind, l, Lc, S)
#   (for mix=false the heavy-quark spin is a spectator; S is set when the
#    state is built)
CHANNELS = {
    'Ql0': [('Q', 0, 0, None)], 'Ql1': [('Q', 1, 1, None)],
    'Ql2': [('Q', 2, 2, None)], 'Ql3': [('Q', 3, 3, None)],
    'Hjc0': [('H', 1, 0, None)],
    'Hjc1': [('H', 1, 1, None), ('H', 2, 1, None), ('H', 0, 1, None)],
    'Hjc2': [('H', 2, 2, None), ('H', 3, 2, None), ('H', 1, 2, None)],
    # mixing families (charm). S fixed by the physics of the 1/m mixing
    'MixJ1': [('Q', 1, 1, 1), ('H', 1, 1, 0), ('Q', 2, 2, 1), ('Q', 0, 0, 1),
              ('H', 2, 1, 0), ('H', 0, 1, 0)],
    'MixJ0': [('Q', 1, 1, 1), ('H', 1, 0, 0)],
    'MixJ2': [('Q', 2, 2, 1), ('H', 2, 2, 0), ('Q', 3, 3, 1), ('Q', 1, 1, 1),
              ('H', 3, 2, 0), ('H', 1, 2, 0)],
    # spin-0 quarkonium mixed with spin-1 hybrids (QQbarS0J0): channel 2 =
    # (l = J+2 = 2) and channel 3 = (l = J = 0) of the J = 1 (s/d) hybrid
    # with heavy-quark spin 1 coupled to J = 0 (same Sigma/Pi structure as
    # GQQbarS0Jcal1, cf. v22, v33, v23 of QQbarS0J0.m)
    'MixS0J0': [('Q', 0, 0, 0), ('H', 2, 1, 1), ('H', 0, 1, 1)],
    'MixS0J1': [('Q', 1, 1, 0)] + [('X', None, None, 1)] * 4,
}


class Component:
    def __init__(self, kind, l, Lc, S, c):
        self.kind, self.l, self.Lc, self.S, self.c = kind, l, Lc, S, np.asarray(c)

    @property
    def weight(self):
        return float(np.sum(self.c ** 2))


class State:
    def __init__(self, name, J, comps, energy=None, mass=None):
        self.name, self.J, self.comps = name, J, comps
        self.energy = energy      # model energy (GeV)
        self.mass = mass          # physical mass used for the photon energy (GeV)

    def weights(self):
        return {(c.kind, c.l, c.Lc, c.S): c.weight for c in self.comps}

    def __repr__(self):
        return f"State({self.name}, J={self.J}, E={self.energy})"


class Library:
    """Wave-function library for one flavor and one mixing setting."""

    def __init__(self, flavor='c', mix=False, data_dir=DATA_DIR):
        fn = os.path.join(data_dir, f"wf_{flavor}_{'mix' if mix else 'nomix'}.mat")
        d = loadmat(fn, squeeze_me=False)
        self.flavor, self.mix = flavor, mix
        self.r = np.asarray(d['r']).ravel()
        self.h = float(np.asarray(d['h']).ravel()[0])
        self.lam = np.asarray(d['lambda']).ravel()
        self.m = float(np.asarray(d['m']).ravel()[0])
        self.fam = {}
        for k in d:
            if k.endswith('_E'):
                f = k[:-2]
                E = np.asarray(d[f + '_E']).ravel()
                C = np.asarray(d[f + '_C'])
                if C.ndim == 2:            # single channel saved as N x nst
                    C = C[None, :, :]
                self.fam[f] = (E, C)

    # ------------------------------------------------------------------
    def family(self, f):
        return self.fam[f]

    def u(self, c):
        """reduced radial function at the mesh points from coefficients"""
        return c / np.sqrt(self.h * self.lam)

    def raw_state(self, f, k):
        """k-th eigenstate (0-based, energy order) of family f: (E, list of comps)"""
        E, C = self.fam[f]
        comps = []
        for ich, (kind, l, Lc, S) in enumerate(CHANNELS[f]):
            if kind == 'X':
                continue
            comps.append(Component(kind, l, Lc, S, C[ich, :, k]))
        return E[k], comps

    def describe(self, f, nmax=None):
        E, C = self.fam[f]
        nmax = len(E) if nmax is None else nmax
        out = []
        for k in range(nmax):
            w = np.sum(C[:, :, k] ** 2, axis=1)
            out.append((k, E[k], w))
        return out

    # ------------------------------------------------------------------
    def quarkonium(self, n, l, S=0, J=None, mass=None, name=None):
        """n-th (1-based) unmixed quarkonium state with orbital l and spin S."""
        if self.mix:
            raise ValueError('use mixed(...) for the mix=true library')
        J = l if (S == 0 and J is None) else J
        E, comps = self.raw_state(f'Ql{l}', n - 1)
        for c in comps:
            c.S = S
        return State(name or f'Q{n}{"SPDF"[l]}', J, comps, E, mass)

    def hybrid(self, Jc, k, S=0, J=None, mass=None, name=None):
        """k-th (0-based, energy order) state of the hybrid family with J = Jc (S=0 angular momentum).
        Notation of the notes: Jc = J (orbital + gluon), and the argument J here is the
        total angular momentum, i.e. Jcal for S=1 hybrids (J = Jc for S=0)."""
        J = Jc if (S == 0 and J is None) else J
        E, comps = self.raw_state(f'Hjc{Jc}', k)
        for c in comps:
            c.S = S
        return State(name or f'H{Jc}_{k}', J, comps, E, mass)

    def hybrid_index(self, Jc, kind, n):
        """Index (0-based) of the n-th (1-based) hybrid of a given kind in family
        Hjc<Jc>.  kind = 'minus' (pure Pi, l = J: p1, d2) or 'plus'
        ((s/d)1, (p/f)2) or 'sigma' (p0)."""
        E, C = self.fam[f'Hjc{Jc}']
        cnt = 0
        for k in range(len(E)):
            w = np.sum(C[:, :, k] ** 2, axis=1)
            if Jc == 0:
                typ = 'sigma'
            else:
                typ = 'minus' if w[0] > 0.5 else 'plus'
            if typ == kind:
                cnt += 1
                if cnt == n:
                    return k
        raise IndexError('state not found')

    def mixed(self, f, k, J, mass=None, name=None):
        E, comps = self.raw_state(f, k)
        return State(name or f'{f}_{k}', J, comps, E, mass)
