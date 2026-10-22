"""Bound states built from the QQbarSpectrum wave functions (exported by
matlab/exportWavefunctions.m).

Notation (as in the notes, App. app:s1):
  quarkonium  l (orbital), S, total J:        |(l S) J M>
  hybrid      l (orbital), J = l (x) 1_gluon, S, total Jcal:  |(J S) Jcal Mcal>
  For S = 0, J = l (quarkonium) and Jcal = J (hybrids).

A State is a superposition of components. Each component is
    kind  : 'Q' (quarkonium, gluonic Sigma_g^+) or 'H' (hybrid, 1+- gluonic vector)
    l     : orbital angular momentum of the heavy pair
    J0    : the angular momentum of the component WITHOUT the heavy-quark spin,
            i.e. the total angular momentum it would have for S = 0; it is what
            couples with S:  J0 = l for 'Q',  J0 = J (= l (x) 1_gluon) for 'H'
    S     : heavy-quark spin (0 or 1)
    c     : Lagrange-mesh coefficients (length N), u(r_i) = c_i/sqrt(h lambda_i);
            int f u_a u_b dr = sum_i c_a,i c_b,i f(r_i).
The angular state of a component is |(J0 S) Jtot Mtot>, with Jtot = State.J the
total angular momentum of the state (J for quarkonium, Jcal for hybrids):
    'Q': sum_s C(l S J; M-s, s) Y_l^{M-s} chi^S_s
    'H': sum_s C(J S Jcal; Mcal-s, s) [sum_mu C(l 1 J; .., mu) Y_l chi_mu] chi^S_s
i.e. the same coupling order (orbital first) as in Eqs. (eq:psi_hybrid_chi),
(eq:S1_Qstate), (eq:S1_Hstate) of the notes and as in the QQbarSpectrum
potential matrices.

Families (names of the exported .mat fields):
  Ql0..Ql3   quarkonium, from QQbarS0J0..QQbarS0J3 (S = 0, J = l)
  Hj0..Hj2   hybrids with J = 0, 1, 2, from GQQbarS0Jcal0..2 (S = 0, so Jcal = J).
             The S = 1 hybrids are built from these by coupling J with S = 1 to
             Jcal (heavy-quark spin symmetry; the GQQbarS1Jcal files, which add
             the hyperfine terms, are not used). Files written by older versions
             of exportWavefunctions.m call them Hjc0..Hjc2; Library renames them.
  Mix*       mixed families (mix = true, charm), see CHANNELS.

A hybrid state of given J is a solution of the coupled Pi_u--Sigma_u^-
equations: it is not labelled as Pi or Sigma. Its Pi_u and Sigma_u^- content is
what the photon vertex sees, the two radial projections of Eqs. (eq:lab_RSigma)
and (eq:lab_RPi), returned by pi_sigma_projections().
"""
import os
import re
import numpy as np
from scipy.io import loadmat

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

# channel content of the exported families: list of (kind, l, J0, S)
#   (for mix=false the heavy-quark spin is a spectator; S is set when the
#    state is built)
CHANNELS = {
    'Ql0': [('Q', 0, 0, None)], 'Ql1': [('Q', 1, 1, None)],
    'Ql2': [('Q', 2, 2, None)], 'Ql3': [('Q', 3, 3, None)],
    'Hj0': [('H', 1, 0, None)],
    'Hj1': [('H', 1, 1, None), ('H', 2, 1, None), ('H', 0, 1, None)],
    'Hj2': [('H', 2, 2, None), ('H', 3, 2, None), ('H', 1, 2, None)],
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
    """One channel of a state: kind 'Q'/'H', orbital l, J0 (angular momentum without
    the heavy-quark spin: l for 'Q', J for 'H'), heavy-quark spin S, mesh coefficients c."""

    def __init__(self, kind, l, J0, S, c):
        self.kind, self.l, self.J0, self.S, self.c = kind, l, J0, S, np.asarray(c)

    @property
    def weight(self):
        return float(np.sum(self.c ** 2))


class State:
    """Superposition of components with total angular momentum J (the J of
    quarkonium, the Jcal of hybrids; the two coincide for S = 0)."""

    def __init__(self, name, J, comps, energy=None, mass=None):
        self.name, self.J, self.comps = name, J, comps
        self.energy = energy      # model energy (GeV)
        self.mass = mass          # physical mass used for the photon energy (GeV)

    def weights(self):
        return {(c.kind, c.l, c.J0, c.S): c.weight for c in self.comps}

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
                if re.fullmatch(r'Hjc\d', f):   # name used by older exports
                    f = 'Hj' + f[3:]
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
        for ich, (kind, l, J0, S) in enumerate(CHANNELS[f]):
            if kind == 'X':
                continue
            comps.append(Component(kind, l, J0, S, C[ich, :, k]))
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
        """n-th (1-based) unmixed quarkonium state with orbital l, heavy-quark spin S
        and total angular momentum J (J = l for S = 0)."""
        if self.mix:
            raise ValueError('use mixed(...) for the mix=true library')
        J = l if (S == 0 and J is None) else J
        E, comps = self.raw_state(f'Ql{l}', n - 1)
        for c in comps:
            c.S = S
        return State(name or f'Q{n}{"SPDF"[l]}', J, comps, E, mass)

    def hybrid(self, J, k, S=0, Jcal=None, mass=None, name=None):
        """k-th (0-based, energy order) state of family Hj<J>, i.e. the hybrid with
        J = l (x) 1_gluon, coupled with heavy-quark spin S to total angular momentum
        Jcal (Jcal = J for S = 0; for S = 1, |J-1| <= Jcal <= J+1), Eq. (eq:S1_Hstate)."""
        if Jcal is None:
            if S != 0:
                raise ValueError('S = 1 hybrid: give Jcal')
            Jcal = J
        E, comps = self.raw_state(f'Hj{J}', k)
        for c in comps:
            c.S = S
        return State(name or f'H{J}_{k}', Jcal, comps, E, mass)

    def hybrid_index(self, J, kind, n):
        """Index (0-based) of the n-th (1-based) hybrid of a given kind in family
        Hj<J>.  kind = 'minus' (l = J: p1, d2; R_Sigma = 0 by s_J = 0),
        'plus' (l = J -+ 1: (s/d)1, (p/f)2; both R_Pi and R_Sigma) or
        'sigma' (J = 0: p0; R_Pi = 0 by p_1 = 0 at J = 0)."""
        E, C = self.fam[f'Hj{J}']
        cnt = 0
        for k in range(len(E)):
            w = np.sum(C[:, :, k] ** 2, axis=1)
            if J == 0:
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


def _s_l(J, l):
    """s_l of Eq. (eq:lab_s)"""
    return {J - 1: np.sqrt(J / (2 * J + 1)), J: 0.0, J + 1: -np.sqrt((J + 1) / (2 * J + 1))}[l]


def _p_l(J, l):
    """p_l of Eq. (eq:lab_p_values)"""
    return {J - 1: np.sqrt((J + 1) / (2 * J + 1)), J: -1.0, J + 1: np.sqrt(J / (2 * J + 1))}[l]


def pi_sigma_projections(state, S=None):
    """Pi_u and Sigma_u^- projections of the hybrid components (kind 'H',
    heavy-quark spin S, all with the same J0 = J) of a state:
        R_Sigma = sum_l s_l R^l                         (Eq. eq:lab_RSigma)
        R_Pi    = sum_{l: (-1)^l = (-1)^(J+1)} p_l R^l   (l = J -+ 1, Eq. eq:lab_RPi)
        R_Pi'   = p_J R^J = -R^J                        (l = J)
    as Lagrange-mesh coefficient vectors, and the weights
    P_Sigma = int R_Sigma^2, P_Pi = int (R_Pi^2 + R_Pi'^2)  (P_Pi + P_Sigma = hybrid norm)."""
    comps = [c for c in state.comps if c.kind == 'H' and (S is None or c.S == S)]
    if not comps:
        return None
    J = comps[0].J0
    N = len(comps[0].c)
    cS, cP, cPJ = np.zeros(N), np.zeros(N), np.zeros(N)
    for c in comps:
        cS += _s_l(J, c.l) * c.c
        if c.l == J:
            cPJ += _p_l(J, c.l) * c.c
        else:
            cP += _p_l(J, c.l) * c.c
    wS = float(np.sum(cS ** 2)); wP = float(np.sum(cP ** 2) + np.sum(cPJ ** 2))
    return {'J': J, 'R_Sigma': cS, 'R_Pi': cP, 'R_Pi_lJ': cPJ, 'P_Sigma': wS, 'P_Pi': wP}
