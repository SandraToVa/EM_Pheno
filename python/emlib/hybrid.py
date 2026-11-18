"""Hybrid -> quarkonium + photon: the 1+- vertex of Eq. (eq:lab_vertex) with the r-dependent string coupling of
Eq. (eq:kappaH) of the notes,

    c_1(r) = c_2(r) = c_3(r) = c_4(r) = e kappa_H(r),     kappa_H(r) = (B_Q r + kappa_Q x)/(1 + x),
    x = (r/r0)^n,  n = 2,  r0 = sqrt(pi/(6 sigma))  (the same n, r0 as g_E1),

with kappa_Q the long-distance charge correction of the quarkonium fit (Eq. (eq:gE1_interp)) and B_Q (GeV) the
short-distance slope of the hybrid vertex, B_c = (e_b/e_c) B_b = -B_b/2 (Eq. (eq:flavour_link_H)).

Vertex (photon along +z, helicity h), with the EST LEFs of Table tab:HQ_lef and the phase convention
|H^0> <-> -i|Sigma_u^->:
    V_h = E { h G1^Pi eps*_perp + h G1^Sigma rhat (rhat.eps*) - i G2 rhat x (Q_h rhat) },
    G1^Pi    = g_B_perp + (E/2)(g_E_par - g_E_perp),   g_B_perp = 2 c2/sqrt(pi sigma),
               g_E_par = -2 c3 r/(pi sqrt(pi sigma)),   g_E_perp = 2 c1 r/(pi sqrt(pi sigma)),
    G1^Sigma = g_B_par + E g_E^Sigma,                  g_B_par = -4 c4/(sigma r),  g_E^Sigma = 4 c3/(3 pi sigma),
    G2       = (E/2)(g_E_par + g_E_perp)               (= 0 for c1 = c3).
The transverse terms ('Pi' channel) only see psi - rhat(rhat.psi), the longitudinal ones ('Sigma' channel) only
rhat.psi. Non-local form (NL): every density rho_X(z, r) of Table tab:HQ_lef is evaluated at R + z rhat, which
multiplies it by its Fourier transform in q_z = E cos theta (Q = q_z r):
    cos(pi z/r)  -> f  = pi^2 cos(Q/2)/(pi^2 - Q^2),   cos(3 pi z/r) -> F3 = 9 pi^2 cos(Q/2)/(9 pi^2 - Q^2)
(normalized to their integrals), so that FT[rho_B_par] = g_B_par f - (c2/(2 sigma r))(f - F3) and
FT[rho_E^Sigma] = g_E^Sigma (9 f - F3)/8; the c2 part of rho_B_par integrates to zero (Eq. (eq:rhoSigma)).

The angular integrals are done numerically on the sphere (no transition-specific algebra), the radial ones as Gauss
quadratures on the mesh; the heavy-quark spin is a spectator,
    M(Mf, Mi, h) = sum_s C(lf S Jf; Mf-s, s) C(J S Jcal; Mi-s, s) M_orb(Mf-s, Mi-s, h).
Width (sum over Mf, h, average over Mi), Eq. (eq:Gamma_hyb_ps):  Gamma = E/(2 pi) 1/(2 Jcal + 1) sum |M|^2.
M = M_Pi + M_Sigma, so Gamma = Gamma_PiPi + Gamma_SigmaSigma + Gamma_int exactly (Eq. (eq:Gamma_split)).

Since the vertex is linear in kappa_H, M = B_Q M_B + kappa_Q M_kappa (radial weights f_B = r/(1+x),
f_kappa = x/(1+x)), and every width is a quadratic form in v = (B_Q, kappa_Q):
    Gamma_XY = v^T H_XY v,   X, Y in {Pi, Sigma}.
"""
import numpy as np
from .angular import AngularGrid, cg, eps_star
from .e1 import E_CHARGE, CHARGE, SIGMA, GEV2KEV, interp_x, photon_energy

CHANNELS = ('Pi', 'Sigma')
_GRID = AngularGrid(48, 64)


def kappa_H(r, B, kappa):
    """kappa_H(r) of Eq. (eq:kappaH): the string coupling c(r) in units of e (B in GeV)"""
    x = interp_x(r)
    return (B * np.asarray(r, dtype=float) + kappa * x) / (1.0 + x)


def basis_weights(r):
    """(f_B, f_kappa): kappa_H = B_Q f_B + kappa_Q f_kappa"""
    x = interp_x(r)
    return np.asarray(r, dtype=float) / (1.0 + x), x / (1.0 + x)


def flavour_B(B_c, fl):
    """B_Q of flavour fl from B_c: B_Q e_Q flavour independent, B_b = (e_c/e_b) B_c = -2 B_c"""
    return B_c * CHARGE['c'] / CHARGE[fl]


# ---------------------------------------------------------------------------------------------- non-local form factors
def _ff_pole(Q, n):
    """(n pi)^2 cos(Q/2)/((n pi)^2 - Q^2): FT of cos(n pi z/r) over [-r/2, r/2] divided by its integral
    (regular at |Q| = n pi, where it is n pi/4 (-1)^((n-1)/2))"""
    Q = np.asarray(Q, dtype=float)
    a = (n * np.pi) ** 2
    near = np.abs(np.abs(Q) - n * np.pi) < 1e-6
    out = np.empty_like(Q)
    out[~near] = a * np.cos(Q[~near] / 2.0) / (a - Q[~near] ** 2)
    out[near] = n * np.pi / 4.0 * (-1) ** ((n - 1) // 2)
    return out


# ------------------------------------------------------------------------------------------------------- the vertex
# (term, channel, coupling index c1..c4 -> 0..3, angular structure)
TERMS = [('B',  'Pi',    1, 'perp'),     # g_B_perp      (c2)
         ('Ea', 'Pi',    2, 'Ta'),       # g_E_par       (c3)
         ('Eb', 'Pi',    0, 'Tb'),       # g_E_perp      (c1)
         ('B4', 'Sigma', 3, 'long'),     # g_B_par       (c4)
         ('B2', 'Sigma', 1, 'long'),     # c2 part of rho_B_par (non-local only)
         ('E3', 'Sigma', 2, 'long')]     # g_E^Sigma     (c3)


def _radial_weights(E, r, nonloc, sigma=SIGMA):
    """radial weight of each term per unit coupling (the common prefactor 2 E e/sqrt(pi sigma) is applied in
    radial_integrands) and its non-local form factor on the (r, cos theta) grid (None: absent in the L form)"""
    sq = np.sqrt(np.pi / sigma)
    w = {'B': np.ones_like(r), 'Ea': E * r / np.pi, 'Eb': E * r / np.pi,
         'B4': -2.0 * sq / r, 'B2': -0.25 * sq / r, 'E3': 2.0 * sq / r * E * r / (3.0 * np.pi)}
    Q = E * np.outer(r, _GRID.costh)
    if nonloc:
        f, F3 = _ff_pole(Q, 1), _ff_pole(Q, 3)
        FF = {'B': f, 'Ea': f, 'Eb': f, 'B4': f, 'B2': f - F3, 'E3': (9.0 * f - F3) / 8.0}
    else:
        one = np.ones_like(Q)
        FF = {'B': one, 'Ea': one, 'Eb': one, 'B4': one, 'B2': None, 'E3': one}
    return w, FF


def _structures(h, vec):
    """angular structures of the vertex contracted with a vector field vec[theta, phi, i]"""
    g = _GRID
    rh = g.rhat
    es = eps_star(h)
    rde = rh @ es
    rdv = np.einsum('abi,abi->ab', rh, vec)
    rxe = np.cross(rh, np.broadcast_to(es, rh.shape))
    rxz = np.cross(rh, np.broadcast_to(np.array([0.0, 0.0, 1.0]), rh.shape))
    return {'perp': h * (vec @ es - rde * rdv),                              # h eps*_perp . psi
            'Ta': 1j * g.COS * np.einsum('abi,abi->ab', rxe, vec),           # i (q.rhat)(rhat x eps*) . psi
            'Tb': -1j * rde * np.einsum('abi,abi->ab', rxz, vec),            # -i (rhat.eps*)(rhat x q) . psi
            'long': h * rde * rdv}                                           # h (rhat.eps*)(rhat.psi)


_PROFILE_CACHE = {}


def _profiles(lf, l, J):
    """{(mf, M, h): {term: theta profile}}: phi-integrated Y_lf^mf* (structure . Y^l_{JM})"""
    key = (lf, l, J)
    if key not in _PROFILE_CACHE:
        g = _GRID
        out = {}
        for M in range(-J, J + 1):
            vec = g.vecY(l, J, M)
            for h in (+1, -1):
                st = _structures(h, vec)
                for mf in range(-lf, lf + 1):
                    yf = np.conj(g.Y(lf, mf))
                    out[(mf, M, h)] = {k: np.sum(g.W * yf * st[a], axis=1) for k, _, _, a in TERMS}
        _PROFILE_CACHE[key] = out
    return _PROFILE_CACHE[key]


def radial_integrands(final, hyb, E, r, couplings, nonloc, sigma=SIGMA):
    """{(Mf, Mi, h): {X: array over the mesh}}: the radial integrand of the amplitude of channel X ('Pi',
    'Sigma'), summed over the hybrid components and recoupled with the heavy-quark spin; the amplitude is its sum
    over the mesh. couplings: (c1, c2, c3, c4) on the mesh, in units of e."""
    w, FF = _radial_weights(E, r, nonloc, sigma)
    pref = 2.0 * E / np.sqrt(np.pi * sigma) * E_CHARGE
    orb = {}                                                     # (mf, M, h) -> {X: integrand}
    for l, c in hyb.comps:
        cc = final.c * c
        for key, prof in _profiles(final.l, l, hyb.J).items():
            acc = orb.setdefault(key, {X: np.zeros(len(r), dtype=complex) for X in CHANNELS})
            for k, X, ic, _ in TERMS:
                if FF[k] is None:
                    continue
                acc[X] += pref * cc * couplings[ic] * w[k] * (FF[k] @ prof[k])
    S = final.S
    if hyb.S != S:
        raise ValueError('the hybrid vertex does not change the heavy-quark spin')
    out = {}
    zero = {X: 0.0 for X in CHANNELS}
    for Mf in range(-final.J, final.J + 1):
        for Mi in range(-hyb.Jcal, hyb.Jcal + 1):
            for h in (+1, -1):
                acc = {X: np.zeros(len(r), dtype=complex) for X in CHANNELS}
                for s in range(-S, S + 1):
                    a = cg(final.l, Mf - s, S, s, final.J, Mf)
                    b = cg(hyb.J, Mi - s, S, s, hyb.Jcal, Mi)
                    if a != 0.0 and b != 0.0:
                        o = orb.get((Mf - s, Mi - s, h), zero)
                        for X in CHANNELS:
                            acc[X] = acc[X] + a * b * o[X]
                out[(Mf, Mi, h)] = acc
    return out


class HybridTransition:
    """hybrid -> final quarkonium + photon at photon energy E (GeV), L or NL form. Stores the radial integrands
    of the two basis couplings (f_B, f_kappa), so that widths at any (B_Q, kappa_Q) are quadratic forms."""

    def __init__(self, final, hyb, E, r, nonloc, sigma=SIGMA):
        self.final, self.hyb, self.E, self.nonloc = final, hyb, E, nonloc
        self.norm = E / (2.0 * np.pi) / (2 * hyb.Jcal + 1) * GEV2KEV          # GeV -> keV
        fB, fk = basis_weights(r)
        ints = [radial_integrands(final, hyb, E, r, (f, f, f, f), nonloc, sigma) for f in (fB, fk)]
        self.keys = list(ints[0])
        # T[key, channel, basis, mesh]
        self.T = np.array([[[ints[a][k][X] for a in range(2)] for X in CHANNELS] for k in self.keys])
        A = self.T.sum(axis=3)                                                   # [key, channel, basis]
        self.H = {}
        for i, X in enumerate(CHANNELS):
            for j, Y in enumerate(CHANNELS):
                self.H[(X, Y)] = self.norm * np.einsum('ka,kb->ab', A[:, i, :], np.conj(A[:, j, :])).real

    def split(self, v):
        """{'Pi', 'Sigma', 'int', 'total'} widths (keV) at v = (B_Q, kappa_Q)"""
        v = np.asarray(v, dtype=float)
        PP = v @ self.H[('Pi', 'Pi')] @ v
        SS = v @ self.H[('Sigma', 'Sigma')] @ v
        I = v @ (self.H[('Pi', 'Sigma')] + self.H[('Sigma', 'Pi')]) @ v
        return {'Pi': PP, 'Sigma': SS, 'int': I, 'total': PP + SS + I}

    def width(self, v):
        return self.split(v)['total']

    def form(self):
        """2x2 matrix H (keV): Gamma = v^T H v"""
        return sum(self.H.values())

    def width_abs(self, v):
        """width with the radial integrand of every amplitude replaced by its absolute value (no cancellation in
        the radial integral, neither from the nodes of the wave functions nor from the zero of the vertex): the
        Gamma_bar of the theory error, Eq. (eq:sigma_th)"""
        v = np.asarray(v, dtype=float)
        integ = np.einsum('kxan,a->kn', self.T, v)                          # [key, mesh], channels summed
        return self.norm * float(np.sum(np.sum(np.abs(integ), axis=1) ** 2))

    def amplitudes(self, v):
        """{(Mf, Mi, h): M} (GeV^-1/2 units of the notes) at v = (B_Q, kappa_Q)"""
        v = np.asarray(v, dtype=float)
        A = np.einsum('kxan,a->k', self.T, v)
        return dict(zip(self.keys, A))


def width_general(final, hyb, E, r, couplings, nonloc, sigma=SIGMA):
    """{'Pi', 'Sigma', 'int', 'total'} (keV) for arbitrary couplings (c1, c2, c3, c4)(r) in units of e
    (used by the tests)"""
    ints = radial_integrands(final, hyb, E, r, couplings, nonloc, sigma)
    norm = E / (2.0 * np.pi) / (2 * hyb.Jcal + 1) * GEV2KEV
    P = sum(abs(np.sum(v['Pi'])) ** 2 for v in ints.values())
    S = sum(abs(np.sum(v['Sigma'])) ** 2 for v in ints.values())
    I = sum(2.0 * (np.sum(v['Pi']) * np.conj(np.sum(v['Sigma']))).real for v in ints.values())
    return {'Pi': norm * P, 'Sigma': norm * S, 'int': norm * I, 'total': norm * (P + S + I)}
