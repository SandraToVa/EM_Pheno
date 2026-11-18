"""Hybrid -> quarkonium + photon: the 1+- vertex of Eq. (eq:lab_vertex) with interpolated LEFs, Eq. (eq:gH_interp)
of the notes (v12).

Each LEF g_X(r) of Table tab:HQ_lef interpolates between a short-distance behaviour and its EST (long-string) shape,
with the long-distance coupling identified with the quarkonium kappa_Q (c_i -> e kappa_Q), as in Eq. (eq:gE1_interp):

    magnetic, O(p^2):  g_X(r) = e [ B_Q r + kappa_Q ghat_X(r) x ] / (1 + x),      X = B_perp, B_par
    electric, O(p^3):  g_X(r) = e  kappa_Q ghat_X(r) x / (1 + x),                X = E_par, E_perp, E^Sigma
    x = (r/r0)^n,  n = 2,  r0 = sqrt(pi/(6 sigma))  (the same n, r0 as g_E1),

with the EST shapes per unit coupling (Table tab:HQ_lef, c_i = 1)
    ghat_B_perp = 2/sqrt(pi sigma),          ghat_B_par = -4/(sigma r),
    ghat_E_par  = -2 r/(pi sqrt(pi sigma)),  ghat_E_perp = +2 r/(pi sqrt(pi sigma)),  ghat_E^Sigma = 4/(3 pi sigma).
At short distance g_B_perp -> e B_Q r. Since ghat_B_par x = -4 r/(sigma r0^2) is itself O(r), g_B_par ->
e (B_Q - 4 kappa_Q/(sigma r0^2)) r = e (B_Q - (24/pi) kappa_Q) r: the two magnetic LEFs are equal at r -> 0 only for
kappa_Q = 0 (see the notes). The electric LEFs (dimension -2) have no B_Q term and vanish like kappa_Q r^2 ghat_X. B_Q is dimensionless
(e B_Q r has the dimension -1 of g_B), with B_c = (e_b/e_c) B_b = -B_b/2 (Eq. (eq:flavour_link_H)).

Vertex (photon along +z, helicity h), phase convention |H^0> <-> -i|Sigma_u^->:
    V_h = E { h G1^Pi eps*_perp + h G1^Sigma rhat (rhat.eps*) - i G2 rhat x (Q_h rhat) },
    G1^Pi = g_B_perp + (E/2)(g_E_par - g_E_perp),  G1^Sigma = g_B_par + E g_E^Sigma,  G2 = (E/2)(g_E_par + g_E_perp)
(G2 = 0 here, since ghat_E_par = -ghat_E_perp). Only the LOCAL form is implemented in v12: the non-local form needs
an interpolation of the densities rho_X(z, r), which is not defined yet (nonloc=True raises NotImplementedError).

The angular integrals are done numerically on the sphere, the radial ones as Gauss quadratures on the mesh; the
heavy-quark spin is a spectator,
    M(Mf, Mi, h) = sum_s C(lf S Jf; Mf-s, s) C(J S Jcal; Mi-s, s) M_orb(Mf-s, Mi-s, h).
Width (sum over Mf, h, average over Mi), Eq. (eq:Gamma_hyb_ps):  Gamma = E/(2 pi) 1/(2 Jcal + 1) sum |M|^2.
M = M_Pi + M_Sigma, so Gamma = Gamma_PiPi + Gamma_SigmaSigma + Gamma_int exactly (Eq. (eq:Gamma_split)).
Every LEF is linear in (B_Q, kappa_Q), so every width is a quadratic form, Gamma_XY = v^T H_XY v, v = (B_Q, kappa_Q).
"""
import numpy as np
from .angular import AngularGrid, cg, eps_star
from .e1 import E_CHARGE, CHARGE, SIGMA, GEV2KEV, interp_x, photon_energy

CHANNELS = ('Pi', 'Sigma')
LEFS = ('B_perp', 'E_par', 'E_perp', 'B_par', 'E_Sigma')
MAGNETIC = ('B_perp', 'B_par')
_GRID = AngularGrid(48, 64)

# vertex terms: (LEF, channel, angular structure, power p of E, sign s): term = s E^p g_X(r) x structure
TERMS = [('B_perp',  'Pi',    'perp', 1, +1),    #   E h g_B_perp eps*_perp . psi
         ('E_par',   'Pi',    'Ta',   2, -1),    # -i E^2 g_E_par T_a,  'Ta' = i (q.rhat)(rhat x eps*) . psi
         ('E_perp',  'Pi',    'Tb',   2, +1),    # -i E^2 g_E_perp T_b, 'Tb' = -i (rhat.eps*)(rhat x q) . psi
         ('B_par',   'Sigma', 'long', 1, +1),    #   E h g_B_par (rhat.eps*)(rhat.psi)
         ('E_Sigma', 'Sigma', 'long', 2, +1)]    #   E^2 h g_E^Sigma (rhat.eps*)(rhat.psi)


def lef_shapes(r, sigma=SIGMA):
    """EST long-distance shapes ghat_X(r) per unit coupling (Table tab:HQ_lef with c_i = 1), GeV units"""
    r = np.asarray(r, dtype=float)
    a = 2.0 / np.sqrt(np.pi * sigma)
    return {'B_perp': np.full_like(r, a), 'E_par': -a * r / np.pi, 'E_perp': a * r / np.pi,
            'B_par': -4.0 / (sigma * r), 'E_Sigma': np.full_like(r, 4.0 / (3.0 * np.pi * sigma))}


def lef_basis(r, sigma=SIGMA):
    """({X: f_B,X}, {X: f_kappa,X}): g_X = B_Q f_B,X + kappa_Q f_kappa,X, Eq. (eq:gH_interp), including e"""
    r = np.asarray(r, dtype=float)
    x = interp_x(r)
    gh = lef_shapes(r, sigma)
    fB = {X: (E_CHARGE * r / (1.0 + x) if X in MAGNETIC else np.zeros_like(r)) for X in LEFS}
    fk = {X: E_CHARGE * gh[X] * x / (1.0 + x) for X in LEFS}
    return fB, fk


def lefs(r, B, kappa, sigma=SIGMA):
    """the interpolated LEFs g_X(r) of Eq. (eq:gH_interp) (GeV^-1 for B_perp, B_par; GeV^-2 for the electric ones)"""
    fB, fk = lef_basis(r, sigma)
    return {X: B * fB[X] + kappa * fk[X] for X in LEFS}


def est_lefs(r, c, sigma=SIGMA):
    """the pure EST LEFs with a constant coupling c (in units of e): g_X = e c ghat_X (used by the tests)"""
    gh = lef_shapes(r, sigma)
    return {X: E_CHARGE * c * gh[X] for X in LEFS}


def flavour_B(B_c, fl):
    """B_Q of flavour fl from B_c: B_Q e_Q flavour independent, B_b = (e_c/e_b) B_c = -2 B_c"""
    return B_c * CHARGE['c'] / CHARGE[fl]


# ------------------------------------------------------------------------------------------------- angular part
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
    """{(mf, M, h): {structure: integral over the sphere}} of Y_lf^mf* (structure . Y^l_{JM}) (local form: the
    angular integral does not depend on r)"""
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
                    out[(mf, M, h)] = {a: complex(np.sum(g.W * yf * st[a])) for a in ('perp', 'Ta', 'Tb', 'long')}
        _PROFILE_CACHE[key] = out
    return _PROFILE_CACHE[key]


def radial_integrands(final, hyb, E, r, g, nonloc=False):
    """{(Mf, Mi, h): {X: array over the mesh}}: radial integrand of the amplitude of channel X ('Pi', 'Sigma'),
    summed over the hybrid components and recoupled with the heavy-quark spin; the amplitude is its sum over the
    mesh. g: {LEF: array on the mesh} (e.g. lefs(r, B, kappa))."""
    if nonloc:
        raise NotImplementedError('non-local form with the interpolated LEFs of Eq. (eq:gH_interp): not defined yet')
    orb = {}                                                     # (mf, M, h) -> {X: integrand}
    for l, c in hyb.comps:
        cc = final.c * c
        for key, prof in _profiles(final.l, l, hyb.J).items():
            acc = orb.setdefault(key, {X: np.zeros(len(r), dtype=complex) for X in CHANNELS})
            for X, ch, a, p, s in TERMS:
                if prof[a] != 0.0:
                    acc[ch] += s * E ** p * prof[a] * cc * g[X]
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


def width_lefs(final, hyb, E, r, g):
    """{'Pi', 'Sigma', 'int', 'total'} (keV) for arbitrary LEFs g = {LEF: array on the mesh} (tests, checks)"""
    ints = radial_integrands(final, hyb, E, r, g)
    norm = E / (2.0 * np.pi) / (2 * hyb.Jcal + 1) * GEV2KEV
    P = sum(abs(np.sum(v['Pi'])) ** 2 for v in ints.values())
    S = sum(abs(np.sum(v['Sigma'])) ** 2 for v in ints.values())
    I = sum(2.0 * (np.sum(v['Pi']) * np.conj(np.sum(v['Sigma']))).real for v in ints.values())
    return {'Pi': norm * P, 'Sigma': norm * S, 'int': norm * I, 'total': norm * (P + S + I)}


class HybridTransition:
    """hybrid -> final quarkonium + photon at photon energy E (GeV), local form. Stores the amplitudes of the two
    basis LEF sets (f_B, f_kappa of lef_basis), so that widths at any v = (B_Q, kappa_Q) are quadratic forms."""

    def __init__(self, final, hyb, E, r, nonloc=False, sigma=SIGMA):
        if nonloc:
            raise NotImplementedError('non-local form with the interpolated LEFs: not defined yet')
        self.final, self.hyb, self.E = final, hyb, E
        self.norm = E / (2.0 * np.pi) / (2 * hyb.Jcal + 1) * GEV2KEV          # GeV -> keV
        fB, fk = lef_basis(r, sigma)
        ints = [radial_integrands(final, hyb, E, r, f) for f in (fB, fk)]
        self.keys = list(ints[0])
        A = np.array([[[np.sum(ints[a][k][X]) for a in range(2)] for X in CHANNELS] for k in self.keys])
        self.A = A                                                              # [key, channel, basis]
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

    def amplitudes(self, v):
        """{(Mf, Mi, h): M} at v = (B_Q, kappa_Q)"""
        v = np.asarray(v, dtype=float)
        return dict(zip(self.keys, np.einsum('kxa,a->k', self.A, v)))
