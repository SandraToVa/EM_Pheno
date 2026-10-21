"""Generic one-photon transition amplitudes and widths in the BOEFT.

Everything is computed by explicit angular quadrature (no transition-specific
angular integrals): the amplitude of Eq. (111) of the notes,
    M = int d^3r  psi_B^dagger  O_h(r)  psi_A ,
with the photon along +z and helicity h, for
  * 'E1'  quarkonium -> quarkonium (and hybrid -> hybrid, gluon index
          contracted) with g_E1(r) = c r              [Eqs. (74), (95)]
          O_h = -i E c r (rhat . eps*_h) * FF_E1
  * 'HQ'  hybrid -> quarkonium with the 1+- vertex of Eq. (110)
          O_h = 2E/sqrt(pi sigma) [ c2 h (eps* - rhat (rhat.eps*))
                + i c3 (E r/pi) (z.rhat)(rhat x eps*) - i c1 (E r/pi)(rhat.eps*)(rhat x z) ] * FF_H
  * 'QH'  quarkonium -> hybrid (hermitian conjugate vertex; only relevant for
          mixed final states)
FF = 1 in the local (BOEFT, multipole-expanded) limit. In the non-local
(long-distance, string) version the LEF densities rho(z,r) of Eqs. (39), (50),
(61), (66) give
          FF_E1 = j0(q_z r/2),   FF_H = pi^2 cos(q_z r/2)/(pi^2 - q_z^2 r^2),
q_z = E cos(theta) (theta = angle between rhat and the photon).

Heavy-quark spin is a spectator: for initial |(Li S) Ji Mi> and final
|(Lf S) Jf Mf>
    M = sum_s C(Lf S Jf; Mf-s, s) C(Li S Ji; Mi-s, s) M_orb(Mf-s, Mi-s).
The width (sum over Mf, h, average over Mi, rest frame) is Eq. (139):
    Gamma = E/(2 pi) * 1/(2Ji+1) * sum |M|^2 .
"""
import numpy as np
from .angular import AngularGrid, cg, eps_star

ALPHA_EM = 1.0 / 137.035999
E_CHARGE = np.sqrt(4.0 * np.pi * ALPHA_EM)
CHARGE = {'c': 2.0 / 3.0, 'b': -1.0 / 3.0}

_GRID = None


def grid():
    global _GRID
    if _GRID is None:
        _GRID = AngularGrid(48, 64)
    return _GRID


def ff_E1(x):
    """j0(x/2), x = q_z r"""
    return np.sinc(x / (2.0 * np.pi))


def ff_H(x):
    """pi^2 cos(x/2)/(pi^2 - x^2), regular at |x| = pi (limit pi/4)"""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    near = np.abs(np.abs(x) - np.pi) < 1e-6
    xx = x[~near]
    out[~near] = np.pi ** 2 * np.cos(xx / 2.0) / (np.pi ** 2 - xx ** 2)
    out[near] = np.pi / 4.0
    return out


class Params:
    """Couplings. kappa_Q multiplies e e_Q in g_E1 (c1 of quarkonium);
    kH = (c1, c2, c3) of the hybrid vertex in units of e e_Q."""

    def __init__(self, flavor, kappa_Q=1.0, kH=(1.0, 1.0, 1.0), sigma=0.21,
                 nonloc=False, include_HH=True, include_QH=True, gap='string', dV=None):
        self.flavor = flavor
        self.eeQ = E_CHARGE * CHARGE[flavor]
        self.kappa_Q = kappa_Q
        self.kH = tuple(kH)
        self.sigma = sigma
        self.nonloc = nonloc
        self.include_HH = include_HH
        self.include_QH = include_QH
        # gap entering the hybrid vertex: 'string' -> E_1 = pi/r (notes, Eq. (33));
        # 'lattice' -> E_1 = V_Pi(r) - V_Sigma(r) of the BOEFT potentials (dV on the mesh)
        self.gap = gap
        self.dV = None if dV is None else np.asarray(dV)
        if gap == 'lattice' and dV is None:
            raise ValueError('gap=lattice needs dV')


# ----------------------------------------------------------------------
# angular base integrands
def _ang_functions(comp, Lc):
    g = grid()
    out = {}
    for M in range(-Lc, Lc + 1):
        if comp.kind == 'Q':
            out[M] = g.Y(comp.l, M)
        else:
            out[M] = g.vecY(comp.l, Lc, M)
    return out


def _orbital_bases(fc, ic, op, E, P):
    """Returns dict[(mf, mi, h)] -> list of (power p, theta-profile array):
    the amplitude is  pref * sum_k c_f c_i sum_p r_k^p sum_theta prof(theta) FF(E r_k cos theta)."""
    g = grid()
    Ff = _ang_functions(fc, fc.Lc)
    Fi = _ang_functions(ic, ic.Lc)
    rh = g.rhat
    z = np.array([0.0, 0.0, 1.0])
    rxz = np.cross(rh, np.broadcast_to(z, rh.shape))
    cos = g.COS
    c1, c2, c3 = P.kH
    out = {}
    for h in (+1, -1):
        es = eps_star(h)
        rde = rh @ es
        rxe = np.cross(rh, np.broadcast_to(es, rh.shape))
        for mi, fi in Fi.items():
            for mf, ff in Ff.items():
                if op == 'E1':
                    if fc.kind == 'Q':
                        integ = np.conj(ff) * rde * fi
                    else:   # H -> H, gluon index contracted
                        integ = np.einsum('abi,abi->ab', np.conj(ff), fi) * rde
                    terms = [('r', integ)]
                elif op == 'HQ':
                    # final scalar, initial vector
                    edotF = fi @ es
                    rdotF = np.einsum('abi,abi->ab', rh, fi)
                    TB = c2 * h * (edotF - rde * rdotF)
                    TE = (1j * c3 * cos * np.einsum('abi,abi->ab', rxe, fi)
                          - 1j * c1 * rde * np.einsum('abi,abi->ab', rxz, fi))
                    yb = np.conj(ff)
                    terms = [('B', yb * TB), ('E', yb * TE)]
                elif op == 'QH':
                    # initial scalar, final vector (conjugated); conjugate vertex
                    ffc = np.conj(ff)
                    edotF = ffc @ es
                    rdotF = np.einsum('abi,abi->ab', rh, ffc)
                    TB = c2 * h * (edotF - rde * rdotF)
                    TE = (-1j * c3 * cos * np.einsum('abi,abi->ab', rxe, ffc)
                          + 1j * c1 * rde * np.einsum('abi,abi->ab', rxz, ffc))
                    terms = [('B', TB * fi), ('E', TE * fi)]
                else:
                    raise ValueError(op)
                prof = [(p, np.sum(g.W * t, axis=1)) for p, t in terms]
                out[(mf, mi, h)] = prof
    return out


def orbital_amplitudes(fc, ic, op, E, P, r, absval=False):
    """M_orb[(mf, mi, h)] for one pair of components (radial sum included)."""
    g = grid()
    bases = _orbital_bases(fc, ic, op, E, P)
    cc = fc.c * ic.c
    if absval:
        cc = np.abs(cc)
    x = E * np.outer(r, g.costh)             # q_z r on (r_k, theta_j)
    if op == 'E1':
        pref = -1j * E * P.kappa_Q * P.eeQ
        FF = ff_E1(x) if P.nonloc else np.ones_like(x)
    else:
        pref = 2.0 * E / np.sqrt(np.pi * P.sigma) * P.eeQ
        FF = ff_H(x) if P.nonloc else np.ones_like(x)
    # radial weights of each term type
    if P.gap == 'string':
        wts = {'r': r, 'B': np.ones_like(r), 'E': E * r / np.pi}
    else:   # E_1(r) = dV(r): amplitude x sqrt(E_1 r/pi), (E r/pi) -> E/E_1
        s = np.sqrt(P.dV * r / np.pi)
        wts = {'r': r, 'B': s, 'E': s * E / P.dV}
    out = {}
    for key, prof in bases.items():
        val = 0j
        for p, pr in prof:
            K = FF @ pr                          # (N,) angular integral at each r_k
            val += np.sum(cc * wts[p] * K)
        out[key] = pref * val
    return out


def _op_for(fc, ic, P):
    if fc.S != ic.S:
        return None
    if fc.kind == 'Q' and ic.kind == 'Q':
        return 'E1'
    if fc.kind == 'Q' and ic.kind == 'H':
        return 'HQ'
    if fc.kind == 'H' and ic.kind == 'H':
        return 'E1' if P.include_HH else None
    if fc.kind == 'H' and ic.kind == 'Q':
        return 'QH' if P.include_QH else None
    return None


def amplitudes(final, initial, E, P, r, absval=False, only=None):
    """Full amplitudes M[(Mf, Mi, h)] between two States.
    only: optional set of ops to keep (e.g. {'E1'} or {'HQ'})."""
    Jf, Ji = final.J, initial.J
    M = {(Mf, Mi, h): 0j for Mf in range(-Jf, Jf + 1)
         for Mi in range(-Ji, Ji + 1) for h in (1, -1)}
    for fc in final.comps:
        for ic in initial.comps:
            op = _op_for(fc, ic, P)
            if op is None or (only is not None and op not in only):
                continue
            Morb = orbital_amplitudes(fc, ic, op, E, P, r, absval)
            S = fc.S
            for (Mf, Mi, h) in M:
                acc = 0j
                for s in range(-S, S + 1):
                    a = cg(fc.Lc, Mf - s, S, s, Jf, Mf)
                    b = cg(ic.Lc, Mi - s, S, s, Ji, Mi)
                    if a == 0.0 or b == 0.0:
                        continue
                    acc += a * b * Morb.get((Mf - s, Mi - s, h), 0j)
                M[(Mf, Mi, h)] += acc
    return M


def width(final, initial, E, P, r, absval=False, only=None):
    """Partial width in GeV (Eq. (139) of the notes)."""
    M = amplitudes(final, initial, E, P, r, absval, only)
    s = sum(abs(v) ** 2 for v in M.values())
    return E / (2.0 * np.pi) * s / (2 * initial.J + 1)


def photon_energy(Mi, Mf):
    """Photon energy in the rest frame of the decaying state (GeV)."""
    return (Mi ** 2 - Mf ** 2) / (2.0 * Mi)


def theory_sigma(G, Gabs, eps):
    """Theory uncertainty of a width G whose amplitude carries an absolute
    error eps * A_abs (A_abs = same amplitude with |u_f u_i| in the radial
    integrand, G_abs its width): sigma^2 = 4 eps^2 G Gabs + 2 eps^4 Gabs^2."""
    return np.sqrt(4.0 * eps ** 2 * G * Gabs + 2.0 * eps ** 4 * Gabs ** 2)
