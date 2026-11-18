"""E1 transition amplitudes and widths with the interpolated LEF of Eq. (eq:gE1_interp) of the notes,

    g_E1(r) = e r [ e_Q + (A_Q r + kappa_Q x)/(1 + x) ],     x = (r/r0)^n,  n = 2,  r0 = sqrt(pi/(6 sigma)),

and the known 1/m corrections, A -> A (1 + delta_1m), Eq. (eq:delta_rel).

Amplitude (photon along +z, helicity h), heavy-quark spin as a spectator:
    M(Mf, Mi, h) = sum_s C(lf S Jf; Mf-s, s) C(li S Ji; Mi-s, s) M_orb(Mf-s, Mi-s, h),
    M_orb(mf, mi, h) = -i E int dr u_f u_i g_E1(r) int dOmega Y_lf^mf* (rhat . eps*_h) Y_li^mi F(E r cos theta),
with F = 1 for the local (L) LEF and F = j0(E r cos theta / 2) for the non-local (NL) one, Eq. (eq:Gamma_E1_NL).
The angular integral is done numerically on the sphere and the radial one as a Gauss quadrature on the mesh.
Width (sum over Mf, h, average over Mi):  Gamma = E/(2 pi) 1/(2 Ji + 1) sum |M|^2  (Eq. (eq:Gamma_hyb_ps)).

The amplitude is linear in (A_Q, kappa_Q): M = M_0 + A_Q M_A + kappa_Q M_kappa, with the radial weights
    w_0 = e e_Q r,   w_A = e r^2/(1 + x),   w_kappa = e r x/(1 + x).
Every width is therefore a quadratic form,
    Gamma = (1 + delta_1m)^2 v^T G v,     v = (1, A_Q, kappa_Q),
with the 3x3 matrix G computed once per transition (width_form). The LO width is G[0, 0] (A_Q = kappa_Q = 0,
delta_1m = 0); the LO + 1/m width is (1 + delta_1m)^2 G[0, 0].
"""
import numpy as np
from .angular import AngularGrid, cg, eps_star

ALPHA_EM = 1.0 / 137.035999
E_CHARGE = np.sqrt(4.0 * np.pi * ALPHA_EM)
CHARGE = {'c': 2.0 / 3.0, 'b': -1.0 / 3.0}
M_QUARK = {'c': 1.496, 'b': 4.88}           # GeV, masses of the QQbarSpectrum wave functions
GEV2KEV = 1e6
SIGMA = 0.21                                 # GeV^2, string tension of the potentials
N_INTERP = 2                                 # exponent n of the interpolation
R0 = np.sqrt(np.pi / (6.0 * SIGMA))          # GeV^-1 (= 1.58 GeV^-1 = 0.31 fm): sigma r^2 = 2 pi |N - 1/12|, N = 0
HBARC = 0.1973269804                         # GeV fm

_GRID = AngularGrid(48, 64)


def photon_energy(Mi, Mf):
    """photon energy in the rest frame of the decaying state (GeV), recoil included"""
    return (Mi ** 2 - Mf ** 2) / (2.0 * Mi)


def interp_x(r):
    """x = (r/r0)^n of Eq. (eq:gE1_interp)"""
    return (np.asarray(r, dtype=float) / R0) ** N_INTERP


def g_E1(r, flavor, A, kappa):
    """the LEF g_E1(r) of Eq. (eq:gE1_interp), in GeV^-1 (A in GeV, kappa in units of e)"""
    x = interp_x(r)
    return E_CHARGE * r * (CHARGE[flavor] + (A * r + kappa * x) / (1.0 + x))


def radial_weights(r, flavor):
    """(w_0, w_A, w_kappa): g_E1 = w_0 + A_Q w_A + kappa_Q w_kappa"""
    x = interp_x(r)
    return (E_CHARGE * CHARGE[flavor] * r, E_CHARGE * r * r / (1.0 + x), E_CHARGE * r * x / (1.0 + x))


def _orbital_profiles(lf, li):
    """{(mf, mi, h): theta profile}: the phi-integrated angular integrand Y_lf^mf* (rhat.eps*_h) Y_li^mi"""
    g = _GRID
    out = {}
    for h in (+1, -1):
        rde = g.rhat @ eps_star(h)
        for mi in range(-li, li + 1):
            for mf in range(-lf, lf + 1):
                out[(mf, mi, h)] = np.sum(g.W * np.conj(g.Y(lf, mf)) * rde * g.Y(li, mi), axis=1)
    return out


def amplitudes(final, initial, E, weight, r, nonloc, absval=False):
    """M[(Mf, Mi, h)] for g_E1(r) = weight(r) (array on the mesh). absval: |u_f u_i| instead of u_f u_i
    (no-cancellation amplitude, used for the theory error up to v10; not used since v11)."""
    g = _GRID
    cc = final.c * initial.c
    if absval:
        cc = np.abs(cc)
    x = E * np.outer(r, g.costh)
    F = np.sinc(x / (2.0 * np.pi)) if nonloc else np.ones_like(x)         # j0(x/2)
    orb = {k: -1j * E * np.sum(cc * weight * (F @ p)) for k, p in _orbital_profiles(final.l, initial.l).items()}
    S = final.S
    if initial.S != S:
        raise ValueError('E1 does not change the heavy-quark spin')
    M = {}
    for Mf in range(-final.J, final.J + 1):
        for Mi in range(-initial.J, initial.J + 1):
            for h in (+1, -1):
                acc = 0j
                for s in range(-S, S + 1):
                    a = cg(final.l, Mf - s, S, s, final.J, Mf)
                    b = cg(initial.l, Mi - s, S, s, initial.J, Mi)
                    if a != 0.0 and b != 0.0:
                        acc += a * b * orb.get((Mf - s, Mi - s, h), 0j)
                M[(Mf, Mi, h)] = acc
    return M


def width_form(final, initial, E, r, nonloc, absval=False):
    """3x3 matrix G (keV): Gamma = v^T G v, v = (1, A_Q, kappa_Q), without the 1/m factor"""
    Ms = [amplitudes(final, initial, E, w, r, nonloc, absval) for w in radial_weights(r, initial.flavor)]
    norm = E / (2.0 * np.pi) / (2 * initial.J + 1) * GEV2KEV
    G = np.zeros((3, 3))
    for i in range(3):
        for j in range(i, 3):
            G[i, j] = G[j, i] = norm * sum((Ms[i][k] * np.conj(Ms[j][k])).real for k in Ms[0])
    return G


def delta_1m(initial, final, E):
    """1/m correction of the E1 amplitude, A -> A (1 + delta), Eq. (eq:delta_rel):
        delta = k/(8 m_Q) ( s [J_P (J_P + 1) - 4] - Delta_l/3 ),   Delta_l = l_i (l_i + 1) - l_f (l_f + 1),
    k = E the photon energy; spin term only for spin-triplet S <-> P transitions, J_P the J of the P wave,
    s = -1 if the P wave is the initial state, +1 if it is the final one (Eichten et al. 1980)."""
    dl = initial.l * (initial.l + 1) - final.l * (final.l + 1)
    sp = 0.0
    if initial.S == 1 and {initial.l, final.l} == {0, 1}:
        JP, s = (initial.J, -1.0) if initial.l == 1 else (final.J, +1.0)
        sp = s * (JP * (JP + 1) - 4)
    return E / (8.0 * M_QUARK[initial.flavor]) * (sp - dl / 3.0)


def theory_sigma(G, eps):
    """theory error of a width G, Eq. (eq:sigma_th) (v11): the amplitude has a relative error eps (standard
    deviation of delta in A -> A (1 + delta)); the variance of Gamma (1 + delta)^2 is 4 eps^2 Gamma^2 + 2 eps^4 Gamma^2.
    The eps^4 term changes sigma_th by a factor (1 + eps^2/2)^(1/2), i.e. by 2.2% for eps = 0.3 and 0.25% for
    eps = 0.1, and is dropped: sigma_th = 2 eps Gamma, with Gamma the model width."""
    return 2.0 * eps * abs(G)
