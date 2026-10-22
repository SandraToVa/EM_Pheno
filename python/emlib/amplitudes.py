"""One-photon transition amplitudes and widths in the BOEFT (EM_Pheno v3).

The amplitude is that of Eq. (eq:amp_hybrid_vec) of the notes,
    M = int d^3r  psi_B^dagger  V_h(r) . psi_A ,
for a photon along +z with helicity h. Angular integrals are done by explicit
quadrature on the sphere (no transition-specific angular algebra), so the same
code handles pure, coupled (Pi_u + Sigma_u^-) and mixed (quarkonium + hybrid)
states, local and non-local vertices.

Operators
---------
* 'E1'  quarkonium -> quarkonium (and hybrid -> hybrid, gluonic index
        contracted): V = -i E kappa_Q e e_Q r (rhat . eps*_h) * FF_E1.
        One channel: 'E1'.
* 'HQ'  hybrid -> quarkonium, the 1+- vertex of Eq. (eq:lab_vertex),
            V_h = E { h G1^Pi eps*_perp + h G1^Sigma rhat (rhat.eps*)
                      - i G2 rhat x (Q_h rhat) } .
        The vertex has two channels, treated on the same footing:
          'Pi'    = the transverse operators (g_B_perp, g_E_par, g_E_perp),
                    which only see psi_perp = psi - rhat (rhat.psi)  [Pi_u];
          'Sigma' = the longitudinal operators (g_B_par, g_E^Sigma),
                    which only see rhat.psi                          [Sigma_u^-].
        Each channel has its own string gap Delta_X and its own on-shell zero
        (E_q = Delta_X). Table tab:HQ_lef, with the relative phase of the two
        channels fixed by the r -> 0 limit, g_B_par(0) = g_B_perp(0) for c4 = c2
        (both real, same sign):
            g_B_perp = 2 c2/sqrt(pi sigma),  g_E_par = -2 c3 r/(pi sqrt(pi sigma)),
            g_E_perp = 2 c1 r/(pi sqrt(pi sigma)),
            g_B_par  = 4 c4/(sigma r),       g_E^Sigma = -4 c3/(3 pi sigma).
* 'QH'  quarkonium -> hybrid, the hermitian-conjugate vertex (needed for mixed
        final states): the terms with an explicit i change sign.

Gaps (Params.gap)
-----------------
'string'  (variant A)  Delta_Pi = pi/r,  Delta_Sigma = 3 pi/r  (EST, as matched).
'lattice' (variant B)  Delta_Pi = dV = V_Pi_u - V_Sigma_g,
                       Delta_Sigma = dVS = V_Sigma_u - V_Sigma_g (from QQbarSpectrum).
The same rule is applied to both channels X = Pi, Sigma (n_Pi = 1, n_Sigma = 3):
the channel amplitude is multiplied by s_X = sqrt(Delta_X r/(n_X pi)) and
E r/(n_X pi) -> E/Delta_X. For 'string', s_X = 1.

Non-local form factors (q_z = E cos theta, x = q_z r)
------------------------------------------------------
    FF_E1 = j0(x/2),  f = pi^2 cos(x/2)/(pi^2 - x^2),  F3 = 9pi^2 cos(x/2)/(9pi^2 - x^2).
Pi channel: all three densities ~ cos(pi z/r) -> f.
Sigma channel (Eq. eq:rhoSigma): FT[rho_B_par] = g_B_par f + (c2/(2 sigma r)) (f - F3),
FT[rho_E^Sigma] = g_E^Sigma (9 f - F3)/8. The c2 piece integrates to zero.

Heavy-quark spin is a spectator: for |(Li S) Ji Mi> -> |(Lf S) Jf Mf>
    M = sum_s C(Lf S Jf; Mf-s, s) C(Li S Ji; Mi-s, s) M_orb(Mf-s, Mi-s).
Width (sum over Mf, h, average over Mi, rest frame), Eq. (eq:Gamma_hyb_ps):
    Gamma = E/(2 pi) * 1/(2Ji+1) * sum |M|^2 .
Since M = M_Pi + M_Sigma (+ M_E1 for mixed states), the width splits exactly as
    Gamma = Gamma_PiPi + Gamma_SigmaSigma + Gamma_int,
    Gamma_int = E/(2 pi)/(2Ji+1) * sum 2 Re(M_Pi M_Sigma^*)     (width_split).
"""
import numpy as np
from .angular import AngularGrid, cg, eps_star

ALPHA_EM = 1.0 / 137.035999
E_CHARGE = np.sqrt(4.0 * np.pi * ALPHA_EM)
CHARGE = {'c': 2.0 / 3.0, 'b': -1.0 / 3.0}

HYBRID_CHANNELS = ('Pi', 'Sigma')
N_GAP = {'Pi': 1.0, 'Sigma': 3.0}     # string gaps n_X pi / r

_GRID = None


def grid():
    global _GRID
    if _GRID is None:
        _GRID = AngularGrid(48, 64)
    return _GRID


def ff_E1(x):
    """j0(x/2), x = q_z r"""
    return np.sinc(x / (2.0 * np.pi))


def _ff_pole(x, n):
    """(n pi)^2 cos(x/2)/((n pi)^2 - x^2): normalized FT of cos(n pi z/r);
    regular at |x| = n pi (limit n pi/4 (-1)^((n-1)/2))"""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    a = (n * np.pi) ** 2
    near = np.abs(np.abs(x) - n * np.pi) < 1e-6
    xx = x[~near]
    out[~near] = a * np.cos(xx / 2.0) / (a - xx ** 2)
    out[near] = n * np.pi / 4.0 * (-1) ** ((n - 1) // 2)
    return out


def ff_H(x):
    """f: one-phonon (cos(pi z/r)) form factor"""
    return _ff_pole(x, 1)


def ff_H3(x):
    """F3: three-node (cos(3 pi z/r)) form factor"""
    return _ff_pole(x, 3)


class Params:
    """Couplings and model choices.
    kappa_Q : g_E1 = kappa_Q e e_Q r (quarkonium E1).
    kH      : (c1, c2, c3, c4) of the hybrid vertex in units of e e_Q; a 3-tuple
              sets c4 = c2 (the r -> 0 condition g_B_par(0) = g_B_perp(0)).
    gap     : 'string' (A) or 'lattice' (B); 'lattice' needs dV and dVS on the mesh.
    nonloc  : non-local (string) densities instead of the multipole-expanded LEFs."""

    def __init__(self, flavor, kappa_Q=1.0, kH=(1.0, 1.0, 1.0, 1.0), sigma=0.21,
                 nonloc=False, include_HH=True, include_QH=True, gap='string', dV=None, dVS=None):
        self.flavor = flavor
        self.eeQ = E_CHARGE * CHARGE[flavor]
        self.kappa_Q = kappa_Q
        kH = tuple(kH)
        if len(kH) == 3:
            kH = kH + (kH[1],)
        self.kH = kH
        self.sigma = sigma
        self.nonloc = nonloc
        self.include_HH = include_HH
        self.include_QH = include_QH
        if gap not in ('string', 'lattice'):
            raise ValueError(gap)
        self.gap = gap
        self.dV = None if dV is None else np.asarray(dV)
        self.dVS = None if dVS is None else np.asarray(dVS)
        if gap == 'lattice' and (self.dV is None or self.dVS is None):
            raise ValueError('gap=lattice needs both static gaps dV (Pi_u) and dVS (Sigma_u^-)')

    def gaps(self, r):
        """Delta_Pi(r), Delta_Sigma(r) on the mesh"""
        if self.gap == 'string':
            return {'Pi': np.pi / r, 'Sigma': 3.0 * np.pi / r}
        return {'Pi': self.dV, 'Sigma': self.dVS}


# ----------------------------------------------------------------------
# The hybrid vertex: one list of terms per channel.
#   (key, channel, coupling index, angular structure, sign in the conjugate vertex)
# angular structures (contracted with the hybrid vector psi):
#   'perp'  h eps*_perp . psi
#   'Ta'    i (q.rhat)(rhat x eps*) . psi            [g_E_par operator]
#   'Tb'   -i (rhat.eps*)(rhat x q) . psi            [g_E_perp operator]
#   'long'  h (rhat.eps*)(rhat.psi)
HQ_TERMS = [
    ('B',  'Pi',    1, 'perp', +1),    # g_B_perp      (c2)
    ('Ea', 'Pi',    2, 'Ta',   -1),    # g_E_par       (c3)
    ('Eb', 'Pi',    0, 'Tb',   -1),    # g_E_perp      (c1)
    ('B4', 'Sigma', 3, 'long', +1),    # g_B_par       (c4)
    ('B2', 'Sigma', 1, 'long', +1),    # c2 part of rho_B_par (non-local only)
    ('E3', 'Sigma', 2, 'long', -1),    # g_E^Sigma     (c3)
]


def _radial_weights(E, P, r):
    """Radial weight of each term per unit c (the common prefactor
    2 E e e_Q/sqrt(pi sigma) is applied in orbital_amplitudes), and its
    non-local form factor on the (r, cos theta) grid."""
    g = grid()
    D = P.gaps(r)
    s = {X: np.sqrt(D[X] * r / (N_GAP[X] * np.pi)) for X in HYBRID_CHANNELS}   # = 1 for 'string'
    sq = np.sqrt(np.pi / P.sigma)
    w = {'B': s['Pi'], 'Ea': s['Pi'] * E / D['Pi'], 'Eb': s['Pi'] * E / D['Pi'],
         'B4': 2.0 * sq / r * s['Sigma'], 'B2': 0.25 * sq / r * s['Sigma'],
         'E3': -2.0 * sq / r * s['Sigma'] * E / D['Sigma']}
    x = E * np.outer(r, g.costh)
    if P.nonloc:
        f, F3 = ff_H(x), ff_H3(x)
        FF = {'B': f, 'Ea': f, 'Eb': f, 'B4': f, 'B2': f - F3, 'E3': (9.0 * f - F3) / 8.0}
    else:
        one = np.ones_like(x)
        FF = {'B': one, 'Ea': one, 'Eb': one, 'B4': one, 'B2': None, 'E3': one}
    return w, FF


# ----------------------------------------------------------------------
def _ang_functions(comp, J0):
    """angular functions of a component for each projection of J0 (no heavy-quark
    spin): Y_l^M (quarkonium, J0 = l) or the vector harmonic Y^l_{J M} (hybrid, J0 = J)"""
    g = grid()
    out = {}
    for M in range(-J0, J0 + 1):
        out[M] = g.Y(comp.l, M) if comp.kind == 'Q' else g.vecY(comp.l, J0, M)
    return out


def _structures(h, vec):
    """the four angular structures contracted with a vector field vec[a,b,i]"""
    g = grid()
    rh = g.rhat
    es = eps_star(h)
    z = np.array([0.0, 0.0, 1.0])
    rde = rh @ es
    rdv = np.einsum('abi,abi->ab', rh, vec)
    rxe = np.cross(rh, np.broadcast_to(es, rh.shape))
    rxz = np.cross(rh, np.broadcast_to(z, rh.shape))
    return {'perp': h * (vec @ es - rde * rdv),
            'Ta': 1j * g.COS * np.einsum('abi,abi->ab', rxe, vec),
            'Tb': -1j * rde * np.einsum('abi,abi->ab', rxz, vec),
            'long': h * rde * rdv}


def _orbital_bases(fc, ic, op):
    """dict[(mf, mi, h)] -> list of (term key, theta profile): the amplitude is
    pref * sum_k c_f,k c_i,k sum_terms coupling * w_term(r_k) sum_theta FF(r_k, theta) profile(theta)."""
    g = grid()
    Ff = _ang_functions(fc, fc.J0)
    Fi = _ang_functions(ic, ic.J0)
    out = {}
    for h in (+1, -1):
        for mi, fi in Fi.items():
            for mf, ff in Ff.items():
                if op == 'E1':
                    rde = g.rhat @ eps_star(h)
                    if fc.kind == 'Q':
                        integ = np.conj(ff) * rde * fi
                    else:   # H -> H, gluonic index contracted
                        integ = np.einsum('abi,abi->ab', np.conj(ff), fi) * rde
                    terms = [('r', integ)]
                elif op == 'HQ':       # final scalar, initial vector
                    st = _structures(h, fi)
                    terms = [(k, np.conj(ff) * st[a]) for k, _, _, a, _ in HQ_TERMS]
                elif op == 'QH':       # initial scalar, final vector: conjugate vertex
                    st = _structures(h, np.conj(ff))
                    terms = [(k, sgn * st[a] * fi) for k, _, _, a, sgn in HQ_TERMS]
                else:
                    raise ValueError(op)
                out[(mf, mi, h)] = [(k, np.sum(g.W * t, axis=1)) for k, t in terms]
    return out


def orbital_amplitudes(fc, ic, op, E, P, r, absval=False):
    """Orbital amplitudes of one pair of components, per channel:
    {channel: {(mf, mi, h): M}} with channel 'E1' or 'Pi', 'Sigma'."""
    g = grid()
    bases = _orbital_bases(fc, ic, op)
    cc = fc.c * ic.c
    if absval:
        cc = np.abs(cc)
    if op == 'E1':
        pref = -1j * E * P.kappa_Q * P.eeQ
        x = E * np.outer(r, g.costh)
        FF = {'r': ff_E1(x) if P.nonloc else np.ones_like(x)}
        w, chan, coup = {'r': r}, {'r': 'E1'}, {'r': 1.0}
    else:
        pref = 2.0 * E / np.sqrt(np.pi * P.sigma) * P.eeQ
        w, FF = _radial_weights(E, P, r)
        chan = {k: X for k, X, _, _, _ in HQ_TERMS}
        coup = {k: P.kH[i] for k, _, i, _, _ in HQ_TERMS}
    out = {X: {} for X in dict.fromkeys(chan.values())}
    for key, prof in bases.items():
        acc = {X: 0j for X in out}
        for k, pr in prof:
            if FF[k] is None or coup[k] == 0.0:       # absent in the local limit / zero coupling
                continue
            acc[chan[k]] += coup[k] * np.sum(cc * w[k] * (FF[k] @ pr))
        for X in out:
            out[X][key] = pref * acc[X]
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


def channel_amplitudes(final, initial, E, P, r, absval=False, only=None):
    """Amplitudes per channel between two States: {channel: {(Mf, Mi, h): M}}.
    only: optional set of operators to keep ({'E1'}, {'HQ'}, ...)."""
    Jf, Ji = final.J, initial.J
    keys = [(Mf, Mi, h) for Mf in range(-Jf, Jf + 1) for Mi in range(-Ji, Ji + 1) for h in (1, -1)]
    out = {}
    for fc in final.comps:
        for ic in initial.comps:
            op = _op_for(fc, ic, P)
            if op is None or (only is not None and op not in only):
                continue
            Morb = orbital_amplitudes(fc, ic, op, E, P, r, absval)
            S = fc.S
            for X, MX in Morb.items():
                M = out.setdefault(X, {k: 0j for k in keys})
                for (Mf, Mi, h) in keys:
                    acc = 0j
                    for s in range(-S, S + 1):
                        a = cg(fc.J0, Mf - s, S, s, Jf, Mf)
                        b = cg(ic.J0, Mi - s, S, s, Ji, Mi)
                        if a == 0.0 or b == 0.0:
                            continue
                        acc += a * b * MX.get((Mf - s, Mi - s, h), 0j)
                    M[(Mf, Mi, h)] += acc
    return out


def amplitudes(final, initial, E, P, r, absval=False, only=None):
    """Full amplitude M[(Mf, Mi, h)] = sum over channels."""
    ch = channel_amplitudes(final, initial, E, P, r, absval, only)
    Jf, Ji = final.J, initial.J
    M = {(Mf, Mi, h): 0j for Mf in range(-Jf, Jf + 1) for Mi in range(-Ji, Ji + 1) for h in (1, -1)}
    for MX in ch.values():
        for k, v in MX.items():
            M[k] += v
    return M


CHANNEL_ORDER = ('E1', 'Pi', 'Sigma')


def width_split(final, initial, E, P, r, absval=False, only=None):
    """Partial width (GeV) and its exact channel decomposition:
    {'total', 'Pi', 'Sigma', 'int', ...}: 'X' = Gamma_XX (channel X alone),
    'int' = Gamma_PiSigma = interference of the Pi_u and Sigma_u^- channels;
    for mixed states also 'E1' and the interferences 'E1xPi', 'E1xSigma'."""
    ch = channel_amplitudes(final, initial, E, P, r, absval, only)
    norm = E / (2.0 * np.pi) / (2 * initial.J + 1)
    names = sorted(ch, key=CHANNEL_ORDER.index)
    out = {X: norm * sum(abs(v) ** 2 for v in ch[X].values()) for X in names}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            key = 'int' if (a, b) == ('Pi', 'Sigma') else f'{a}x{b}'
            out[key] = norm * sum(2.0 * (ch[a][k] * np.conj(ch[b][k])).real for k in ch[a])
    for X in HYBRID_CHANNELS:
        out.setdefault(X, 0.0)
    out.setdefault('int', 0.0)
    out['total'] = sum(out.values())
    return out


def width(final, initial, E, P, r, absval=False, only=None):
    """Partial width in GeV, Eq. (eq:Gamma_hyb_ps)."""
    M = amplitudes(final, initial, E, P, r, absval, only)
    return E / (2.0 * np.pi) * sum(abs(v) ** 2 for v in M.values()) / (2 * initial.J + 1)


def photon_energy(Mi, Mf):
    """Photon energy in the rest frame of the decaying state (GeV)."""
    return (Mi ** 2 - Mf ** 2) / (2.0 * Mi)


def theory_sigma(G, Gabs, eps):
    """Theory uncertainty of a width G whose amplitude carries an absolute
    error eps * A_abs (A_abs = same amplitude with |u_f u_i| in the radial
    integrand, G_abs its width): sigma^2 = 4 eps^2 G Gabs + 2 eps^4 Gabs^2."""
    return np.sqrt(4.0 * eps ** 2 * G * Gabs + 2.0 * eps ** 4 * Gabs ** 2)
