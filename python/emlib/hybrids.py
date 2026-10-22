"""Hybrid -> quarkonium + photon: predictions, variants and data constraints.

The hybrid vertex has two channels on the same footing (amplitudes.py):
  Pi_u    transverse operators g_B_perp (c2), g_E_par (c3), g_E_perp (c1), gap Delta_Pi;
  Sigma_u^- longitudinal operators g_B_par (c4), g_E^Sigma (c3),          gap Delta_Sigma.
Variants of the gaps (Sec. sec:pheno_strategy):
  'A'  Delta_Pi = pi/r,              Delta_Sigma = 3 pi/r                (EST, as matched)
  'B'  Delta_Pi = V_Pi_u - V_Sigma_g, Delta_Sigma = V_Sigma_u - V_Sigma_g (BOEFT potentials)
each in the local (L, multipole-expanded) or non-local (NL, string) form.
Couplings: c_i = kappa_i e e_Q, with c1 = c2 = c3 = c4 (one string coupling)
unless stated otherwise. Every width is returned with its exact split
Gamma = Gamma_PiPi + Gamma_SigmaSigma + Gamma_int (width_split), so the
channel strengths kappa_Pi, kappa_Sigma enter as
    Gamma(kappa_Pi, kappa_Sigma) = kappa_Pi^2 G_Pi + kappa_Sigma^2 G_Sigma + kappa_Pi kappa_Sigma G_int.
"""
import numpy as np
from scipy.io import loadmat
import os
from .states import Library, DATA_DIR
from .amplitudes import Params, width, width_split, photon_energy
from .states import pi_sigma_projections

GEV2KEV = 1e6

# --------------------------------------------------------------- hybrid states
# (label, J (S=0 hybrid angular momentum, l x 1_gluon), kind, n, JPC of the spin-0 member, candidate, mass [GeV], mass
#  source, Gamma_tot [MeV] of the candidate or None)
HYBRIDS = {
    'c': [
        ('1(s/d)1', 1, 'plus', 1, '1--', 'psi(4040)', 4.040, 'exp', 84.0),
        ('1p1', 1, 'minus', 1, '1++', 'chi_c1(4140)', 4.1465, 'exp', 19.0),
        ('1(p/f)2', 2, 'plus', 1, '2++', '-', 4.269, 'model', None),
        ('1d2', 2, 'minus', 1, '2--', '-', 4.386, 'model', None),
        ('2(s/d)1', 1, 'plus', 2, '1--', 'psi(4415)', 4.415, 'exp', 110.0),
        ('1p0', 0, 'sigma', 1, '0++', 'chi_c0(4500)', 4.484, 'exp', 87.0),
        ('3(s/d)1', 1, 'plus', 3, '1--', 'psi(4660)', 4.623, 'exp', 55.0),
    ],
    'b': [
        ('1(s/d)1', 1, 'plus', 1, '1--', 'Upsilon(10753)', 10.7566, 'exp', 29.0),
        ('1p1', 1, 'minus', 1, '1++', '-', 10.822, 'model', None),
        ('1(p/f)2', 2, 'plus', 1, '2++', '-', 10.872, 'model', None),
        ('2(s/d)1', 1, 'plus', 2, '1--', 'Upsilon(10860)', 10.8852, 'exp', 37.0),
        ('1d2', 2, 'minus', 1, '2--', '-', 10.928, 'model', None),
        ('3(s/d)1', 1, 'plus', 3, '1--', 'Upsilon(11020)', 11.000, 'exp', 24.0),
        ('1p0', 0, 'sigma', 1, '0++', '-', 11.022, 'model', None),
    ],
}
# spin-0 final quarkonia: (label, n, l, mass, source)
FINALS = {
    'c': [('eta_c(1S)', 1, 0, 2.98409, 'exp'), ('eta_c(2S)', 2, 0, 3.6378, 'exp'),
          ('eta_c(3S)', 3, 0, 4.159, 'model'), ('h_c(1P)', 1, 1, 3.52537, 'exp'),
          ('h_c(2P)', 2, 1, 3.973, 'model'), ('eta_c2(1D)', 1, 2, 3.782, 'model'),
          ('eta_c2(2D)', 2, 2, 4.219, 'model'), ('h_c3(1F)', 1, 3, 4.044, 'model')],
    'b': [('eta_b(1S)', 1, 0, 9.3987, 'exp'), ('eta_b(2S)', 2, 0, 9.999, 'exp'),
          ('eta_b(3S)', 3, 0, 10.345, 'est'), ('h_b(1P)', 1, 1, 9.8993, 'exp'),
          ('h_b(2P)', 2, 1, 10.2598, 'exp'), ('h_b(3P)', 3, 1, 10.52, 'est'),
          ('eta_b2(1D)', 1, 2, 10.1637, 'est'), ('eta_b2(2D)', 2, 2, 10.469, 'est'),
          ('h_b3(1F)', 1, 3, 10.35, 'est')],
}

VARIANTS = {'A-NL': ('string', True), 'A-L': ('string', False),
            'B-NL': ('lattice', True), 'B-L': ('lattice', False)}


def load_dV(flavor):
    return loadmat(os.path.join(DATA_DIR, f'wf_{flavor}_nomix.mat'))['dV'].ravel()


def load_dVS(flavor):
    """V_Sigma_u^- - V_Sigma_g^+ on the mesh (GeV)"""
    return loadmat(os.path.join(DATA_DIR, f'wf_{flavor}_nomix.mat'))['dVSig'].ravel()


def load_gaps(flavor):
    return load_dV(flavor), load_dVS(flavor)


def hybrid_state(lib, h, S=0, Jcal=None):
    """hybrid multiplet h of HYBRIDS (h[1] = J), with heavy-quark spin S and total Jcal"""
    label, J, kind, n = h[0], h[1], h[2], h[3]
    k = lib.hybrid_index(J, kind, n)
    st = lib.hybrid(J, k, S=S, Jcal=Jcal, mass=h[6], name=label)
    return st


def params(flavor, variant, kH, gaps):
    """gaps: (dV, dVS) from load_gaps"""
    gap, nl = VARIANTS[variant]
    dV, dVS = gaps
    return Params(flavor, kH=kH, nonloc=nl, gap=gap, dV=dV, dVS=dVS)


def gamma_split(flavor, lib, gaps, h, fin, variant, kH, S=0, Jcal=None, Jf=None):
    """{'total', 'Pi', 'Sigma', 'int'} widths in keV of hybrid h -> final (label,n,l,mass) + gamma,
    and the photon energy. S: heavy-quark spin; Jcal: total angular momentum of the
    hybrid; Jf: total angular momentum J' of the final quarkonium (both = J, l' for S = 0)."""
    st = hybrid_state(lib, h, S=S, Jcal=Jcal)
    f = lib.quarkonium(fin[1], fin[2], S=S, J=Jf, mass=fin[3])
    E = photon_energy(h[6], fin[3])
    sp = width_split(f, st, E, params(flavor, variant, kH, gaps), lib.r)
    return {k: v * GEV2KEV for k, v in sp.items()}, E


def gamma(flavor, lib, gaps, h, fin, variant, kH, S=0, Jcal=None, Jf=None):
    """width in keV of hybrid h -> final (label,n,l,mass) + gamma, and the photon energy"""
    sp, E = gamma_split(flavor, lib, gaps, h, fin, variant, kH, S, Jcal, Jf)
    return sp['total'], E


def quadratic_form(flavor, lib, gaps, h, fin, variant):
    """Gamma = c^T Q c with c = (c1, c2, c3, c4)/(e e_Q). Returns Q (4x4, keV)."""
    n = 4
    e = np.eye(n)
    G = {}
    for i in range(n):
        G[(i, i)] = gamma(flavor, lib, gaps, h, fin, variant, tuple(e[i]))[0]
    Q = np.zeros((n, n))
    for i in range(n):
        Q[i, i] = G[(i, i)]
        for j in range(i + 1, n):
            gij = gamma(flavor, lib, gaps, h, fin, variant, tuple(e[i] + e[j]))[0]
            Q[i, j] = Q[j, i] = 0.5 * (gij - G[(i, i)] - G[(j, j)])
    return Q


def channel_bounds(split, UL):
    """Limits from Gamma(kappa_Pi, kappa_Sigma) < UL with the split at kappa = 1:
    common kappa (kappa_Pi = kappa_Sigma), kappa_Pi alone, kappa_Sigma alone."""
    return {'kappa_H_max': np.sqrt(UL / split['total']),
            'kappa_Pi_max': np.sqrt(UL / split['Pi']) if split['Pi'] > 0 else np.inf,
            'kappa_Sigma_max': np.sqrt(UL / split['Sigma']) if split['Sigma'] > 0 else np.inf}


def allowed_channel(h, fin):
    """parity selection rule (-1)^l' = (-1)^l of the radiating components;
    |l' - J| <= 2 (local rank <= 2)"""
    J = h[1]
    if h[2] == 'sigma':
        # p0 (0++, R_Pi = 0): l' = 1 (local, rank 1); l' = 3 only through the non-local rank-3 term
        return fin[2] in (1, 3)
    xi = -1 if h[2] == 'minus' else +1
    return ((-1) ** fin[2] == -xi * (-1) ** J) and abs(fin[2] - J) <= 2


def predictions(flavor, kappa, variants=('A-NL', 'A-L', 'B-NL', 'B-L')):
    lib = Library(flavor)
    gaps = load_gaps(flavor)
    rows = []
    for h in HYBRIDS[flavor]:
        st = hybrid_state(lib, h)
        pr = pi_sigma_projections(st)
        for fin in FINALS[flavor]:
            if fin[3] >= h[6] or not allowed_channel(h, fin):
                continue
            res = {'flavor': flavor, 'hybrid': h[0], 'JPC': h[4], 'candidate': h[5],
                   'M_i': h[6], 'M_i_src': h[7], 'P_Pi': pr['P_Pi'], 'P_Sigma': pr['P_Sigma'],
                   'final': fin[0], 'M_f': fin[3], 'M_f_src': fin[4], 'Gtot_MeV': h[8]}
            kk = (kappa,) * 4
            for v in variants:
                sp, E = gamma_split(flavor, lib, gaps, h, fin, v, kk)
                res[v] = sp['total']
                for X in ('Pi', 'Sigma', 'int'):
                    res[f'{v}_{X}'] = sp[X]
            res['k_MeV'] = E * 1e3
            # magnetic couplings only (g_B_perp and g_B_par: c2 = c4 = kappa, c1 = c3 = 0)
            for v in ('A-NL', 'B-NL'):
                res[f'mag_{v}'] = gamma(flavor, lib, gaps, h, fin, v, (0.0, kappa, 0.0, kappa))[0]
            rows.append(res)
    return rows


# --------------------------------------------------- BESIII gamma eta_c limits
HBARC2_PB = 0.3893794e9   # GeV^2 pb


def sigma_bw(sqrt_s, M, Gtot, Bee, Bf):
    """Born cross section (pb) of e+e- -> R -> f for a single J=1 Breit-Wigner"""
    s = sqrt_s ** 2
    return (12 * np.pi / s) * Bee * Bf * M ** 2 * Gtot ** 2 / ((s - M ** 2) ** 2 + M ** 2 * Gtot ** 2) * HBARC2_PB


def derived_ul_gamma_etac():
    """Upper limits on Gamma(R -> gamma eta_c) (keV) from the BESIII 90% CL
    limits on sigma(e+e- -> gamma eta_c) [PRD 96, 051101 (2017), Table I],
    assuming the cross section at the nearest energy point is saturated by
    the single resonance R (no interference). Uses PDG 2026 B(e+e-), Gamma."""
    out = {}
    # psi(4040): sqrt(s) = 4.009 GeV, UL 2.4 pb; B_ee = (1.02 +- 0.17)e-5, Gamma = 84 +- 12 MeV
    for name, sqs, ul, M, G, Bee in [('psi(4040)', 4.009, 2.4, 4.040, 0.084, 1.02e-5),
                                     ('psi(4415)', 4.42, 1.6, 4.415, 0.110, 3.2e-6)]:
        unit = sigma_bw(sqs, M, G, Bee, 1.0)       # pb per unit branching fraction
        Bul = ul / unit
        out[name] = {'sqrt_s': sqs, 'sigma_UL_pb': ul, 'B_UL': Bul, 'Gamma_UL_keV': Bul * G * 1e6,
                     'sigma_per_B_pb': unit}
    return out
