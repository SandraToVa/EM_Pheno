"""Radiative transitions of the mixed 1-- charmonium states (quarkonium-hybrid
mixing of the QQbarSpectrum code, mix = true, charm only).

The initial 1-- states are eigenstates of GQQbarS0Jcal1 = QQbarS1J1 (mix=true):
spin-1 quarkonium (l = 0, 2) + spin-0 hybrid (s/d)1 (l = 0, 2) [+ the 1++
block]. Transitions:
  * spin-1 quarkonium components -> chi_cJ(nP) (spin 1): E1, g_E1 = kappa_Q e e_Q r
  * spin-0 hybrid components     -> eta_c(nS) (spin 0): hybrid vertex, with its
    Pi_u and Sigma_u^- channels (the hybrid components are solutions of the
    coupled Pi--Sigma equations; the split is returned per channel)
  * spin-1 quarkonium -> spin-1 hybrid component of eta_c: conjugate vertex
    (enters the hindered psi(2S) -> gamma eta_c)
  * hybrid -> hybrid components with the same heavy-quark spin: E1 operator
    g_E1 = kappa_Q e e_Q r with the gluonic index contracted (the photon couples
    to the quark charges; the gluonic state is a spectator).
Final chi_cJ and eta_c are also taken from the mix=true families.
"""
import numpy as np
from .states import Library, State
from .amplitudes import Params, width, width_split, photon_energy
from .hybrids import load_dV, load_dVS

GEV2KEV = 1e6

# initial mixed 1-- states: (index in MixJ1, model content, candidate, mass used)
INITIAL = [
    (2, 'psi(2S)', 'psi(2S)', 3.686097),
    (3, 'psi(1D)', 'psi(3770)', 3.7737),
    (5, '1(s/d)1 hybrid', 'psi(4040)', 4.040),
    (6, 'psi(3S)', 'psi(4160)', 4.191),
    (7, 'psi(2D)', 'psi(4230)', 4.2217),
    (10, '2(s/d)1 hybrid', 'psi(4415)', 4.415),
]
# finals: label -> (family, index, J, mass)
FINAL = {
    'eta_c(1S)': ('MixS0J0', 0, 0, 2.98409), 'eta_c(2S)': ('MixS0J0', 1, 0, 3.6378),
    'chi_c0(1P)': ('MixJ0', 0, 0, 3.41550), 'chi_c1(1P)': ('MixJ1', 1, 1, 3.51067),
    'chi_c2(1P)': ('MixJ2', 0, 2, 3.55617),
    'chi_c0(2P)': ('MixJ0', 1, 0, 3.9221), 'chi_c1(2P)=X(3872)': ('MixJ1', 4, 1, 3.87164),
    'chi_c2(2P)': ('MixJ2', 2, 2, 3.9225),
}


def table(kappa_Q, kappa_H, variants=('A-NL', 'B-NL')):
    lib = Library('c', mix=True)
    dV = load_dV('c'); dVS = load_dVS('c')
    from .hybrids import VARIANTS
    rows = []
    for (idx, content, cand, Mi) in INITIAL:
        st = lib.mixed('MixJ1', idx, 1, mass=Mi, name=cand)
        w = st.weights()
        wQ = sum(v for k, v in w.items() if k[0] == 'Q')
        for fname, (fam, k, J, Mf) in FINAL.items():
            if Mf >= Mi:
                continue
            f = lib.mixed(fam, k, J, mass=Mf, name=fname)
            E = photon_energy(Mi, Mf)
            row = {'state': cand, 'content': content, 'P_Q': wQ, 'final': fname, 'k_MeV': 1e3 * E}
            for v in variants:
                gap, nl = VARIANTS[v]
                P = Params('c', kappa_Q=kappa_Q, kH=(kappa_H,) * 4, nonloc=nl, gap=gap, dV=dV, dVS=dVS)
                sp = width_split(f, st, E, P, lib.r)
                row[v] = sp['total'] * GEV2KEV
                for X in ('E1', 'Pi', 'Sigma', 'int'):
                    row[f'{v}_{X}'] = sp.get(X, 0.0) * GEV2KEV
            rows.append(row)
    return rows


def hindered_psi2S_etac(kappa_H, variants=('A-NL', 'B-NL')):
    """psi(2S) -> gamma eta_c(1S) through the hybrid admixtures: the two paths
    (spin-0 hybrid component of psi(2S) -> eta_c; psi(2S) -> spin-1 hybrid
    component of eta_c) and their coherent sum."""
    lib = Library('c', mix=True)
    dV = load_dV('c'); dVS = load_dVS('c')
    from .hybrids import VARIANTS
    out = []
    for (iname, idx, Mi, fname, kf, Mf) in [('psi(2S)', 2, 3.686097, 'eta_c(1S)', 0, 2.98409),
                                            ('J/psi', 0, 3.096900, 'eta_c(1S)', 0, 2.98409)]:
        st = lib.mixed('MixJ1', idx, 1)
        f = lib.mixed('MixS0J0', kf, 0)
        E = photon_energy(Mi, Mf)
        def sub(s, keep):
            return State(s.name, s.J, [c for c in s.comps if (c.kind, c.S) in keep])
        for v in variants:
            gap, nl = VARIANTS[v]
            P = Params('c', kappa_Q=1.0, kH=(kappa_H,) * 4, nonloc=nl, gap=gap, dV=dV, dVS=dVS)
            g1 = width(sub(f, {('Q', 0)}), sub(st, {('H', 0)}), E, P, lib.r) * GEV2KEV
            g2 = width(sub(f, {('H', 1)}), sub(st, {('Q', 1)}), E, P, lib.r) * GEV2KEV
            sp = width_split(f, st, E, P, lib.r)
            out.append({'transition': f'{iname}->gamma {fname}', 'variant': v, 'k_MeV': 1e3 * E,
                        'path_H_to_Q': g1, 'path_Q_to_H': g2, 'coherent': sp['total'] * GEV2KEV,
                        'coherent_Pi': sp['Pi'] * GEV2KEV, 'coherent_Sigma': sp['Sigma'] * GEV2KEV,
                        'coherent_int': sp['int'] * GEV2KEV})
    return out


# experimental gamma eta_c(1S) widths/limits (keV) used with the scan
# (RPP 2026: B x Gamma_tot; 90% CL limit for psi(3770))
GAMMA_ETAC_EXP = {'J/psi': (1.31, 0.33), 'psi(2S)': (1.05, 0.15), 'psi(3770)': ('UL', 21.8)}


def c4_scan(kappa, ts, variants=('B-NL', 'A-NL')):
    """Gamma(V -> gamma eta_c(1S)) of the mixed 1-- states for c1 = c2 = c3 = kappa e e_c
    and c4 = t kappa e e_c (t = ts). c4 is the only matching coefficient of the
    hybrid vertex that is not shared by the two channels (it fixes g_B_par), so the
    scan locates the value allowed by data."""
    lib = Library('c', mix=True)
    dV = load_dV('c'); dVS = load_dVS('c')
    from .hybrids import VARIANTS
    f = lib.mixed('MixS0J0', 0, 0)
    states = [('J/psi', 0, 3.096900), ('psi(2S)', 2, 3.686097), ('psi(3770)', 3, 3.7737),
              ('psi(4040)', 5, 4.040), ('psi(4415)', 10, 4.415)]
    out = []
    for t in ts:
        for (name, idx, Mi) in states:
            st = lib.mixed('MixJ1', idx, 1)
            E = photon_energy(Mi, 2.98409)
            for v in variants:
                gap, nl = VARIANTS[v]
                P = Params('c', kH=(kappa, kappa, kappa, t * kappa), nonloc=nl, gap=gap, dV=dV, dVS=dVS)
                out.append({'c4_over_c': float(t), 'state': name + ' (mixed)', 'variant': v,
                            'Gamma_keV': width(f, st, E, P, lib.r) * GEV2KEV})
    return out
