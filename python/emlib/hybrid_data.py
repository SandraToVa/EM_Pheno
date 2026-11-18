"""Inputs of the hybrid analysis: lattice QCD data used in the fit of B_Q, the hybrid multiplets with their
experimental candidates, the final quarkonia, and the results of other approaches used in the comparison.

Masses in GeV, widths in keV.
"""
import numpy as np

# ------------------------------------------------------------------------------------------------------ fit data
# Quenched lattice QCD, Dudek, Edwards, Thomas, PRD 79 (2009) 094504 [arXiv:0902.2241] (Ref. Dudek:2009kk):
#   Table PVtab:  Y_hyb -> eta_c gamma = 42(18) keV  (sink level 5 of T1--, M = 4400(60) MeV, Table vectorstates)
#   Sec. "Exotic transitions": eta_c1 -> J/psi gamma = 115(16) keV (ground state of T1-+, M = 4300(50) MeV)
#   "All errors are purely lattice statistical"; the widths use the lattice hybrid masses and the physical masses
#   of eta_c and J/psi ("where we have experimental masses we use these to compute the phase space").
# In the BOEFT both are the lowest (s/d)1 multiplet: Y = S = 0 (1--), eta_c1 = S = 1, Jcal = 1 (1-+).
LATTICE_FIT = [
    {'id': 'Y->eta_c', 'hybrid': '1(s/d)1', 'S': 0, 'Jcal': 1, 'final': 'eta_c(1S)',
     'M_i': 4.400, 'dM_i': 0.060, 'M_f': 2.98409, 'value': 42.0, 'err': 18.0},
    {'id': 'eta_c1->J/psi', 'hybrid': '1(s/d)1', 'S': 1, 'Jcal': 1, 'final': 'J/psi(1S)',
     'M_i': 4.300, 'dM_i': 0.050, 'M_f': 3.096900, 'value': 115.0, 'err': 16.0},
]
# Same reference, not in the fit: Y_hyb -> chi_c0 gamma, E1(0) = 0.00(3) -> Gamma <~ 20 keV (Table SVfit).
# Delta S = 1 (S = 0 hybrid -> S = 1 quarkonium): zero at our order.
LATTICE_UL = {'id': 'Y->chi_c0', 'M_i': 4.400, 'M_f': 3.41550, 'UL': 20.0}
# Conventional transitions of the same lattice set-up (Tables SVfit, PVtab, text), used only to gauge the
# lattice systematics (Sec. sec:pheno_H_fit): (lattice, error, experiment quoted there, error)
LATTICE_CONVENTIONAL = {
    'chi_c0->J/psi': (199.0, 6.0, 131.0, 14.0), 'chi_c1->J/psi': (270.0, 70.0, 320.0, 25.0),
    'chi_c2->J/psi': (380.0, 50.0, 406.0, 31.0), 'psi(2S)->chi_c0': (26.0, 11.0, 30.0, 2.0),
    'J/psi->eta_c': (2.51, 0.08, 1.85, 0.29),
}

EPS_H = 0.3          # theory error of the hybrid amplitudes, eps = v_c^2 (as for the charmonium observables)

# ------------------------------------------------------------------------------- hybrid multiplets (S = 0 member)
# (label, J = l (x) 1_gluon, kind, n, JPC of S = 0, JPC of S = 1 (Jcal = |J-1|..J+1), candidate, mass, mass source,
#  total width of the candidate (MeV) or None)
# Charm: candidates of the BOEFT analysis of Ref. Oncala:2025mqj (RPP 2026 masses); model masses otherwise.
# Bottom: model masses shifted so that 1(s/d)1 is the Upsilon(10753) (as in v8).
HYBRIDS = {
    'c': [
        ('1(s/d)1', 1, 'plus', 1, '1--', '(0,1,2)-+', 'psi(4040)', 4.040, 'exp', 84.0),
        ('1p1', 1, 'minus', 1, '1++', '(0,1,2)+-', 'chi_c1(4140)', 4.1465, 'exp', 19.0),
        ('1(p/f)2', 2, 'plus', 1, '2++', '(1,2,3)+-', '-', 4.269, 'model', None),
        ('1d2', 2, 'minus', 1, '2--', '(1,2,3)-+', '-', 4.386, 'model', None),
        ('2(s/d)1', 1, 'plus', 2, '1--', '(0,1,2)-+', 'psi(4415)', 4.415, 'exp', 110.0),
        ('1p0', 0, 'sigma', 1, '0++', '1+-', 'chi_c0(4500)', 4.484, 'exp', 87.0),
        ('3(s/d)1', 1, 'plus', 3, '1--', '(0,1,2)-+', 'psi(4660)', 4.623, 'exp', 55.0),
    ],
    'b': [
        ('1(s/d)1', 1, 'plus', 1, '1--', '(0,1,2)-+', 'Upsilon(10753)', 10.7566, 'exp', 29.0),
        ('1p1', 1, 'minus', 1, '1++', '(0,1,2)+-', '-', 10.822, 'model', None),
        ('1(p/f)2', 2, 'plus', 1, '2++', '(1,2,3)+-', '-', 10.872, 'model', None),
        ('2(s/d)1', 1, 'plus', 2, '1--', '(0,1,2)-+', 'Upsilon(10860)', 10.8852, 'exp', 37.0),
        ('1d2', 2, 'minus', 1, '2--', '(1,2,3)-+', '-', 10.928, 'model', None),
        ('3(s/d)1', 1, 'plus', 3, '1--', '(0,1,2)-+', 'Upsilon(11020)', 11.000, 'exp', 24.0),
        ('1p0', 0, 'sigma', 1, '0++', '1+-', '-', 11.022, 'model', None),
    ],
}

# ------------------------------------------------------------------------------------------------ final quarkonia
# S = 0 finals: (label, n, l, mass, source). S = 1 finals: (label, n, l, J, mass, source); for the multiplets
# without established masses the S = 0 (model) mass is used for all J.
FINALS_S0 = {
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
FINALS_S1 = {
    'c': [('J/psi', 1, 0, 1, 3.096900, 'exp'), ('psi(2S)', 2, 0, 1, 3.686097, 'exp'),
          ('chi_c0(1P)', 1, 1, 0, 3.41550, 'exp'), ('chi_c1(1P)', 1, 1, 1, 3.51067, 'exp'),
          ('chi_c2(1P)', 1, 1, 2, 3.55617, 'exp'),
          ('chi_c0(2P)', 2, 1, 0, 3.973, 'model'), ('chi_c1(2P)', 2, 1, 1, 3.973, 'model'),
          ('chi_c2(2P)', 2, 1, 2, 3.973, 'model'),
          ('psi(3770)', 1, 2, 1, 3.7737, 'exp'), ('psi_2(3823)', 1, 2, 2, 3.82351, 'exp'),
          ('psi_3(3842)', 1, 2, 3, 3.84271, 'exp')],
    'b': [('Upsilon(1S)', 1, 0, 1, 9.46040, 'exp'), ('Upsilon(2S)', 2, 0, 1, 10.02326, 'exp'),
          ('Upsilon(3S)', 3, 0, 1, 10.3552, 'exp'),
          ('chi_b0(1P)', 1, 1, 0, 9.85944, 'exp'), ('chi_b1(1P)', 1, 1, 1, 9.89278, 'exp'),
          ('chi_b2(1P)', 1, 1, 2, 9.91221, 'exp'),
          ('chi_b0(2P)', 2, 1, 0, 10.2325, 'exp'), ('chi_b1(2P)', 2, 1, 1, 10.25546, 'exp'),
          ('chi_b2(2P)', 2, 1, 2, 10.26865, 'exp'),
          ('Upsilon_1(1D)', 1, 2, 1, 10.1637, 'est'), ('Upsilon_2(1D)', 1, 2, 2, 10.1637, 'exp'),
          ('Upsilon_3(1D)', 1, 2, 3, 10.1637, 'est')],
}

# --------------------------------------------------------------------------------------- other approaches (keV)
# Jafarzade & Swanson, arXiv:2610.22088, Table tab:physical: Coulomb-gauge constituent-gluon model, at the masses
# M_Y = 4.400, M_eta_c1 = 4.300, M_H1(0-+) = 4.279, M_H1(2-+) = 4.456 GeV and physical final masses.
# (initial S, Jcal, M_i, final label as in FINALS, J&S width)
JS_MODEL = [
    ('Y->eta_c', 0, 1, 4.400, 'eta_c(1S)', 22.0),
    ('Y->chi_c0', 0, 1, 4.400, 'chi_c0(1P)', 10.0),
    ('Y->chi_c1', 0, 1, 4.400, 'chi_c1(1P)', 10.0),
    ('Y->chi_c2', 0, 1, 4.400, 'chi_c2(1P)', 15.0),
    ('H1(0-+)->J/psi', 1, 0, 4.279, 'J/psi', 6.0),
    ('H1(0-+)->h_c', 1, 0, 4.279, 'h_c(1P)', 36.0),
    ('eta_c1->J/psi', 1, 1, 4.300, 'J/psi', 14.0),
    ('eta_c1->h_c', 1, 1, 4.300, 'h_c(1P)', 7.0),
    ('H1(2-+)->J/psi', 1, 2, 4.456, 'J/psi', 41.0),
    ('H1(2-+)->h_c', 1, 2, 4.456, 'h_c(1P)', 9.0),
]
# Flux-tube model, quoted in Dudek:2009kk (footnote, from the DPhil thesis of J. J. Dudek):
#   Gamma(eta_c1 -> J/psi gamma) ~ Gamma(Y_hyb -> eta_c gamma) ~ 30-60 keV
FLUX_TUBE = (30.0, 60.0)
# Coulomb-gauge model of Guo, Yepez-Martinez, Szczepaniak (Guo:2014zva), common mass 4.35 GeV, normalized to the
# quenched lattice Y or eta_c1 amplitude: Gamma(Y -> eta_c gamma) = 39-126, Gamma((0,1,2)-+ -> J/psi gamma) = 32-116
GYS = {'M': 4.35, 'Y->eta_c': (39.0, 126.0), 'S1->J/psi': (32.0, 116.0)}
# eta_c1 in N_f = 2 lattice QCD (Shi:2023sdy): mass and total width from its two-body strong decays
ETAC1_LATTICE = {'M': 4.329, 'dM': 0.036, 'Gamma_tot_MeV': 500.0, 'dGamma_tot_MeV': 180.0}

# ------------------------------------------------------------------------------------- BESIII gamma eta_c limits
HBARC2_PB = 0.3893794e9   # GeV^2 pb


def sigma_bw(sqrt_s, M, Gtot, Bee, Bf):
    """Born cross section (pb) of e+e- -> R -> f for a single J = 1 Breit-Wigner resonance"""
    s = sqrt_s ** 2
    return (12 * np.pi / s) * Bee * Bf * M ** 2 * Gtot ** 2 / ((s - M ** 2) ** 2 + M ** 2 * Gtot ** 2) * HBARC2_PB


def besiii_gamma_etac_limits():
    """Upper limits on Gamma(R -> gamma eta_c) (keV) from the BESIII 90% CL limits on sigma(e+e- -> gamma eta_c)
    [BESIII:2017etac, Table I] at the energy point next to R, assuming the cross section there is saturated by R
    (single Breit-Wigner, no interference); M, Gamma, B(e+e-) of RPP 2026."""
    out = {}
    for name, sqs, ul, M, G, Bee in [('psi(4040)', 4.009, 2.4, 4.040, 0.084, 1.02e-5),
                                     ('psi(4415)', 4.42, 1.6, 4.415, 0.110, 3.2e-6)]:
        unit = sigma_bw(sqs, M, G, Bee, 1.0)
        out[name] = {'sqrt_s': sqs, 'sigma_UL_pb': ul, 'Gamma_UL_keV': ul / unit * G * 1e6}
    return out
