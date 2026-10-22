"""Comparison with the lattice QCD results of Delaney, Thomas, Ryan (HSC),
arXiv:2301.08213 [N_f = 2+1, m_pi ~ 391 MeV, a_s ~ 0.12 fm, xi ~ 3.5,
a_t^-1 = 5.666 GeV, single lattice spacing, statistical errors only].

The paper has NO hybrid states; the quantities that test our framework are
  * charge radii <r^2_ch>^(1/2) of eta_c, J/psi, eta_c', chi_c0
    (<r^2_ch> = -6 dF/dQ^2 at Q^2 = 0). For an equal-mass Q Qbar pair with the
    quarks at +-r/2, F(Q^2) = <j0(|q| r/2)> and <r^2_ch> = <r^2>/4, so
    <r^2_ch>^(1/2) = r_rms/2 with r_rms the rms Q Qbar distance of our LO
    BOEFT wave functions (spin-independent: eta_c and J/psi share 1S).
  * Gamma(chi_c0 -> J/psi gamma) = 208.6(3) keV with experimental masses
    (247.4(2) keV with lattice masses), from E1(0) of the Exp2 fit
    E1(Q^2) = F0 (1 + alpha Q^2) exp(-Q^2/(16 beta^2)), beta^2 = 0.304(33) GeV^2.
  * beta^2 is the harmonic-oscillator parameter of the quark-model form
    (psi ~ exp(-beta^2 r^2/2)): for SHO wave functions <r^2>_1S = 3/(2 beta^2),
    <r^2>_1P = 5/(2 beta^2). We quote the beta^2 implied by our 1S and 1P sizes.
"""
import numpy as np

HBARC = 0.1973269804   # GeV fm

LATTICE = {
    'arXiv': '2301.08213',
    'radii_fm': {'eta_c(1S)': (0.243, 0.007), 'J/psi': (0.249, 0.004),
                 'eta_c(2S)': (0.62, 0.18), 'chi_c0(1P)': (0.339, 0.018)},
    'Gamma_chic0_Jpsi_keV': {'exp_masses': (208.6, 0.3), 'lat_masses': (247.4, 0.2)},
    'other_lattice_range_keV': (180.0, 288.0),   # Dudek 2006/2009 (quenched), Li 2021 (N_f=2)
    'chen2011_range_keV': (60.0, 90.0),
    'E1_exp2': {'F0_lat': (0.13882, 0.00077), 'beta2_GeV2': (0.304, 0.033), 'alpha_GeV-2': (0.903, 0.031),
                'chi2_ndf': 8.97},
    'a_t_inv_GeV': 5.666,
}
# (our state label, (n, l)) for each lattice state
STATE_MAP = {'eta_c(1S)': (1, 0), 'J/psi': (1, 0), 'eta_c(2S)': (2, 0), 'chi_c0(1P)': (1, 1)}


def rms(lib, n, l):
    st = lib.quarkonium(n, l)
    w = sum(c.c ** 2 for c in st.comps)
    return np.sqrt(np.sum(w * lib.r ** 2) / np.sum(w))


def radii_table(lib):
    rows = []
    for lab, (n, l) in STATE_MAP.items():
        rr = rms(lib, n, l)
        lat, err = LATTICE['radii_fm'][lab]
        ours = 0.5 * rr * HBARC
        rows.append({'state': lab, 'nl': f'{n}{"SPDF"[l]}', 'r_rms_fm': rr * HBARC, 'r_ch_ours_fm': ours,
                     'r_ch_lat_fm': lat, 'err_lat_fm': err, 'ratio': ours / lat, 'pull': (ours - lat) / err})
    return rows


def beta2_from_sizes(lib):
    r1s, r1p = rms(lib, 1, 0), rms(lib, 1, 1)
    return {'beta2_from_1S': 1.5 / r1s ** 2, 'beta2_from_1P': 2.5 / r1p ** 2,
            'beta2_lattice': LATTICE['E1_exp2']['beta2_GeV2']}


def chic0_comparison(G_lo_local, G_lo_nonlocal, kappa_c, dk, G_exp, G_exp_err):
    """Gamma(chi_c0 -> J/psi gamma) [keV]: LO (kappa = 1), fitted kappa_c, lattice, PDG."""
    glat = LATTICE['Gamma_chic0_Jpsi_keV']['exp_masses'][0]
    return {'Gamma_LO_local_keV': G_lo_local, 'Gamma_LO_nonlocal_keV': G_lo_nonlocal,
            'Gamma_fit_keV': kappa_c ** 2 * G_lo_nonlocal, 'dGamma_fit_keV': 2 * kappa_c * dk * G_lo_nonlocal,
            'Gamma_lattice_keV': glat, 'Gamma_lattice_latmasses_keV': LATTICE['Gamma_chic0_Jpsi_keV']['lat_masses'][0],
            'Gamma_PDG_keV': G_exp, 'dGamma_PDG_keV': G_exp_err,
            'kappa_equiv_lattice': np.sqrt(glat / G_lo_nonlocal),
            'kappa_equiv_PDG': np.sqrt(G_exp / G_lo_nonlocal),
            'lattice_over_PDG': glat / G_exp}


# ---------------------------------------------------------------------------
# Hybrid radiative transitions on the lattice.
#  * quenched: Dudek, Edwards, Thomas, PRD 79, 094504 (2009), as quoted in
#    Jafarzade & Swanson, arXiv:2610.22088, with their kinematics
#    (M_Y = 4.400, M_eta_c1 = 4.300 GeV, physical final masses).
#  * N_f = 2+1: J. Delaney, PhD thesis, Cambridge (2023), doi:10.17863/CAM.103271
#    (same ensemble as arXiv:2301.08213: m_pi ~ 391 MeV, a_t^-1 = 5.666 GeV).
#    Table 4.3: a_t m = 0.52331 (eta_c), 0.53752 (J/psi), 0.758 (psi_hyb),
#    0.754 (eta_c1) -> 2.965, 3.046, 4.295, 4.272 GeV. Widths use lattice masses
#    (checked: Eq. (3.32) with Tables 6.11, 6.12 and these masses gives 104.6 keV
#    for eta_c1 -> J/psi gamma, quoted 105(8) keV, Eq. (6.17)).
#    psi_hyb -> eta_c gamma: quoted 59(6) keV, Eq. (6.13). Eq. (6.6) with the
#    tabulated F(0) = 0.2532 (Table 6.8) and lattice masses gives 29.9 keV, which is
#    the value 30.3(1.8) keV used by Jafarzade & Swanson (we scale the quoted 10%
#    error: 29.9(3.0) keV); the same factor ~2
#    appears for psi' -> eta_c gamma (0.245 vs quoted 0.485(60) keV), so it is a
#    normalization issue internal to the thesis for V -> P. We keep both.
# Statistical errors only; Ref. 2610.22088 estimates an extra ~30% systematic.
GEV_AT = 5.666
LAT_M = {k: v * GEV_AT for k, v in {'eta_c': 0.52331, 'J/psi': 0.53752, 'psi_hyb': 0.758,
                                     'eta_c1': 0.754, "psi'": 0.6483, "psi''": 0.6740}.items()}
# (channel, dataset, S, Jcal, M_i, final (label, n, l, M_f), Gamma, dGamma)
LATTICE_HYBRID = [
    ('Y->eta_c', 'quenched', 0, None, 4.400, ('eta_c(1S)', 1, 0, 2.98409), 42.0, 18.0),
    ('eta_c1->J/psi', 'quenched', 1, 1, 4.300, ('J/psi', 1, 0, 3.096900), 115.0, 16.0),
    ('Y->eta_c', 'Nf3', 0, None, LAT_M['psi_hyb'], ('eta_c(1S)', 1, 0, LAT_M['eta_c']), 59.0, 6.0),
    ('Y->eta_c', 'Nf3-F0', 0, None, LAT_M['psi_hyb'], ('eta_c(1S)', 1, 0, LAT_M['eta_c']), 29.9, 3.0),
    ('eta_c1->J/psi', 'Nf3', 1, 1, LAT_M['eta_c1'], ('J/psi', 1, 0, LAT_M['J/psi']), 105.0, 8.0),
]
# model of Jafarzade & Swanson (keV), their kinematics
JS_MODEL = {'Y->eta_c': 22.0, 'eta_c1->J/psi': 14.0, 'H1(0-+)->J/psi': 6.0, 'H1(2-+)->J/psi': 41.0,
            'Y->chi_c0': 10.0, 'Y->chi_c1': 10.0, 'Y->chi_c2': 15.0, 'H1(0-+)->h_c': 36.0,
            'eta_c1->h_c': 7.0, 'H1(2-+)->h_c': 9.0}
JS_PARTNERS = [('H1(0-+)->J/psi', 1, 0, 4.279), ('H1(2-+)->J/psi', 1, 2, 4.456)]
# vector -> eta_c gamma of conventional states in the thesis (full QCD, M1 included), keV
THESIS_V_ETAC = {"psi(2S)": (0.485, 0.060), "psi(3770)": (0.33, 0.11), 'J/psi': (1.5, None)}
LAT_SYST = 0.30


def hybrid_lattice_table(flavor_lib, gaps, hyb, variants, kappa):
    """Our LO widths (keV) at the kinematics of each lattice determination, and
    the kappa_H that reproduces it; plus the S=1 partners at the J&S kinematics."""
    from . import hybrids as HY
    rows = []
    entries = [e + (None,) for e in LATTICE_HYBRID] + \
              [(ch, 'JS-kin', S, Jcal, M, ('J/psi', 1, 0, 3.096900), np.nan, np.nan, None) for ch, S, Jcal, M in JS_PARTNERS]
    for ch, ds, S, Jcal, Mi, fin, G, dG, _ in entries:
        h = tuple(hyb[:6]) + (Mi,) + tuple(hyb[7:])
        Jf = None if S == 0 else 1
        for v in variants:
            sp, E = HY.gamma_split('c', flavor_lib, gaps, h, fin, v, (1.0,) * 4, S=S, Jcal=Jcal, Jf=Jf)
            g1 = sp['total']
            rows.append({'channel': ch, 'dataset': ds, 'M_i': Mi, 'M_f': fin[3], 'E_q': E,
                         'variant': v, 'Gamma_k1_keV': g1, 'Gamma_k1_Pi_keV': sp['Pi'],
                         'Gamma_k1_Sigma_keV': sp['Sigma'], 'Gamma_k1_int_keV': sp['int'],
                         'Gamma_kappac_keV': kappa ** 2 * g1, 'lat_keV': G, 'dlat_keV': dG,
                         'kappaH': np.sqrt(G / g1) if G == G else np.nan,
                         'JS_keV': JS_MODEL.get(ch, np.nan)})
    return rows


def gamma_vs_Eq(lib, gaps, hyb, variants, Es):
    """Gamma(1(s/d)_1 -> eta_c(1S) gamma)(E_q) at kappa_H = 1 (keV). At our order this is
    also Gamma(S=1 (s/d)_1, any Jcal -> J/psi gamma) at the same E_q."""
    from . import hybrids as HY
    from .amplitudes import width_split
    st = HY.hybrid_state(lib, hyb)
    f = lib.quarkonium(1, 0)
    out = []
    for v in variants:
        P = HY.params('c', v, (1.0,) * 4, gaps)
        for E in Es:
            sp = width_split(f, st, E, P, lib.r)
            out.append({'variant': v, 'E_q': float(E), 'Gamma_k1_keV': sp['total'] * 1e6,
                        'Pi_keV': sp['Pi'] * 1e6, 'Sigma_keV': sp['Sigma'] * 1e6, 'int_keV': sp['int'] * 1e6})
    return out


def channel_solutions(rows, variants=('A-NL', 'A-L', 'B-NL', 'B-L')):
    """Channel strengths (kappa_Pi, kappa_Sigma) that reproduce two lattice widths of the
    same data set at once, Gamma = kappa_Pi^2 G_PiPi + kappa_Sigma^2 G_SigmaSigma +
    kappa_Pi kappa_Sigma G_int (G at kappa = 1). All solutions with kappa_Pi > 0."""
    from scipy.optimize import fsolve
    def get(ch, ds, v):
        r = [x for x in rows if x['channel'] == ch and x['dataset'] == ds and x['variant'] == v][0]
        return (r['Gamma_k1_Pi_keV'], r['Gamma_k1_Sigma_keV'], r['Gamma_k1_int_keV'], r['lat_keV'])
    G = lambda k, a: k[0] ** 2 * a[0] + k[1] ** 2 * a[1] + k[0] * k[1] * a[2]
    out = []
    for v in variants:
        for yset, hset in (('quenched', 'quenched'), ('Nf3', 'Nf3'), ('Nf3-F0', 'Nf3')):
            Y, H = get('Y->eta_c', yset, v), get('eta_c1->J/psi', hset, v)
            sols = set()
            for x0 in [(0.5, 0.5), (0.5, -0.5), (0.2, 1), (0.2, -1), (1, 1), (0.3, 0.1), (0.5, 2), (0.5, -2), (0.1, 3), (0.1, -3)]:
                s, info, ier, msg = fsolve(lambda k: [G(k, Y) - Y[3], G(k, H) - H[3]], x0, full_output=True)
                if ier == 1 and s[0] > 0 and abs(G(s, Y) / Y[3] - 1) < 1e-8:
                    sols.add((round(float(s[0]), 4), round(float(s[1]), 4)))
            for kP, kS in sorted(sols, key=lambda t: abs(t[1])):
                out.append({'variant': v, 'Y_dataset': yset, 'etac1_dataset': hset, 'kappa_Pi': kP, 'kappa_Sigma': kS,
                            'kSigma_over_kPi': kS / kP})
    return out
