"""Experimental inputs (PDG 2026 = RPP 2026, F. Takahashi et al., IJMPA 41 (2026)
2630011, unless stated otherwise) and the state dictionary.

Masses in GeV. Widths in keV.
Every entry of DATA is one observable:
  kind  : 'width'   partial width Gamma(i -> f gamma)      value +- err (keV)
          'ratio'   Gamma(i -> f1 gamma)/Gamma(i -> f2 gamma)
          'ul_width', 'ul_ratio' : 90% CL upper limits
  init, final (, final2 for ratios): state labels (see STATES)
  group : observables sharing a total width (correlated errors)
  set   : 'core' (states below open-flavour threshold), 'ext' (above
          threshold / strongly coupled channels), 'test' (exotic, not fitted)
"""
import numpy as np

# ---------------------------------------------------------------- masses
MASS = {
    # charmonium
    'etac1S': 2.98409, 'Jpsi': 3.096900, 'chic0': 3.41550, 'chic1': 3.51067,
    'hc': 3.52537, 'chic2': 3.55617, 'etac2S': 3.6378, 'psi2S': 3.686097,
    'psi3770': 3.7737, 'psi2_3823': 3.82351, 'psi3_3842': 3.84271,
    'X3872': 3.87164,
    # charmonium-like 1-- (hybrid candidates / higher quarkonia)
    'psi4040': 4.040, 'psi4160': 4.191, 'psi4230': 4.2217, 'psi4360': 4.373,
    'psi4415': 4.415, 'psi4660': 4.623, 'chic1_4140': 4.1465, 'chic0_4500': 4.484,
    # bottomonium
    'etab1S': 9.3987, 'Y1S': 9.46040, 'chib0': 9.85944, 'chib1': 9.89278,
    'hb': 9.8993, 'chib2': 9.91221, 'etab2S': 9.9990, 'Y2S': 10.0234,
    'Y1D': 10.1637, 'chib0_2P': 10.2325, 'chib1_2P': 10.25546, 'hb2P': 10.2598,
    'chib2_2P': 10.26865, 'Y3S': 10.3551, 'chib1_3P': 10.5134, 'chib2_3P': 10.5240,
    'Y4S': 10.5794, 'Y10753': 10.7566, 'Y10860': 10.8852, 'Y11020': 11.000,
}

# ------------------------------------------------- state labels -> quantum numbers
# (flavor, n, l, S, J): n is the radial index inside the family of orbital l
STATES = {
    'etac1S': ('c', 1, 0, 0, 0), 'Jpsi': ('c', 1, 0, 1, 1),
    'chic0': ('c', 1, 1, 1, 0), 'chic1': ('c', 1, 1, 1, 1), 'chic2': ('c', 1, 1, 1, 2),
    'hc': ('c', 1, 1, 0, 1), 'etac2S': ('c', 2, 0, 0, 0), 'psi2S': ('c', 2, 0, 1, 1),
    'psi3770': ('c', 1, 2, 1, 1), 'psi2_3823': ('c', 1, 2, 1, 2), 'psi3_3842': ('c', 1, 2, 1, 3),
    'X3872': ('c', 2, 1, 1, 1),
    'etab1S': ('b', 1, 0, 0, 0), 'Y1S': ('b', 1, 0, 1, 1),
    'chib0': ('b', 1, 1, 1, 0), 'chib1': ('b', 1, 1, 1, 1), 'chib2': ('b', 1, 1, 1, 2),
    'hb': ('b', 1, 1, 0, 1), 'etab2S': ('b', 2, 0, 0, 0), 'Y2S': ('b', 2, 0, 1, 1),
    'Y1D': ('b', 1, 2, 1, 2),
    'chib0_2P': ('b', 2, 1, 1, 0), 'chib1_2P': ('b', 2, 1, 1, 1), 'chib2_2P': ('b', 2, 1, 1, 2),
    'hb2P': ('b', 2, 1, 0, 1), 'Y3S': ('b', 3, 0, 1, 1),
}

# total widths used to convert branching fractions (keV)
GTOT = {'psi2S': (293.0, 9.0), 'chic0': (12400.0, 270.0), 'chic1': (820.0, 40.0),
        'chic2': (1940.0, 90.0), 'hc': (780.0, 280.0), 'psi3770': (27200.0, 1000.0),
        'Y2S': (31.98, 2.63), 'Y3S': (20.32, 1.85), 'X3872': (1190.0, 210.0)}


def _w(init, br, dbr):
    """width = Gamma_tot * BR with uncorrelated error from BR and the
    error of Gamma_tot kept separately (for the correlation matrix)."""
    G, dG = GTOT[init]
    return G * br, G * dbr, br * dG          # value, err_BR, err_Gtot


DATA = []


def _add(**kw):
    DATA.append(kw)


# ---------------------------- charmonium, below threshold
for J, (br, dbr) in zip((0, 1, 2), [(0.1006, 0.0027), (0.0985, 0.0026), (0.0963, 0.0023)]):
    v, e1, e2 = _w('psi2S', br, dbr)
    _add(id=f'psi2S->chic{J}', kind='width', init='psi2S', final=f'chic{J}', value=v,
         err=e1, err_corr=e2, group='psi2S', set='core', flavor='c')
for J, (br, dbr) in zip((0, 1, 2), [(0.0135, 0.0009), (0.339, 0.012), (0.190, 0.008)]):
    v, e1, e2 = _w(f'chic{J}', br, dbr)
    _add(id=f'chic{J}->Jpsi', kind='width', init=f'chic{J}', final='Jpsi', value=v,
         err=np.hypot(e1, e2), err_corr=0.0, group=None, set='core', flavor='c')
v, e1, e2 = _w('hc', 0.64, 0.05)
_add(id='hc->etac', kind='width', init='hc', final='etac1S', value=v, err=np.hypot(e1, e2),
     err_corr=0.0, group=None, set='core', flavor='c')
# ---------------------------- charmonium, above DD threshold / D-waves
v, e1, e2 = _w('psi3770', 7.1e-3, 0.6e-3)
_add(id='psi3770->chic0', kind='width', init='psi3770', final='chic0', value=v, err=e1,
     err_corr=e2, group='psi3770', set='ext', flavor='c')
v, e1, e2 = _w('psi3770', 2.49e-3, 0.23e-3)
_add(id='psi3770->chic1', kind='width', init='psi3770', final='chic1', value=v, err=e1,
     err_corr=e2, group='psi3770', set='ext', flavor='c')
_add(id='psi3770->chic2', kind='ul_width', init='psi3770', final='chic2', value=6.4e-4 * 27200.0,
     err=0.0, err_corr=0.0, group=None, set='ext', flavor='c')
_add(id='psi2(3823): chic2/chic1', kind='ratio', init='psi2_3823', final='chic2', final2='chic1',
     value=0.28, err=0.125, err_up=0.14, err_dn=0.11, err_corr=0.0, group=None, set='ext', flavor='c')
_add(id='psi2(3823): chic0/chic1', kind='ul_ratio', init='psi2_3823', final='chic0', final2='chic1',
     value=0.24, err=0.0, err_corr=0.0, group=None, set='ext', flavor='c')
# ---------------------------- exotic test (not fitted)
v, e1, e2 = _w('X3872', 1.0e-2, 0.4e-2)
_add(id='X(3872)->Jpsi', kind='width', init='X3872', final='Jpsi', value=v, err=np.hypot(e1, e2),
     err_corr=0.0, group=None, set='test', flavor='c')

# ---------------------------- bottomonium
for J, (br, dbr) in zip((0, 1, 2), [(0.038, 0.004), (0.069, 0.004), (0.0715, 0.0035)]):
    v, e1, e2 = _w('Y2S', br, dbr)
    _add(id=f'Y2S->chib{J}(1P)', kind='width', init='Y2S', final=f'chib{J}', value=v, err=e1,
         err_corr=e2, group='Y2S', set='core', flavor='b')
for J, (br, dbr) in zip((0, 1, 2), [(0.059, 0.006), (0.126, 0.012), (0.131, 0.016)]):
    v, e1, e2 = _w('Y3S', br, dbr)
    _add(id=f'Y3S->chib{J}(2P)', kind='width', init='Y3S', final=f'chib{J}_2P', value=v, err=e1,
         err_corr=e2, group='Y3S', set='core', flavor='b')
for J, (br, dbr) in zip((0, 1, 2), [(2.7e-3, 0.4e-3), (9e-4, 5e-4), (10.0e-3, 1.0e-3)]):
    v, e1, e2 = _w('Y3S', br, dbr)
    _add(id=f'Y3S->chib{J}(1P)', kind='width', init='Y3S', final=f'chib{J}', value=v, err=e1,
         err_corr=e2, group='Y3S', set='core', flavor='b')
# ratios from branching fractions (independent errors)
for J, (a, da, b, db) in zip((0, 1, 2), [(0.38e-2, 0.17e-2, 1.38e-2, 0.30e-2),
                                         (9.9e-2, 1.0e-2, 18.1e-2, 1.9e-2),
                                         (6.6e-2, 0.8e-2, 8.9e-2, 1.2e-2)]):
    R = a / b
    _add(id=f'chib{J}(2P): Y1S/Y2S', kind='ratio', init=f'chib{J}_2P', final='Y1S', final2='Y2S',
         value=R, err=R * np.hypot(da / a, db / b), err_corr=0.0, group=None, set='core', flavor='b')
R = 0.22 / 0.48
_add(id='hb(2P): etab1S/etab2S', kind='ratio', init='hb2P', final='etab1S', final2='etab2S',
     value=R, err=R * np.hypot(0.05 / 0.22, 0.13 / 0.48), err_corr=0.0, group=None, set='core', flavor='b')


def covariance(rows):
    """Experimental covariance matrix (keV^2 or ratio^2) for a list of rows."""
    n = len(rows)
    C = np.zeros((n, n))
    for i, a in enumerate(rows):
        C[i, i] = a['err'] ** 2 + a.get('err_corr', 0.0) ** 2
        for j, b in enumerate(rows):
            if j > i and a.get('group') and a['group'] == b.get('group'):
                c = a['err_corr'] * b['err_corr']
                C[i, j] = C[j, i] = c
    return C


# ------------------------------------------------------------------ BESIII inputs (Y region)
# Born cross sections (pb) at sqrt(s) = 4.23 and 4.26 GeV
#  gamma eta_c(1S): BESIII, PRD 96, 051101 (2017) [arXiv:1705.06853], Table I
#  gamma chi_c1,2 : BESIII, PRD 92, 012008 (2015) [arXiv:1411.6336], Table II
#  gamma X(3872)  : BESIII, PRL 112, 092001 (2014) [arXiv:1310.4101]: sigma*B(X->pi pi J/psi)
BES = {
    4.23: {'etac1S': (1.34, np.hypot(0.59, 0.22)), 'chic1': (0.7, np.hypot(0.5, 0.1), 1.7),
           'chic2': (2.7, np.hypot(1.1, 0.3), 5.0), 'X3872xB': (0.27, np.hypot(0.09, 0.02))},
    4.26: {'etac1S': (2.17, np.hypot(0.70, 0.39)), 'chic1': (0.4, np.hypot(0.4, 0.1), 1.1),
           'chic2': (2.0, np.hypot(1.1, 0.2), 4.2), 'X3872xB': (0.33, np.hypot(0.12, 0.02))},
}
B_X_PIPIJPSI = (0.043, 0.014)     # PDG 2026, chi_c1(3872) -> pi+ pi- J/psi
# PDG 2026 branching-fraction upper limits (90% CL) of 1-- states
PDG_UL = {
    'psi4040': {'chic1': 3.4e-3, 'chic2': 5e-3},
    'psi4160': {'chic1': 5e-3, 'chic2': 1.3e-2, 'X3872': 1.6e-3},
    'psi4415': {'chic1': 1.3e-3, 'chic2': 7e-3},
}
GTOT_XYZ = {'psi4040': (84e3, 12e3), 'psi4160': (69e3, 10e3), 'psi4230': (51e3, 7e3),
            'psi4360': (124e3, 13e3), 'psi4415': (110e3, 13e3), 'psi4660': (55e3, 9e3),
            'Y10753': (29e3, 9e3), 'Y10860': (37e3, 4e3), 'Y11020': (24e3, 7e3)}
