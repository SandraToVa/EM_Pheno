"""Experimental inputs: RPP 2026 (F. Takahashi et al. [Particle Data Group], IJMPA 41 (2026) 2630011).

Masses in GeV, widths in keV. Every entry of DATA is one observable:
  kind   'width'      partial width Gamma(init -> final gamma), value +- err (keV)
         'ratio'      Gamma(init -> final gamma)/Gamma(init -> final2 gamma)
         'ul_width', 'ul_ratio'   90% CL upper limits
  init, final (, final2): state labels (STATES)
  err    error not shared with other observables; err_corr: error from the total width of init,
         common to the observables of the same group (correlated); err_up, err_dn: asymmetric errors
  set    'core' (states below the open-flavour threshold, without hindered transitions) or 'ext' (psi(3770),
         psi_2(3823): D waves; hindered Upsilon(3S) -> gamma chi_bJ(1P): never fitted, v11)
  flavor 'c' or 'b'
"""
import numpy as np

MASS = {
    'etac1S': 2.98409, 'Jpsi': 3.096900, 'chic0': 3.41550, 'chic1': 3.51067, 'hc': 3.52537,
    'chic2': 3.55617, 'psi2S': 3.686097, 'psi3770': 3.7737, 'psi2_3823': 3.82351,
    'etab1S': 9.3987, 'Y1S': 9.46040, 'chib0': 9.85944, 'chib1': 9.89278, 'chib2': 9.91221,
    'etab2S': 9.9990, 'Y2S': 10.0234, 'chib0_2P': 10.2325, 'chib1_2P': 10.25546, 'hb2P': 10.2598,
    'chib2_2P': 10.26865, 'Y3S': 10.3551,
}

# state label -> (flavor, n, l, S, J); n is the radial index within the family of orbital l
STATES = {
    'etac1S': ('c', 1, 0, 0, 0), 'Jpsi': ('c', 1, 0, 1, 1),
    'chic0': ('c', 1, 1, 1, 0), 'chic1': ('c', 1, 1, 1, 1), 'chic2': ('c', 1, 1, 1, 2),
    'hc': ('c', 1, 1, 0, 1), 'psi2S': ('c', 2, 0, 1, 1),
    'psi3770': ('c', 1, 2, 1, 1), 'psi2_3823': ('c', 1, 2, 1, 2),
    'etab1S': ('b', 1, 0, 0, 0), 'Y1S': ('b', 1, 0, 1, 1),
    'chib0': ('b', 1, 1, 1, 0), 'chib1': ('b', 1, 1, 1, 1), 'chib2': ('b', 1, 1, 1, 2),
    'etab2S': ('b', 2, 0, 0, 0), 'Y2S': ('b', 2, 0, 1, 1),
    'chib0_2P': ('b', 2, 1, 1, 0), 'chib1_2P': ('b', 2, 1, 1, 1), 'chib2_2P': ('b', 2, 1, 1, 2),
    'hb2P': ('b', 2, 1, 0, 1), 'Y3S': ('b', 3, 0, 1, 1),
}

# total widths used to convert branching fractions (keV)
GTOT = {'psi2S': (293.0, 9.0), 'chic0': (12400.0, 270.0), 'chic1': (820.0, 40.0),
        'chic2': (1940.0, 90.0), 'hc': (780.0, 280.0), 'psi3770': (27200.0, 1000.0),
        'Y2S': (31.98, 2.63), 'Y3S': (20.32, 1.85)}


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
         err_corr=e2, group='Y3S', set='ext', flavor='b')
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
