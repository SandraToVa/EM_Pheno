"""Size and photon-energy estimates for the power counting of the hybrid
EM transitions (BOEFT multipole parameter E_q r, EST parameter 1/(r sqrt(sigma))).

    python3 r_moments.py          (from the python/ folder)

Reads only data/wf_*.mat (exported by matlab/exportWavefunctions.m) through
emlib.states.Library, and the state/mass tables of emlib.hybrids. Nothing in
QQbarSpectrum or in emlib is modified.

For every state:      <r>, sqrt(<r^2>), and for hybrids the Sigma_u^- / Pi_u weights.
For every transition: E_q = (M_i^2 - M_f^2)/(2 M_i), the overlap-weighted size
                      r_tr = int r |u_f u_i| / int |u_f u_i|,
                      and the counting parameters r sqrt(sigma), E_q r, E_q r^2 sqrt(sigma).
Output: results/r_moments_states.csv, results/r_moments_transitions.csv
"""
import os, sys, csv
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from emlib.states import Library, pi_sigma_projections
from emlib import hybrids as HY
from emlib.amplitudes import photon_energy, Params

HBARC = 0.1973269804                       # GeV fm
SIGMA = Params('c').sigma                  # string tension used in emlib (GeV^2)
SQS = np.sqrt(SIGMA)
RES = os.path.join(os.path.dirname(__file__), '..', 'results')


def density(st):
    """|psi|^2 r^2 on the mesh, summed over channels, as Gauss weights c_i^2"""
    return sum(c.c ** 2 for c in st.comps)


def moments(lib, st):
    w = density(st)
    norm = w.sum()
    r1 = (w * lib.r).sum() / norm
    r2 = (w * lib.r ** 2).sum() / norm
    return r1, np.sqrt(r2)


def sigma_pi_weights(st, J):
    """Sigma_u^- and Pi_u weights of a hybrid state: int R_Sigma^2 and int R_Pi^2
    with R_Pi, R_Sigma the projections of Eqs. (eq:lab_RPi), (eq:lab_RSigma)."""
    pr = pi_sigma_projections(st)
    return pr['P_Sigma'], pr['P_Pi']


def overlap_size(lib, sf, si):
    """r_tr = int r |u_f u_i| / int |u_f u_i| (u_i summed in quadrature over channels)"""
    af = np.sqrt(density(sf)); ai = np.sqrt(density(si))
    w = af * ai
    return (w * lib.r).sum() / w.sum()


def allowed(h, fin):
    J, kind, lf = h[1], h[2], fin[2]
    if kind == 'sigma':
        return lf == 1
    xi = -1 if kind == 'minus' else +1
    return ((-1) ** lf == -xi * (-1) ** J) and abs(lf - J) <= 2


def main():
    os.makedirs(RES, exist_ok=True)
    srows, trows = [], []
    for fl in ('c', 'b'):
        lib = Library(fl)
        for fin in HY.FINALS[fl]:
            st = lib.quarkonium(fin[1], fin[2])
            r1, rr = moments(lib, st)
            srows.append({'flavor': fl, 'state': fin[0], 'type': 'quarkonium', 'M_GeV': fin[3],
                          'r_mean_GeVinv': r1, 'r_rms_GeVinv': rr, 'r_mean_fm': r1 * HBARC,
                          'r_rms_fm': rr * HBARC, 'w_Sigma': '', 'w_Pi': ''})
        for h in HY.HYBRIDS[fl]:
            st = HY.hybrid_state(lib, h)
            r1, rr = moments(lib, st)
            wS, wP = sigma_pi_weights(st, h[1])
            srows.append({'flavor': fl, 'state': h[0] + ' ' + h[4], 'type': 'hybrid', 'M_GeV': h[6],
                          'r_mean_GeVinv': r1, 'r_rms_GeVinv': rr, 'r_mean_fm': r1 * HBARC,
                          'r_rms_fm': rr * HBARC, 'w_Sigma': wS, 'w_Pi': wP})
            for fin in HY.FINALS[fl]:
                if fin[3] >= h[6] or not allowed(h, fin):
                    continue
                sf = lib.quarkonium(fin[1], fin[2])
                Eq = photon_energy(h[6], fin[3])
                rt = overlap_size(lib, sf, st)
                trows.append({'flavor': fl, 'hybrid': h[0], 'JPC': h[4], 'final': fin[0],
                              'E_q_GeV': Eq, 'r_tr_fm': rt * HBARC,
                              'r_sqrt_sigma': rt * SQS, 'Eq_r': Eq * rt,
                              'Eq_r2_sqrt_sigma': Eq * rt ** 2 * SQS, 'w_Sigma_i': wS})
    for name, rows in (('r_moments_states.csv', srows), ('r_moments_transitions.csv', trows)):
        with open(os.path.join(RES, name), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow({k: (f'{v:.4g}' if isinstance(v, float) else v) for k, v in r.items()})

    print(f'sigma = {SIGMA} GeV^2, sqrt(sigma) = {SQS:.3f} GeV, 1/sqrt(sigma) = {HBARC/SQS:.3f} fm\n')
    print(f"{'fl':2} {'state':22} {'M':>7} {'<r>fm':>6} {'rms fm':>6} {'w_Sig':>6}")
    for r in srows:
        ws = f"{r['w_Sigma']:.2f}" if r['w_Sigma'] != '' else ''
        print(f"{r['flavor']:2} {r['state']:22} {r['M_GeV']:7.3f} {r['r_mean_fm']:6.3f} {r['r_rms_fm']:6.3f} {ws:>6}")
    print(f"\n{'fl':2} {'hybrid':9} {'final':11} {'E_q':>6} {'r_tr fm':>7} {'r√σ':>5} {'E_q r':>6} {'E_q r²√σ':>8} {'w_Sig':>6}")
    for r in trows:
        print(f"{r['flavor']:2} {r['hybrid']:9} {r['final']:11} {r['E_q_GeV']:6.3f} {r['r_tr_fm']:7.3f} "
              f"{r['r_sqrt_sigma']:5.2f} {r['Eq_r']:6.2f} {r['Eq_r2_sqrt_sigma']:8.2f} {r['w_Sigma_i']:6.2f}")


if __name__ == '__main__':
    main()
