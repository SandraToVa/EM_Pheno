# EM_Pheno_v9: quarkonium E1 transitions

Code of Sec. 5 (phenomenology) of the *EM Transitions Notes*, quarkonium E1 part. It does exactly what the text
describes, and nothing else:

* LEF of Eq. (eq:gE1_interp):
  `g_E1(r) = e r [ e_Q + (A_Q r + kappa_Q x)/(1 + x) ]`, `x = (r/r0)^2`, `r0 = sqrt(pi/(6 sigma)) = 1.58 GeV^-1`
  (`sigma = 0.21 GeV^2`).
* Known 1/m corrections of the E1 amplitude, `A -> A (1 + delta_1m)`, Eq. (eq:delta_rel) (spin term for
  spin-triplet S <-> P, Delta l term for all transitions). The 1/m^2 term is not included.
* Flavour-independent fit: `A_Q = A/e_Q`, `kappa_Q = kappa/e_Q`, i.e. `A_c = -A_b/2`, `kappa_c = -kappa_b/2`;
  two parameters `(A_b, kappa_b)` fitted to the charmonium and bottomonium data together.
* Theory error `eps = v^2` (`eps_c = 0.3` for the charmonium observables, `eps_b = 0.1` for the bottomonium ones)
  or `eps = 0` (experimental errors only).
* Local (L) and non-local (NL) forms; reference fit: NL, core data.
* The LO widths (`A_Q = kappa_Q = 0`, no 1/m) are kept and the corrections are added on top of them:
  every width is `Gamma = (1 + delta_1m)^2 v^T G v`, `v = (1, A_Q, kappa_Q)`, and LO is `G[0,0]`.

The hybrid part of the analysis (and the quarkonium-hybrid mixing) is not in this version; it is in
`EM_Pheno_v8` and will be added back on top of this code.

## How to run

```
# 1. (only if the wave functions change) MATLAB or octave-cli, from EM_Pheno_v9/matlab
exportWavefunctions('<path>/Ruben Mixing/QQbarSpectrum', '../data')
# 2. from EM_Pheno_v9/python  (about 15 s)
python3 run_all.py            # -> results/*.csv, results/summary.json
python3 make_tables.py        # -> TeX file/tex_pheno/tab_*.tex (or $EMPHENO_TEX, or EM_Pheno_v9/tex)
python3 ../tests/test_e1.py   # must end with ALL PASS
# 3. from "TeX file/tex_pheno" (needs ../mainEM.aux: compile mainEM once)
pdflatex pheno_standalone && bibtex pheno_standalone && pdflatex pheno_standalone && pdflatex pheno_standalone
```

Requirements: Python 3 with `numpy`, `scipy` (>= 1.15, for `sph_harm_y`); `sympy` for the tests only.

## Files

```
data/wf_c_nomix.mat, wf_b_nomix.mat   LO wave functions (QQbarSpectrum, families Ql0..Ql3 on a common mesh)
matlab/exportWavefunctions.m          writes the two .mat files
python/emlib/angular.py               Clebsch-Gordan, 6j, spin factor S_if of the E1 widths, angular grid, Y_l^m
python/emlib/states.py                Library (reads data/), State n (l S) J
python/emlib/data.py                  RPP 2026 E1 data (widths, ratios, upper limits, correlations), masses
python/emlib/e1.py                    g_E1(r), E1 amplitudes (L and NL), quadratic forms G, delta_1m, theory error
python/emlib/fit.py                   Observable, flavour-independent Fit: chi^2, minimum, errors, pulls
python/run_all.py                     all the numbers (results/)
python/make_tables.py                 LaTeX table bodies
tests/test_e1.py                      checks against closed formulas
results/                              fits.csv, widths.csv, j_ratios.csv, lattice.csv, summary.json, *.log
```

## What `run_all.py` computes

1. `fits.csv`: for L and NL, and for the data sets `core` (charm and bottom states below the open-flavour
   threshold, 20 observables) and `core+ext` (+ psi(3770), psi_2(3823)): the chi^2 at LO and at LO + 1/m (no free
   parameter), and the fits at `eps = v^2` and `eps = 0`: `A_b`, `kappa_b` with Delta chi^2 = 1 profile errors,
   their correlation (Hessian) and chi^2 = chi^2_c + chi^2_b. -> Table `tab:gE1_fit`.
2. `widths.csv`: every observable at the reference fit (NL, core): data, LO, delta_1m, LO + 1/m, fit at
   `eps = v^2` (with theory error, fit error and pull) and at `eps = 0` (with pull); `fitted = False` for the
   `ext` observables, which are then predictions. -> Table `tab:E1_widths`.
3. `j_ratios.csv`: J dependence of the spin multiplets. At LO the radial integral is the same for all J and
   `Gamma_J = (4/3) alpha e_Q^2 E_J^3 S_J |I|^2`, `S_J = (2Jf+1) max(li,lf) {Ji Jf 1; lf li 1}^2`, so the reduced
   width `Gamma_J/(S_J E_J^3)` relative to a reference J is 1. Data, LO (NL), LO + 1/m and the fit.
   -> Table `tab:J_ratios`.
4. `lattice.csv`: lattice E1 widths (HSC, ETMC, CLQCD) versus LO (L, NL) and the reference fit with its error.
   -> Tables `tab:lattice`, `tab:quarkonium_lattice`.
5. `summary.json`: reference parameters (and A_c, kappa_c), chi^2, g_E1/(e e_Q r) of the fit at a few r, lattice J
   ratios.

## Fit details

* chi^2 with the full experimental covariance (widths sharing a total width are correlated), asymmetric errors,
  90% CL upper limits as half-normal likelihoods, and the theory error
  `sigma_th^2 = 4 eps^2 Gamma Gamma_abs + 2 eps^4 Gamma_abs^2` (`Gamma_abs`: same width with `|u_f u_i|`).
* Physical region: `-0.8 <= kappa_Q/e_Q <= 1` and `|A_Q/e_Q| <= 2 GeV` for both flavours (outside it the correction
  flips the sign of the LO amplitude and the chi^2 has mirror minima). Minimum: grid over the region, then
  Nelder-Mead from the four lowest local minima of the grid.
* Errors of predictions: propagated from the covariance of `(A_b, kappa_b)` (inverse Hessian of chi^2/2).

## Tests (`tests/test_e1.py`, 87 checks)

CG and 6j against sympy; LO local widths of all the data-set transitions against the closed formula with the 6j;
spin sum rule; NL widths against the multipole expansion of `j0(q_z r/2)`; quadratic form against the direct width
with `g_E1(r)`; `delta_1m`; flavour relations; equal reduced LO widths in a multiplet; theory error limits.

## Changes with respect to v8

Removed: the one-parameter fits `g_E1 = (1 + dkappa_Q) e e_Q r` (corr = LO, 1m, 1m2), the one-parameter
interpolation `e r [e_Q + kappa_Q f(r)]`, the per-flavour two-parameter fits, n = 7 and r0 = 1 GeV^-1, the 1/m^2
option, the epsilon of chi^2/ndf = 1 and epsilon = v^3, the X(3872) test, the hybrid, mixing and hybrid-lattice
code (hybrids.py, mixing.py, lattice.py, the hybrid vertex of amplitudes.py), r_moments.py, figures.py, pandas.
The fits reproduce those of v8 (`interp2.Interp2Fitter`, mode linked, corr = 1m) to all digits for the same eps.
