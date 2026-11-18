# EM_Pheno (v12): quarkonium E1 and hybrid-to-quarkonium transitions

Git repository `SandraToVa/EM_Pheno` (this folder: `EM transitions/EM_Pheno`). Versions are commits; the TeX files
are outside the repo, in `EM transitions/TeX file/tex_pheno`. See `CLAUDE.md` for the working conventions.

## Changes with respect to v11 (hybrids only; quarkonium unchanged)

* Hybrid LEFs, Eq. (eq:gH_interp): each LEF interpolates between a short-distance term and its own EST shape, with the
  long-distance coupling identified with the quarkonium `kappa_Q` (`c_i -> e kappa_Q`):
  `g_B(r) = e [B_Q r + kappa_Q ghat_B(r) x]/(1 + x)` for `g_B_perp` (`ghat = 2/sqrt(pi sigma)`) and `g_B_par`
  (`ghat = -4/(sigma r)`); `g_E(r) = e kappa_Q ghat_E(r) x/(1 + x)` for the electric LEFs (`ghat_E_par = -2r/(pi
  sqrt(pi sigma))`, `ghat_E_perp = +2r/(pi sqrt(pi sigma))`, `ghat_E^Sigma = 4/(3 pi sigma)`): no short-distance term of
  order r (dimension -2). `B_Q` is now dimensionless; `B_c = -B_b/2`. Replaces `c_i = e kappa_H(r)` of v10.
* Only the LOCAL form (the interpolation of the non-local densities is not defined yet; `nonloc=True` raises
  `NotImplementedError`).
* Fit as quarkonium v11: `sigma_th = 2 eps Gamma` at the model width (eps = 0.3); no t0, no Gamma_bar. Errors on
  `B_c` and on every prediction: range over the central fit and the fits to `Gamma_lat + sigma_lat` and
  `Gamma_lat - sigma_lat` (both widths moved together, as Eq. (eq:exp_range)); no Delta chi^2 = 1.
* `kappa_c = -0.0231` (v11 quarkonium fit). Result: `B_c = -0.186 +0.009 -0.010` (chi^2 = 1.4/1; other branch
  `B_c = +0.120`, chi^2 = 1.5). The Pi_u channel now dominates.
* `fig_H_kappaH` replaced by `fig_H_LEFs` (g_B_perp, g_B_par). tests/test_hybrid.py rewritten (149 checks).

## Earlier: changes of v11 with respect to v10 (quarkonium only)

* Theory error `sigma_th = 2 eps Gamma` (model width), Eq. (eq:sigma_th); the eps^4 term of the variance and the
  no-cancellation width Gamma_bar are dropped (`e1.theory_sigma(G, eps)`). Ratios: `2 sqrt(2) eps R`.
* Errors of `A_b`, `kappa_b`: fits to Gamma_exp,max = d + sigma_exp and Gamma_exp,min = d - sigma_exp (all measured
  widths and ratios moved together; err_up/err_dn where asymmetric; same covariance, theory error and upper limits),
  Eq. (eq:exp_range). The interval is the range spanned by the central and the two shifted fits; predicted widths get
  the range over the same three fits (`Fit.exp_shifted`, `Fit.run`, `Fit.prediction_range`). No Delta chi^2 = 1
  errors, no Hessian, no rho.
* Hindered Upsilon(3S) -> gamma chi_bJ(1P) moved from core to ext and never fitted (core: 7 charm + 10 bottom).
* Reference fit (NL, core, eps = v^2): `A_b = -0.059 +0.067 -0.067 GeV`, `kappa_b = +0.046 +0.026 -0.025`,
  chi^2 = 8.8/15. Max fit: (-0.126, +0.072); min fit: (+0.008, +0.021).
* `summary.json` `ref_v2.errors_b` now holds these errors. `run_hybrids.py` reads it (kappa_c +-1 sigma check of
  Table tab:H_fit) and `kappa_c`: if it is rerun, the hybrid results change. The hybrid tables in the TeX folder are
  still those of v10.
* tests/test_e1.py: 142 checks (theory error, data sets, shifted fits).

Code of Sec. 5 (phenomenology) of the *EM Transitions Notes*. The quarkonium part is that of v9, unchanged (same
numbers); v10 adds the hybrid part on top of it.

## Quarkonium E1 (as in v9)

* LEF of Eq. (eq:gE1_interp): `g_E1(r) = e r [ e_Q + (A_Q r + kappa_Q x)/(1 + x) ]`, `x = (r/r0)^2`,
  `r0 = sqrt(pi/(6 sigma)) = 1.58 GeV^-1`, with the known 1/m corrections, Eq. (eq:delta_rel).
* Flavour-independent fit `A_c = -A_b/2`, `kappa_c = -kappa_b/2`; theory error `eps = v^2` (0.3 c, 0.1 b) or 0.
* Reference fit: NL, core data: `A_b = -0.057 GeV`, `kappa_b = +0.044` (`kappa_c = -0.022`).

## Hybrids (v12)

* Vertex of Eq. (eq:lab_vertex) with the interpolated LEFs of Eq. (eq:gH_interp) (see the v12 changes above), local
  form. `kappa_Q` from the reference quarkonium fit (read from `results/summary.json`), `B_c` fitted, `B_b = -2 B_c`.
* Fit of `B_c` to the quenched lattice widths of Dudek, Edwards, Thomas, PRD 79 (2009) 094504 [arXiv:0902.2241]:
  `Gamma(Y_hyb -> gamma eta_c) = 42(18) keV` (M_Y = 4.400 GeV), `Gamma(eta_c1 -> gamma J/psi) = 115(16) keV`
  (M = 4.300 GeV); statistical errors + theory error `sigma_th = 2 eps Gamma`, `eps = v_c^2 = 0.3`. chi^2 has two
  minima (sign of `B_c`); both are kept, the reference is the lower one.
* Predictions: all S = 0 hybrids (7 multiplets per flavour) -> S = 0 quarkonia, all S = 1 members -> S = 1 quarkonia,
  with the split `Gamma = Gamma_PiPi + Gamma_SigmaSigma + Gamma_int` (Eq. eq:Gamma_split).
* Comparisons: J&S model (arXiv:2610.22088) at its kinematics, flux tube, GYS (Guo:2014zva), N_f = 2 eta_c1 mass
  (Shi:2023sdy), BESIII limits on psi(4040), psi(4415) -> gamma eta_c.

## How to run

```
# 1. (only if the wave functions change) MATLAB or octave-cli, from EM_Pheno/matlab
exportWavefunctions('<path>/Ruben Mixing/QQbarSpectrum', '../data')
# 2.-5. from EM_Pheno/python
python3 run_all.py              # quarkonium, ~15 s  -> results/*.csv, summary.json
python3 run_hybrids.py          # hybrids,    ~3 s   -> results/hybrid_*  (reads summary.json)
python3 make_tables.py          # quarkonium tables  -> TeX file/tex_pheno/tab_*.tex
python3 make_hybrid_tables.py   # hybrid tables      -> TeX file/tex_pheno/tab_H_*.tex
python3 make_figures.py         # hybrid figures     -> TeX file/tex_pheno/fig_H_*.pdf and figures/
python3 ../tests/test_e1.py     # 142 checks, must end with ALL PASS
python3 ../tests/test_hybrid.py # 149 checks, must end with ALL PASS
# 6. from "TeX file/tex_pheno" (needs ../mainEM.aux: compile mainEM once)
pdflatex pheno_standalone && bibtex pheno_standalone && pdflatex pheno_standalone && pdflatex pheno_standalone
```

The hybrid part is a separate script because it only needs `kappa_Q` from the quarkonium fit (read from
`results/summary.json`); it can be rerun alone when only the hybrid inputs change.

Requirements: Python 3 with `numpy`, `scipy` (>= 1.15, for `sph_harm_y`); `matplotlib` for the figures; `sympy`
for `test_e1.py` only. The output folder of the tables/figures is `../TeX file/tex_pheno` if it exists, otherwise
`$EMPHENO_TEX` or `EM_Pheno/tex`.

## Files

```
data/wf_c_nomix.mat, wf_b_nomix.mat   LO wave functions (QQbarSpectrum): Ql0..Ql3 and hybrids Hj0..Hj2
                                      (named Hjc0..Hjc2 in the current files, exported with v8; both are read)
matlab/exportWavefunctions.m          writes the two .mat files (v10: hybrid families restored)
python/emlib/angular.py               CG, 6j, spin factor, angular grid, Y_l^m, vector harmonics Y^l_JM
python/emlib/states.py                Library, State (quarkonium), HybridState, hybrid(), R_Pi/R_Sigma projections
python/emlib/data.py                  RPP 2026 E1 data
python/emlib/e1.py                    g_E1(r), E1 amplitudes, quadratic forms, delta_1m, theory error
python/emlib/fit.py                   quarkonium Observable and Fit
python/emlib/hybrid.py                interpolated LEFs g_X(r), hybrid vertex (local form), HybridTransition: 2x2
                                      forms in (B_Q, kappa_Q), Pi/Sigma/int split
python/emlib/hybrid_data.py           lattice data, hybrid multiplets and candidates, final states, other models,
                                      BESIII limits
python/emlib/hybrid_fit.py            HybridFit: chi^2(B_c), minima per branch, fits to Gamma_lat +- sigma_lat
python/run_all.py                     quarkonium results
python/run_hybrids.py                 hybrid results
python/make_tables.py                 quarkonium table bodies
python/make_hybrid_tables.py          hybrid table bodies
python/make_figures.py                fig_H_gamma_E.pdf, fig_H_LEFs.pdf
tests/test_e1.py, tests/test_hybrid.py
results/                              csv, json, logs
figures/                              pdf and png of the figures
```

## What `run_hybrids.py` computes

1. `hybrid_fit.csv`: fits of `B_c` (both minima, each with its fits to Gamma_lat +- sigma_lat): reference
   (eps = 0.3) and checks (eps = 0; lattice masses +-1 sigma; kappa_c of the max and min quarkonium fits).
   -> Table `tab:H_fit`.
2. `hybrid_lattice.csv`: widths at the J&S kinematics (Y, H1(0-+), eta_c1, H1(2-+)), with lattice and J&S values;
   Delta S = 1 channels are zero at our order. -> Table `tab:H_lattice`.
3. `hybrid_predictions.csv`: S = 0 predictions, charm and bottom: `G` with split and range `G_lo`, `G_hi` (three fits),
   other branch `G_alt`, `G_eQ` (bottom with B_Q proportional to e_Q), BR. -> Tables `tab:H_pred_c`, `tab:H_pred_b`.
4. `hybrid_spin1.csv`: S = 1 members. -> Table `tab:H_spin1`.
5. `hybrid_besiii.csv`: psi(4040), psi(4415) -> gamma eta_c vs BESIII. -> Table `tab:H_besiii`.
6. `hybrid_curves.csv`, `hybrid_summary.json`: figure data (Gamma(E_q), g_B(r)) and numbers quoted in the text.

## Tests (`tests/test_hybrid.py`, 149 checks)

Closed formula (eq:lab_Gamma) with five different arbitrary LEFs for all kinds of hybrids and l' = 0..3 (total and
split); selection rules (incl. G2 = 0 for the interpolated LEFs); P_Pi + P_Sigma = 1; heavy-quark spin sum rules;
NotImplementedError for the non-local form; quadratic form vs direct width; regression against v8/v10 (pure EST
LEFs, c = e e_c, local); limits of the interpolated LEFs and flavour relation; the fit (stationarity, branches,
sigma_th = 2 eps Gamma).

## Changes with respect to v9

Added: `emlib/hybrid.py`, `hybrid_data.py`, `hybrid_fit.py`, `run_hybrids.py`, `make_hybrid_tables.py`,
`make_figures.py`, `tests/test_hybrid.py`; `angular.py` (vector harmonics), `states.py` (hybrid states, mesh
`h`, `lambda`, `Library.u`), `exportWavefunctions.m` (hybrid families). Quarkonium code and results unchanged.
Not included (still only in v8): quarkonium-hybrid mixing, hybrid -> hybrid transitions.
