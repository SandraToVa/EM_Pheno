# EM transitions — phenomenology code, v7

**EM_Pheno_v7**: quarkonium E1 fits at LO, with the 1/m corrections, with the 1/m^2 term and with
the interpolated g_E1(r); hybrid predictions with the string gaps of the matching (pi/r, 3pi/r)
in the non-local (NL) and local (L) forms. One run script, one table script.

## How to run

```matlab
% 1. MATLAB or octave-cli, from EM_Pheno_v7/matlab (only if data/ has to be regenerated)
exportWavefunctions('<path>/Ruben Mixing/QQbarSpectrum', '../data')
```
```bash
# 2-6. from EM_Pheno_v7/python
python3 run_all.py            # everything (about 2 min); python3 run_all.py --e1: quarkonium E1 only (15 s)
python3 r_moments.py          # sizes and counting parameters
python3 make_tables.py        # all tab_*.tex, written to ../TeX file/tex_pheno (or $EMPHENO_TEX)
python3 figures.py            # figures/*.pdf|png
python3 ../tests/test_engine.py   # 269 checks, must end with ALL PASS
```

## What each script does

| script | does | writes |
|---|---|---|
| `run_all.py` | 1. E1 fits with constant dkappa_Q for corr = LO, 1m, 1m2 | `results/quarkonium_fits.csv`, `quarkonium_pulls.csv` |
| | 2. E1 fits with the interpolated g_E1(r) + 1/m (corr = 1m) | `interp_fits.csv`, `interp_pulls.csv`, `interp_f_of_r.csv` |
| | 3. quarkonium widths versus lattice QCD (all fits above) | `quarkonium_lattice.csv` |
| | 4.-7. hybrids, mixing, hybrid lattice comparison, external models | `hybrid_*.csv`, `mixing_*.csv`, `lattice_*.csv`, `c4_scan.csv`, `gamma_vs_Eq.csv`, `summary.json` |
| `make_tables.py` | all the LaTeX tables (no hand-copied numbers) | `tab_*.tex` (19 tables) |
| `figures.py` | figures | `fig_E1_ratios`, `fig_hybrid_integrand`, `fig_LEFs_B`, `fig_lattice_hybrids` |
| `r_moments.py` | sizes, P_Pi, P_Sigma, E_q r | `results/r_moments_*.csv` |
| `tests/test_engine.py` | checks of the amplitude engine and of the E1 corrections | `results/test_engine.log` |

The relativistic corrections are one option, `corr`, of `quarkonium.rel_delta`, `predict_row`,
`Fitter` (and `interp.predict_parts`, `InterpFitter`):

| corr | delta of the amplitude, A -> A (1 + delta) | tables (label in sec_pheno.tex) | eps check |
|---|---|---|---|
| `'LO'` | 0 | `tab_kappa`, `tab_E1_fit` (tab:kappa, tab:E1) | v^2 (0.3, 0.1) |
| `'1m'` | EGTKLY O(k/m): k/(8m) (s [J_P(J_P+1) - 4] - Delta l/3) | `tab_kappa_1m`, `tab_E1_1m` (tab:kappa_1m, tab:E1_1m) | v^3 (0.164, 0.032) |
| `'1m2'` | 1m + c_S^em 1/m^2 term (cancels the spin term): -(Delta l/3) k/(8m) | `tab_kappa_1m2`, `tab_E1_1m2` (tab:kappa_1m2, tab:E1_1m2) | v^3 |
| interp | g_E1 = e r [e_Q + kappa_Q f(r)], f = Hill n = 2, 7, r0 = 1.58 GeV^-1, with `'1m'` | `tab_kappa_interp`, `tab_E1_interp` | v^3 |

Comparison with lattice QCD: `tab_lattice_chic0`, `tab_quarkonium_lattice` (all the fits above).
The hybrid normalization c_i = dkappa_Q e e_Q uses the LO fit.

## Changes with respect to the previous v7 (5 Nov 2026)

* **The gap option is removed.** The hybrid vertex always uses the string gaps of the matching,
  Delta_Pi = pi/r, Delta_Sigma = 3 pi/r. `Params` has no `gap`, `dV`, `dVS`; `hybrids.VARIANTS` is
  {'NL', 'L'}; `load_dV`, `load_dVS`, `load_gaps` and every `gaps` argument
  (`gamma_split`, `gamma`, `quadratic_form`, `params`, `hybrid_lattice_table`, `gamma_vs_Eq`) are
  gone; `fig_potential_gaps` and `tab_potential_gaps` are gone; `exportWavefunctions.m` no longer
  exports `dV`, `dVSig`. (The existing `data/*.mat` still contain these two fields from the old
  export; nothing reads them, and they disappear the next time the export is run.)
  All hybrid, mixing and lattice-hybrid results are identical bit for bit to the NL/L results before.
* **One option `corr` for the E1 corrections** (`'LO'`, `'1m'`, `'1m2'`) replaces `rel=True/False`
  and the module flag `SPIN`. Interpolated fits use `'1m'` (spin + Delta l), as in v6.
* **epsilon = v^3** (0.3^1.5 = 0.164 charm, 0.1^1.5 = 0.032 bottom) instead of v^2 for the second
  theory-error option of the 1m, 1m2 and interpolated fits (columns `eps_alt`, `kappa_eps_alt`, ...).
* **One run script and one table script**: `run_rel.py`, `run_interp.py` merged into `run_all.py`;
  `make_tables_v7.py`, `make_tables_interp.py` merged into `make_tables.py`. New table names
  (`tab_*_1m`, `tab_*_1m2`, `tab_*_interp`, `tab_lattice_chic0`, `tab_quarkonium_lattice`); the old
  `_v6`, `_v7`, `_rel` table files in tex_pheno are no longer written or used.
* `quarkonium_external.csv` and `interp_lattice.csv` are replaced by `quarkonium_lattice.csv`.
* Test 7 now checks the E1 corrections and the interpolation (f = 1 reproduces (1 + dkappa)^2 LO for
  each corr; 1m - 1m2 = spin term). 269 checks, ALL PASS.
* `app_codes.tex` (code appendix) rewritten for this version.

## Main results (non-local, core)

The results use epsilon = v^2 (LO) or v^3 (1m, 1m2, interp) as theory error; the epsilon that gives
chi2/ndf = 1 is only reported in the fit tables (tab_kappa*, as a goodness measure).

| | epsilon | dkappa_c or kappa_c | chi2/ndf | chi2/ndf at eps = 0 | bottom |
|---|---|---|---|---|---|
| LO | v^2 = 0.3 | dkappa_c = -0.149 +0.134 -0.092 | 0.4/6 | 161/6 | dkappa_b = +0.002 +0.062 -0.057 (2.9/12) |
| 1m (spin + Delta l) | v^3 = 0.164 | -0.146 +0.066 -0.055 | 0.8/6 | 72/6 | +0.002 +- 0.038 (17/12) |
| 1m2 (Delta l only) | v^3 | -0.138 +0.067 -0.056 | 1.3/6 | 177/6 | -0.004 +- 0.038 (17/12) |
| interp n = 2 + 1m | v^3 | kappa_c = -0.117 +0.055 -0.045 | 1.0/6 | 76/6 | kappa_b = +0.016 (16/12) |
| interp n = 7 + 1m | v^3 | kappa_c = -0.100 +0.047 -0.038 | 1.0/6 | 74/6 | +0.012 (16/12) |

The improvement of 1m comes from the c_F^em spin term, which is cancelled by the 1/m^2 c_S^em
term (1m2). With the spin term, the interpolation does not improve charmonium.
Since 6 Nov 2026 the pulls, the lattice comparison, fig_E1_ratios and the hybrid normalization
(c_i = dkappa_Q e e_Q, dkappa_c = -0.149) use the epsilon = v^2 (v^3) fits.

----

## Earlier history

### v7 (first version, 4 Nov 2026): 1/m corrections, Delta l term only

The c_F^em spin-orbit term of the O(k/m) correction is cancelled by the 1/m^2 c_S^em spin-orbit
operator of Brambilla, Pietrulewicz, Vairo (2012), Eq. (86): with the LO Schroedinger equation,
Eq. (119), (I2' + 2 I1)/(m^2 I3) = +-k_LO/(2m) (checked with our wave functions to three digits),
so the two spin terms add up to kappa_Q^em k/(2m)[J(J+1)/2 - 2] (their Eq. (122)). This is now
`corr='1m2'`.

### v6: interpolated g_E1

**EM_Pheno_v6_interp** = EM_Pheno_v5 + the quarkonium E1 fit with the interpolated LEF of
Sec. "Interpolation of g_E1(r)" of the notes (label sec:pheno_E1_interpol):

    g_E1(r) = e r [e_Q + kappa_Q f(r)],   f(r) = (r/r0)^n / (1 + (r/r0)^n),

with the known O(k/m_Q) ("1/m") corrections of EGTKLY applied to the whole amplitude at all r.
kappa_Q (end-point charge in units of e) is fitted separately for charm and bottom, with the
three theory-error options of v5 (epsilon from chi2/ndf = 1, epsilon = v^2, epsilon = 0).
Baseline n = 2, variation n = 7, r0 = sqrt(pi/(6 sigma)) = 1.58 GeV^-1 (0.31 fm); checks:
r0 = 1.0 GeV^-1 and n = 50 (step limit). EM_Pheno_v5 is kept unchanged.

Changes of v6 with respect to v5:

* `python/emlib/amplitudes.py`: one change. `Params.fE1` (optional array on the mesh) multiplies
  the E1 radial weight, g_E1 = kappa_Q e e_Q r fE1(r). Default None = LO dipole, so every v5
  result is unchanged (tests/test_engine.py: ALL PASS, see results/test_engine.log).
* `python/emlib/interp.py`: `f_hill`, `predict_parts` (each width as the quadratic form
  G00 + 2 lambda G0f + lambda^2 Gff, lambda = kappa_Q/e_Q, plus the |u_f u_i| version for the
  theory error), `InterpFitter` (same chi^2 and theory error as `quarkonium.Fitter`; global
  minimum on a grid, Delta chi2 = 1 errors). Check: with f = 1 it reproduces the v5 rel=True
  fits exactly (kappa_Q = e_Q dkappa_Q).
* `python/run_interp.py` -> `results/interp_fits.csv`, `interp_pulls.csv`, `interp_lattice.csv`,
  `interp_f_of_r.csv`, `interp_summary.json` (log: `results/run_interp.log`).
* `python/make_tables_interp.py` -> in `../TeX file/tex_pheno/`: `tab_kappa_interp.tex`,
  `tab_E1_interp.tex`, `tab_lattice_chic0_v6.tex`, `tab_quarkonium_lattice_v6.tex`
  (the v5 tables are not touched).

(In the current version these scripts are part of `run_all.py` and `make_tables.py`.)

Main results of v6 (non-local, core, epsilon from chi2/ndf = 1, epsilon = v^2 as second option):

| | n = 2 | n = 7 |
|---|---|---|
| kappa_c | -0.114 +0.021 -0.020 (eps 0.060) | -0.098 +0.017 -0.016 (eps 0.058) |
| kappa_b | +0.012 +- 0.016 (eps 0.040) | +0.011 +- 0.012 (eps 0.039) |
| kappa_c - kappa_b | -0.126 +- 0.026 | -0.109 +- 0.021 |

kappa_Q is not flavour universal: the charmonium value is dominated by flavour-dependent
corrections, not by screening. Shape systematic about 0.02 (charm), <= 0.005 (bottom).
The hybrid normalization still uses dkappa_Q of v5 (Table tab:kappa).

----

### v3: one hybrid vertex with two channels (inherited by all later versions)

**EM_Pheno_v3** (and every later version) treats the hybrid vertex as one object with two channels on the
same footing, as in the lab-basis notes (Secs. "Hybrid Amplitude" and "Hybrid
Decay Width", Eqs. eq:lab_vertex, eq:lab_M, eq:lab_Gamma_c):

* **Π_u channel**: transverse operators g_B⊥ (c2), g_E∥ (c3), g_E⊥ (c1); gap Δ_Π.
* **Σ_u⁻ channel**: longitudinal operators g_B∥ (c4), g_E^Σ (c3); gap Δ_Σ.

A hybrid state is a solution of the coupled Π–Σ equations; it is never labelled
"Π" or "Σ". What the photon sees are its projections R_Π and R_Σ
(`states.pi_sigma_projections`, Eqs. eq:lab_RPi, eq:lab_RSigma). Every hybrid
width is returned with its exact split

    Γ = Γ_ΠΠ + Γ_ΣΣ + Γ_int            (amplitudes.width_split)

so channel strengths κ_Π, κ_Σ enter as Γ = κ_Π² Γ_ΠΠ + κ_Σ² Γ_ΣΣ + κ_Πκ_Σ Γ_int.
There is no on/off switch for Σ.

Changes of v3 with respect to v2 (v2 is kept unchanged as a reference):

* `amplitudes.py`: the vertex is a table of terms (`HQ_TERMS`) grouped by channel;
  one gap rule for both channels (`Params.gaps`; the alternative gaps from the potentials of v3–v6 were removed in v7);
  `channel_amplitudes`, `width_split`. `sigma_mode` ('EST'/'norm'/'off') removed:
  the LEFs are those of Table tab:HQ_lef, with the phase convention of the notes
  (Sec. sec:phase_conv, |H⁰⟩ ↔ −i|Σu⁻⟩): all real, g_B∥ = −4c4/(σr) and g_B⊥ of opposite sign for c4 = c2.
* `states.py`: `pi_sigma_projections` (R_Π, R_Σ, P_Π, P_Σ); notation of the notes
  (see "Angular-momentum notation" below).
* `hybrids.py`: `gamma_split`, `channel_bounds` (κ_H, κ_Π, κ_Σ limits),
  Π/Σ/int columns in the predictions, magnetic-only = (c2, c4).
* `mixing.py`: split columns; `c4_scan` (was `sigma_scan`).
* `lattice.py`: split columns; lattice ratio R for each channel.
* `run_all.py`: new `hybrid_channel_plane.csv` (replaces `hybrid_sigma_plane.csv`),
  `c4_scan.csv` (replaces `sigma_scan.csv`), c4 windows in `summary.json`.
* `make_tables.py`, `figures.py`: tables and figures in the channel language;
  new `fig_channel_plane`, `fig_c4_scan` (replaces `fig_sigma_scan`).
* `tests/test_engine.py`: every hybrid check is done channel by channel against the
  lab-basis formula Eq. eq:lab_Gamma_c; selection rules of the two channels;
  248 checks, ALL PASS.
* The numbers of v2 with `sigma_mode='EST'` are reproduced exactly (totals);
  the old `'off'` numbers are now the Γ_ΠΠ columns.

### Angular-momentum notation (as in the notes, App. "Spin 1 treatment")

* Quarkonium |(l S) J M⟩; hybrids |(J S) 𝒥 ℳ⟩ with J = l ⊗ 1_gluon. For S = 0, J = l
  (quarkonium) and 𝒥 = J (hybrids).
* Families in `data/*.mat`: `Ql0`–`Ql3` (from QQbarS0J0–3) and `Hj0`–`Hj2` (from
  GQQbarS0Jcal0–2, S = 0 hybrids labelled by J). Files exported before 22 Oct 2026
  call them `Hjc0`–`Hjc2`; `Library` renames them when reading.
* `Library.hybrid(J, k, S=0, Jcal=None, mass)`: k-th state of `Hj<J>`, coupled with
  heavy-quark spin S to total 𝒥 = `Jcal` (= J for S = 0). S = 1 hybrids are always built
  this way from the S = 0 states (heavy-quark spin symmetry); the GQQbarS1Jcal files
  (hyperfine terms) are not used. `hybrid_index(J, kind, n)`.
* `Component(kind, l, J0, S, c)`: `J0` = angular momentum of the component without the
  heavy-quark spin (the one that couples with S): `J0 = l` for quarkonium, `J0 = J` for
  hybrids. `State(name, J, comps)`: `J` = total angular momentum (J of quarkonium, 𝒥 of
  hybrids).
* `hybrids.gamma(…, S, Jcal, Jf)`: `Jcal` = 𝒥 of the initial hybrid, `Jf` = J′ of the
  final quarkonium. `results/hybrid_spin1.csv`: columns `Jcal_i`, `J_f`.
* The renaming of 22 Oct 2026 (Jc → J, Hjc → Hj, Lc → J0, hybrid(…, J=) → hybrid(…, Jcal=))
  changed no number: all results, tables and figures are identical bit for bit.

----

## General description of the code

Phenomenology of one-photon transitions in the BOEFT (quarkonium E1 and
hybrid → quarkonium + γ) for the EM-transitions paper. The `QQbarSpectrum`
folder is only *read* (added to the MATLAB path); nothing there is modified.

### Pipeline

```
matlab/exportWavefunctions.m   QQbarSpectrum  ->  data/wf_*.mat   (MATLAB or Octave)
python/run_all.py              fits + predictions -> results/*.csv, results/summary.json
python/make_tables.py          results -> ../TeX file/tex_pheno/tab_*.tex (or $EMPHENO_TEX, or tex/)
python/r_moments.py            sizes, Σ/Π weights -> results/r_moments_*.csv
python/figures.py              results -> figures/*.pdf|png
tests/test_engine.py           checks of the amplitude engine (must print ALL PASS)
../TeX file/tex_pheno/sec_pheno.tex         the phenomenology section (\input in mainEM.tex)
../TeX file/tex_pheno/pheno_standalone.tex  standalone compilation of the section (+ pheno.bib)
```

Run:

```matlab
% in MATLAB (or octave-cli), from EM_Pheno_v7/matlab
exportWavefunctions('<path>/Ruben Mixing/QQbarSpectrum', '../data')
```
```bash
cd python
python3 run_all.py && python3 make_tables.py && python3 figures.py
python3 ../tests/test_engine.py
```
Python ≥ 3.10 with numpy, scipy (≥ 1.15, `sph_harm_y`), pandas, matplotlib;
the tests also use sympy.

### What the code does

* **Wave functions** (`exportWavefunctions.m`): takes the potential handle of
  each QQbarSpectrum file (`'potential'` option) and solves it with
  QQbarSpectrum's own `solveRadialSchrodinger` (Lagrange–Laguerre mesh,
  N = 100, rmax = 30 GeV⁻¹; converged: N = 140, rmax = 40 changes nothing at the
  10⁻⁵ level). Exports the mesh coefficients `c` so that
  ∫ f(r) u_a u_b dr = Σ_i c_a,i c_b,i f(r_i). Quarkonium l = 0–3 comes from QQbarS0J0–QQbarS0J3. Charm m = 1.496 GeV, bottom m = 4.88 GeV;
  mixing (mix = true) only for charm (the code switches cF off otherwise).
* **Amplitude engine** (`emlib/amplitudes.py`): generic numerical angular
  integration of M = ∫d³r ψ_B† O_h ψ_A for E1 (g_E1 = κ e e_Q r) and for the
  1+- hybrid vertex of Eq. eq:lab_vertex (Π and Σ channels), local or non-local
  (j0(q_z r/2); f = π²cos(q_z r/2)/(π²−q_z²r²) and F3 for the Σ densities), with
  the string gaps π/r, 3π/r of the matching (variants NL, L). Heavy-quark spin is a spectator; states can
  have any number of components (quarkonium/hybrid, Σ/Π channels, spin 0/1),
  so mixed states and spin-1 multiplets need no extra angular algebra.
  Validated against the quarkonium E1 widths, the 6j spin-1 formulas, the
  lab-basis hybrid width eq:lab_Gamma_c channel by channel, the non-local
  multipole formulas, sum rules and the local limit of the non-local forms.
* **Data** (`emlib/data.py`): RPP 2026 E1 inputs, with correlations through
  shared total widths, asymmetric errors and 90% CL limits; BESIII
  e+e−→γη_c, γχ_cJ, γX(3872) cross sections.
* **Fit** (`emlib/quarkonium.py`): χ² with experimental covariance, limits as
  half-normal likelihoods, amplitude-level theory error
  σ_th² = 4ε²ΓΓ̄ + 2ε⁴Γ̄² (Γ̄: same width with |u_f u_i|), ε fixed by
  χ²/ndf = 1. Parameter κ_Q = 1 + δκ_Q: g_E1 = κ_Q e e_Q r, with the relativistic corrections `corr`.
* **Interpolated g_E1** (`emlib/interp.py`): g_E1 = e r [e_Q + κ̄_Q f(r)], quadratic form in κ̄_Q/e_Q.
* **Hybrids** (`emlib/hybrids.py`), **mixing** (`emlib/mixing.py`):
  predictions with the Π/Σ/interference split, κ_H, κ_Π, κ_Σ limits from BESIII
  γη_c, quadratic forms in (c1,c2,c3,c4), c4 scans.

### v5 (October 28, 2026): relativistic corrections to E1

`emlib/quarkonium.py`: `rel_delta(init, final, k)` = O(k/m_Q) correction of the E1 amplitude
from EGTKLY (Eichten et al. 1980; App. E of the notes),
delta = k/(8 m_Q) (s [J_P(J_P+1) - 4] - Delta_l/3), s = -1 for 3P_J -> 3S_1, +1 for 3S_1 -> 3P_J,
Delta_l = l_i(l_i+1) - l_f(l_f+1); spin term only for triplet S <-> P (now `corr='1m'`). Switch (v5–v6): `rel=True` in
`predict_row` / `Fitter` (default False). `run_all.py` fits both (column `rel`); the
hybrid normalization keeps using the rel=False fits. New tables: `tab_E1_rel`, `tab_kappa_rel`;
`tab_E1_fit` now also has the epsilon=0 predictions and pulls.
Result (charm core, NL): chi2/ndf at epsilon=0 161/6 -> 72/6, epsilon 0.073 -> 0.054,
dkappa_c -0.144 -> -0.143; bottom unchanged (hindered transitions dominate).

### v4 (October 27, 2026): κ_Q = 1 + δκ_Q

The E1 normalization is fitted as κ_Q = 1 + δκ_Q (LO = 1, all corrections in δκ_Q), and the
hybrid vertex, a correction computed with our model, is normalized with δκ_Q alone:
c_i = δκ_Q e e_Q (v3 used κ_Q). Hybrid widths are stored at κ_H = 1 and at δκ_Q
(`*_at_dk` columns of `hybrid_predictions.csv`); everything compared with BESIII, lattice and
the thesis uses δκ_Q. ψ(3770) → γη_c limit updated to BESIII arXiv:2609.33606 (< 5.5 keV).
v3 is kept unchanged in `../EM_Pheno_v3`.

### Main results (v4, gaps π/r, 3π/r, non-local form)

| quantity | value |
|---|---|
| δκ_b | −0.002 ± 0.040 (ε = 4%): hybrid radiative widths of bottomonium vanish within errors |
| δκ_c | −0.144 +0.031 −0.028 (core, ε = 7%); −0.172 with ψ(3770), ψ2(3823) |
| ψ(4040) → γη_c at κ_H = δκ_c | 219 keV (Γ_ΠΠ 13, Γ_ΣΣ 341, Γ_int −135), 6.6 × BESIII limit; |κ_H| < 0.06 |
| c4/c at c = δκ_c | 0.08–0.59 (BESIII); 0.08–0.31 with J/ψ, ψ(2S), ψ(3770) → γη_c (mixing appendix) |
| lattice R = Γ(η_c1)/Γ(Y) | Π channel alone 1.61, both 0.78 (lattice 1.78 ± 0.23); 1.0–2.6 for c4/c = 0.1–0.25 |
| lattice widths | need κ_Π ≈ 0.46–0.94, i.e. 3–6.5 × |δκ_c| |

See `../TeX file/tex_pheno/pheno_standalone.pdf` for the discussion. A detailed description of every file (QQbarSpectrum summary, exporter,
`emlib` modules, scripts, tests and libraries) is the appendix
`../TeX file/tex_pheno/app_codes.tex`. Equation labels refer to the current notes (mainEM.tex).
