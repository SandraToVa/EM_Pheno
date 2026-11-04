# EM transitions — phenomenology code, v7 (1/m corrections: Delta l term only)

**EM_Pheno_v7** = EM_Pheno_v6_interp with the O(k/m_Q) correction of the E1 amplitude reduced to its
spin-independent part. The c_F^em spin-orbit term (included in v5 and v6 through the EGTKLY delta) is
cancelled by the 1/m^2 c_S^em spin-orbit operator of Brambilla, Pietrulewicz, Vairo (2012), Eq. (86):
with the LO Schroedinger equation, Eq. (119), (I2' + 2 I1)/(m^2 I3) = +-k_LO/(2m), so the two spin
terms add up to kappa_Q^em k/(2m)[J(J+1)/2 - 2] (their Eq. (122)), beyond our accuracy. Checked with
our wave functions: the ratio equals k_LO/(2m) to three digits.

    delta_1/m = -(Delta l / 3) k/(8 m_Q),   Delta l = l_i(l_i+1) - l_f(l_f+1)
              = -k/(12 m) for P -> S, +k/(12 m) for S -> P

## Changes with respect to v6

* `python/emlib/quarkonium.py`: `SPIN = False` (module flag) and `rel_delta(init, final, E, spin=None)`.
  `spin=True` (or `SPIN = True`) gives back the EGTKLY delta of v5/v6 (checked: the v5 rel=True fits
  are reproduced exactly).
* `python/run_rel.py` (new): fits with constant dkappa_Q and the 1/m correction (Tables tab:kappa_rel,
  tab:E1_rel), both without (v7) and with (reference) the spin term -> `results/rel_fits.csv`,
  `rel_pulls.csv`, `rel_summary.json`, log `run_rel.log`.
* `python/run_interp.py`: unchanged except that it now uses the Delta l correction; the lattice
  comparison uses the constant-dkappa fit with the same correction -> `results/interp_*.csv`.
* `python/make_tables_v7.py` (replaces make_tables_interp.py) -> in `../TeX file/tex_pheno/`:
  `tab_kappa_rel_v7.tex`, `tab_E1_rel_v7.tex`, `tab_kappa_interp_v7.tex`, `tab_E1_interp_v7.tex`,
  `tab_lattice_chic0_v7.tex`, `tab_quarkonium_lattice_v7.tex`. Earlier tables are not touched.

Run, from `python/`: `python3 run_rel.py && python3 run_interp.py && python3 make_tables_v7.py`.

## Main results (non-local, core)

| | eps (chi2/ndf=1) | dkappa or kappa | chi2/ndf at eps = 0 |
|---|---|---|---|
| charm, LO (no 1/m) | 0.073 | dkappa = -0.144 | 161/6 |
| charm, 1/m with spin term (v5/v6) | 0.054 | -0.143 | 72/6 |
| charm, 1/m Delta l only (v7) | 0.072 | dkappa = -0.136 | 177/6 |
| charm, interpolated n = 2 (v7) | 0.073 | kappa_c = -0.111 | 164/6 |
| bottom, 1/m Delta l only (v7) | 0.041 | dkappa = -0.004 | 105/12 |
| bottom, interpolated n = 2 (v7) | 0.039 | kappa_b = +0.015 | 73/12 |

The improvement of v5 came from the spin term; once it is cancelled, the J dependence of the data
has to come from the spin-dependent wave-function corrections R(J) (not included).

----

## v6 (inherited): interpolated g_E1

**EM_Pheno_v6_interp** = EM_Pheno_v5 + the quarkonium E1 fit with the interpolated LEF of
Sec. "Interpolation of g_E1(r)" of the notes (label sec:pheno_E1_interpol):

    g_E1(r) = e r [e_Q + kappa_Q f(r)],   f(r) = (r/r0)^n / (1 + (r/r0)^n),

with the known O(k/m_Q) ("1/m") corrections of EGTKLY applied to the whole amplitude at all r.
kappa_Q (end-point charge in units of e) is fitted separately for charm and bottom, with the
three theory-error options of v5 (epsilon from chi2/ndf = 1, epsilon = v^2, epsilon = 0).
Baseline n = 2, variation n = 7, r0 = sqrt(pi/(6 sigma)) = 1.58 GeV^-1 (0.31 fm); checks:
r0 = 1.0 GeV^-1 and n = 50 (step limit). EM_Pheno_v5 is kept unchanged.

### Changes of v6 with respect to v5

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

Run, from `python/`: `python3 run_interp.py && python3 make_tables_interp.py` (about 15 s).

## Main results (non-local, core, epsilon from chi2/ndf = 1)

| | n = 2 | n = 7 |
|---|---|---|
| kappa_c | -0.114 +0.021 -0.020 (eps 0.060) | -0.098 +0.017 -0.016 (eps 0.058) |
| kappa_b | +0.012 +- 0.016 (eps 0.040) | +0.011 +- 0.012 (eps 0.039) |
| kappa_c - kappa_b | -0.126 +- 0.026 | -0.109 +- 0.021 |

kappa_Q is not flavour universal: the charmonium value is dominated by flavour-dependent
corrections, not by screening. Shape systematic about 0.02 (charm), <= 0.005 (bottom).
The hybrid normalization still uses dkappa_Q of v5 (Table tab:kappa).

----

## v3: one hybrid vertex with two channels (inherited by all later versions)

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
  one gap rule for both channels (`Params.gaps`: π/r, 3π/r in A; ΔV, ΔV_Σ in B);
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

## Angular-momentum notation (as in the notes, App. "Spin 1 treatment")

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
----

## General description of the code (from v1)

Phenomenology of one-photon transitions in the BOEFT (quarkonium E1 and
hybrid → quarkonium + γ) for the EM-transitions paper. The `QQbarSpectrum`
folder is only *read* (added to the MATLAB path); nothing there is modified.

## Pipeline

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
% in MATLAB (or octave-cli), from EM_Pheno_v4/matlab
exportWavefunctions('<path>/Ruben Mixing/QQbarSpectrum', '../data')
```
```bash
cd python
python3 run_all.py && python3 make_tables.py && python3 figures.py
python3 ../tests/test_engine.py
```
Python ≥ 3.10 with numpy, scipy (≥ 1.15, `sph_harm_y`), pandas, matplotlib;
the tests also use sympy.

## What the code does

* **Wave functions** (`exportWavefunctions.m`): takes the potential handle of
  each QQbarSpectrum file (`'potential'` option) and solves it with
  QQbarSpectrum's own `solveRadialSchrodinger` (Lagrange–Laguerre mesh,
  N = 100, rmax = 30 GeV⁻¹; converged: N = 140, rmax = 40 changes nothing at the
  10⁻⁵ level). Exports the mesh coefficients `c` so that
  ∫ f(r) u_a u_b dr = Σ_i c_a,i c_b,i f(r_i). Quarkonium l = 0–3 comes from QQbarS0J0–QQbarS0J3. Also the static gap
  ΔV(r) = V_Πu − V_Σg on the mesh. Charm m = 1.496 GeV, bottom m = 4.88 GeV;
  mixing (mix = true) only for charm (the code switches cF off otherwise).
* **Amplitude engine** (`emlib/amplitudes.py`): generic numerical angular
  integration of M = ∫d³r ψ_B† O_h ψ_A for E1 (g_E1 = κ e e_Q r) and for the
  1+- hybrid vertex of Eq. eq:lab_vertex (Π and Σ channels), local or non-local
  (j0(q_z r/2); f = π²cos(q_z r/2)/(π²−q_z²r²) and F3 for the Σ densities), with
  gaps π/r, 3π/r (`gap='matched'`, variants NL, L: the results of Sec. 5) or ΔV(r), ΔV_Σ(r) (`gap='potential'`, variants pot-NL, pot-L: appendix "Gaps from the BOEFT static potentials"). Heavy-quark spin is a spectator; states can
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
  χ²/ndf = 1. Parameter κ_Q: g_E1 = κ_Q e e_Q r.
* **Hybrids** (`emlib/hybrids.py`), **mixing** (`emlib/mixing.py`):
  predictions with the Π/Σ/interference split, κ_H, κ_Π, κ_Σ limits from BESIII
  γη_c, quadratic forms in (c1,c2,c3,c4), c4 scans.

## v5 (October 28, 2026): relativistic corrections to E1

`emlib/quarkonium.py`: `rel_delta(init, final, k)` = O(k/m_Q) correction of the E1 amplitude
from EGTKLY (Eichten et al. 1980; App. E of the notes),
delta = k/(8 m_Q) (s [J_P(J_P+1) - 4] - Delta_l/3), s = -1 for 3P_J -> 3S_1, +1 for 3S_1 -> 3P_J,
Delta_l = l_i(l_i+1) - l_f(l_f+1); spin term only for triplet S <-> P. Switch: `rel=True` in
`predict_row` / `Fitter` (default False). `run_all.py` fits both (column `rel`); the
hybrid normalization keeps using the rel=False fits. New tables: `tab_E1_rel`, `tab_kappa_rel`;
`tab_E1_fit` now also has the epsilon=0 predictions and pulls.
Result (charm core, NL): chi2/ndf at epsilon=0 161/6 -> 72/6, epsilon 0.073 -> 0.054,
dkappa_c -0.144 -> -0.143; bottom unchanged (hindered transitions dominate).

## v4 (October 27, 2026): κ_Q = 1 + δκ_Q

The E1 normalization is fitted as κ_Q = 1 + δκ_Q (LO = 1, all corrections in δκ_Q), and the
hybrid vertex, a correction computed with our model, is normalized with δκ_Q alone:
c_i = δκ_Q e e_Q (v3 used κ_Q). Hybrid widths are stored at κ_H = 1 and at δκ_Q
(`*_at_dk` columns of `hybrid_predictions.csv`); everything compared with BESIII, lattice and
the thesis uses δκ_Q. ψ(3770) → γη_c limit updated to BESIII arXiv:2609.33606 (< 5.5 keV).
v3 is kept unchanged in `../EM_Pheno_v3`.

## Main results (v4, matched gaps π/r, 3π/r, non-local form)

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
