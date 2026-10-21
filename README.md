# EM transitions — phenomenology code, version 2 (Σu⁻)

**EM_Pheno_v2** = EM_Pheno + the Σu⁻ component of the hybrid vertex of the v10
notes (Sec. "Hybrids: the Σu⁻ component", Eqs. eq:gSigma, eq:rhoSigma,
eq:Lfull, eq:Gfull) + comparison with lattice QCD (arXiv:2301.08213).
EM_Pheno (v1) is kept unchanged; `sigma_mode='off'` in v2 reproduces it.

What is new in v2:

* `emlib/amplitudes.py`: longitudinal vertex h (r̂·ε*)(r̂·ψ)(g_B∥ + E g_E^Σ),
  g_B∥ = 4c4/(σr), g_E^Σ = −4c3/(3πσ); non-local form factors of the Σ
  densities (f, F3 = 9π²cos(x/2)/(9π²−x²)); couplings kH = (c1,c2,c3,c4);
  `sigma_mode` = 'EST' (notes) | 'norm' (g_B∥ rescaled to g_B⊥, r→0 condition)
  | 'off'; variant B uses dVS = V_Σu − V_Σg for the Σ gap.
* `matlab/exportWavefunctions.m`: also exports `dVSig` (wave functions identical).
* `emlib/hybrids.py`: 4×4 quadratic forms, p0 predictions, Σ-mode columns.
* `emlib/mixing.py`: Σ modes, `sigma_scan` (c4 scan of V → γη_c).
* `emlib/lattice.py`: charge radii, Γ(χc0→J/ψγ), SHO β² vs lattice.
* `python/r_moments.py`: sizes, Σ/Π weights, counting parameters.
* `tests/test_engine.py`: tests 10–14 (Σ vertex vs Eq. eq:Gfull, p0 width,
  non-local Σ, FT of ρ_Σ, spin-1 with Σ). 193 checks, ALL PASS.
* `make_tables.py` writes to `$EMPHENO_TEX` if set.

----

# EM transitions — phenomenology code

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
% in MATLAB (or octave-cli), from Pheno/matlab
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
  1+- hybrid vertex of Eq. (110) of the notes, local or non-local
  (j0(q_z r/2) and π²cos(q_z r/2)/(π²−q_z²r²)), with gap π/r (variant A, as
  matched) or ΔV(r) (variant B). Heavy-quark spin is a spectator; states can
  have any number of components (quarkonium/hybrid, Σ/Π channels, spin 0/1),
  so mixed states and spin-1 multiplets need no extra angular algebra.
  Validated against Eq. (106), the 6j spin-1 E1 formula, the hybrid width
  Eq. (140) (also for mixed Σ/Π states with R_Π of Eq. (146), eq:RPi_eff),
  sum rules, and the local limit of the non-local forms.
* **Data** (`emlib/data.py`): RPP 2026 E1 inputs, with correlations through
  shared total widths, asymmetric errors and 90% CL limits; BESIII
  e+e−→γη_c, γχ_cJ, γX(3872) cross sections.
* **Fit** (`emlib/quarkonium.py`): χ² with experimental covariance, limits as
  half-normal likelihoods, amplitude-level theory error
  σ_th² = 4ε²ΓΓ̄ + 2ε⁴Γ̄² (Γ̄: same width with |u_f u_i|), ε fixed by
  χ²/ndf = 1. Parameter κ_Q: g_E1 = κ_Q e e_Q r.
* **Hybrids** (`emlib/hybrids.py`), **mixing** (`emlib/mixing.py`):
  predictions, κ_H limits from BESIII γη_c, quadratic forms in (c1,c2,c3).

## Main results (October 2026)

| quantity | value |
|---|---|
| κ_b (g_E1 = κ e e_b r) | 0.998 ± 0.040 (ε = 4%) |
| κ_c | 0.856 +0.031 −0.028 (core), 0.828 with ψ(3770), ψ2(3823) |
| hybrid LEFs | c1 = c2 = c3 = κ_H e e_Q (EST); data: κ_H < 0.6–0.8 (variant B), < 0.23 (variant A) |
| g_B⊥ only | excluded by BESIII γη_c limits (factor 6–250 in width) |
| X(3872) c c̄ probability | Z ≈ 0.44 ± 0.19 (from γJ/ψ), Z ≲ 0.3 (from ψ(4160)→γX) |

See `../TeX file/tex_pheno/pheno_standalone.pdf` for the discussion. A detailed description of every file (QQbarSpectrum summary, exporter,
`emlib` modules, scripts, tests and libraries) is the appendix
`../TeX file/tex_pheno/app_codes.tex`. Equation numbers refer to EM_transitions_v9.
