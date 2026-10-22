# EM transitions — phenomenology code, version 3

**EM_Pheno_v3** treats the hybrid vertex as one object with two channels on the
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

Changes with respect to v2 (v2 is kept unchanged as a reference):

* `amplitudes.py`: the vertex is a table of terms (`HQ_TERMS`) grouped by channel;
  one gap rule for both channels (`Params.gaps`: π/r, 3π/r in A; ΔV, ΔV_Σ in B);
  `channel_amplitudes`, `width_split`. `sigma_mode` ('EST'/'norm'/'off') removed:
  the LEFs are those of the notes, with the relative Π–Σ phase fixed by the
  r→0 limit g_B∥(0) = g_B⊥(0) (real, same sign).
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
% in MATLAB (or octave-cli), from EM_Pheno_v3/matlab
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
  gaps π/r, 3π/r (variant A, as matched) or ΔV(r), ΔV_Σ(r) (variant B). Heavy-quark spin is a spectator; states can
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

## Main results (October 2026, v3)

| quantity | value |
|---|---|
| κ_b (g_E1 = κ e e_b r) | 0.998 ± 0.040 (ε = 4%) |
| κ_c | 0.856 +0.031 −0.028 (core), 0.828 with ψ(3770), ψ2(3823) |
| hybrid LEFs | c1 = c2 = c3 = c4 = κ_H e e_Q; BESIII γη_c: κ_H < 0.94, 0.63 (B, NL), < 0.04 (A) |
| channels (B, NL, ψ(4040)) | Γ_ΠΠ = 57, Γ_ΣΣ = 188, Γ_int = −208 keV at κ_H = 1: strong destructive interference |
| c4 at κ_c (B, NL) | 0.81 < c4/c < 0.96 from the hybrid candidates; 0.08–0.2 from J/ψ, ψ(2S), ψ(3770) → γη_c |
| lattice R = Γ(η_c1)/Γ(Y) | A: Π channel alone 1.61, both channels 0.90; B: 0.25 (lattice 1.78 ± 0.24, or 3.5 from F(0)) |
| X(3872) c c̄ probability | Z ≈ 0.44 ± 0.19 (from γJ/ψ), Z ≲ 0.3 (from ψ(4160)→γX) |

See `../TeX file/tex_pheno/pheno_standalone.pdf` for the discussion. A detailed description of every file (QQbarSpectrum summary, exporter,
`emlib` modules, scripts, tests and libraries) is the appendix
`../TeX file/tex_pheno/app_codes.tex`. Equation labels refer to the current notes (mainEM.tex).
