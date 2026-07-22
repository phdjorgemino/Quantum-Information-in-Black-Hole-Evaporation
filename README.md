# Quantum Information in Black-Hole Evaporation

## Reproducibility Archive for Information Preservation, Computational Recovery, and Species-Resolved Hawking Transport

A channel-based framework is developed to separate three logically distinct questions
in black-hole evaporation: information preservation under an assumed global isometry,
accessibility to resource-bounded decoders, and species-dependent transport through
the exterior curvature potential. A literal entanglement “backup” is excluded by the
no-cloning theorem and replaced by entanglement-assisted recovery after dynamical en-
coding, while Hawking pair creation, greybody scattering, and superradiance are repre-
sented as distinct channel dilations. For Schwarzschild evaporation, inclusive number
and power spectra at future null infinity, I +, are computed by summing numerically
converged angular modes of massless scalar, Weyl, photon, and graviton fields, with
each greybody factor applied exactly once. The normalized number spectra yield pair-
wise Jensen–Shannon divergences of 0.09054–0.53261 nats, compared with maximum
frequency-grid and boundary sensitivities of 2.46×10−4 and 8.05×10−6 nats, respectively.
Reconstructed Weyl, photon, and graviton emission powers agree with Page’s benchmark
values within 0.21%. Complementary reference-tagged and Yoshida–Yao circuits distin-
guish correlation transfer from decoder-specific heralded recovery through mismatched-
unitary and dephasing controls. These results support species-dependent semiclassical
transport but neither establish cryptographic hardness nor provide a microscopic encod-
ing mechanism for infalling states. Information conservation therefore remains condi-
tional on the assumed isometry and is not derived for realistic four-dimensional evapo
ration

This repository contains the computational workflow, numerical data, statistical analyses, figures, manuscript sources, bibliography, environment specifications, and integrity checks supporting the associated study of quantum information in black-hole evaporation.

The repository addresses three conceptually distinct questions:

1. information preservation under an assumed global isometry;
2. accessibility of the encoded information to finite or resource-bounded decoders; and
3. species-dependent Hawking transport through the Schwarzschild curvature potential.

The numerical workflow performs four separate analyses:

1. lowest-multipole greybody transmission calculations for massless fields in Schwarzschild spacetime;
2. inclusive number and power spectra for massless scalar, Weyl, photon, and graviton fields;
3. event-conditioned correlation transfer in a 17-qubit reference-tagged model; and
4. probabilistic Yoshida--Yao decoding with matched-unitary, mismatched-unitary, and complete-dephasing controls.

> **Scientific scope.**  
> The inclusive spectral calculations support the kinematic component of the species-dependent transport hypothesis. They do not constitute a microscopic simulation of black-hole evaporation, a cryptographic reduction, a derivation of microstate-dependent information encoding, or a proof of gravitational unitarity. Information conservation remains conditional on the assumed global isometry.

---

## Physical convention: the greybody factor is applied exactly once

Natural units are used throughout:

\[
G=c=\hbar=k_{\mathrm B}=1,
\qquad 2M=1.
\]

The modal number spectrum is constructed directly at future null infinity:

\[
q_s^{N}(\omega)
=
\frac{1}{2\pi}
\sum_a
g_a
\frac{\Gamma_a(\omega)}
{\exp(\omega/T_{\mathrm H})-(-1)^{2s}},
\]

where:

- \(s\) denotes the field spin;
- \(a\) indexes the angular and polarization modes;
- \(g_a\) is the corresponding degeneracy;
- \(\Gamma_a(\omega)\) is the greybody transmission factor; and
- \(T_{\mathrm H}\) is the Hawking temperature.

Because the spectrum already describes particles transmitted to future null infinity, each \(\Gamma_a(\omega)\) enters the sampling distribution exactly once.

The qubit model is conditioned on the event that the emitted quantum has already reached infinity. Its effective channel transmissivity is therefore equal to one. The executable guard in `pagesim.py` terminates the simulation if any other transmissivity is supplied, preventing accidental greybody-factor double counting.

A second guard verifies the port convention:

\[
\Gamma=1:
\qquad
\lvert 10\rangle_{SR}
\longmapsto
\lvert 01\rangle_{SR}
\]

up to an irrelevant phase.

---

## Main numerical results

### Inclusive species-resolved spectra

The production calculation generated 735 mode--frequency records, including one angular-momentum truncation sentinel for each field.

All values below are expressed in the scaled units used by the numerical pipeline.

| Field | Integrated number rate | Integrated power | Median number frequency | Number-spectrum peak |
|---|---:|---:|---:|---:|
| Real scalar, \(s=0\) | \(1.32965\times10^{-3}\) | \(2.97513\times10^{-4}\) | 0.202287 | 0.172290 |
| Weyl, \(s=\tfrac12\) | \(4.85779\times10^{-4}\) | \(1.63685\times10^{-4}\) | 0.326672 | 0.317287 |
| Photon, \(s=1\) | \(2.95975\times10^{-4}\) | \(1.34568\times10^{-4}\) | 0.450719 | 0.455128 |
| Graviton, \(s=2\) | \(2.21809\times10^{-5}\) | \(1.53445\times10^{-5}\) | 0.694940 | 0.706662 |

### Jensen--Shannon separation

Pairwise Jensen--Shannon divergences between the normalized inclusive number spectra are reported in nats:

|  | Scalar | Weyl | Photon | Graviton |
|---|---:|---:|---:|---:|
| **Scalar** | 0 | 0.10077 | 0.28027 | 0.53261 |
| **Weyl** | 0.10077 | 0 | 0.09054 | 0.42396 |
| **Photon** | 0.28027 | 0.09054 | 0 | 0.25619 |
| **Graviton** | 0.53261 | 0.42396 | 0.25619 | 0 |

The pairwise divergences range from

\[
0.09054 \leq D_{\mathrm{JS}} \leq 0.53261
\quad \text{nats}.
\]

The recorded numerical sensitivities are substantially smaller:

- maximum thinned-frequency-grid sensitivity:
  \(2.45\times10^{-4}\) nats;
- maximum boundary sensitivity:
  \(8.05\times10^{-6}\) nats;
- maximum sentinel contribution to the number rate:
  \(3.00\times10^{-5}\);
- maximum sentinel contribution to the power:
  \(6.08\times10^{-5}\).

These diagnostics establish numerical separation of the field-resolved spectra within the declared semiclassical model. They do not establish that field species encode or anonymize arbitrary infalling microstates.

### Validation against Page's emission powers

The reconstructed coefficients are compared with Table I of [Page, *Physical Review D* **13**, 198 (1976)](https://doi.org/10.1103/PhysRevD.13.198).

| Field | Reconstructed coefficient | Page reference | Relative error |
|---|---:|---:|---:|
| Weyl | \(1.6368519\times10^{-4}\) | \(1.6360\times10^{-4}\) | \(+0.0521\%\) |
| Photon | \(3.3642076\times10^{-5}\) | \(3.3710\times10^{-5}\) | \(-0.2015\%\) |
| Graviton | \(3.8361363\times10^{-6}\) | \(3.8400\times10^{-6}\) | \(-0.1006\%\) |

The maximum absolute discrepancy is approximately \(0.21\%\).

### Greybody solver validation

The lowest-multipole production calculation generated 196 accepted mode--frequency records.

The principal numerical diagnostics are:

| Diagnostic | Maximum recorded value |
|---|---:|
| Normalized flux-balance residual | \(5.30\times10^{-12}\) |
| Boundary relative change | \(3.92\times10^{-4}\) |
| ODE-tolerance relative change | \(2.71\times10^{-9}\) |
| Maximum-step relative change | \(6.83\times10^{-11}\) |
| Dirac supersymmetric-partner difference | \(6.81\times10^{-5}\) |

### Correlation-transfer and decoder benchmarks

The operational correlation deficit is defined as

\[
\mathcal A_{\mathrm{op}}(t)
=
1-
\frac{I(Q:R_t)}{2S(Q)}.
\]

The reference-tagged channel ensemble contains 128 realizations. The Yoshida--Yao ensemble contains 256 paired realizations.

| Estimand | Point estimate | 95% BCa interval |
|---|---:|---:|
| \(I(Q:R)/(2\ln2)\), Haar circuit at \(t=4\) | 0.499756 | [0.497509, 0.502002] |
| \(\mathcal A_{\mathrm{op}}^{\mathrm{brick}}-\mathcal A_{\mathrm{op}}^{\mathrm{Haar}}\), \(t=4\) | -0.201068 | [-0.210107, -0.192367] |
| Matched conditional fidelity \(F_{\mathrm{cond}}\), \(d=4\) | 0.988599 | [0.988451, 0.988749] |
| Matched success probability \(p_{\mathrm{succ}}\), \(d=7\) | 0.250000 | exact across the ensemble |
| \(F_{\mathrm{matched}}-F_{\mathrm{mismatched}}\), \(d=4\) | 0.739609 | [0.736203, 0.742957] |
| \(F_{\mathrm{matched}}-F_{\mathrm{dephased}}\), \(d=4\) | 0.011494 | [0.011351, 0.011642] |

For the ideal matched decoder,

\[
p_{\mathrm{succ}}F_{\mathrm{cond}}=\frac14
\]

at every evaluated collection size, with a maximum numerical error of
\(8.60\times10^{-16}\).

These circuit results are finite-dimensional recovery diagnostics inspired by the [Yoshida--Yao protocol](https://doi.org/10.1103/PhysRevX.9.011006). They are not microscopic models of gravitational evaporation.

---

## Software requirements

The production metadata records the following environment:

| Component | Version |
|---|---:|
| Python | 3.12.13 |
| NumPy | 2.3.5 |
| SciPy | 1.17.0 |
| Matplotlib | 3.10.8 |

The exact Python dependency versions are specified in `requirements-lock.txt`.

Compiling the manuscript additionally requires a LaTeX distribution providing:

- `pdflatex`;
- `bibtex`; and
- the standard packages loaded by the included SciPost class.

---

## Reproduction instructions

### 1. Clone the repository

```bash
git clone https://github.com/<USERNAME>/<REPOSITORY>.git
cd <REPOSITORY>
```

Replace `<USERNAME>` and `<REPOSITORY>` with the final GitHub location.

### 2. Create the Python environment

On Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
```

Confirm the interpreter before running the production analysis:

```bash
python --version
```

Python 3.12.x is recommended. The archived production run used Python 3.12.13.

### 3. Run the reduced smoke test

```bash
python run_all.py --quick
```

The reduced test writes its outputs to:

```text
data_quick/
figures_quick/
```

This mode verifies the execution path using reduced frequency grids and smaller ensembles. It must not be used as a substitute for the production results reported in the manuscript.

### 4. Run the complete production workflow

```bash
python run_all.py
```

The complete workflow executes, in order:

1. lowest-multipole greybody calculations;
2. inclusive H3 spectra;
3. the reference-tagged correlation-transfer model;
4. the Yoshida--Yao decoder benchmarks;
5. statistical summaries; and
6. production-figure generation.

Production outputs are written to:

```text
data/
figures/
```

A successful execution ends with:

```text
[run_all] pipeline completed without failed checks
```

### 5. Compile the SciPost manuscript

From the repository root:

```bash
mkdir -p build

pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory=build main_REV10_SciPost_BibTeX.tex

bibtex build/main_REV10_SciPost_BibTeX

pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory=build main_REV10_SciPost_BibTeX.tex

pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory=build main_REV10_SciPost_BibTeX.tex
```

The compiled manuscript is written to:

```text
build/main_REV10_SciPost_BibTeX.pdf
```

### 6. Refresh and verify the integrity manifest

After compiling the final manuscript, refresh the SHA-256 manifest:

```bash
python run_all.py --manifest-only
```

Verify all archived files on GNU/Linux:

```bash
sha256sum -c MANIFEST.sha256
```

On macOS:

```bash
shasum -a 256 -c MANIFEST.sha256
```

Because `README.md` is included in the integrity manifest, the manifest must be refreshed after any README modification.

---

## Reproducibility and numerical controls

The workflow implements the following controls:

- No machine-dependent absolute paths are used.
- Fixed pseudorandom seeds are recorded in the metadata.
- BLAS execution is restricted to a single thread by `run_all.py`.
- The Schwarzschild radial coordinate is evaluated with `wrightomega` and is not interpolated inside the ODE integrator.
- DOP853 uses \(\pi/(8\omega)\) as the recorded maximum step.
- Every greybody mode is repeated with half of the nominal maximum step.
- Every mode is repeated with extended boundaries and stricter ODE tolerances.
- Transmission comparisons use logarithmic ratios, including for very small greybody factors.
- The supersymmetric Dirac partner potentials must agree within the recorded numerical threshold.
- The two Dirac partner potentials are not counted as independent physical channels.
- Inclusive spectra explicitly sum angular modes and degeneracies over a common frequency grid.
- One additional mode per field acts as an angular-truncation sentinel.
- Jensen--Shannon divergences are recomputed using alternative boundaries and a thinned frequency grid.
- Weyl, photon, and graviton power coefficients are automatically compared with Page's published benchmark.
- The correlation-transfer model does not reapply the greybody factor.
- The decoder analysis reports conditional fidelity, exact Born success probability, and \(p_{\mathrm{succ}}F_{\mathrm{cond}}\).
- Matched, independently mismatched, and completely dephased decoder controls are evaluated using paired realizations.
- Pointwise circuit intervals use 95% bias-corrected and accelerated bootstrap intervals with 20,000 resamples.
- Curve-level bands use the maximum absolute bootstrap deviation.
- No confirmatory \(p\)-values are reported because no null-hypothesis test was preregistered before the production seeds were fixed.
- `MANIFEST.sha256` records the integrity of the production scripts, data, figures, manuscript, bibliography, and compiled PDF.

The recorded random seeds are:

| Analysis | Seed |
|---|---:|
| Correlation-transfer ensemble | 20260720 |
| Yoshida--Yao ensemble | 20260721 |
| Bootstrap statistics | 20260722 |

---

## Repository contents

### Analysis scripts

| File | Purpose |
|---|---|
| [`greybody.py`](greybody.py) | Computes lowest-multipole Schwarzschild greybody factors and the at-infinity sampling schedule. |
| [`inclusive_h3.py`](inclusive_h3.py) | Computes inclusive scalar, Weyl, photon, and graviton number and power spectra. |
| [`pagesim.py`](pagesim.py) | Runs the 17-qubit event-conditioned correlation-transfer model. |
| [`yysim.py`](yysim.py) | Runs the matched, mismatched, and dephased Yoshida--Yao decoder benchmarks. |
| [`plots.py`](plots.py) | Produces statistical summaries and manuscript figures. |
| [`run_all.py`](run_all.py) | Orchestrates the complete workflow and generates the SHA-256 manifest. |

### Production data

| File | Contents |
|---|---|
| [`data/greybody_modes.csv`](data/greybody_modes.csv) | 196 lowest-multipole transmission and convergence records. |
| [`data/greybody.npz`](data/greybody.npz) | Greybody arrays and at-infinity sampling nodes. |
| [`data/greybody_metadata.json`](data/greybody_metadata.json) | Solver configuration, convergence diagnostics, and sampling convention. |
| [`data/h3_inclusive_modes.csv`](data/h3_inclusive_modes.csv) | 735 inclusive mode--frequency and sentinel records. |
| [`data/h3_species_summary.csv`](data/h3_species_summary.csv) | Species-resolved rates, powers, spectral statistics, truncation diagnostics, and Page comparison. |
| [`data/h3_inclusive.npz`](data/h3_inclusive.npz) | Spectral densities, cumulative distributions, and divergence matrices. |
| [`data/h3_inclusive_metadata.json`](data/h3_inclusive_metadata.json) | H3 protocol, epistemic scope, numerical sensitivities, and integrity hashes. |
| [`data/channel_ensemble_long.csv`](data/channel_ensemble_long.csv) | Long-form correlation-transfer ensemble results. |
| [`data/channel_metadata.json`](data/channel_metadata.json) | Channel scope, seeds, guards, and normalization diagnostics. |
| [`data/yy_ensemble_long.csv`](data/yy_ensemble_long.csv) | Long-form matched, mismatched, and dephased decoder results. |
| [`data/yy_metadata.json`](data/yy_metadata.json) | Decoder conventions, seeds, controls, and integrity hashes. |
| [`data/results_summary.csv`](data/results_summary.csv) | Principal point estimates and 95% BCa intervals. |
| [`data/statistics_metadata.json`](data/statistics_metadata.json) | Bootstrap protocol and statistical reporting conventions. |



## Data and code availability

The repository contains the complete computational protocol, source code, numerical data, environment specifications, generated figures, canonical BibTeX database, and SHA-256 checksums required for independent verification and reproduction of the reported analyses.

The immutable release should additionally be archived in a permanent research repository.



## Contact

For questions concerning the numerical workflow or scientific interpretation, contact:

**Jorge Fernando Miño Ayala**  
Institutional email: jorge.mino@epn.edu.ec
