#!/usr/bin/env python3
"""Inclusive Schwarzschild Hawking spectra for the field-resolved H3 test.

The script sums declared angular modes for massless fields of spin 0, 1/2,
1, and 2 on a common frequency grid.  The greybody factor is used exactly
once, inside the at-infinity number and power spectra.  The resulting
spectra are deterministic field-theory observables; they are not interpreted
as microscopic black-hole message channels.

The lowest multipoles are reused from ``greybody.py``.  Higher multipoles are
solved with the same boundary-extension, tolerance-tightening, maximum-step,
flux-balance, and Dirac-partner checks.  One additional angular mode per
field is computed but excluded from the registered sum as an empirical
truncation sentinel.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import scipy
from scipy.integrate import cumulative_trapezoid, quad
from scipy.interpolate import PchipInterpolator

import greybody


TH = greybody.TH
MASS = 0.5  # 2M=1 convention
PAGE_POWER_ALPHA = {
    # Page, Phys. Rev. D 13, 198 (1976), Table I.  The neutrino value
    # contains four one-helicity species; photon and graviton values contain
    # their two physical helicities.  Units: hbar c^6 G^-2 M^-2.
    "weyl": 1.636e-4,
    "photon": 3.371e-5,
    "graviton": 3.84e-6,
}


@dataclass(frozen=True)
class FieldSpec:
    slug: str
    label: str
    spin: float
    statistics: str
    physical_dof: int
    page_multiplicity: int


@dataclass(frozen=True)
class ModeSpec:
    field: FieldSpec
    mode_slug: str
    mode_label: str
    angular_value: float
    degeneracy: int
    included: bool
    low_power: int
    potential: Callable[[float], float]
    partner: Callable[[float], float] | None = None
    reuse_species: str | None = None


FIELDS = (
    FieldSpec("scalar", r"real scalar $s=0$", 0.0, "bose", 1, 1),
    FieldSpec("weyl", r"Weyl $s=1/2$", 0.5, "fermi", 1, 4),
    FieldSpec("photon", r"photon $s=1$", 1.0, "bose", 2, 1),
    FieldSpec("graviton", r"graviton $s=2$", 2.0, "bose", 2, 1),
)
FIELD_BY_SLUG = {field.slug: field for field in FIELDS}


def integer_mode(field_slug: str, spin: int, ell: int, included: bool, reuse: str | None = None) -> ModeSpec:
    field = FIELD_BY_SLUG[field_slug]
    return ModeSpec(
        field=field,
        mode_slug=f"l{ell}",
        mode_label=rf"$\ell={ell}$",
        angular_value=float(ell),
        degeneracy=field.physical_dof * (2 * ell + 1),
        included=included,
        low_power=2 * ell + 2,
        potential=lambda x, s=spin, value=ell: greybody.potential_integer(x, s, value),
        reuse_species=reuse,
    )


def dirac_mode(lam: int, included: bool, reuse: str | None = None) -> ModeSpec:
    field = FIELD_BY_SLUG["weyl"]
    j = lam - 0.5
    return ModeSpec(
        field=field,
        mode_slug=f"k{lam}",
        mode_label=rf"$j={lam}-1/2$",
        angular_value=j,
        # One Weyl helicity.  V_+ and V_- are supersymmetric radial
        # components of the same channel and are checked, not double-counted.
        degeneracy=2 * lam,
        included=included,
        low_power=2 * lam,
        potential=lambda x, value=float(lam): greybody.potential_dirac_plus(x, value),
        partner=lambda x, value=float(lam): greybody.potential_dirac_minus(x, value),
        reuse_species=reuse,
    )


MODES = (
    integer_mode("scalar", 0, 0, True, "scalar"),
    integer_mode("scalar", 0, 1, True),
    integer_mode("scalar", 0, 2, True),
    integer_mode("scalar", 0, 3, False),
    dirac_mode(1, True, "dirac"),
    dirac_mode(2, True),
    dirac_mode(3, True),
    dirac_mode(4, False),
    integer_mode("photon", 1, 1, True, "gauge"),
    integer_mode("photon", 1, 2, True),
    integer_mode("photon", 1, 3, True),
    integer_mode("photon", 1, 4, False),
    integer_mode("graviton", 2, 2, True, "graviton"),
    integer_mode("graviton", 2, 3, True),
    integer_mode("graviton", 2, 4, False),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_lowest_rows(data_dir: Path) -> dict[tuple[str, float], dict[str, object]]:
    with (data_dir / "greybody_modes.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {(str(row["species"]), float(row["omega"])): dict(row) for row in rows}


def solve_or_reuse_mode(
    mode: ModeSpec,
    omega: np.ndarray,
    lowest: dict[tuple[str, float], dict[str, object]],
) -> list[dict[str, object]]:
    result_rows: list[dict[str, object]] = []
    for value in omega:
        key = (str(mode.reuse_species), float(value))
        if mode.reuse_species is not None and key in lowest:
            diagnostics = dict(lowest[key])
            diagnostics["reused_lowest_mode"] = True
        else:
            diagnostics = greybody.converged_mode(float(value), mode.potential)
            diagnostics["reused_lowest_mode"] = False
            if mode.partner is not None:
                partner = greybody.converged_mode(float(value), mode.partner)
                partner_gamma = float(partner["gamma"])
                gamma = float(diagnostics["gamma"])
                partner_log = abs(np.log(gamma / partner_gamma))
                diagnostics["dirac_partner_gamma"] = partner_gamma
                diagnostics["dirac_partner_delta_abs"] = abs(gamma - partner_gamma)
                diagnostics["dirac_partner_delta_rel"] = abs(gamma - partner_gamma) / max(
                    gamma, partner_gamma, 1.0e-300
                )
                diagnostics["dirac_partner_delta_log"] = partner_log
                diagnostics["dirac_partner_accepted"] = bool(partner["accepted"]) and partner_log <= 6.0e-3
                diagnostics["accepted"] = bool(diagnostics["accepted"]) and bool(
                    diagnostics["dirac_partner_accepted"]
                )
        diagnostics.update(
            species=mode.field.slug,
            field_label=mode.field.label,
            spin=mode.field.spin,
            statistics=mode.field.statistics,
            physical_dof=mode.field.physical_dof,
            mode_slug=mode.mode_slug,
            mode_label=mode.mode_label,
            angular_value=mode.angular_value,
            degeneracy=mode.degeneracy,
            included=mode.included,
            truncation_sentinel=not mode.included,
            low_frequency_power=mode.low_power,
        )
        result_rows.append(diagnostics)
    return result_rows


def gamma_curve(dense: np.ndarray, omega: np.ndarray, gamma: np.ndarray, low_power: int) -> np.ndarray:
    if np.any(gamma <= 0.0):
        raise ValueError("positive transmission is required for log-log interpolation")
    interpolation = PchipInterpolator(np.log(omega), np.log(gamma), extrapolate=False)
    result = np.empty_like(dense)
    below = dense < omega[0]
    result[below] = gamma[0] * (dense[below] / omega[0]) ** low_power
    result[~below] = np.exp(interpolation(np.log(dense[~below])))
    return np.clip(result, 0.0, 1.0)


def occupation_denominator(field: FieldSpec, omega: np.ndarray) -> np.ndarray:
    x = omega / TH
    if field.statistics == "bose":
        return np.expm1(x)
    return np.exp(x) + 1.0


def build_field_spectra(
    rows: list[dict[str, object]],
    omega: np.ndarray,
    gamma_key: str,
    thin: bool = False,
) -> dict[str, dict[str, np.ndarray | float]]:
    dense = np.unique(
        np.r_[np.geomspace(1.0e-7, omega[0], 900), np.linspace(omega[0], omega[-1], 16000)]
    )
    by_mode: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        by_mode.setdefault((str(row["species"]), str(row["mode_slug"])), []).append(row)
    output: dict[str, dict[str, np.ndarray | float]] = {}
    for field in FIELDS:
        total_gamma = np.zeros_like(dense)
        sentinel_gamma = np.zeros_like(dense)
        degeneracy_sum = 0
        for mode in [item for item in MODES if item.field.slug == field.slug]:
            mode_rows = sorted(by_mode[(field.slug, mode.mode_slug)], key=lambda row: float(row["omega"]))
            mode_omega = np.asarray([float(row["omega"]) for row in mode_rows])
            mode_gamma = np.asarray([float(row[gamma_key]) for row in mode_rows])
            if thin:
                indices = np.unique(np.r_[0, np.arange(1, len(mode_omega) - 1, 2), len(mode_omega) - 1])
                mode_omega = mode_omega[indices]
                mode_gamma = mode_gamma[indices]
            weighted = mode.degeneracy * gamma_curve(dense, mode_omega, mode_gamma, mode.low_power)
            if mode.included:
                total_gamma += weighted
                degeneracy_sum += mode.degeneracy
            else:
                sentinel_gamma += weighted
        denominator = occupation_denominator(field, dense)
        number_density = total_gamma / denominator / (2.0 * np.pi)
        power_density = dense * number_density
        sentinel_number = sentinel_gamma / denominator / (2.0 * np.pi)
        sentinel_power = dense * sentinel_number
        number_rate = float(np.trapezoid(number_density, dense))
        power = float(np.trapezoid(power_density, dense))
        sentinel_number_rate = float(np.trapezoid(sentinel_number, dense))
        sentinel_power_rate = float(np.trapezoid(sentinel_power, dense))
        p_number = number_density / number_rate
        p_power = power_density / power
        cdf_number = cumulative_trapezoid(p_number, dense, initial=0.0)
        cdf_number /= cdf_number[-1]
        cdf_power = cumulative_trapezoid(p_power, dense, initial=0.0)
        cdf_power /= cdf_power[-1]
        quantiles = np.asarray((0.10, 0.25, 0.50, 0.75, 0.90))
        number_quantiles = np.interp(quantiles, cdf_number, dense)
        power_quantiles = np.interp(quantiles, cdf_power, dense)

        # Conservative thermal bound for the finite set of included modes,
        # using Gamma<=1.  It does not bound angular modes beyond the sentinel.
        bose_number_tail = -TH * np.log1p(-np.exp(-omega[-1] / TH))
        number_tail_bound = degeneracy_sum * bose_number_tail / (2.0 * np.pi)
        energy_integral, _ = quad(
            lambda value: value * np.exp(-value / TH) / (1.0 - np.exp(-value / TH)),
            float(omega[-1]),
            np.inf,
            epsabs=1.0e-20,
            epsrel=1.0e-10,
        )
        power_tail_bound = degeneracy_sum * energy_integral / (2.0 * np.pi)
        output[field.slug] = {
            "omega": dense,
            "number_density": number_density,
            "power_density": power_density,
            "p_number": p_number,
            "p_power": p_power,
            "cdf_number": cdf_number,
            "cdf_power": cdf_power,
            "number_rate": number_rate,
            "power": power,
            "mean_number_frequency": float(power / number_rate),
            "number_peak_frequency": float(dense[int(np.argmax(number_density))]),
            "power_peak_frequency": float(dense[int(np.argmax(power_density))]),
            "quantiles": quantiles,
            "number_quantiles": number_quantiles,
            "power_quantiles": power_quantiles,
            "sentinel_number_fraction": sentinel_number_rate / number_rate,
            "sentinel_power_fraction": sentinel_power_rate / power,
            "finite_mode_number_tail_bound_fraction": number_tail_bound / number_rate,
            "finite_mode_power_tail_bound_fraction": power_tail_bound / power,
            "included_degeneracy_sum": degeneracy_sum,
        }
    return output


def js_divergence(p: np.ndarray, q: np.ndarray, omega: np.ndarray) -> float:
    midpoint = 0.5 * (p + q)
    left = np.where(p > 0.0, p * np.log(p / midpoint), 0.0)
    right = np.where(q > 0.0, q * np.log(q / midpoint), 0.0)
    return float(0.5 * np.trapezoid(left, omega) + 0.5 * np.trapezoid(right, omega))


def pairwise_jsd(spectra: dict[str, dict[str, np.ndarray | float]], density_key: str) -> np.ndarray:
    matrix = np.zeros((len(FIELDS), len(FIELDS)), dtype=float)
    for i, first in enumerate(FIELDS):
        for j, second in enumerate(FIELDS):
            if j <= i:
                continue
            omega = np.asarray(spectra[first.slug]["omega"], dtype=float)
            value = js_divergence(
                np.asarray(spectra[first.slug][density_key], dtype=float),
                np.asarray(spectra[second.slug][density_key], dtype=float),
                omega,
            )
            matrix[i, j] = matrix[j, i] = value
    return matrix


def serializable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def run(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    lowest = load_lowest_rows(data_dir)
    omega = np.asarray(sorted({frequency for _, frequency in lowest}), dtype=float)
    all_rows: list[dict[str, object]] = []
    for mode in MODES:
        print(
            f"[inclusive_h3] {mode.field.slug}/{mode.mode_slug}: {len(omega)} frequencies"
            + (" (reused)" if mode.reuse_species else ""),
            flush=True,
        )
        all_rows.extend(solve_or_reuse_mode(mode, omega, lowest))

    rejected = [row for row in all_rows if str(row.get("accepted", "True")).lower() not in ("true", "1")]
    if rejected:
        examples = ", ".join(
            f"{row['species']}/{row['mode_slug']}@{float(row['omega']):.4g}" for row in rejected[:8]
        )
        raise RuntimeError(f"{len(rejected)} inclusive H3 modes failed convergence: {examples}")

    csv_path = data_dir / "h3_inclusive_modes.csv"
    fieldnames = sorted({key for row in all_rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    final = build_field_spectra(all_rows, omega, "gamma")
    base = build_field_spectra(all_rows, omega, "gamma_base")
    thinned = build_field_spectra(all_rows, omega, "gamma", thin=True)
    js_number = pairwise_jsd(final, "p_number")
    js_power = pairwise_jsd(final, "p_power")
    js_number_base = pairwise_jsd(base, "p_number")
    js_number_thinned = pairwise_jsd(thinned, "p_number")
    nonzero = js_number[np.triu_indices(len(FIELDS), 1)]

    arrays: dict[str, np.ndarray] = {
        "species": np.asarray([field.slug for field in FIELDS]),
        "labels": np.asarray([field.label for field in FIELDS]),
        "jsd_number": js_number,
        "jsd_power": js_power,
    }
    for field in FIELDS:
        for key in (
            "omega",
            "number_density",
            "power_density",
            "p_number",
            "p_power",
            "cdf_number",
            "cdf_power",
            "quantiles",
            "number_quantiles",
            "power_quantiles",
        ):
            arrays[f"{key}__{field.slug}"] = np.asarray(final[field.slug][key])
    npz_path = data_dir / "h3_inclusive.npz"
    np.savez_compressed(npz_path, **arrays)

    summary_rows: list[dict[str, object]] = []
    page_comparison: dict[str, dict[str, float]] = {}
    for field in FIELDS:
        values = final[field.slug]
        row = {
            "species": field.slug,
            "number_rate_2M1": values["number_rate"],
            "power_2M1": values["power"],
            "mean_frequency": values["mean_number_frequency"],
            "number_peak_frequency": values["number_peak_frequency"],
            "power_peak_frequency": values["power_peak_frequency"],
            "median_number_frequency": np.asarray(values["number_quantiles"])[2],
            "median_power_frequency": np.asarray(values["power_quantiles"])[2],
            "sentinel_number_fraction": values["sentinel_number_fraction"],
            "sentinel_power_fraction": values["sentinel_power_fraction"],
            "finite_mode_number_tail_bound_fraction": values["finite_mode_number_tail_bound_fraction"],
            "finite_mode_power_tail_bound_fraction": values["finite_mode_power_tail_bound_fraction"],
        }
        if field.slug in PAGE_POWER_ALPHA:
            converted_alpha = (
                float(values["power"]) * MASS**2 * field.page_multiplicity
            )
            reference = PAGE_POWER_ALPHA[field.slug]
            relative_error = (converted_alpha - reference) / reference
            row.update(page_alpha_reconstructed=converted_alpha, page_alpha_reference=reference, page_relative_error=relative_error)
            page_comparison[field.slug] = {
                "reconstructed_alpha": converted_alpha,
                "reference_alpha": reference,
                "relative_error": relative_error,
            }
        summary_rows.append(row)
    summary_path = data_dir / "h3_species_summary.csv"
    summary_fields = sorted({key for row in summary_rows for key in row})
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    metadata = {
        "scope": "inclusive over declared Schwarzschild angular modes; one truncation sentinel per field",
        "epistemic_scope": "field-resolved kinematics only; no microscopic information coupling is inferred",
        "units": "G=c=hbar=k_B=1 and 2M=1",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "fields": [field.__dict__ for field in FIELDS],
        "modes": [
            {
                "species": mode.field.slug,
                "mode_slug": mode.mode_slug,
                "mode_label": mode.mode_label,
                "angular_value": mode.angular_value,
                "degeneracy": mode.degeneracy,
                "included": mode.included,
                "low_frequency_power": mode.low_power,
                "reused_lowest_mode": mode.reuse_species,
            }
            for mode in MODES
        ],
        "primary_estimand": "pairwise Jensen-Shannon divergence of normalized inclusive at-infinity number spectra",
        "jsd_units": "nats; maximum ln(2)",
        "jsd_number": js_number.tolist(),
        "jsd_power": js_power.tolist(),
        "minimum_pairwise_jsd_number": float(np.min(nonzero)),
        "maximum_jsd_boundary_sensitivity": float(np.max(np.abs(js_number - js_number_base))),
        "maximum_jsd_thinned_grid_sensitivity": float(np.max(np.abs(js_number - js_number_thinned))),
        "page_1976_power_validation": page_comparison,
        "greybody_factor_usage": "Gamma enters each modal at-infinity spectrum once; no subsequent channel attenuation is applied",
        "field_summaries": {
            field.slug: {
                key: serializable(value)
                for key, value in final[field.slug].items()
                if key not in ("omega", "number_density", "power_density", "p_number", "p_power", "cdf_number", "cdf_power")
            }
            for field in FIELDS
        },
        "sha256": {
            csv_path.name: sha256(csv_path),
            summary_path.name: sha256(summary_path),
            npz_path.name: sha256(npz_path),
        },
    }
    metadata_path = data_dir / "h3_inclusive_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        "[inclusive_h3] all modes passed; min pairwise JSD(number) =",
        f"{metadata['minimum_pairwise_jsd_number']:.6g}",
        "; max Page power relative error =",
        f"{max(abs(item['relative_error']) for item in page_comparison.values()):.3%}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.data_dir.resolve())
