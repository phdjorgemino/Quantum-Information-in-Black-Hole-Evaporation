#!/usr/bin/env python3
"""Auditable lowest-multipole Schwarzschild greybody calculation.

Conventions
-----------
Geometrized units G=c=hbar=1 and 2M=1 are used.  The radial equation is

    d2 psi/dr_*2 + [omega**2 - V(r)] psi = 0,
    r_* = r + log(r-1).

The horizon solution is normalized to exp(-i omega r_*).  At a finite outer
boundary it is decomposed into incoming/outgoing plane waves and
Gamma=1/|A_in|**2 is returned.  Every production value is recomputed after
(i) extending both boundaries and (ii) tightening the ODE tolerances.  These
two changes, rather than the formal solve_ivp tolerance alone, define the
reported numerical-error diagnostics.  The scalar sampling nodes are drawn
from the number spectrum already transmitted to future null infinity,

    p_inf(omega) proportional to Gamma_00(omega)/(exp(omega/T_H)-1).

Consequently Gamma enters this sampling density exactly once.  The emitted
event-conditioned qubit proxy receives unit channel transmissivity; it must
not apply Gamma(omega_j) a second time.

This is deliberately a lowest-multipole, Schwarzschild demonstration.  It is
not an inclusive Standard-Model luminosity and does not cover Kerr
superradiance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import scipy
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.special import wrightomega


TH = 1.0 / (4.0 * np.pi)  # Hawking temperature for 2M=1


@dataclass(frozen=True)
class Species:
    slug: str
    label: str
    spin: float
    multipole: str
    potential: Callable[[float], float]


@dataclass
class ModeResult:
    gamma: float
    a_in_abs2: float
    a_out_abs2: float
    wronskian_residual: float
    flux_balance_residual: float
    rstar_min: float
    rstar_max: float
    left_potential_ratio: float
    right_potential_ratio: float
    nfev: int
    max_step: float
    success: bool


def radius_data(rstar: float) -> tuple[float, float]:
    """Return (r,f) without numerical inversion or near-horizon cancellation.

    With x=r-1, x+log(x)=r_*-1.  scipy.special.wrightomega solves this
    equation directly and remains stable for the negative arguments used at
    the horizon boundary.
    """

    x = float(np.real(wrightomega(rstar - 1.0)))
    if not np.isfinite(x) or x <= 0.0:
        raise FloatingPointError(f"invalid r-1={x!r} at r_*={rstar!r}")
    r = 1.0 + x
    f = x / r
    return r, f


def potential_integer(rstar: float, spin: int, ell: int) -> float:
    r, f = radius_data(rstar)
    return f * (ell * (ell + 1.0) / r**2 + (1.0 - spin**2) / r**3)


def potential_dirac_plus(rstar: float, lam: float = 1.0) -> float:
    """Massless Dirac supersymmetric partner V_+=W^2+dW/dr_*.

    W=lambda sqrt(f)/r, f=1-1/r.  The partner potentials have the same
    transmission probability; lambda=1 is the lowest j=1/2 channel.
    """

    r, f = radius_data(rstar)
    sf = np.sqrt(f)
    w_super = lam * sf / r
    dw_dr = lam * (1.0 / (2.0 * sf * r**3) - sf / r**2)
    return float(w_super**2 + f * dw_dr)


def potential_dirac_minus(rstar: float, lam: float = 1.0) -> float:
    """Massless Dirac supersymmetric partner V_-=W^2-dW/dr_*."""

    r, f = radius_data(rstar)
    sf = np.sqrt(f)
    w_super = lam * sf / r
    dw_dr = lam * (1.0 / (2.0 * sf * r**3) - sf / r**2)
    return float(w_super**2 - f * dw_dr)


SPECIES = (
    Species("scalar", r"$s=0,\ell=0$", 0.0, "ell=0", lambda x: potential_integer(x, 0, 0)),
    Species("dirac", r"$s=1/2,j=1/2$", 0.5, "lambda=1", potential_dirac_plus),
    Species("gauge", r"$s=1,\ell=1$", 1.0, "ell=1", lambda x: potential_integer(x, 1, 1)),
    Species("graviton", r"$s=2,\ell=2$", 2.0, "ell=2", lambda x: potential_integer(x, 2, 2)),
)


def solve_mode(
    omega: float,
    potential: Callable[[float], float],
    rstar_min: float,
    rstar_max: float,
    rtol: float,
    atol: float,
    max_step: float,
) -> ModeResult:
    if omega <= 0.0 or rstar_max <= rstar_min:
        raise ValueError("omega and integration interval must be positive")

    def rhs(x: float, y: np.ndarray) -> np.ndarray:
        return np.asarray((y[1], (potential(x) - omega**2) * y[0]), dtype=complex)

    phase = np.exp(-1j * omega * rstar_min)
    y0 = np.asarray((phase, -1j * omega * phase), dtype=complex)
    sol = solve_ivp(
        rhs,
        (rstar_min, rstar_max),
        y0,
        method="DOP853",
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        dense_output=False,
    )
    if not sol.success or sol.t[-1] != rstar_max:
        raise RuntimeError(f"DOP853 failed at omega={omega}: {sol.message}")

    x = float(sol.t[-1])
    psi, dpsi = sol.y[:, -1]
    a_in = np.exp(1j * omega * x) * (psi + 1j * dpsi / omega) / 2.0
    a_out = np.exp(-1j * omega * x) * (psi - 1j * dpsi / omega) / 2.0
    ain2 = float(abs(a_in) ** 2)
    aout2 = float(abs(a_out) ** 2)
    gamma = 1.0 / ain2
    # The raw Wronskian subtraction is deliberately retained for audit, but
    # it is ill-conditioned when Gamma is tiny because ain2 and aout2 are
    # both O(1/Gamma).  The normalized balance 1-|R|^2-Gamma is the stable
    # acceptance diagnostic.
    flux_balance_residual = abs(1.0 - aout2 / ain2 - gamma)
    return ModeResult(
        gamma=gamma,
        a_in_abs2=ain2,
        a_out_abs2=aout2,
        wronskian_residual=abs(ain2 - aout2 - 1.0),
        flux_balance_residual=flux_balance_residual,
        rstar_min=rstar_min,
        rstar_max=rstar_max,
        left_potential_ratio=abs(potential(rstar_min)) / omega**2,
        right_potential_ratio=abs(potential(rstar_max)) / omega**2,
        nfev=int(sol.nfev),
        max_step=float(max_step),
        success=bool(sol.success),
    )


def converged_mode(omega: float, potential: Callable[[float], float]) -> dict[str, float | int | bool]:
    """Separate boundary-location error from integration-tolerance error."""

    base_left = -22.0
    base_right = max(100.0, 80.0 / omega)
    extended_left = -28.0
    extended_right = 1.4 * base_right

    registered_max_step = np.pi / (8.0 * omega)
    refined_max_step = registered_max_step / 2.0
    base = solve_mode(
        omega, potential, base_left, base_right, 1.0e-9, 1.0e-11, registered_max_step
    )
    extended = solve_mode(
        omega, potential, extended_left, extended_right, 1.0e-9, 1.0e-11, registered_max_step
    )
    tightened = solve_mode(
        omega, potential, extended_left, extended_right, 1.0e-11, 1.0e-13, registered_max_step
    )
    final = solve_mode(
        omega, potential, extended_left, extended_right, 1.0e-11, 1.0e-13, refined_max_step
    )

    delta_boundary = abs(extended.gamma - base.gamma)
    delta_tolerance = abs(tightened.gamma - extended.gamma)
    delta_max_step = abs(final.gamma - tightened.gamma)
    scale = max(final.gamma, 1.0e-14)
    if min(base.gamma, extended.gamma, tightened.gamma, final.gamma) <= 0.0:
        raise FloatingPointError(f"non-positive transmission at omega={omega}")
    delta_boundary_log = abs(np.log(extended.gamma / base.gamma))
    delta_tolerance_log = abs(np.log(tightened.gamma / extended.gamma))
    delta_max_step_log = abs(np.log(final.gamma / tightened.gamma))
    accepted = (
        0.0 < final.gamma <= 1.0 + 2.0e-8
        and final.flux_balance_residual <= 2.0e-10
        and delta_boundary_log <= 5.0e-3
        and delta_tolerance_log <= 5.0e-5
        and delta_max_step_log <= 5.0e-5
    )
    row: dict[str, float | int | bool] = {
        "omega": omega,
        "gamma": final.gamma,
        "gamma_base": base.gamma,
        "gamma_extended": extended.gamma,
        "delta_boundary_abs": delta_boundary,
        "delta_boundary_rel": delta_boundary / scale,
        "delta_boundary_log": delta_boundary_log,
        "delta_tolerance_abs": delta_tolerance,
        "delta_tolerance_rel": delta_tolerance / scale,
        "delta_tolerance_log": delta_tolerance_log,
        "delta_max_step_abs": delta_max_step,
        "delta_max_step_rel": delta_max_step / scale,
        "delta_max_step_log": delta_max_step_log,
        "accepted": accepted,
    }
    row.update({f"final_{key}": value for key, value in asdict(final).items() if key != "gamma"})
    return row


def frequency_grid(quick: bool) -> np.ndarray:
    if quick:
        return np.unique(np.r_[np.geomspace(0.01, 0.20, 8), np.linspace(0.25, 1.60, 8)])
    return np.unique(np.r_[np.geomspace(0.01, 0.20, 20), np.linspace(0.22, 1.60, 29)])


def _scalar_transmitted_spectrum(
    omega: np.ndarray, gamma_scalar: np.ndarray, steps: int
) -> dict[str, np.ndarray | float]:
    """Return deterministic quadrature nodes of the transmitted spectrum.

    Below the first solved frequency the analytic low-frequency law 4 omega^2
    is used.  On the solved interval a shape-preserving log-log interpolant is
    used.  The omitted upper tail is bounded with Gamma<=1.  The returned
    physical Gamma values are diagnostics only; channel_transmissivity is one
    because the nodes condition on quanta that have already reached infinity.
    """

    if np.any(gamma_scalar <= 0.0):
        raise ValueError("log interpolation requires positive scalar transmission")
    log_interp = PchipInterpolator(np.log(omega), np.log(gamma_scalar), extrapolate=False)
    dense = np.unique(np.r_[np.geomspace(1.0e-7, omega[0], 800), np.linspace(omega[0], omega[-1], 12000)])
    gamma_dense = np.empty_like(dense)
    low = dense < omega[0]
    gamma_dense[low] = 4.0 * dense[low] ** 2
    gamma_dense[~low] = np.exp(log_interp(np.log(dense[~low])))
    gamma_dense = np.clip(gamma_dense, 0.0, 1.0)

    bose = np.expm1(dense / TH)
    number_density = gamma_dense / bose
    cumulative = cumulative_trapezoid(number_density, dense, initial=0.0)
    integral = float(cumulative[-1])
    if not integral > 0.0:
        raise FloatingPointError("non-positive integrated scalar number spectrum")
    cdf = cumulative / integral
    quantiles = (np.arange(steps, dtype=float) + 0.5) / steps
    omega_at_infinity = np.interp(quantiles, cdf, dense)
    gamma_at_samples = np.interp(omega_at_infinity, dense, gamma_dense)
    upper_tail_bound = float(-TH * np.log1p(-np.exp(-omega[-1] / TH)))
    return {
        "schedule_quantiles": quantiles,
        "omega_at_infinity": omega_at_infinity,
        "gamma_at_sampled_frequencies": gamma_at_samples,
        "channel_transmissivity": np.ones(steps, dtype=float),
        "spectrum_omega": dense,
        "transmitted_number_density": number_density / integral,
        "spectrum_integral_truncated": integral,
        "upper_tail_absolute_bound": upper_tail_bound,
        "upper_tail_relative_bound": upper_tail_bound / integral,
    }


def scalar_at_infinity_schedule(
    omega: np.ndarray, gamma_scalar: np.ndarray, steps: int = 8
) -> dict[str, np.ndarray | float]:
    """Build the at-infinity schedule and a thinned-grid sensitivity check."""

    result = _scalar_transmitted_spectrum(omega, gamma_scalar, steps)
    coarse_indices = np.unique(np.r_[0, np.arange(1, len(omega) - 1, 2), len(omega) - 1])
    coarse = _scalar_transmitted_spectrum(omega[coarse_indices], gamma_scalar[coarse_indices], steps)
    full_nodes = np.asarray(result["omega_at_infinity"], dtype=float)
    coarse_nodes = np.asarray(coarse["omega_at_infinity"], dtype=float)
    delta = np.abs(full_nodes - coarse_nodes)
    result["quantile_frequency_grid_delta_abs"] = delta
    result["quantile_frequency_grid_delta_rel"] = delta / np.maximum(full_nodes, 1.0e-15)
    result["coarse_frequency_count"] = int(len(coarse_indices))
    if not np.all(np.asarray(result["channel_transmissivity"]) == 1.0):
        raise RuntimeError("double-count guard failed: at-infinity events must have unit proxy transmissivity")
    return result


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run(output_dir: Path, quick: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    omega = frequency_grid(quick)
    rows: list[dict[str, object]] = []
    gamma_matrix = np.empty((len(SPECIES), len(omega)), dtype=float)

    for si, species in enumerate(SPECIES):
        print(f"[greybody] {species.slug}: {len(omega)} modes", flush=True)
        for wi, value in enumerate(omega):
            diagnostics = converged_mode(float(value), species.potential)
            if species.slug == "dirac":
                partner = converged_mode(float(value), potential_dirac_minus)
                partner_scale = max(float(diagnostics["gamma"]), float(partner["gamma"]), 1.0e-14)
                partner_delta = abs(float(diagnostics["gamma"]) - float(partner["gamma"]))
                partner_delta_log = abs(
                    np.log(float(diagnostics["gamma"]) / float(partner["gamma"]))
                )
                diagnostics["dirac_partner_gamma"] = float(partner["gamma"])
                diagnostics["dirac_partner_delta_abs"] = partner_delta
                diagnostics["dirac_partner_delta_rel"] = partner_delta / partner_scale
                diagnostics["dirac_partner_delta_log"] = partner_delta_log
                diagnostics["dirac_partner_accepted"] = bool(partner["accepted"]) and (
                    partner_delta_log <= 6.0e-3
                )
                diagnostics["accepted"] = bool(diagnostics["accepted"]) and bool(
                    diagnostics["dirac_partner_accepted"]
                )
            else:
                diagnostics["dirac_partner_gamma"] = np.nan
                diagnostics["dirac_partner_delta_abs"] = np.nan
                diagnostics["dirac_partner_delta_rel"] = np.nan
                diagnostics["dirac_partner_delta_log"] = np.nan
                diagnostics["dirac_partner_accepted"] = True
            diagnostics.update(
                species=species.slug,
                label=species.label,
                spin=species.spin,
                multipole=species.multipole,
            )
            gamma_matrix[si, wi] = float(diagnostics["gamma"])
            rows.append(diagnostics)

    rejected = [row for row in rows if not bool(row["accepted"])]
    if rejected:
        examples = ", ".join(f"{r['species']}@{float(r['omega']):.4g}" for r in rejected[:8])
        raise RuntimeError(f"{len(rejected)} modes failed declared convergence criteria: {examples}")

    schedule = scalar_at_infinity_schedule(omega, gamma_matrix[0])
    csv_path = output_dir / "greybody_modes.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    npz_path = output_dir / "greybody.npz"
    np.savez_compressed(
        npz_path,
        omega=omega,
        species=np.asarray([s.slug for s in SPECIES]),
        labels=np.asarray([s.label for s in SPECIES]),
        gamma=gamma_matrix,
        **schedule,
    )

    scalar_low_ratio = gamma_matrix[0, :3] / (4.0 * omega[:3] ** 2)
    metadata = {
        "scope": "lowest multipoles in Schwarzschild, 2M=1; not inclusive luminosity",
        "quick": quick,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "ode_method": "DOP853",
        "frequency_count": int(len(omega)),
        "frequency_min": float(omega[0]),
        "frequency_max": float(omega[-1]),
        "accepted_modes": len(rows),
        "max_raw_wronskian_residual_ill_conditioned_for_tiny_gamma": max(
            float(r["final_wronskian_residual"]) for r in rows
        ),
        "max_normalized_flux_balance_residual": max(float(r["final_flux_balance_residual"]) for r in rows),
        "max_boundary_relative_change": max(float(r["delta_boundary_rel"]) for r in rows),
        "max_tolerance_relative_change": max(float(r["delta_tolerance_rel"]) for r in rows),
        "max_max_step_relative_change": max(float(r["delta_max_step_rel"]) for r in rows),
        "max_boundary_log_change": max(float(r["delta_boundary_log"]) for r in rows),
        "max_tolerance_log_change": max(float(r["delta_tolerance_log"]) for r in rows),
        "max_max_step_log_change": max(float(r["delta_max_step_log"]) for r in rows),
        "max_dirac_partner_relative_difference": max(
            float(r["dirac_partner_delta_rel"]) for r in rows if r["species"] == "dirac"
        ),
        "scalar_low_frequency_ratios": scalar_low_ratio.tolist(),
        "sampling_model": "mid-quantiles of the scalar number spectrum already transmitted to future null infinity",
        "greybody_factor_usage": "Gamma enters the sampling density once and is not reapplied as channel attenuation",
        "double_count_guard": "channel_transmissivity is identically one for event-conditioned emitted modes",
        "schedule": {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in schedule.items()},
        "sha256": {csv_path.name: sha256(csv_path), npz_path.name: sha256(npz_path)},
    }
    (output_dir / "greybody_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        "[greybody] all modes passed; at-infinity omega quantiles =",
        np.round(schedule["omega_at_infinity"], 6),
        "; proxy transmissivity = 1",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--quick", action="store_true", help="reduced grid for smoke tests only")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.output_dir.resolve(), quick=args.quick)
