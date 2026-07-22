#!/usr/bin/env python3
"""Reference-tagged correlation-transfer toy model conditioned at infinity.

Registers are Q(1 reference) | S(8 system qubits) | R(8 vacuum modes).  Q and
S0 start in a Bell state.  S is scrambled and each selected emitted event is
represented by a fresh R_j.  The frequencies are deterministic quantiles of
the scalar number spectrum already transmitted to future null infinity.
Therefore each event-conditioned proxy transfer has unit transmissivity:
the physical greybody factor has already entered the sampling law and is not
applied again as a partial-swap probability.

The observable S(R_t) is *not* called a Page curve here.  Because rho_Q is
fixed and maximally mixed, the ensemble is not Page's Haar ensemble on all
nine active qubits.  Even with full proxy transfers this finite construction
is not a microscopic model of complete evaporation.

Haar, finite-depth brickwork, and identity dynamics are compared.  The
frequency quantiles are quadrature representatives, not a chronological
evaporation history, and do not modify the qubit gate in this conditioned
model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np


NQ, NS, NR = 1, 8, 8
NTOT = NQ + NS + NR
LN2 = np.log(2.0)


def haar(dimension: int, rng: np.random.Generator) -> np.ndarray:
    z = (rng.standard_normal((dimension, dimension)) + 1j * rng.standard_normal((dimension, dimension))) / np.sqrt(2.0)
    q, r = np.linalg.qr(z)
    diagonal = np.diagonal(r)
    phases = diagonal / np.abs(diagonal)
    return q * phases


def apply_group(psi: np.ndarray, unitary: np.ndarray, axes: list[int]) -> np.ndarray:
    nqubits = psi.ndim
    permutation = list(axes) + [axis for axis in range(nqubits) if axis not in axes]
    inverse = np.argsort(permutation)
    matrix = np.transpose(psi, permutation).reshape(2 ** len(axes), -1)
    matrix = unitary @ matrix
    return np.transpose(matrix.reshape((2,) * nqubits), inverse)


def apply_two_qubit(psi: np.ndarray, unitary: np.ndarray, axis_a: int, axis_b: int) -> np.ndarray:
    return apply_group(psi, unitary, [axis_a, axis_b])


def partial_swap(transmissivity: float) -> np.ndarray:
    if not 0.0 <= transmissivity <= 1.0:
        raise ValueError(f"transmissivity outside [0,1]: {transmissivity}")
    theta = np.arcsin(np.sqrt(transmissivity))
    swap = np.asarray(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
        dtype=complex,
    )
    return np.cos(theta) * np.eye(4, dtype=complex) + 1j * np.sin(theta) * swap


def entropy(psi: np.ndarray, keep: list[int]) -> float:
    """Von Neumann entropy in nats, evaluated through the smaller Schmidt side."""

    rest = [axis for axis in range(psi.ndim) if axis not in keep]
    if len(keep) > len(rest):
        keep, rest = rest, keep
    matrix = np.transpose(psi, keep + rest).reshape(2 ** len(keep), -1)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    probabilities = singular_values**2
    probabilities = probabilities[probabilities > 1.0e-14]
    return float(-np.sum(probabilities * np.log(probabilities)))


def initial_state() -> np.ndarray:
    psi = np.zeros((2,) * NTOT, dtype=complex)
    zero = [0] * NTOT
    psi[tuple(zero)] = 1.0 / np.sqrt(2.0)
    one = zero.copy()
    one[0] = 1
    one[1] = 1
    psi[tuple(one)] = 1.0 / np.sqrt(2.0)
    return psi


def brickwork(psi: np.ndarray, axes: list[int], depth: int, rng: np.random.Generator) -> np.ndarray:
    for layer in range(depth):
        for local in range(layer % 2, len(axes) - 1, 2):
            psi = apply_two_qubit(psi, haar(4, rng), axes[local], axes[local + 1])
    return psi


def evaluate_schedule(psi: np.ndarray, gammas: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    q_axis = 0
    s_axes = list(range(1, 1 + NS))
    r_axes = list(range(1 + NS, NTOT))
    s_r = np.zeros(NR, dtype=float)
    i_qr = np.zeros(NR, dtype=float)
    for step in range(NR):
        psi = apply_two_qubit(psi, partial_swap(float(gammas[step])), s_axes[step], r_axes[step])
        s_r[step] = entropy(psi, r_axes[: step + 1])
        s_qr = entropy(psi, [q_axis] + r_axes[: step + 1])
        i_qr[step] = LN2 + s_r[step] - s_qr
    a_op = 1.0 - i_qr / (2.0 * LN2)
    norm_error = abs(float(np.vdot(psi, psi).real) - 1.0)
    return s_r, i_qr, a_op, norm_error


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(spectrum_path: Path, output_dir: Path, ensemble_size: int, seed: int, brick_depth: int) -> None:
    if ensemble_size < 2:
        raise ValueError("ensemble-size must be at least two")
    spectrum = np.load(spectrum_path, allow_pickle=False)
    event_transmissivity = np.asarray(spectrum["channel_transmissivity"], dtype=float)
    omega_at_infinity = np.asarray(spectrum["omega_at_infinity"], dtype=float)
    gamma_in_sampling_law = np.asarray(spectrum["gamma_at_sampled_frequencies"], dtype=float)
    if event_transmissivity.shape != (NR,) or omega_at_infinity.shape != (NR,):
        raise ValueError("expected eight at-infinity sampling nodes and transfer entries")
    if gamma_in_sampling_law.shape != (NR,):
        raise ValueError("expected eight diagnostic greybody factors at the sampled frequencies")
    if not np.array_equal(event_transmissivity, np.ones(NR)):
        raise RuntimeError(
            "double-count guard: a spectrum sampled after greybody transmission must not reapply Gamma"
        )
    probe = np.asarray([0.0, 0.0, 1.0, 0.0], dtype=complex)  # |10> in (S,R) ordering
    transferred = partial_swap(1.0) @ probe
    if not np.isclose(abs(transferred[1]) ** 2, 1.0) or not np.isclose(abs(transferred[2]) ** 2, 0.0):
        raise RuntimeError("port-convention guard: Gamma=1 must transfer |10>_SR to |01>_SR")
    schedules = {
        "haar_infinity": event_transmissivity,
        "brick_infinity": event_transmissivity,
        "identity_infinity": event_transmissivity,
    }
    scenarios = tuple(schedules)
    quantities = ("S_R", "I_QR", "A_op")
    data = {name: {quantity: np.zeros((ensemble_size, NR)) for quantity in quantities} for name in scenarios}
    norm_errors = {name: np.zeros(ensemble_size) for name in scenarios}

    root = np.random.SeedSequence(seed)
    haar_root, brick_root = root.spawn(2)
    haar_seeds = haar_root.spawn(ensemble_size)
    brick_seeds = brick_root.spawn(ensemble_size)
    s_axes = list(range(1, 1 + NS))

    identity_result = evaluate_schedule(initial_state(), schedules["identity_infinity"])
    for quantity, values in zip(quantities, identity_result[:3], strict=True):
        data["identity_infinity"][quantity][:] = values
    norm_errors["identity_infinity"][:] = identity_result[3]

    for realization in range(ensemble_size):
        rng_haar = np.random.default_rng(haar_seeds[realization])
        scrambled = apply_group(initial_state(), haar(2**NS, rng_haar), s_axes)
        result = evaluate_schedule(scrambled.copy(), schedules["haar_infinity"])
        for quantity, values in zip(quantities, result[:3], strict=True):
            data["haar_infinity"][quantity][realization] = values
        norm_errors["haar_infinity"][realization] = result[3]

        rng_brick = np.random.default_rng(brick_seeds[realization])
        brick_state = brickwork(initial_state(), s_axes, brick_depth, rng_brick)
        result = evaluate_schedule(brick_state, schedules["brick_infinity"])
        for quantity, values in zip(quantities, result[:3], strict=True):
            data["brick_infinity"][quantity][realization] = values
        norm_errors["brick_infinity"][realization] = result[3]

        if (realization + 1) % max(1, ensemble_size // 8) == 0:
            print(f"[pagesim] {realization + 1}/{ensemble_size}", flush=True)

    for scenario in scenarios:
        if np.max(norm_errors[scenario]) > 2.0e-10:
            raise RuntimeError(f"normalization failure in {scenario}")
        i_qr = data[scenario]["I_QR"]
        a_op = data[scenario]["A_op"]
        if np.min(i_qr) < -2.0e-10 or np.max(i_qr) > 2.0 * LN2 + 2.0e-10:
            raise RuntimeError(f"mutual-information bound violated in {scenario}")
        if np.min(np.diff(i_qr, axis=1)) < -2.0e-9:
            raise RuntimeError(f"nested-radiation monotonicity violated in {scenario}")
        if np.min(a_op) < -2.0e-10 or np.max(a_op) > 1.0 + 2.0e-10:
            raise RuntimeError(f"A_op bound violated in {scenario}")

    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "channel_ensemble.npz"
    payload: dict[str, np.ndarray] = {
        "scenarios": np.asarray(scenarios),
        "steps": np.arange(1, NR + 1),
        "omega_at_infinity": omega_at_infinity,
        "gamma_in_sampling_law": gamma_in_sampling_law,
        "channel_transmissivity": event_transmissivity,
    }
    for scenario in scenarios:
        payload[f"gammas__{scenario}"] = schedules[scenario]
        payload[f"norm_error__{scenario}"] = norm_errors[scenario]
        for quantity in quantities:
            payload[f"{quantity}__{scenario}"] = data[scenario][quantity]
    np.savez_compressed(npz_path, **payload)

    csv_path = output_dir / "channel_ensemble_long.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "realization",
                "scenario",
                "collected_event",
                "omega_at_infinity",
                "gamma_used_only_in_sampling_law",
                "event_conditioned_transmissivity",
                "S_R_nats",
                "I_QR_nats",
                "A_op",
            )
        )
        for scenario in scenarios:
            for realization in range(ensemble_size):
                for index in range(NR):
                    writer.writerow(
                        (
                            realization,
                            scenario,
                            index + 1,
                            omega_at_infinity[index],
                            gamma_in_sampling_law[index],
                            schedules[scenario][index],
                            data[scenario]["S_R"][realization, index],
                            data[scenario]["I_QR"][realization, index],
                            data[scenario]["A_op"][realization, index],
                        )
                    )

    metadata = {
        "scope": "17-qubit reference-tagged event-conditioned correlation-transfer toy model; not a microscopic evaporation model",
        "ensemble_size": ensemble_size,
        "master_seed": seed,
        "bit_generator": "PCG64 via numpy.random.default_rng",
        "sampling_model": "frequency nodes are mid-quantiles of the number spectrum already transmitted to future null infinity",
        "greybody_factor_usage": "Gamma is used by greybody.py in the sampling density only; it is not reapplied by this channel",
        "double_count_guard_passed": bool(np.array_equal(event_transmissivity, np.ones(NR))),
        "port_convention_guard_passed": True,
        "brick_depth": brick_depth,
        "identity_control_is_deterministic": True,
        "proxy_caveat": "a unit-transmissivity swap is conditioned on an emitted event at infinity and is not a microscopic source-to-radiation map",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "max_norm_error": {scenario: float(np.max(norm_errors[scenario])) for scenario in scenarios},
        "sha256": {npz_path.name: sha256(npz_path), csv_path.name: sha256(csv_path)},
    }
    (output_dir / "channel_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("[pagesim] all algebraic and normalization checks passed", flush=True)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectrum", type=Path, default=root / "data" / "greybody.npz")
    parser.add_argument("--output-dir", type=Path, default=root / "data")
    parser.add_argument("--ensemble-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--brick-depth", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(
        arguments.spectrum.resolve(),
        arguments.output_dir.resolve(),
        arguments.ensemble_size,
        arguments.seed,
        arguments.brick_depth,
    )
