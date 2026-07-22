#!/usr/bin/env python3
"""Matched, mismatched, and dephased Yoshida--Yao decoder simulation.

Registers (16 qubits): R(1) | A(1) | B(6) | B'(6) | A'(1) | R'(1).
The matched arm applies U to AB and U* to A'B'.  One negative control replaces
U* by an independently drawn V*.  A second control applies complete phase
damping to output A after U, represented exactly as the equal mixture of the
I and Z branches, before the matched U* decoder.  For d=1,...,7 output pairs
are projected onto Phi+.  The script reports

  F_cond : Bell fidelity of RR' conditioned on the heralded projection,
  P      : exact Born probability of that projection, and
  Y=P*F  : success-weighted Bell yield.

Y is not called an unconditional decoder fidelity: a deterministic channel
would also have to specify and score every failed measurement outcome.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np


NB = 6
R_AXIS, A_AXIS = 0, 1
B_AXES = list(range(2, 2 + NB))
BP_AXES = list(range(2 + NB, 2 + 2 * NB))
AP_AXIS, RP_AXIS = 2 + 2 * NB, 3 + 2 * NB
NTOT = 4 + 2 * NB


def haar(dimension: int, rng: np.random.Generator) -> np.ndarray:
    z = (rng.standard_normal((dimension, dimension)) + 1j * rng.standard_normal((dimension, dimension))) / np.sqrt(2.0)
    q, r = np.linalg.qr(z)
    diagonal = np.diagonal(r)
    return q * (diagonal / np.abs(diagonal))


def apply_group(psi: np.ndarray, unitary: np.ndarray, axes: list[int]) -> np.ndarray:
    permutation = list(axes) + [axis for axis in range(psi.ndim) if axis not in axes]
    inverse = np.argsort(permutation)
    matrix = np.transpose(psi, permutation).reshape(2 ** len(axes), -1)
    matrix = unitary @ matrix
    return np.transpose(matrix.reshape((2,) * psi.ndim), inverse)


def apply_z(psi: np.ndarray, axis: int) -> np.ndarray:
    """Apply Pauli Z on one tensor axis without constructing a dense matrix."""

    result = psi.copy()
    selector = [slice(None)] * psi.ndim
    selector[axis] = 1
    result[tuple(selector)] *= -1.0
    return result


def bell_project(psi: np.ndarray, axis_a: int, axis_b: int) -> np.ndarray:
    permutation = [axis_a, axis_b] + [axis for axis in range(psi.ndim) if axis not in (axis_a, axis_b)]
    moved = np.transpose(psi, permutation)
    return (moved[0, 0] + moved[1, 1]) / np.sqrt(2.0)


def initial_state() -> np.ndarray:
    bell = np.asarray([[1, 0], [0, 1]], dtype=complex) / np.sqrt(2.0)
    tensor = bell
    for _ in range(NB + 1):
        tensor = np.tensordot(tensor, bell, axes=0)
    current_axis_names = [R_AXIS, A_AXIS]
    for index in range(NB):
        current_axis_names.extend((B_AXES[index], BP_AXES[index]))
    current_axis_names.extend((AP_AXIS, RP_AXIS))
    return np.transpose(tensor, np.argsort(current_axis_names))


def decode_curve(psi0: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ds = np.arange(1, NB + 2)
    fidelity = np.zeros(len(ds), dtype=float)
    probability = np.zeros(len(ds), dtype=float)
    outputs = [A_AXIS] + B_AXES
    mirror_outputs = [AP_AXIS] + BP_AXES

    for index, d_collected in enumerate(ds):
        psi = psi0
        names = list(range(NTOT))
        for pair_index in range(int(d_collected)):
            axis_a = names.index(outputs[pair_index])
            axis_b = names.index(mirror_outputs[pair_index])
            psi = bell_project(psi, axis_a, axis_b)
            del names[max(axis_a, axis_b)]
            del names[min(axis_a, axis_b)]

        success = float(np.vdot(psi, psi).real)
        probability[index] = success
        if success <= 1.0e-15:
            continue
        psi = psi / np.sqrt(success)
        r_position = names.index(R_AXIS)
        rp_position = names.index(RP_AXIS)
        permutation = [r_position, rp_position] + [
            axis for axis in range(psi.ndim) if axis not in (r_position, rp_position)
        ]
        reshaped = np.transpose(psi, permutation).reshape(2, 2, -1)
        bell_amplitude = (reshaped[0, 0] + reshaped[1, 1]) / np.sqrt(2.0)
        fidelity[index] = float(np.vdot(bell_amplitude, bell_amplitude).real)
    return fidelity, probability, fidelity * probability


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(output_dir: Path, ensemble_size: int, seed: int) -> None:
    if ensemble_size < 2:
        raise ValueError("ensemble-size must be at least two")
    ds = np.arange(1, NB + 2)
    protocols = ("matched", "mismatched", "dephased")
    arrays = {
        protocol: {
            "F_cond": np.zeros((ensemble_size, len(ds))),
            "P_success": np.zeros((ensemble_size, len(ds))),
            "Y_weighted": np.zeros((ensemble_size, len(ds))),
        }
        for protocol in protocols
    }

    root = np.random.SeedSequence(seed)
    dynamics_root, mismatch_root = root.spawn(2)
    dynamics_seeds = dynamics_root.spawn(ensemble_size)
    mismatch_seeds = mismatch_root.spawn(ensemble_size)
    initial = initial_state()
    if abs(float(np.vdot(initial, initial).real) - 1.0) > 1.0e-13:
        raise RuntimeError("initial Bell-pair state is not normalized")

    for realization in range(ensemble_size):
        rng_u = np.random.default_rng(dynamics_seeds[realization])
        rng_v = np.random.default_rng(mismatch_seeds[realization])
        u = haar(2 ** (NB + 1), rng_u)
        v = haar(2 ** (NB + 1), rng_v)
        evolved = apply_group(initial, u, [A_AXIS] + B_AXES)

        states = {
            "matched": apply_group(evolved, u.conj(), [AP_AXIS] + BP_AXES),
            "mismatched": apply_group(evolved, v.conj(), [AP_AXIS] + BP_AXES),
        }
        for protocol in ("matched", "mismatched"):
            result = decode_curve(states[protocol])
            for name, values in zip(("F_cond", "P_success", "Y_weighted"), result, strict=True):
                arrays[protocol][name][realization] = values

        # Exact complete phase damping on A: rho -> (rho + Z_A rho Z_A)/2.
        # Born probabilities and unnormalized Bell yields are linear in rho;
        # conditional fidelity is their ratio after mixing the two branches.
        dephased_branches = []
        for branch in (evolved, apply_z(evolved, A_AXIS)):
            decoded = apply_group(branch, u.conj(), [AP_AXIS] + BP_AXES)
            dephased_branches.append(decode_curve(decoded))
        probability = 0.5 * (dephased_branches[0][1] + dephased_branches[1][1])
        weighted_yield = 0.5 * (dephased_branches[0][2] + dephased_branches[1][2])
        fidelity = np.divide(
            weighted_yield,
            probability,
            out=np.zeros_like(weighted_yield),
            where=probability > 1.0e-15,
        )
        arrays["dephased"]["F_cond"][realization] = fidelity
        arrays["dephased"]["P_success"][realization] = probability
        arrays["dephased"]["Y_weighted"][realization] = weighted_yield

        if (realization + 1) % max(1, ensemble_size // 8) == 0:
            print(f"[yysim] {realization + 1}/{ensemble_size}", flush=True)

    for protocol in protocols:
        for name, values in arrays[protocol].items():
            if np.min(values) < -2.0e-12 or np.max(values) > 1.0 + 2.0e-12:
                raise RuntimeError(f"probability/fidelity bound violated: {protocol}/{name}")
    matched_identity_error = float(np.max(np.abs(arrays["matched"]["Y_weighted"] - 0.25)))
    if matched_identity_error > 5.0e-12:
        raise RuntimeError(f"matched P*F identity failed: {matched_identity_error}")

    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "yy_ensemble.npz"
    payload: dict[str, np.ndarray] = {"ds": ds, "protocols": np.asarray(protocols)}
    for protocol in protocols:
        for name, values in arrays[protocol].items():
            payload[f"{name}__{protocol}"] = values
    np.savez_compressed(npz_path, **payload)

    csv_path = output_dir / "yy_ensemble_long.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("realization", "protocol", "d_collected", "F_cond", "P_success", "Y_weighted"))
        for protocol in protocols:
            for realization in range(ensemble_size):
                for index, d_collected in enumerate(ds):
                    writer.writerow(
                        (
                            realization,
                            protocol,
                            int(d_collected),
                            arrays[protocol]["F_cond"][realization, index],
                            arrays[protocol]["P_success"][realization, index],
                            arrays[protocol]["Y_weighted"][realization, index],
                        )
                    )

    metadata = {
        "scope": "finite Haar decoder benchmark with exact statevectors, a complete single-output phase-damping control, and no deterministic decoder",
        "ensemble_size": ensemble_size,
        "master_seed": seed,
        "bit_generator": "PCG64 via numpy.random.default_rng",
        "pairing": "matched, mismatched, and dephased protocols share U in each realization; mismatch uses independent V",
        "dephasing_control": "complete phase damping on output A after U, evaluated as the exact equal mixture of I and Z branches",
        "fidelity_convention": "Bell-state overlap of normalized RR' state conditional on successful Phi+ projections",
        "success_probability_convention": "exact Born probability, not finite-shot frequency",
        "weighted_yield_caveat": "P_success*F_cond is not an unconditional deterministic-decoder fidelity",
        "matched_weighted_yield_identity_max_error": matched_identity_error,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sha256": {npz_path.name: sha256(npz_path), csv_path.name: sha256(csv_path)},
    }
    (output_dir / "yy_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("[yysim] all probability bounds and matched P*F=1/4 identity passed", flush=True)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=root / "data")
    parser.add_argument("--ensemble-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.output_dir.resolve(), arguments.ensemble_size, arguments.seed)
