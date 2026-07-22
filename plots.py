#!/usr/bin/env python3
"""Generate publication figures and auditable statistical summaries.

Pointwise uncertainty intervals are 95% BCa bootstrap intervals for ensemble
means.  Schedule contrasts are formed realization by realization before
resampling.  Their shaded curve bands use a separately labelled 95% bootstrap
max-deviation construction; they are not called max-t bands.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.lines import Line2D
from scipy.stats import DegenerateDataWarning, bootstrap


LN2 = np.log(2.0)
N_RESAMPLES = 20_000

plt.rcParams.update(
    {
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7.2,
        "figure.dpi": 160,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def bca_mean(samples: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = np.asarray(samples, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    mean = np.mean(array, axis=0)
    low = np.empty(array.shape[1])
    high = np.empty(array.shape[1])
    for index in range(array.shape[1]):
        column = array[:, index]
        if np.ptp(column) <= 1.0e-14:
            low[index] = high[index] = mean[index]
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("error", DegenerateDataWarning)
            result = bootstrap(
                (column,),
                np.mean,
                confidence_level=0.95,
                n_resamples=N_RESAMPLES,
                method="BCa",
                rng=rng,
            )
        low[index] = float(result.confidence_interval.low)
        high[index] = float(result.confidence_interval.high)
    return mean, low, high


def simultaneous_mean_band(
    samples: np.ndarray,
    rng: np.random.Generator,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = np.asarray(samples, dtype=float)
    n = array.shape[0]
    mean = np.mean(array, axis=0)
    indices = rng.integers(0, n, size=(N_RESAMPLES, n))
    bootstrap_means = np.mean(array[indices], axis=1)
    radius = float(np.quantile(np.max(np.abs(bootstrap_means - mean), axis=1), 0.95))
    low = mean - radius
    high = mean + radius
    if lower_bound is not None:
        low = np.maximum(low, lower_bound)
    if upper_bound is not None:
        high = np.minimum(high, upper_bound)
    return mean, low, high


def save_figure(fig: plt.Figure, figure_dir: Path, stem: str) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / f"{stem}.pdf")
    fig.savefig(figure_dir / f"{stem}.png", dpi=600)
    plt.close(fig)


def figure_greybody(data_dir: Path, figure_dir: Path) -> None:
    gb = np.load(data_dir / "greybody.npz", allow_pickle=False)
    omega = gb["omega"]
    species = [str(item) for item in gb["species"]]
    gamma = gb["gamma"]
    spectrum_omega = gb["spectrum_omega"]
    transmitted_density = gb["transmitted_number_density"]
    omega_at_infinity = gb["omega_at_infinity"]

    with (data_dir / "greybody_modes.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.0))
    styles = {
        "scalar": (r"scalar $s=0,\ell=0$", "-"),
        "dirac": (r"Dirac $s=1/2,j=1/2$", "--"),
        "gauge": (r"gauge $s=1,\ell=1$", "-."),
        "graviton": (r"graviton $s=2,\ell=2$", ":"),
    }
    for index, slug in enumerate(species):
        label, linestyle = styles[slug]
        axes[0].semilogy(omega, gamma[index], linestyle, linewidth=1.5, label=label)
    low = omega <= 0.05
    axes[0].semilogy(omega[low], 4.0 * omega[low] ** 2, color="0.35", linewidth=0.9, label=r"$4\omega^2$ asymptote")
    axes[0].set_xlabel(r"dimensionless frequency $2M\omega$")
    axes[0].set_ylabel(r"transmission $\Gamma_{s\ell}$")
    axes[0].set_ylim(1.0e-14, 1.4)
    axes[0].set_title("(a) Lowest-multipole transmissions")
    axes[0].legend(loc="lower right")

    convergence_colors = dict(zip(species, ("C0", "C1", "C2", "C3"), strict=True))
    species_handles = []
    for slug in species:
        selected = [row for row in rows if row["species"] == slug]
        selected_omega = np.asarray([float(row["omega"]) for row in selected])
        boundary_change = np.asarray([float(row["delta_boundary_rel"]) for row in selected])
        tolerance_change = np.asarray([float(row["delta_tolerance_rel"]) for row in selected])
        step_change = np.asarray([float(row["delta_max_step_rel"]) for row in selected])
        color = convergence_colors[slug]
        axes[1].loglog(selected_omega, boundary_change, color=color)
        axes[1].loglog(selected_omega, tolerance_change, color=color, linestyle=":")
        axes[1].loglog(selected_omega, step_change, color=color, linestyle="--")
        species_handles.append(Line2D([0], [0], color=color, label=slug))
    axes[1].axhline(5.0e-3, color="0.25", linestyle="-.", linewidth=0.8)
    axes[1].set_xlabel(r"dimensionless frequency $2M\omega$")
    axes[1].set_ylabel("relative change in transmission")
    axes[1].set_title("(b) Numerical convergence diagnostics")
    species_legend = axes[1].legend(handles=species_handles, ncol=2, loc="lower left", fontsize=6.2, title="field", title_fontsize=6.2)
    axes[1].add_artist(species_legend)
    diagnostic_handles = [
        Line2D([0], [0], color="0.2", linestyle="-", label="boundary extension"),
        Line2D([0], [0], color="0.2", linestyle=":", label="tolerance tightening"),
        Line2D([0], [0], color="0.2", linestyle="--", label="maximum-step halving"),
        Line2D([0], [0], color="0.2", linestyle="-.", label="boundary log-ratio limit"),
    ]
    axes[1].legend(handles=diagnostic_handles, loc="upper right", fontsize=5.9)

    axes[2].semilogx(spectrum_omega, transmitted_density, color="C0", linewidth=1.4)
    density_at_nodes = np.interp(omega_at_infinity, spectrum_omega, transmitted_density)
    axes[2].scatter(omega_at_infinity, density_at_nodes, color="C3", s=17, zorder=3, label="mid-quantile nodes")
    axes[2].set_xlabel(r"dimensionless frequency $2M\omega$")
    axes[2].set_ylabel(r"normalized $p^\infty_{00}(\omega)$")
    axes[2].set_title("(c) Spectrum already transmitted to infinity")
    axes[2].legend(loc="upper right", fontsize=6.5)
    fig.tight_layout()
    save_figure(fig, figure_dir, "fig_greybody_repro")


def figure_h3_inclusive(data_dir: Path, figure_dir: Path) -> None:
    h3 = np.load(data_dir / "h3_inclusive.npz", allow_pickle=False)
    species = [str(item) for item in h3["species"]]
    labels = {
        "scalar": r"real scalar $s=0$",
        "weyl": r"Weyl $s=1/2$",
        "photon": r"photon $s=1$",
        "graviton": r"graviton $s=2$",
    }
    styles = {
        "scalar": ("C0", "-"),
        "weyl": ("C1", "--"),
        "photon": ("C2", "-."),
        "graviton": ("C3", ":"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.05))
    for slug in species:
        omega = h3[f"omega__{slug}"]
        color, linestyle = styles[slug]
        axes[0].plot(
            omega,
            h3[f"p_number__{slug}"],
            color=color,
            linestyle=linestyle,
            linewidth=1.5,
            label=labels[slug],
        )
        axes[1].plot(
            omega,
            h3[f"p_power__{slug}"],
            color=color,
            linestyle=linestyle,
            linewidth=1.5,
            label=labels[slug],
        )
    for axis in axes[:2]:
        axis.set_xlim(0.0, 1.15)
        axis.set_xlabel(r"dimensionless frequency $2M\omega$")
    axes[0].set_ylabel(r"normalized number density $p_s^{N}(\omega)$")
    axes[0].set_title("(a) Inclusive number spectra")
    axes[0].legend(loc="upper right", fontsize=6.7)
    axes[1].set_ylabel(r"normalized power density $p_s^{P}(\omega)$")
    axes[1].set_title("(b) Inclusive power spectra")

    normalized_jsd = h3["jsd_number"] / np.log(2.0)
    heatmap = axes[2].imshow(normalized_jsd, vmin=0.0, vmax=1.0, cmap="magma")
    short_labels = ("scalar", "Weyl", "photon", "graviton")
    axes[2].set_xticks(np.arange(len(species)), short_labels, rotation=35, ha="right")
    axes[2].set_yticks(np.arange(len(species)), short_labels)
    axes[2].set_title(r"(c) Number-spectrum $D_{\rm JS}/\ln2$")
    for row in range(len(species)):
        for column in range(len(species)):
            value = normalized_jsd[row, column]
            axes[2].text(
                column,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                color="white" if value < 0.25 or value > 0.65 else "black",
                fontsize=6.5,
            )
    colorbar = fig.colorbar(heatmap, ax=axes[2], fraction=0.046, pad=0.04)
    colorbar.set_label(r"$D_{\rm JS}/\ln2$")
    fig.tight_layout()
    save_figure(fig, figure_dir, "fig_h3_inclusive")


def plot_bca_curve(
    axis: plt.Axes,
    x: np.ndarray,
    samples: np.ndarray,
    label: str,
    color: str,
    rng: np.random.Generator,
    linestyle: str = "-",
) -> None:
    mean, low, high = bca_mean(samples, rng)
    axis.plot(x, mean, color=color, linestyle=linestyle, label=label)
    axis.fill_between(x, low, high, color=color, alpha=0.18, linewidth=0)


def figure_channel(data_dir: Path, figure_dir: Path, rng: np.random.Generator) -> None:
    channel = np.load(data_dir / "channel_ensemble.npz", allow_pickle=False)
    steps = channel["steps"]
    fig, axes = plt.subplots(1, 3, figsize=(10.1, 3.0))
    curves = (
        ("haar_infinity", "Haar, at-infinity events", "C0", "-"),
        ("brick_infinity", "brickwork depth 8", "C2", "--"),
    )
    for scenario, label, color, linestyle in curves:
        plot_bca_curve(
            axes[0],
            steps,
            channel[f"S_R__{scenario}"] / LN2,
            label,
            color,
            rng,
            linestyle,
        )
    axes[0].plot(steps, channel["S_R__identity_infinity"][0] / LN2, color="0.25", linestyle=":", label="identity control")
    full_swap_bound = np.minimum(steps, 9 - steps)
    axes[0].plot(steps, full_swap_bound, color="black", linewidth=0.8, linestyle="-.", label="full-swap dimension bound")
    axes[0].set_xlabel("collected modes $t$")
    axes[0].set_ylabel(r"$S(R_t)/\ln 2$")
    axes[0].set_title("(a) Reference-tagged radiation entropy")
    axes[0].legend(loc="upper left")

    for scenario, label, color, linestyle in curves:
        plot_bca_curve(axes[1], steps, channel[f"A_op__{scenario}"], label, color, rng, linestyle)
    axes[1].plot(steps, channel["A_op__identity_infinity"][0], color="0.25", linestyle=":", label="identity control")
    axes[1].set_xlabel("collected modes $t$")
    axes[1].set_ylabel(r"$\mathcal{A}_{\rm op}=1-I(Q{:}R_t)/(2\ln2)$")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_title("(b) Correlation-transfer diagnostic")

    contrast = channel["A_op__brick_infinity"] - channel["A_op__haar_infinity"]
    mean, low, high = simultaneous_mean_band(contrast, rng)
    axes[2].plot(steps, mean, color="C2", label="brickwork - Haar")
    axes[2].fill_between(steps, low, high, color="C2", alpha=0.22, linewidth=0, label="95% max-deviation band")
    axes[2].axhline(0.0, color="0.25", linewidth=0.8, linestyle=":")
    axes[2].set_xlabel("collected modes $t$")
    axes[2].set_ylabel(r"paired $\Delta\mathcal{A}_{\rm op}$")
    axes[2].set_title("(c) Paired architecture contrast")
    axes[2].legend(loc="best")
    fig.tight_layout()
    save_figure(fig, figure_dir, "fig_channel_repro")


def figure_yy(data_dir: Path, figure_dir: Path, rng: np.random.Generator) -> None:
    yy = np.load(data_dir / "yy_ensemble.npz", allow_pickle=False)
    ds = yy["ds"]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9))
    specifications = (
        ("F_cond", r"conditional Bell fidelity $F_{\rm cond}$", "(a) Heralded fidelity"),
        ("P_success", r"Born success probability $p_{\rm succ}$", "(b) Projection success"),
        ("Y_weighted", r"weighted Bell yield $p_{\rm succ}F_{\rm cond}$", "(c) Success-weighted yield"),
    )
    for axis, (quantity, ylabel, title) in zip(axes, specifications, strict=True):
        for protocol, label, color, linestyle in (
            ("matched", "matched $U^*$", "C0", "-"),
            ("mismatched", "independent $V^*$ control", "C1", "--"),
            ("dephased", "matched with phase damping", "C3", ":"),
        ):
            mean, low, high = bca_mean(yy[f"{quantity}__{protocol}"], rng)
            axis.errorbar(ds, mean, yerr=(mean - low, high - mean), color=color, linestyle=linestyle, marker="o", markersize=3, capsize=2, label=label)
        axis.set_xlabel("projected Bell pairs $d$")
        axis.set_ylabel(ylabel)
        axis.set_ylim(-0.03, 1.03)
        axis.set_title(title)
    axes[1].axhline(0.25, color="0.3", linestyle=":", linewidth=0.8, label=r"matched large-$d$: $d_A^{-2}=1/4$")
    axes[2].axhline(0.25, color="0.3", linestyle=":", linewidth=0.8, label="matched identity $pF=1/4$")
    axes[0].legend(loc="lower right")
    axes[1].legend(loc="upper right")
    axes[2].legend(loc="upper right")
    fig.tight_layout()
    save_figure(fig, figure_dir, "fig_yy_repro")


def scalar_summary(
    rows: list[dict[str, object]],
    estimand: str,
    point: str,
    values: np.ndarray,
    rng: np.random.Generator,
    paired: bool,
) -> None:
    mean, low, high = bca_mean(np.asarray(values)[:, None], rng)
    rows.append(
        {
            "estimand": estimand,
            "point": point,
            "mean": float(mean[0]),
            "ci_low": float(low[0]),
            "ci_high": float(high[0]),
            "interval": "95% BCa bootstrap of ensemble mean",
            "paired_contrast": paired,
            "n": int(len(values)),
        }
    )


def write_summaries(data_dir: Path, rng: np.random.Generator) -> None:
    channel = np.load(data_dir / "channel_ensemble.npz", allow_pickle=False)
    yy = np.load(data_dir / "yy_ensemble.npz", allow_pickle=False)
    rows: list[dict[str, object]] = []
    scalar_summary(
        rows,
        "A_op(brickwork)-A_op(Haar)",
        "t=4",
        channel["A_op__brick_infinity"][:, 3] - channel["A_op__haar_infinity"][:, 3],
        rng,
        True,
    )
    scalar_summary(rows, "I(Q:R)/(2 ln 2), Haar at infinity", "t=4", channel["I_QR__haar_infinity"][:, 3] / (2.0 * LN2), rng, False)
    scalar_summary(rows, "F_cond matched", "d=4", yy["F_cond__matched"][:, 3], rng, False)
    scalar_summary(rows, "P_success matched", "d=7", yy["P_success__matched"][:, 6], rng, False)
    scalar_summary(
        rows,
        "F_cond(matched)-F_cond(mismatched)",
        "d=4",
        yy["F_cond__matched"][:, 3] - yy["F_cond__mismatched"][:, 3],
        rng,
        True,
    )
    scalar_summary(
        rows,
        "F_cond(matched)-F_cond(dephased)",
        "d=4",
        yy["F_cond__matched"][:, 3] - yy["F_cond__dephased"][:, 3],
        rng,
        True,
    )

    csv_path = data_dir / "results_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "pointwise_interval": "95% bias-corrected and accelerated (BCa) bootstrap",
        "curve_contrast_band": "95% bootstrap max absolute deviation of resampled mean from observed mean",
        "resamples": N_RESAMPLES,
        "rng_seed": 20260722,
        "p_values": "none; no confirmatory null test was preregistered before these production seeds",
        "greybody_sampling": "at-infinity spectrum; Gamma is not reapplied as channel attenuation",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
    }
    (data_dir / "statistics_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--figure-dir", type=Path, default=root / "figures")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    data_dir = arguments.data_dir.resolve()
    figure_dir = arguments.figure_dir.resolve()
    random = np.random.default_rng(20260722)
    figure_greybody(data_dir, figure_dir)
    figure_h3_inclusive(data_dir, figure_dir)
    figure_channel(data_dir, figure_dir, random)
    figure_yy(data_dir, figure_dir, random)
    write_summaries(data_dir, random)
    print("[plots] vector PDFs, 600-dpi PNGs, and statistical summaries written", flush=True)
