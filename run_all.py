#!/usr/bin/env python3
"""Run the complete deterministic analysis pipeline from the project root."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def execute(arguments: list[str], environment: dict[str, str]) -> None:
    print("[run_all]", " ".join(arguments), flush=True)
    subprocess.run(arguments, cwd=ROOT, env=environment, check=True)


def write_manifest() -> None:
    paths = sorted(
        path
        for directory in (ROOT / "data", ROOT / "figures")
        for path in directory.rglob("*")
        if path.is_file()
    )
    paths.extend(sorted(ROOT.glob("*.py")))
    paths.extend(sorted(ROOT.glob("requirements*.txt")))
    paths.extend(sorted(ROOT.glob("*.tex")))
    paths.extend(sorted(ROOT.glob("*.bib")))
    paths.extend(sorted(ROOT.glob("*.md")))
    paths.extend(sorted(ROOT.glob("*.cls")))
    paths.extend(sorted(ROOT.glob("*.bst")))
    built_pdf = ROOT / "build" / "main_REV10_SciPost_BibTeX.pdf"
    if built_pdf.exists():
        paths.append(built_pdf)
    lines = []
    for path in sorted(set(paths)):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(ROOT).as_posix()}")
    (ROOT / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="smoke test with reduced grids/ensembles")
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="refresh MANIFEST.sha256 after compiling the manuscript without rerunning simulations",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.manifest_only:
        write_manifest()
        print("[run_all] manifest refreshed", flush=True)
        raise SystemExit(0)
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
    # Fixed single-threaded BLAS avoids machine-dependent oversubscription and
    # makes the recorded wall-clock behavior reproducible.
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    data = ROOT / ("data_quick" if args.quick else "data")
    figures = ROOT / ("figures_quick" if args.quick else "figures")
    data.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)

    greybody_command = [sys.executable, "greybody.py", "--output-dir", str(data)]
    if args.quick:
        greybody_command.append("--quick")
    execute(greybody_command, env)
    execute(
        [
            sys.executable,
            "inclusive_h3.py",
            "--data-dir",
            str(data),
        ],
        env,
    )
    execute(
        [
            sys.executable,
            "pagesim.py",
            "--spectrum",
            str(data / "greybody.npz"),
            "--output-dir",
            str(data),
            "--ensemble-size",
            "16" if args.quick else "128",
        ],
        env,
    )
    execute(
        [
            sys.executable,
            "yysim.py",
            "--output-dir",
            str(data),
            "--ensemble-size",
            "32" if args.quick else "256",
        ],
        env,
    )
    execute([sys.executable, "plots.py", "--data-dir", str(data), "--figure-dir", str(figures)], env)
    if not args.quick:
        write_manifest()
    print("[run_all] pipeline completed without failed checks", flush=True)
