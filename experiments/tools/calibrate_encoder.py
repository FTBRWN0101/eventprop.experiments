"""Calibrate a delta encoder's threshold, by reconstruction quality and by firing rate.

Two criteria, because they answer different questions.

**Reconstruction (SNR).** Petro, Kasabov & Kiss (IEEE TNNLS 2020) optimise temporal
contrast encoders by the error between the original signal and the signal rebuilt from
the spike train, and recommend SNR as the metric. Each spike stands for a move of
+/-theta, so the rebuild is ``x[0] + cumsum(spikes * theta)``. This picks the best
threshold for an encoder on its own terms, independent of any downstream model.

**Firing rate.** delta_adaptive fires markedly more than static delta at multiplier
1.0, so a win in the encoder comparison would be unattributable -- better events, or
merely more of them. Matching the train firing rate leaves "which events" as the only
difference.

Both read only the train split, never a validation score, so neither is model selection.

    python experiments/tools/calibrate_encoder.py --input-window 45

Notes on method. The earlier version of this tool bisected on firing rate, assuming the
rate falls monotonically as the threshold rises. That is obvious for TBR but not
provable for the accumulator, where raising the threshold changes *which* steps fire and
can defer a spike into a step that fires anyway. This version sweeps a grid, checks
monotonicity empirically, reports the curve, and only then refines -- so a violation is
visible in the output instead of silently corrupting a bisection.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  #noqa: E402

from core.config import ExperimentConfig  #noqa: E402
from core.dataset import VrpDataset  #noqa: E402
from encoders.base import ENCODERS  #noqa: E402

#geometric so the low end is resolved as finely as the high end
GRID = np.geomspace(0.05, 20.0, 61)
REFINE_STEPS = 24


def reconstruct(X: np.ndarray, up: np.ndarray, down: np.ndarray,
                theta: np.ndarray) -> np.ndarray:
    """Rebuild ``X`` from its spikes: each spike is a step of +/-theta from ``x[0]``."""
    steps = (up.astype(float) - down.astype(float)) * theta
    walked = X[:, :1, :] + np.cumsum(steps, axis=1)
    return np.concatenate([X[:, :1, :], walked], axis=1)


def snr_db(X: np.ndarray, recon: np.ndarray) -> float:
    """SNR in dB of the reconstruction.

    Signal power is measured about each window's own mean, not about zero: the encoder
    is given ``x[0]`` for free, so counting the level would credit it for information
    it never transmitted.
    """
    signal = X - X.mean(axis=1, keepdims=True)
    noise = X - recon
    signal_power = float(np.sum(signal ** 2))
    noise_power = float(np.sum(noise ** 2))
    if noise_power <= 0.0:
        return float("inf")
    if signal_power <= 0.0:
        return float("-inf")
    return 10.0 * np.log10(signal_power / noise_power)


def spikes_per_cell(up: np.ndarray, down: np.ndarray) -> float:
    """Mean spikes per (example, step, feature). Differencing eats one step."""
    cells = up.size
    return float((up.sum() + down.sum()) / cells) if cells else float("nan")


def _measure(encoding: str, multiplier: float, X: np.ndarray,
             scale: np.ndarray | None) -> tuple[float, float]:
    encoder = ENCODERS.get(encoding)(multiplier=multiplier)
    encoder.fit(X, scale)
    up, down, theta = encoder.spike_masks(X, scale)
    return spikes_per_cell(up, down), snr_db(X, reconstruct(X, up, down, theta))


def sweep(encoding: str, X: np.ndarray, scale: np.ndarray | None,
          grid: np.ndarray = GRID) -> dict[str, np.ndarray]:
    rates, snrs = [], []
    for multiplier in grid:
        rate, snr = _measure(encoding, float(multiplier), X, scale)
        rates.append(rate)
        snrs.append(snr)
    return {"multiplier": np.asarray(grid), "rate": np.asarray(rates),
            "snr": np.asarray(snrs)}


def monotonicity_violations(rate: np.ndarray) -> int:
    """How many grid steps show the rate *rising* as the threshold rises."""
    return int(np.sum(np.diff(rate) > 1e-12))


def match_rate(encoding: str, X: np.ndarray, scale: np.ndarray | None,
               goal: float, curve: dict[str, np.ndarray]) -> float:
    """Multiplier whose firing rate is closest to *goal*.

    Grid-nearest first, then a local bisection between the neighbouring grid points.
    Never assumes global monotonicity -- only that the rate is well behaved between two
    adjacent grid points, which spans a factor of ~1.1.
    """
    grid, rates = curve["multiplier"], curve["rate"]
    index = int(np.argmin(np.abs(rates - goal)))
    low = grid[max(index - 1, 0)]
    high = grid[min(index + 1, len(grid) - 1)]
    best, best_error = float(grid[index]), abs(rates[index] - goal)

    for _ in range(REFINE_STEPS):
        mid = 0.5 * (low + high)
        rate, _ = _measure(encoding, mid, X, scale)
        if abs(rate - goal) < best_error:
            best, best_error = mid, abs(rate - goal)
        if rate > goal:
            low = mid
        else:
            high = mid
    return best


def calibrate(config: ExperimentConfig, candidate: str = "delta_adaptive",
              reference: str = "delta") -> dict[str, object]:
    ENCODERS.discover(_ROOT / "encoders", "encoders")
    data = VrpDataset(config)

    X, _, _ = data.sequences("train", space="raw")
    scale = data.window_scale

    reference_curve = sweep(reference, X, None)
    candidate_curve = sweep(candidate, X, scale)

    reference_rate, reference_snr = _measure(reference, config.delta_multiplier, X, None)
    best_snr_index = int(np.argmax(candidate_curve["snr"]))
    matched = match_rate(candidate, X, scale, reference_rate, candidate_curve)
    matched_rate, matched_snr = _measure(candidate, matched, X, scale)

    return {
        "windows": X.shape,
        "reference": reference,
        "reference_multiplier": config.delta_multiplier,
        "reference_rate": reference_rate,
        "reference_snr": reference_snr,
        "reference_best_snr_multiplier": float(
            reference_curve["multiplier"][int(np.argmax(reference_curve["snr"]))]),
        "reference_best_snr": float(np.max(reference_curve["snr"])),
        "candidate": candidate,
        "candidate_best_snr_multiplier": float(
            candidate_curve["multiplier"][best_snr_index]),
        "candidate_best_snr": float(candidate_curve["snr"][best_snr_index]),
        "candidate_best_snr_rate": float(candidate_curve["rate"][best_snr_index]),
        "matched_multiplier": matched,
        "matched_rate": matched_rate,
        "matched_snr": matched_snr,
        "reference_violations": monotonicity_violations(reference_curve["rate"]),
        "candidate_violations": monotonicity_violations(candidate_curve["rate"]),
        "grid_size": len(GRID),
        "curves": {reference: reference_curve, candidate: candidate_curve},
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Calibrate delta encoder thresholds.")
    parser.add_argument("--horizon", default="weekly")
    parser.add_argument("--target", default="vrp")
    parser.add_argument("--iv-leg", default="vix9d")
    parser.add_argument("--feature-set", default="options+price")
    parser.add_argument("--input-window", type=int, default=45)
    parser.add_argument("--reference", default="delta")
    parser.add_argument("--candidate", default="delta_adaptive")
    parser.add_argument("--show-curve", action="store_true",
                        help="Print the full multiplier/rate/SNR sweep.")
    args = parser.parse_args(argv)

    config = ExperimentConfig.load(
        horizon=args.horizon, target=args.target, iv_leg=args.iv_leg,
        feature_set=args.feature_set, input_window=args.input_window)
    out = calibrate(config, args.candidate, args.reference)

    print(f"train windows {out['windows']}, grid of {out['grid_size']} multipliers\n")
    print(f"{out['reference']} @ x{out['reference_multiplier']:g} (the control arm)")
    print(f"  spikes/cell {out['reference_rate']:.4f}   SNR {out['reference_snr']:.2f} dB")
    print(f"  its own best-SNR multiplier would be x{out['reference_best_snr_multiplier']:.3f}"
          f" ({out['reference_best_snr']:.2f} dB)\n")
    print(f"{out['candidate']}")
    print(f"  best SNR      x{out['candidate_best_snr_multiplier']:.3f} -> "
          f"{out['candidate_best_snr']:.2f} dB at {out['candidate_best_snr_rate']:.4f} spikes/cell")
    print(f"  rate matched  x{out['matched_multiplier']:.3f} -> "
          f"{out['matched_snr']:.2f} dB at {out['matched_rate']:.4f} spikes/cell")
    print(f"  (target rate was {out['reference_rate']:.4f})\n")

    for name in (out["reference"], out["candidate"]):
        violations = out[f"{'reference' if name == out['reference'] else 'candidate'}_violations"]
        verdict = "monotone" if violations == 0 else f"NON-MONOTONE at {violations} grid steps"
        print(f"  rate vs threshold, {name}: {verdict}")

    if args.show_curve:
        print("\nmultiplier      rate       SNR(dB)   encoder")
        for name, curve in out["curves"].items():
            for m, r, s in zip(curve["multiplier"], curve["rate"], curve["snr"]):
                print(f"{m:10.4f} {r:10.4f} {s:10.2f}   {name}")


if __name__ == "__main__":
    main()
