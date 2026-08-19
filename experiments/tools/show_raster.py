"""Summarise and draw the spike rasters ``SnnForecaster`` writes each checkpoint.

The proposal inspects rasters at fixed intervals to catch silent-neuron pathology
early. ``models/snn.py`` writes one ``rasters/epoch<N>.npz`` per checkpoint chunk;
this reads them back without needing the mlgenn env.

    python experiments/tools/show_raster.py --list
    python experiments/tools/show_raster.py weekly_vrp_vix9d_rate_eventprop_L5_s1
    python experiments/tools/show_raster.py <cell> --epoch 20 --example 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  #noqa: E402

CHECKPOINT_ROOT = _ROOT / ".snn_checkpoints"
#rows of the drawn raster; neurons are binned into this many bands
DRAW_ROWS = 32
RAMP = " .:-=+*#%@"


def load(path: Path) -> dict:
    """Read one raster npz into a plain dict."""
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def summarise(raster: dict) -> dict:
    """Spike totals, silent-neuron count and rate for one raster file."""
    num_neurons = int(raster["num_neurons"])
    counts = np.bincount(raster["ids"].astype(int), minlength=num_neurons)
    examples = int(raster["example"].max()) + 1 if raster["example"].size else 0
    timesteps = int(raster["timesteps"])
    denominator = max(examples * timesteps, 1)
    return {
        "neurons": num_neurons,
        "examples": examples,
        "timesteps": timesteps,
        "spikes": int(raster["ids"].size),
        "silent": int((counts == 0).sum()),
        "silent_fraction": float((counts == 0).mean()),
        "mean_rate": float(counts.sum() / num_neurons / denominator),
        "counts": counts,
    }


def draw(raster: dict, example: int) -> str:
    """Render one example's raster as text, neurons binned into ``DRAW_ROWS`` bands."""
    mask = raster["example"].astype(int) == example
    if not mask.any():
        return f"(no spikes recorded for example {example})"
    num_neurons, timesteps = int(raster["num_neurons"]), int(raster["timesteps"])
    ids = raster["ids"][mask].astype(int)
    times = np.clip(raster["times"][mask].astype(int), 0, timesteps - 1)

    rows = min(DRAW_ROWS, num_neurons)
    band = np.minimum(ids * rows // num_neurons, rows - 1)
    grid = np.zeros((rows, timesteps), dtype=int)
    np.add.at(grid, (band, times), 1)

    peak = grid.max()
    lines = []
    for r in range(rows):
        cells = "".join(
            RAMP[min(int(v * (len(RAMP) - 1) / peak), len(RAMP) - 1)] if peak else " "
            for v in grid[r])
        lines.append(f"{r * num_neurons // rows:>4} |{cells}|")
    lines.append(f"     +{'-' * timesteps}+")
    lines.append(f"      t=0{' ' * max(timesteps - 6, 0)}t={timesteps - 1}")
    return "\n".join(lines)


def cell_rasters(cell: Path) -> list[Path]:
    """Raster files for one checkpoint cell, in epoch order."""
    directory = cell / "rasters"
    if not directory.is_dir():
        return []
    return sorted(directory.glob("epoch*.npz"),
                  key=lambda p: int(p.stem.removeprefix("epoch")))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cell", nargs="?", help="Checkpoint directory name.")
    parser.add_argument("--list", action="store_true",
                        help="List cells that have rasters and exit.")
    parser.add_argument("--epoch", type=int, default=None,
                        help="Draw this checkpoint (default: the last one).")
    parser.add_argument("--example", type=int, default=0)
    parser.add_argument("--root", default=str(CHECKPOINT_ROOT))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    root = Path(args.root)

    if args.list or not args.cell:
        found = [d for d in sorted(root.glob("*")) if cell_rasters(d)]
        if not found:
            print(f"no rasters under {root}")
            return
        for directory in found:
            epochs = [p.stem.removeprefix("epoch") for p in cell_rasters(directory)]
            print(f"{directory.name}  epochs: {', '.join(epochs)}")
        return

    cell = root / args.cell
    files = cell_rasters(cell)
    if not files:
        raise SystemExit(f"no rasters in {cell / 'rasters'}")

    print(f"{'epoch':>6} {'spikes':>9} {'silent':>8} {'silent%':>8} {'rate/step':>10}")
    for path in files:
        stats = summarise(load(path))
        print(f"{path.stem.removeprefix('epoch'):>6} {stats['spikes']:>9} "
              f"{stats['silent']:>4}/{stats['neurons']:<3} "
              f"{stats['silent_fraction'] * 100:>7.1f}% {stats['mean_rate']:>10.4f}")

    chosen = files[-1]
    if args.epoch is not None:
        matches = [p for p in files if p.stem == f"epoch{args.epoch}"]
        if not matches:
            raise SystemExit(f"no raster for epoch {args.epoch} in {cell.name}")
        chosen = matches[0]
    print(f"\n{chosen.stem}, example {args.example}:")
    print(draw(load(chosen), args.example))


if __name__ == "__main__":
    main()
