"""Materialise validation-selected SNN forecasts and configuration ensembles.

Training runs remain immutable final-epoch evidence. This tool reads their saved
checkpoints, chooses an epoch using validation ``r2_rv``, writes a separate selected
run tree, and forms one equal-weight seed ensemble per configuration on the test split.
Only the ensemble artefacts contain configuration-level DM tests. Rate-coded forecasts
average a declared number of independent Poisson inference draws per selected checkpoint.

    python src/experiments/tools/materialize_selection.py \
        --campaign-id final_weekly_20260831_v3_holdout \
        --selection-metric val_r2 --rate-draws 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd  # noqa: E402

from core import metrics  # noqa: E402
from core.config import ExperimentConfig  # noqa: E402
from core.dataset import VrpDataset  # noqa: E402
from core.results import RESULTS_ROOT  # noqa: E402
from models.snn import SnnForecaster  # noqa: E402
from run_experiment import _add_dm_columns  # noqa: E402
from tools.aggregate import group_key  # noqa: E402
from tools.epoch_curve import score_epochs  # noqa: E402
from tools.select_epoch import _select, config_from_run, eligible_runs  # noqa: E402

logger = logging.getLogger("materialize_selection")


def _draw_seeds(config: ExperimentConfig, draws: int) -> tuple[int, ...]:
    """Stable seeds reserved for post-training Poisson inference draws."""
    if config.encoding != "rate":
        return (config.seed,)
    return tuple(10_000_000 + config.seed * 100 + draw for draw in range(draws))


def _predict_selected(config: ExperimentConfig, epoch: int, data: VrpDataset,
                      draws: int) -> tuple[pd.Series, tuple[int, ...]]:
    """Mean target-space test forecast across declared inference draws."""
    seeds = _draw_seeds(config, draws)
    forecasts = []
    for inference_seed in seeds:
        model = SnnForecaster(config=ExperimentConfig.load(
            **{**asdict(config), "predict_epoch": epoch}))
        model.warm_start(data)
        forecasts.append(model.predict(data, "test", inference_seed=inference_seed))
    frame = pd.concat(forecasts, axis=1)
    if frame.isna().any().any():
        raise ValueError("selected inference draws did not share one complete test index")
    return frame.mean(axis=1), seeds


def _read_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path / "predictions.csv", parse_dates=["date"])
    return frame.set_index("date").sort_index()


def _read_diagnostics(path: Path) -> dict[str, Any]:
    source = path / "diagnostics.json"
    if not source.is_file():
        return {}
    loaded = json.loads(source.read_text(encoding="utf8"))
    return loaded if isinstance(loaded, dict) else {}


def _metrics_with_selected_snn(source: Path, config: ExperimentConfig,
                               prediction: pd.Series, data: VrpDataset) -> pd.DataFrame:
    """Copy deterministic baseline metrics and replace only the selected SNN row.

    Per-seed DM values are intentionally removed. Configuration-level inference belongs
    to the ensemble produced below, not to averages of seed-level p-values.
    """
    frame = pd.read_csv(source / "metrics.csv", index_col="model")
    frame = frame.drop(columns=[c for c in frame if c.startswith("dm_")], errors="ignore")
    train = data.split("train")
    selected = metrics.evaluate(data.split("test"), prediction,
                                float(train.target.mean()), float(train.rv_fwd.mean()))
    for column, value in selected.items():
        frame.loc["snn", column] = value
    return frame


def _write_selected_run(source: Path, out_dir: Path, config: ExperimentConfig,
                        curve: pd.DataFrame, chosen: pd.Series, rate_draws: int) -> Path:
    """Write one immutable source-derived selected run."""
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing selected run {out_dir}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.", dir=out_dir.parent))
    try:
        data = VrpDataset(config)
        prediction, seeds = _predict_selected(config, int(chosen["epoch"]), data, rate_draws)
        predictions = _read_predictions(source)
        expected = data.split("test").target.index
        if not predictions.index.equals(expected):
            raise ValueError(f"{source.name} baseline predictions do not match the test split")
        predictions["snn"] = prediction.reindex(predictions.index)
        _metrics_with_selected_snn(source, config, prediction, data).to_csv(
            staging / "metrics.csv", index_label="model")
        predictions.to_csv(staging / "predictions.csv", index_label="date")
        shutil.copy2(source / "config.json", staging / "config.json")
        diagnostics = _read_diagnostics(source)
        (staging / "diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2, default=str), encoding="utf8")
        final = curve.loc[curve["epoch"].idxmax()]
        (staging / "selection.json").write_text(json.dumps({
            "source_run": str(source),
            "selection_metric": "val_r2",
            "selection_direction": "max",
            "selected_epoch": int(chosen["epoch"]),
            "selected_validation_r2": float(chosen["val_r2"]),
            "selected_test_r2": float(chosen["test_r2"]),
            "final_epoch": int(final["epoch"]),
            "final_test_r2": float(final["test_r2"]),
            "rate_inference_draws": len(seeds),
            "rate_inference_seeds": list(seeds),
        }, indent=2), encoding="utf8")
        staging.replace(out_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return out_dir


def _assert_selected_run_complete(source: Path, out_dir: Path, config: ExperimentConfig,
                                  chosen: pd.Series, rate_draws: int) -> None:
    """Accept only an atomically published selection derived from this exact raw run."""
    required = ("config.json", "diagnostics.json", "metrics.csv", "predictions.csv", "selection.json")
    missing = [name for name in required if not (out_dir / name).is_file()]
    if missing:
        raise ValueError(f"existing selected run is incomplete: {out_dir} lacks {missing}")
    selection = json.loads((out_dir / "selection.json").read_text(encoding="utf8"))
    if selection.get("source_run") != str(source):
        raise ValueError(f"existing selected run has the wrong source: {out_dir}")
    expected = {
        "selection_metric": "val_r2",
        "selection_direction": "max",
        "selected_epoch": int(chosen["epoch"]),
        "selected_validation_r2": float(chosen["val_r2"]),
        "selected_test_r2": float(chosen["test_r2"]),
        "rate_inference_draws": len(_draw_seeds(config, rate_draws)),
        "rate_inference_seeds": list(_draw_seeds(config, rate_draws)),
    }
    for name, value in expected.items():
        if selection.get(name) != value:
            raise ValueError(f"existing selected run has incompatible {name!r}: {out_dir}")


def _ensemble_key(config: dict[str, Any]) -> str:
    return hashlib.sha256(group_key(config).encode("utf8")).hexdigest()[:16]


def _write_ensembles(selected_root: Path, output_root: Path | None = None) -> pd.DataFrame:
    """Write one equal-weight selected-SNN ensemble and valid DM test per cell."""
    output_root = selected_root if output_root is None else output_root
    members: dict[str, list[Path]] = {}
    configs: dict[str, dict[str, Any]] = {}
    for path in sorted(selected_root.glob("runs/*/config.json")):
        config = json.loads(path.read_text(encoding="utf8"))
        key = group_key(config)
        members.setdefault(key, []).append(path.parent)
        configs[key] = config

    index: list[dict[str, Any]] = []
    for key, paths in sorted(members.items()):
        config = configs[key]
        first = paths[0]
        predictions = [_read_predictions(path) for path in paths]
        baseline_names = [c for c in predictions[0].columns if c not in ("actual", "snn")]
        for frame in predictions[1:]:
            for name in baseline_names:
                if not frame[name].equals(predictions[0][name]):
                    raise ValueError(f"baseline {name!r} differs across seeds in {key}")
        forecast = pd.concat([frame["snn"] for frame in predictions], axis=1).mean(axis=1)
        experiment = config_from_run(first)
        data = VrpDataset(experiment)
        split, train = data.split("test"), data.split("train")
        model_predictions = {"snn_ensemble": forecast,
                             **{name: predictions[0][name] for name in baseline_names}}
        results = {name: metrics.evaluate(split, value, float(train.target.mean()),
                                          float(train.rv_fwd.mean()))
                   for name, value in model_predictions.items()}
        errors = {name: metrics.squared_errors(split, value)
                  for name, value in model_predictions.items()}
        errors_rv = {name: metrics.squared_errors_rv(split, value)
                     for name, value in model_predictions.items()}
        diagnostics = _read_diagnostics(first)
        diagnostics["snn_ensemble"] = diagnostics.get("snn", {})
        _add_dm_columns(results, list(model_predictions), errors, errors_rv,
                        max(experiment.horizon_days - 1, 0), diagnostics,
                        data.daily("full").index)
        destination = output_root / "ensembles" / _ensemble_key(config)
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite ensemble {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.",
                                        dir=destination.parent))
        try:
            pd.DataFrame(results).T.to_csv(staging / "metrics.csv", index_label="model")
            pd.DataFrame({"actual": predictions[0]["actual"], **model_predictions}).to_csv(
                staging / "predictions.csv", index_label="date")
            (staging / "config.json").write_text(json.dumps(config, indent=2), encoding="utf8")
            (staging / "ensemble.json").write_text(json.dumps({
                "member_runs": [str(path) for path in paths],
                "members": len(paths),
                "rule": "equal-weight mean of validation-selected seed forecasts",
                "selection_metric": "val_r2",
            }, indent=2), encoding="utf8")
            staging.replace(destination)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        index.append({"ensemble_dir": str(destination), "members": len(paths), **config})
    frame = pd.DataFrame(index)
    frame.to_csv(output_root / "ensemble_index.csv", index=False)
    return frame


def _shard_candidates(candidates: list[tuple[Path, ExperimentConfig]], shard_index: int,
                      shard_count: int) -> list[tuple[Path, ExperimentConfig]]:
    """Return one deterministic, disjoint partition of a sorted candidate list."""
    if shard_count < 1:
        raise ValueError("shard count must be at least one")
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"shard index {shard_index} is outside 0..{shard_count - 1}")
    return [candidate for number, candidate in enumerate(candidates)
            if number % shard_count == shard_index]


def _shard_stem(shard_index: int, shard_count: int) -> str:
    return f"shard-{shard_index + 1:03d}-of-{shard_count:03d}"


def _atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    """Publish one CSV only after its complete temporary replacement is ready."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.tmp")
    if staging.exists():
        raise FileExistsError(f"stale temporary file requires review: {staging}")
    try:
        frame.to_csv(staging, index=False)
        staging.replace(destination)
    except Exception:
        staging.unlink(missing_ok=True)
        raise


def _atomic_json(value: dict[str, Any], destination: Path) -> None:
    """Publish one JSON record atomically, matching the run-directory contract."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.tmp")
    if staging.exists():
        raise FileExistsError(f"stale temporary file requires review: {staging}")
    try:
        staging.write_text(json.dumps(value, indent=2), encoding="utf8")
        staging.replace(destination)
    except Exception:
        staging.unlink(missing_ok=True)
        raise


def _write_shard(selected_root: Path, candidates: list[tuple[Path, ExperimentConfig]],
                 curves: pd.DataFrame, selections: pd.DataFrame,
                 shard_index: int, shard_count: int) -> None:
    """Write one worker's evidence under a path no other worker may claim."""
    stem = _shard_stem(shard_index, shard_count)
    shard_root = selected_root / "shards"
    destination = shard_root / stem
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite completed shard evidence: {destination}")
    shard_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{stem}.", dir=shard_root))
    try:
        curves.to_csv(staging / "epoch_curves.csv", index=False)
        selections.to_csv(staging / "epoch_selection.csv", index=False)
        (staging / "metadata.json").write_text(json.dumps({
            "schema": 1,
            "shard_index": shard_index,
            "shard_count": shard_count,
            "source_runs": [str(source) for source, _ in candidates],
        }, indent=2), encoding="utf8")
        staging.replace(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def materialize_campaign(results_root: Path, campaign_id: str, metric: str,
                         rate_draws: int, out_root: Path, shard_index: int = 0,
                         shard_count: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Materialise one disjoint shard of a completed campaign without finalising it."""
    if metric != "val_r2":
        raise ValueError("only val_r2 is currently permitted by the pre-specified protocol")
    all_candidates = eligible_runs(results_root, campaign_id)
    if not all_candidates:
        raise FileNotFoundError(f"no eligible runs for {campaign_id!r}")
    candidates = _shard_candidates(all_candidates, shard_index, shard_count)
    selected_root = out_root / campaign_id
    curves, selections = [], []
    for number, (source, config) in enumerate(candidates, start=1):
        logger.info("[materialize] %s %s %d/%d", campaign_id,
                    _shard_stem(shard_index, shard_count), number, len(candidates))
        curve = score_epochs(config, ("val", "test"))
        chosen = _select(curve, metric, higher_is_better=True)
        destination = selected_root / "runs" / source.name
        if destination.exists():
            _assert_selected_run_complete(source, destination, config, chosen, rate_draws)
            logger.info("[materialize] reusing complete selected run %s", destination.name)
        else:
            _write_selected_run(source, destination, config, curve, chosen, rate_draws)
        curve = curve.copy()
        curve.insert(0, "source_run", str(source))
        curve.insert(1, "encoding", config.encoding)
        curve.insert(2, "algorithm", config.algorithm)
        curve.insert(3, "input_window", config.sequence_length)
        curve.insert(4, "seed", config.seed)
        curves.append(curve)
        selections.append({"source_run": str(source), "selected_run": str(destination),
                           "encoding": config.encoding, "algorithm": config.algorithm,
                           "input_window": config.sequence_length, "seed": config.seed,
                           "selected_epoch": int(chosen["epoch"]),
                           "selected_validation_r2": float(chosen["val_r2"]),
                           "selected_test_r2": float(chosen["test_r2"]),
                           "rate_inference_draws": (rate_draws if config.encoding == "rate" else 1)})
    curve_frame = pd.concat(curves, ignore_index=True)
    selection_frame = pd.DataFrame(selections)
    _write_shard(selected_root, candidates, curve_frame, selection_frame, shard_index, shard_count)
    return curve_frame, selection_frame


def _read_completed_shards(selected_root: Path,
                           all_candidates: list[tuple[Path, ExperimentConfig]],
                           shard_count: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate complete, non-overlapping worker evidence before it is merged."""
    expected_runs = {str(source) for source, _ in all_candidates}
    seen_runs: set[str] = set()
    curves, selections = [], []
    for index in range(shard_count):
        stem = _shard_stem(index, shard_count)
        shard_root = selected_root / "shards" / stem
        metadata_path = shard_root / "metadata.json"
        curve_path = shard_root / "epoch_curves.csv"
        selection_path = shard_root / "epoch_selection.csv"
        if not all(path.is_file() for path in (metadata_path, curve_path, selection_path)):
            raise FileNotFoundError(f"missing complete evidence for {stem}")
        metadata = json.loads(metadata_path.read_text(encoding="utf8"))
        if (metadata.get("schema") != 1 or metadata.get("shard_index") != index or
                metadata.get("shard_count") != shard_count):
            raise ValueError(f"invalid shard metadata in {metadata_path}")
        shard_runs = metadata.get("source_runs")
        if not isinstance(shard_runs, list) or not all(isinstance(path, str) for path in shard_runs):
            raise ValueError(f"invalid source-run list in {metadata_path}")
        overlap = seen_runs.intersection(shard_runs)
        if overlap:
            raise ValueError(f"duplicate source runs across shards: {sorted(overlap)}")
        seen_runs.update(shard_runs)
        curve_frame = pd.read_csv(curve_path)
        selection_frame = pd.read_csv(selection_path)
        if set(curve_frame.get("source_run", ())) != set(shard_runs):
            raise ValueError(f"curve coverage does not match metadata for {stem}")
        if set(selection_frame.get("source_run", ())) != set(shard_runs):
            raise ValueError(f"selection coverage does not match metadata for {stem}")
        curves.append(curve_frame)
        selections.append(selection_frame)
    if seen_runs != expected_runs:
        missing, unexpected = expected_runs - seen_runs, seen_runs - expected_runs
        raise ValueError(f"shards do not cover the campaign: missing={len(missing)} "
                         f"unexpected={len(unexpected)}")
    curve_frame = pd.concat(curves, ignore_index=True)
    selection_frame = pd.concat(selections, ignore_index=True)
    if curve_frame["source_run"].nunique() != len(expected_runs):
        raise ValueError("merged checkpoint curves do not have exactly one source-run group")
    if selection_frame["source_run"].nunique() != len(expected_runs):
        raise ValueError("merged selections do not have exactly one source run")
    return curve_frame, selection_frame


def finalize_campaign(results_root: Path, campaign_id: str, out_root: Path,
                      shard_count: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge verified shards and write ensembles exactly once, with no GPU inference."""
    all_candidates = eligible_runs(results_root, campaign_id)
    selected_root = out_root / campaign_id
    if not all_candidates:
        raise FileNotFoundError(f"no eligible runs for {campaign_id!r}")
    expected_names = {source.name for source, _ in all_candidates}
    written_names = {path.parent.name for path in selected_root.glob("runs/*/selection.json")}
    if written_names != expected_names:
        raise ValueError(f"selected-run coverage is incomplete: expected={len(expected_names)} "
                         f"written={len(written_names)}")
    curve_frame, selection_frame = _read_completed_shards(selected_root, all_candidates, shard_count)
    destination = selected_root / "final"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite final campaign evidence: {destination}")
    staging = Path(tempfile.mkdtemp(prefix=".final.", dir=selected_root))
    try:
        curve_frame.to_csv(staging / "epoch_curves.csv", index=False)
        selection_frame.to_csv(staging / "epoch_selection.csv", index=False)
        index = _write_ensembles(selected_root, staging)
        if not (staging / "ensemble_index.csv").is_file() or index.empty:
            raise RuntimeError("ensemble finalisation did not write a complete ensemble index")
        staging.replace(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return curve_frame, selection_frame


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--campaign-id", action="append", required=True,
                        help="Completed holdout campaign to materialise; repeatable.")
    parser.add_argument("--selection-metric", default="val_r2", choices=("val_r2",))
    parser.add_argument("--rate-draws", type=int, default=5)
    parser.add_argument("--results-root", default=str(RESULTS_ROOT))
    parser.add_argument("--out-root", default=str(RESULTS_ROOT.parent / "selected"))
    parser.add_argument("--shard-index", type=int, default=0,
                        help="Zero-based worker index; each worker owns one disjoint shard.")
    parser.add_argument("--shard-count", type=int, default=1,
                        help="Total number of disjoint materialisation workers.")
    parser.add_argument("--finalize-only", action="store_true",
                        help="Merge completed shards and form ensembles without GPU inference.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    args = _parse_args(argv)
    if args.rate_draws < 1:
        raise ValueError("--rate-draws must be at least one")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("--shard-index must be in 0..--shard-count-1")
    if args.finalize_only and args.shard_count == 1:
        raise ValueError("--finalize-only requires explicit shard evidence from at least two workers")
    root, out = Path(args.results_root), Path(args.out_root)
    for campaign in args.campaign_id:
        runs = eligible_runs(root, campaign)
        logger.info("[materialize] %s: %d eligible run(s)", campaign, len(runs))
        if args.dry_run:
            continue
        if args.finalize_only:
            finalize_campaign(root, campaign, out, args.shard_count)
        else:
            materialize_campaign(root, campaign, args.selection_metric, args.rate_draws, out,
                                 args.shard_index, args.shard_count)


if __name__ == "__main__":
    main()
