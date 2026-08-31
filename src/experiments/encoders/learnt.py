"""Parametric temporal encoders trained jointly with the iso-architecture control."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from encoders.base import EncodedInput, Encoder


class _ArctanSpike:
    """Lazy torch surrogate: hard spikes forward, arctangent slope backward."""

    @staticmethod
    def apply(x, width: float):
        import torch

        class Spike(torch.autograd.Function):
            @staticmethod
            def forward(ctx, value):
                ctx.save_for_backward(value)
                return (value >= 0).to(value.dtype)

            @staticmethod
            def backward(ctx, grad_output):
                (value,) = ctx.saved_tensors
                return grad_output * (width / (1.0 + (width * value) ** 2)) / np.pi

        return Spike.apply(x)


def _fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LearntEncoder(Encoder):
    """Base protocol for B1 dense training and B2 frozen spiking export."""

    input_space = "zscore"
    architecture = ""

    def __init__(self, num_neurons: int = 32, spiking: bool = False,
                 checkpoint: Path | None = None, surrogate_width: float = 1.0) -> None:
        self.num_neurons = num_neurons
        self.spiking = spiking
        self.kind = "spikes" if spiking else "dense"
        self.surrogate_width = surrogate_width
        self.checkpoint = checkpoint
        self._module = None
        self._numpy_weights: dict[str, np.ndarray] | None = None
        if checkpoint is not None:
            self._load_artifact(checkpoint)

    def _build_module(self, num_features: int):
        raise NotImplementedError

    def prepare(self, num_features: int) -> None:
        if self._module is None:
            self._module = self._build_module(num_features)

    def fit(self, X: np.ndarray, scale: np.ndarray | None = None) -> None:
        """Fixed API compatibility only. Supervised fitting belongs to ``iso_rnn``."""
        if self._numpy_weights is not None:
            return
        self.prepare(X.shape[2])

    def tensor(self, X):
        if self._module is None:
            raise RuntimeError("learnt encoder is not prepared")
        return self._module(X, self.spiking)

    def encode(self, X: np.ndarray, scale: np.ndarray | None = None) -> EncodedInput:
        if self._numpy_weights is not None:
            values = self._numpy_forward(np.asarray(X, dtype=np.float32))
        else:
            import torch

            self.prepare(X.shape[2])
            self._module.eval()
            with torch.no_grad():
                values = self.tensor(torch.tensor(X, dtype=torch.float32)).cpu().numpy()
        if not self.spiking:
            return EncodedInput("dense", values.astype(np.float32), self.num_neurons)
        from ml_genn.utils.data import preprocess_spikes

        spikes = []
        for row in values:
            times, ids = np.nonzero(row)
            order = np.argsort(times, kind="stable")
            spikes.append(preprocess_spikes(times[order].astype(np.float32),
                                             ids[order].astype(np.int64),
                                             self.num_neurons))
        return EncodedInput("spikes", spikes, self.num_neurons)

    def save_artifact(self, config, num_features: int) -> Path:
        import torch

        if self._module is None:
            raise RuntimeError("cannot save an unprepared learnt encoder")
        root = config.repo_root / "artifacts" / "encoder_checkpoints"
        if config.campaign_id is not None:
            root = root / config.campaign_id
        run_id = (f"{config.horizon}_{config.target}_{config.iv_leg}_{self.name}"
                  f"_L{config.sequence_length}_s{config.seed}_n{self.num_neurons}")
        directory = root / run_id
        directory.mkdir(parents=True, exist_ok=True)
        weights, portable_weights = directory / "encoder.pt", directory / "encoder.npz"
        state = {key: value.detach().cpu().numpy()
                 for key, value in self._module.state_dict().items()}
        torch.save(self._module.state_dict(), weights)
        np.savez_compressed(portable_weights, **state)
        metadata = {
            "architecture": self.name,
            "num_features": num_features,
            "num_neurons": self.num_neurons,
            "surrogate_width": self.surrogate_width,
            "source_seed": config.seed,
            "campaign_id": config.campaign_id,
            "fingerprint": _fingerprint(weights),
            "portable_fingerprint": _fingerprint(portable_weights),
        }
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf8")
        self.checkpoint = directory
        self.fingerprint = metadata["fingerprint"]
        return directory

    def _load_artifact(self, checkpoint: Path) -> None:
        directory = checkpoint if checkpoint.is_dir() else checkpoint.parent
        metadata_path = directory / "metadata.json"
        weights, portable_weights = directory / "encoder.pt", directory / "encoder.npz"
        if not metadata_path.is_file() or not weights.is_file() or not portable_weights.is_file():
            raise FileNotFoundError(f"learnt encoder artifact is incomplete at {directory}")
        metadata = json.loads(metadata_path.read_text(encoding="utf8"))
        if metadata.get("architecture") != self.name:
            raise ValueError(f"artifact is {metadata.get('architecture')!r}, not {self.name!r}")
        if metadata.get("num_neurons") != self.num_neurons:
            raise ValueError("learnt encoder artifact width does not match the requested config")
        if metadata.get("fingerprint") != _fingerprint(weights):
            raise ValueError("learnt encoder artifact fingerprint does not match its weights")
        if metadata.get("portable_fingerprint") != _fingerprint(portable_weights):
            raise ValueError("learnt encoder artifact fingerprint does not match its portable weights")
        if self.spiking:
            with np.load(portable_weights) as archive:
                self._numpy_weights = {key: archive[key] for key in archive.files}
        else:
            import torch

            self.prepare(int(metadata["num_features"]))
            self._module.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.checkpoint = directory
        self.fingerprint = metadata["fingerprint"]

    def _numpy_forward(self, X: np.ndarray) -> np.ndarray:
        """Run a frozen B1 artifact without importing torch in the mlGeNN environment."""
        raise NotImplementedError


class LearntDeltaEncoder(LearntEncoder):
    """Per-timestep learnt sensitivities over first differences."""

    name = "learnt_delta"
    description = "Learnt first-difference encoder, dense for B1 and spike-exportable for B2"

    def _build_module(self, num_features: int):
        import torch
        import torch.nn as nn

        width = self.num_neurons
        threshold = self.surrogate_width

        class Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(num_features, width)
                self.threshold = nn.Parameter(torch.zeros(width))

            def forward(self, x, spiking: bool):
                diffs = torch.cat([torch.zeros_like(x[:, :1]), x[:, 1:] - x[:, :-1]], dim=1)
                logits = self.linear(diffs) - self.threshold
                return _ArctanSpike.apply(logits, threshold) if spiking else torch.sigmoid(logits)

        return Module()

    def _numpy_forward(self, X: np.ndarray) -> np.ndarray:
        if self._numpy_weights is None:
            raise RuntimeError("portable learnt encoder weights are not loaded")
        weights = self._numpy_weights
        diffs = np.concatenate([np.zeros_like(X[:, :1]), X[:, 1:] - X[:, :-1]], axis=1)
        logits = np.einsum("ntf,of->nto", diffs, weights["linear.weight"])
        logits += weights["linear.bias"][None, None, :] - weights["threshold"][None, None, :]
        return (logits >= 0.0).astype(np.float32) if self.spiking else _sigmoid(logits)


class LearntConvEncoder(LearntEncoder):
    """Temporal convolutional encoder with one learnt threshold per output channel."""

    name = "learnt_conv"
    description = "Learnt temporal-convolution encoder, dense for B1 and spike-exportable for B2"

    def _build_module(self, num_features: int):
        import torch
        import torch.nn as nn

        width = self.num_neurons
        threshold = self.surrogate_width

        class Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv1d(num_features, width, kernel_size=3, padding=1)
                self.threshold = nn.Parameter(torch.zeros(width))

            def forward(self, x, spiking: bool):
                logits = self.conv(x.transpose(1, 2)).transpose(1, 2) - self.threshold
                return _ArctanSpike.apply(logits, threshold) if spiking else torch.sigmoid(logits)

        return Module()

    def _numpy_forward(self, X: np.ndarray) -> np.ndarray:
        if self._numpy_weights is None:
            raise RuntimeError("portable learnt encoder weights are not loaded")
        weights = self._numpy_weights
        padded = np.pad(X, ((0, 0), (1, 1), (0, 0)))
        windows = np.lib.stride_tricks.sliding_window_view(padded, 3, axis=1)
        logits = np.einsum("ntfk,ofk->nto", windows, weights["conv.weight"])
        logits += weights["conv.bias"][None, None, :] - weights["threshold"][None, None, :]
        return (logits >= 0.0).astype(np.float32) if self.spiking else _sigmoid(logits)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))
