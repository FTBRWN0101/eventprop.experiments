# SNN env (mlGeNN) - setup notes

`experiments/models/snn.py` needs `ml_genn` + PyGeNN, which have no PyPI wheels - both
build from source. Verified working on this Mac (CPU-only, `single_threaded_cpu` backend);
the target machine is Windows + RTX 3090, which gets the `cuda` backend instead.

## Verified-good versions (built and import-tested on macOS/arm64, 2026-07-21)

- Python 3.12 (3.13 untested/unstated in GeNN docs - 3.12 is the safe choice)
- GeNN tag `5.4.0` (commit `dd258075263c4b2bcb6607d230add658bcc23127`)
- ml_genn commit `4cd6cb6728d016687c7216f11de7d20250fba21c` (installs as `ml_genn==2.3.1`,
  pins `pygenn<6.0.0,>=5.1.0` - 5.4.0 satisfies that)
- Runtime deps: `numpy==2.5.1 pandas==3.0.3 scipy==1.18.0 statsmodels==0.14.6 arch==8.0.0
  scikit-learn==1.9.0` (versions experiments/run_experiment.py was actually run against)

## Build steps (same on Windows, different prerequisites)

```bash
conda create -n mlgenn python=3.12 -y
conda activate mlgenn
pip install pybind11 psutil pkgconfig "setuptools>=61" numpy pandas scipy statsmodels arch scikit-learn

git clone --branch 5.4.0 --depth 1 https://github.com/genn-team/genn.git
cd genn && pip install -e .          # builds libgenn + the CPU backend always;
cd ..                                 # adds the cuda backend automatically if CUDA_PATH is set

git clone --depth 1 https://github.com/genn-team/ml_genn.git
pip install ./ml_genn/ml_genn
```

## macOS -> Windows/CUDA deltas

- **Compiler**: macOS used Apple clang; Windows needs Visual Studio 2019+ with the
  "Desktop development with C++" workload (per GeNN's own install docs).
- **CUDA backend**: `genn`'s `setup.py` only adds the `cuda` extension if `CUDA_PATH` is
  set and exists - set it before `pip install -e .` (the CUDA installer sets it; verify
  with `echo %CUDA_PATH%`). No code changes needed - `EventPropCompiler`/`InferenceCompiler`
  pick the backend automatically from what's built.
- **Batch size**: `single_threaded_cpu` - the only backend buildable here - hard-errors
  above `batch_size=1` (mini-batching is CUDA/HIP-only in GeNN). `snn.py`'s
  `BATCH_SIZE = 32` is correct for the CUDA target; it only needs to drop to 1 if someone
  runs the CPU backend again. Not changed in this pass - left as-is for the 3090 target.
- **libffi/pkg-config**: macOS used Homebrew (`pkg-config`, `libffi` already present via
  system deps). Confirm Windows' CI/install docs - GeNN vendors its own libffi build on
  Windows (see `setup.py`'s `WIN` branch), so this shouldn't need a separate install.

## Known-working proof (this session, CPU backend)

- `pip install -e .` on GeNN 5.4.0 compiled `libgenn_dynamic.dylib` +
  `libgenn_single_threaded_cpu_backend_dynamic.dylib` cleanly, no manual intervention.
- `ml_genn` installed against it without version conflicts.
- All imports in `experiments/models/snn.py` succeed; `GeNNModel` builds and runs a real
  network. The only failure was the batch-size-1 CPU limitation above, which won't apply
  on the CUDA target.
