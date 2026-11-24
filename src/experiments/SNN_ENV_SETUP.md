# SNN env (mlGeNN) - setup notes

`src/experiments/models/snn.py` needs `ml_genn` + PyGeNN, which have no PyPI wheels - both
build from source. Verified working on this Mac (CPU-only, `single_threaded_cpu` backend);
the current target machine is Windows + RTX A5000 Laptop GPU, which gets the `cuda` backend instead.

## Verified-good versions (built and import-tested on macOS/arm64, 2026-07-21)

- Python 3.12 (3.13 untested/unstated in GeNN docs - 3.12 is the safe choice)
- GeNN tag `5.4.0` (commit `dd258075263c4b2bcb6607d230add658bcc23127`)
- ml_genn commit `4cd6cb6728d016687c7216f11de7d20250fba21c` (installs as `ml_genn==2.3.1`,
  pins `pygenn<6.0.0,>=5.1.0` - 5.4.0 satisfies that)
- Runtime deps: `numpy==2.5.1 pandas==3.0.3 scipy==1.18.0 statsmodels==0.14.6 arch==8.0.0
  scikit-learn==1.9.0` (versions `src/experiments/run_experiment.py` was actually run against)

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
  runs the CPU backend again. Not changed in this pass - left as-is for the CUDA target.
- **libffi/pkg-config**: macOS used Homebrew (`pkg-config`, `libffi` already present via
  system deps). Confirm Windows' CI/install docs - GeNN vendors its own libffi build on
  Windows (see `setup.py`'s `WIN` branch), so this shouldn't need a separate install.

## Windows build compatibility (2026-08-27)

The RTX A5000 Laptop GPU is visible to `nvidia-smi`, and CUDA 13.3 and Visual Studio 2026 are installed.
PyGeNN 5.4.0's setuptools MSVC helper can return a duplicate `PATH` and `Path` environment pair.
That loses `MSBuild.exe` during PyGeNN's lookup and makes Visual Studio reject the inherited environment when it starts `cl.exe`.

The editable PyGeNN checkout at `.build-deps/genn/pygenn/genn_model.py` now resolves the installed x64 MSBuild executable explicitly when lookup fails, requests `Release|x64`, and passes the canonical MSVC environment directly to MSBuild.
The launchers used for the runs, which are not published, forwarded the required `Path`, `LIB`, `LIBPATH` and `INCLUDE` values only to the launched process.
An x64 Native Tools Command Prompt for VS provides the same values.
This is a local compatibility patch, not a CUDA, model, or source-layout change.

Verified by `src/experiments/tools/gradient_check.py` on 2026-08-27.
The check compiled every generated arm and passed its numerical EventProp gradient assertions.

Windows error 4551 remains a separate Smart App Control risk for newly generated DLLs.
If it recurs, retry serially before making any machine-level security change.

## Known-working proof (this session, CPU backend)

- `pip install -e .` on GeNN 5.4.0 compiled `libgenn_dynamic.dylib` +
  `libgenn_single_threaded_cpu_backend_dynamic.dylib` cleanly, no manual intervention.
- `ml_genn` installed against it without version conflicts.
- All imports in `src/experiments/models/snn.py` succeed; `GeNNModel` builds and runs a real
  network. The only failure was the batch-size-1 CPU limitation above, which won't apply
  on the CUDA target.
