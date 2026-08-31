# EventProp Experiments

This workspace separates source code, generated artifacts, data and written work.

| Path | Purpose |
|---|---|
| `src/` | Experiment harness and data pipelines. |
| `scripts/windows/` | Windows launchers for CPU and mlGeNN workflows. |
| `artifacts/` | Results, checkpoints and generated GeNN code. |
| `data-save/` | Downloaded and processed data. |
| `docs/` | Plans, decisions and research notes. |
| `dissertation/` | Active dissertation source. |

Run the regular test suite from the repository root with:

```powershell
.venv\Scripts\python.exe -m pytest -q
```

Use the launchers under `scripts/windows/` for mlGeNN work so generated code lands under `artifacts/genn/live/`.
See `docs/workspace-layout.md` for the full layout and evidence-handling rules.
