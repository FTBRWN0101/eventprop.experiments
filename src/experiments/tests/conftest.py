import sys
from pathlib import Path

#same sys.path handling as run_experiment.py
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
