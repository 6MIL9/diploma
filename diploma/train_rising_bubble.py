from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pinn.train import main


if __name__ == "__main__":
    main()
