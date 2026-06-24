"""pytest 共通設定。

torch/mlx/pyannote を入れずに純ロジックを検証できるよう、パッケージを install せずに
src/ を import パスへ通す（CI でも numpy・pyyaml だけで回せる）。
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
