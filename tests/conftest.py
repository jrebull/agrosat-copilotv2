"""Pytest config: añade la raíz del repo al sys.path para imports `ml.*`.

Sin un `src layout` ni instalación editable (`pip install -e .`), pytest no
encuentra paquetes top-level del repo. Esta configuración inyecta la raíz
del repositorio al `sys.path` antes de la colección de tests.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Tests point MLflow at throwaway file:// stores (tmp_path); MLflow >= 3.13 refuses
# them unless this opt-in is set. Production code goes through
# ml.utils.mlflow_utils.resolve_tracking_uri, which sets it only for file URIs.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
