import json
from pathlib import Path

import pandas as pd


def load_metric_files(metric_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(metric_dir.glob("*.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return pd.DataFrame(rows)
