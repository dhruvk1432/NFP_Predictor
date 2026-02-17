"""
Feature importance CSV export.
"""

import pandas as pd
from pathlib import Path
from typing import Dict


def save_feature_importance_csv(importance: Dict[str, float], save_path: Path) -> None:
    """
    Save feature importance as a ranked CSV.

    Args:
        importance: Dict mapping feature name -> importance score (gain).
        save_path: Destination CSV path.
    """
    sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    df = pd.DataFrame(sorted_items, columns=["feature_name", "importance_score"])
    df.insert(0, "rank", range(1, len(df) + 1))
    df.to_csv(save_path, index=False)
