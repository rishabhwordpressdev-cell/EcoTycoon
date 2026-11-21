from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Union
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


def feature_importance(
    model: Union[RandomForestRegressor, XGBRegressor], feature_names: List[str]
) -> pd.DataFrame:
    """Extract feature importance from a fitted model."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "get_score"):
        # XGBoost alternative
        score_dict = model.get_score(importance_type="gain")
        importances = np.array([score_dict.get(f, 0.0) for f in feature_names])
    else:
        raise ValueError("Model does not have feature importance attribute")

    fi_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    )
    return fi_df.sort_values("importance", ascending=False).reset_index(drop=True)