import numpy as np
import pandas as pd

from app.models.clustering.analyzer import discover_patterns


def test_kmeans_finds_patterns_without_target_feature():
    rng = np.random.default_rng(12)
    features = np.vstack([rng.normal(-3, 0.3, (30, 2)), rng.normal(3, 0.3, (30, 2))])
    frame = pd.DataFrame(features, columns=["x", "y"])
    frame["Class"] = [0] * 30 + [1] * 30
    result = discover_patterns(frame, algorithm="kmeans", n_clusters=2, exclude_columns=("Class",))
    assert result["row_count"] == 60
    assert result["cluster_count_excluding_noise"] == 2
    assert result["metrics"]["silhouette"] is not None
    assert "not confirmed fraud labels" in result["notice"]
