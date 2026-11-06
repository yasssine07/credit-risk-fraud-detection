from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE


ALGORITHMS = ("kmeans", "dbscan", "hierarchical", "gaussian_mixture")


def discover_patterns(
    frame: pd.DataFrame,
    *,
    algorithm: str = "kmeans",
    n_clusters: int = 4,
    reducer: str = "pca",
    eps: float = 0.5,
    min_samples: int = 5,
    random_state: int = 42,
    max_visualization_rows: int = 2000,
    exclude_columns: tuple[str, ...] = (),
) -> dict[str, object]:
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unsupported algorithm. Choose: {', '.join(ALGORITHMS)}")
    if reducer not in {"pca", "tsne"}:
        raise ValueError("Reducer must be 'pca' or 'tsne'.")
    if not 2 <= n_clusters <= 20:
        raise ValueError("n_clusters must be between 2 and 20.")
    if eps <= 0 or min_samples < 2:
        raise ValueError("DBSCAN eps must be positive and min_samples must be at least 2.")
    numeric = frame.drop(columns=list(exclude_columns), errors="ignore").select_dtypes(include="number")
    if numeric.shape[1] < 2:
        raise ValueError("Clustering requires at least two numeric feature columns.")
    if len(numeric) < 10:
        raise ValueError("At least 10 rows are required for pattern discovery.")

    transformed = SimpleImputer(strategy="median").fit_transform(numeric)
    transformed = StandardScaler().fit_transform(transformed)
    estimator = _estimator(algorithm, n_clusters, eps, min_samples, random_state)
    labels = estimator.fit_predict(transformed) if algorithm != "gaussian_mixture" else estimator.fit_predict(transformed)
    labels = np.asarray(labels, dtype=int)
    metric_rows = np.ones(len(labels), dtype=bool)
    if algorithm == "dbscan":
        metric_rows = labels != -1
    metric_labels = labels[metric_rows]
    metric_features = transformed[metric_rows]
    unique_clusters = np.unique(metric_labels)
    metrics: dict[str, float | None] = {"silhouette": None, "davies_bouldin": None, "calinski_harabasz": None}
    if 2 <= len(unique_clusters) < len(metric_labels) and len(metric_labels) >= 2:
        metrics = {
            "silhouette": float(silhouette_score(metric_features, metric_labels)),
            "davies_bouldin": float(davies_bouldin_score(metric_features, metric_labels)),
            "calinski_harabasz": float(calinski_harabasz_score(metric_features, metric_labels)),
        }

    if reducer == "tsne":
        if len(transformed) > 5000:
            raise ValueError("t-SNE is limited to 5,000 rows; sample the dataset before requesting this view.")
        embedding = TSNE(n_components=2, perplexity=min(30.0, (len(transformed) - 1) / 3), random_state=random_state, init="pca", learning_rate="auto").fit_transform(transformed)
    else:
        embedding = PCA(n_components=2, random_state=random_state).fit_transform(transformed)
    step = max(1, int(np.ceil(len(labels) / max_visualization_rows)))
    points = [
        {"x": float(embedding[index, 0]), "y": float(embedding[index, 1]), "cluster": int(labels[index]), "row_index": int(numeric.index[index]) if isinstance(numeric.index[index], (int, np.integer)) else str(numeric.index[index])}
        for index in range(0, len(labels), step)
    ]
    counts = pd.Series(labels).value_counts().sort_index()
    return {
        "algorithm": algorithm,
        "reducer": reducer,
        "row_count": len(labels),
        "cluster_count_excluding_noise": int(len(set(labels) - {-1})),
        "noise_row_count": int(np.count_nonzero(labels == -1)),
        "cluster_sizes": [{"cluster": int(cluster), "count": int(count)} for cluster, count in counts.items()],
        "metrics": metrics,
        "visualization_sample_step": step,
        "points": points,
        "notice": "Clusters describe behavioral patterns for review. They are not confirmed fraud labels.",
    }


def _estimator(algorithm: str, n_clusters: int, eps: float, min_samples: int, random_state: int):
    if algorithm == "kmeans":
        return KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    if algorithm == "dbscan":
        return DBSCAN(eps=eps, min_samples=min_samples)
    if algorithm == "hierarchical":
        return AgglomerativeClustering(n_clusters=n_clusters)
    return GaussianMixture(n_components=n_clusters, random_state=random_state, n_init=2)
