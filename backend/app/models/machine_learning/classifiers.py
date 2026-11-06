from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


MODEL_NAMES = (
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "support_vector_machine",
    "k_nearest_neighbors",
    "gradient_boosting",
)


def make_classifier(name: str, random_state: int = 42):
    if name == "logistic_regression":
        return LogisticRegression(max_iter=2000, class_weight=None, random_state=random_state)
    if name == "decision_tree":
        return DecisionTreeClassifier(random_state=random_state)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=250, n_jobs=-1, random_state=random_state)
    if name == "support_vector_machine":
        return SVC(probability=True, random_state=random_state)
    if name == "k_nearest_neighbors":
        return KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    if name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=random_state)
    raise ValueError(f"Unsupported model '{name}'. Available: {', '.join(MODEL_NAMES)}")
