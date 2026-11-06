import pytest

torch = pytest.importorskip("torch")

from app.models.deep_learning.networks import BinaryFocalLoss, FraudClassifier, TransactionAutoencoder


def test_network_output_shapes_and_focal_loss():
    model = FraudClassifier(5, hidden_layers=(8, 4))
    logits = model(torch.randn(6, 5))
    loss = BinaryFocalLoss()(logits, torch.tensor([0, 1, 0, 1, 0, 1]))
    assert logits.shape == (6,)
    assert torch.isfinite(loss)


def test_autoencoder_returns_per_row_anomaly_scores():
    model = TransactionAutoencoder(6, latent_features=2)
    scores = model.anomaly_score(torch.randn(9, 6))
    assert scores.shape == (9,)
    assert torch.all(scores >= 0)
