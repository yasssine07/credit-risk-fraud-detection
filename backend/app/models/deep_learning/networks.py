from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class FraudClassifier(nn.Module):
    """Configurable binary network with batch normalization and dropout."""

    def __init__(self, input_features: int, hidden_layers: tuple[int, ...] = (128, 64), dropout: float = 0.25):
        super().__init__()
        if input_features < 1 or not hidden_layers or any(width < 1 for width in hidden_layers):
            raise ValueError("Input width and hidden layer widths must be positive.")
        if not 0 <= dropout < 1:
            raise ValueError("Dropout must be in [0, 1).")
        layers: list[nn.Module] = []
        previous = input_features
        for width in hidden_layers:
            layers.extend([nn.Linear(previous, width), nn.BatchNorm1d(width), nn.ReLU(), nn.Dropout(dropout)])
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


class TransactionAutoencoder(nn.Module):
    def __init__(self, input_features: int, latent_features: int = 12):
        super().__init__()
        if input_features < 2 or not 1 <= latent_features < input_features:
            raise ValueError("Latent width must be positive and smaller than input width.")
        middle = max(latent_features * 2, input_features // 2)
        self.encoder = nn.Sequential(nn.Linear(input_features, middle), nn.ReLU(), nn.Linear(middle, latent_features))
        self.decoder = nn.Sequential(nn.Linear(latent_features, middle), nn.ReLU(), nn.Linear(middle, input_features))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(values))

    def anomaly_score(self, values: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.square(values - self(values)), dim=1)


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        if not 0 <= alpha <= 1 or gamma < 0:
            raise ValueError("alpha must be within [0,1] and gamma must be non-negative.")
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.to(dtype=logits.dtype)
        base = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probabilities = torch.sigmoid(logits)
        correct_probability = probabilities * targets + (1 - probabilities) * (1 - targets)
        class_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (class_weight * torch.pow(1 - correct_probability, self.gamma) * base).mean()


@dataclass
class TrainingHistory:
    train_loss: list[float]
    validation_loss: list[float]
    best_epoch: int
    stopped_early: bool


def train_classifier(
    model: FraudClassifier,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_validation: torch.Tensor,
    y_validation: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    use_focal_loss: bool = False,
    positive_weight: float | None = None,
    patience: int = 10,
) -> TrainingHistory:
    """Train with validation early stopping; return the best weights in-place."""
    if epochs < 1 or batch_size < 1 or learning_rate <= 0 or patience < 1:
        raise ValueError("epochs, batch_size, learning_rate, and patience must be positive.")
    device = next(model.parameters()).device
    X_train, y_train = X_train.to(device), y_train.float().to(device).reshape(-1)
    X_validation, y_validation = X_validation.to(device), y_validation.float().to(device).reshape(-1)
    if use_focal_loss:
        criterion = BinaryFocalLoss()
    else:
        pos_weight = torch.tensor([positive_weight], device=device) if positive_weight is not None else None
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_history: list[float] = []
    validation_history: list[float] = []
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(len(X_train), device=device)
        epoch_loss = 0.0
        batches = 0
        for indices in permutation.split(batch_size):
            if len(indices) == 1 and any(isinstance(layer, nn.BatchNorm1d) for layer in model.modules()):
                continue
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(X_train[indices]), y_train[indices])
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach())
            batches += 1
        model.eval()
        with torch.no_grad():
            validation_loss = float(criterion(model(X_validation), y_validation))
        train_history.append(epoch_loss / max(batches, 1))
        validation_history.append(validation_loss)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch + 1
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainingHistory(train_history, validation_history, best_epoch, len(train_history) < epochs)


def train_autoencoder(
    model: TransactionAutoencoder,
    X_train_normal: torch.Tensor,
    X_validation_normal: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    patience: int = 10,
) -> TrainingHistory:
    """Fit reconstruction on normal training examples only.

    The caller must supply normal examples from the training partition. The
    validation tensor is used for early stopping and should also contain only
    normal training-period examples to keep the score threshold interpretable.
    """
    if epochs < 1 or batch_size < 1 or learning_rate <= 0 or patience < 1:
        raise ValueError("epochs, batch_size, learning_rate, and patience must be positive.")
    device = next(model.parameters()).device
    X_train_normal = X_train_normal.to(device)
    X_validation_normal = X_validation_normal.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    train_history: list[float] = []
    validation_history: list[float] = []
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(len(X_train_normal), device=device)
        losses = []
        for indices in permutation.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            batch = X_train_normal[indices]
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            validation_loss = float(criterion(model(X_validation_normal), X_validation_normal))
        train_history.append(sum(losses) / max(len(losses), 1))
        validation_history.append(validation_loss)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch + 1
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainingHistory(train_history, validation_history, best_epoch, len(train_history) < epochs)
