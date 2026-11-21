from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class LSTMBundle:
    model: object
    train_loss: float
    feature_names: List[str]
    target_name: str
    sequence_length: int
    scaler_X: object
    scaler_y: object


def build_demo_time_series(rows: int = 500) -> pd.DataFrame:
    """Build a demo time series for LSTM training."""
    rng = np.random.default_rng(42)
    time_index = np.arange(rows)

    # Create correlated time series with trend and seasonality
    trend = 0.001 * time_index
    seasonality = 0.1 * np.sin(2 * np.pi * time_index / 52)  # yearly cycle

    ndvi = 0.6 + trend + seasonality + rng.normal(0, 0.02, rows)
    climate_exposure = 0.3 + 0.0005 * time_index + rng.normal(0, 0.01, rows)
    rainfall = 1200 + 100 * np.sin(2 * np.pi * time_index / 52) + rng.normal(0, 50, rows)

    # Risk target with lagged effects
    risk_target = (
        0.4 * (1 - ndvi)
        + 0.3 * climate_exposure
        + 0.2 * (1 - rainfall / 2000).clip(0, 1)
        + 0.1 * rng.normal(0, 0.05, rows)
    )
    risk_target = np.clip(risk_target, 0, 1)

    return pd.DataFrame(
        {
            "time_index": time_index,
            "ndvi": np.clip(ndvi, 0, 1),
            "climate_exposure": np.clip(climate_exposure, 0, 1),
            "rainfall": rainfall,
            "risk_target": risk_target,
        }
    )


def _create_sequences(
    X: np.ndarray, y: np.ndarray, sequence_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences for LSTM training."""
    Xs, ys = [], []
    for i in range(len(X) - sequence_length):
        Xs.append(X[i : i + sequence_length])
        ys.append(y[i + sequence_length])
    return np.array(Xs), np.array(ys)


def train_lstm_regressor(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    sequence_length: int = 12,
    epochs: int = 20,
    hidden_size: int = 64,
    num_layers: int = 2,
    learning_rate: float = 0.001,
    batch_size: int = 32,
) -> LSTMBundle:
    """Train an LSTM regressor for time series forecasting."""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        raise ImportError("PyTorch required. Install requirements-advanced.txt")

    # Prepare data
    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32).reshape(-1, 1)

    # Scale features and target
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    # Create sequences
    X_seq, y_seq = _create_sequences(X_scaled, y_scaled, sequence_length)

    # Train/test split
    split = int(0.8 * len(X_seq))
    X_train, X_val = X_seq[:split], X_seq[split:]
    y_train, y_val = y_seq[:split], y_seq[split:]

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False
    )

    # Define LSTM model
    class LSTMRegressor(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size=1):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(
                input_size, hidden_size, num_layers, batch_first=True, dropout=0.2
            )
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            return out

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMRegressor(
        input_size=len(feature_cols), hidden_size=hidden_size, num_layers=num_layers
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    train_losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            val_loss += criterion(outputs, batch_y).item()
    val_loss /= len(val_loader)

    return LSTMBundle(
        model=model,
        train_loss=val_loss,
        feature_names=feature_cols,
        target_name=target_col,
        sequence_length=sequence_length,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
    )


def predict_lstm(bundle: LSTMBundle, df: pd.DataFrame) -> float:
    """Predict next time step using trained LSTM."""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = bundle.model.to(device)
    model.eval()

    # Get last sequence_length rows
    X = df[bundle.feature_names].tail(bundle.sequence_length).values.astype(np.float32)
    X_scaled = bundle.scaler_X.transform(X)
    X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_scaled = model(X_tensor).cpu().numpy()

    pred = bundle.scaler_y.inverse_transform(pred_scaled)
    return float(pred[0, 0])


def train_maxent(X: pd.DataFrame, y: pd.Series, beta_multiplier: float = 1.0):
    """Train a MaxEnt model using elapid."""
    try:
        from elapid import MaxentModel
    except ImportError:
        raise ImportError("elapid required. Install requirements-advanced.txt")

    model = MaxentModel(beta_multiplier=beta_multiplier)
    model.fit(X, y)
    return model


def predict_maxent(model, X: pd.DataFrame) -> np.ndarray:
    """Predict using trained MaxEnt model."""
    return model.predict(X)