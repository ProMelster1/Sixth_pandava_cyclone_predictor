"""
Model Definitions
==================
- CycloneDetectionCNN      : binary patch classifier (cyclone / no-cyclone)
- CycloneClassificationCNN : multi-class cyclone stage/category classifier
- ConvLSTMCell / ConvLSTM  : recurrent conv block for spatio-temporal sequences
- TrackIntensityPredictor  : predicts future track (dlat, dlon) + intensity
                             (wind speed, pressure) from a sequence of frames
"""

import torch
import torch.nn as nn


def conv_block(in_ch, out_ch, pool=True):
    layers = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class _CNNBackbone(nn.Module):
    """Shared conv backbone used by both detection and classification heads."""

    def __init__(self, in_channels=4):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(in_channels, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return x.flatten(1)  # (B, 128)


class CycloneDetectionCNN(nn.Module):
    """Binary classifier: does this satellite patch contain a cyclone?"""

    def __init__(self, in_channels=4, dropout=0.3):
        super().__init__()
        self.backbone = _CNNBackbone(in_channels)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        feats = self.backbone(x)
        return self.classifier(feats)


class CycloneClassificationCNN(nn.Module):
    """Multi-class classifier for cyclone pattern / intensity stage."""

    def __init__(self, in_channels=4, num_classes=8, dropout=0.3):
        super().__init__()
        self.backbone = _CNNBackbone(in_channels)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        feats = self.backbone(x)
        return self.classifier(feats)


# ---------------------------------------------------------------------------
# ConvLSTM for spatio-temporal sequence modeling (track + intensity)
# ---------------------------------------------------------------------------
class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x, state):
        h_prev, c_prev = state
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c

    def init_state(self, batch_size, height, width, device):
        shape = (batch_size, self.hidden_channels, height, width)
        return (torch.zeros(shape, device=device), torch.zeros(shape, device=device))


class ConvLSTM(nn.Module):
    def __init__(self, in_channels, hidden_channels=32, kernel_size=3):
        super().__init__()
        self.cell = ConvLSTMCell(in_channels, hidden_channels, kernel_size)
        self.hidden_channels = hidden_channels

    def forward(self, seq):
        # seq: (B, T, C, H, W)
        b, t, c, h, w = seq.shape
        state = self.cell.init_state(b, h, w, seq.device)
        outputs = []
        for step in range(t):
            state = self.cell(seq[:, step], state)
            outputs.append(state[0])
        return torch.stack(outputs, dim=1)  # (B, T, hidden, H, W)


class TrackIntensityPredictor(nn.Module):
    """
    Takes a sequence of preprocessed satellite/reanalysis frames and predicts
    the next `horizon` steps of (delta_lat, delta_lon, wind_speed, pressure).
    """

    def __init__(self, in_channels=4, hidden_channels=32, horizon=4, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(in_channels, 16, pool=True),
            conv_block(16, hidden_channels, pool=True),
        )
        self.convlstm = ConvLSTM(hidden_channels, hidden_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.horizon = horizon
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, horizon * 4),
        )

    def forward(self, seq):
        # seq: (B, T, C, H, W)
        b, t, c, h, w = seq.shape
        encoded = self.encoder(seq.view(b * t, c, h, w))
        _, ch, eh, ew = encoded.shape
        encoded = encoded.view(b, t, ch, eh, ew)

        lstm_out = self.convlstm(encoded)          # (B, T, hidden, eh, ew)
        last = lstm_out[:, -1]                      # (B, hidden, eh, ew)
        pooled = self.pool(last).flatten(1)         # (B, hidden)
        out = self.head(pooled)                     # (B, horizon*4)
        return out.view(b, self.horizon, 4)
