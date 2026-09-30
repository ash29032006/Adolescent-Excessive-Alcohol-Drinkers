"""Small 1D CNN on the raw region time series (comparison only, like the paper's deep models).

Scans have different lengths per site, so training uses random fixed-length crops and the
network ends in global average pooling, which lets it score a full scan of any length.
"""
import numpy as np
import torch
from torch import nn

SEED = 42


class CNN(nn.Module):
    """200 regions as input channels -> two temporal conv layers -> average over time -> logit."""

    def __init__(self, n_regions=200, width=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_regions, width, 5, padding=2), nn.BatchNorm1d(width), nn.ReLU(), nn.Dropout(0.3),
            nn.Conv1d(width, width, 5, padding=2), nn.BatchNorm1d(width), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(width, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _zscore(t):
    """Z-score each region over time (as in the paper) and return (regions, time) float32."""
    return ((t - t.mean(0)) / (t.std(0) + 1e-8)).T.astype(np.float32)


def cnn_oof_probs(ts, y, splits, crop=64, epochs=40, batch=32):
    """Train one CNN per fold and return out-of-fold probabilities of class 1."""
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    X = [_zscore(t) for t in ts]
    crop = min(crop, min(x.shape[1] for x in X))
    prob = np.zeros(len(y))
    for tr, te in splits:
        model = CNN(X[0].shape[0])
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
        pos_weight = torch.tensor((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1), dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # balances the classes
        for _ in range(epochs):
            model.train()
            for b in np.array_split(rng.permutation(tr), max(len(tr) // batch, 1)):
                starts = [rng.integers(0, X[i].shape[1] - crop + 1) for i in b]
                xb = torch.from_numpy(np.stack([X[i][:, s:s + crop] for i, s in zip(b, starts)]))
                opt.zero_grad()
                loss_fn(model(xb), torch.tensor(y[b], dtype=torch.float32)).backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            for i in te:
                prob[i] = torch.sigmoid(model(torch.from_numpy(X[i])[None])).item()
    return prob
