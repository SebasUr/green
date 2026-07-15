"""LSTM forecaster (optional) - a deep-learning comparison point.

This is a *modular experiment*, on the same footing as SARIMAX: it plugs into the
rolling-origin backtest and is compared against the gradient-boosting model, but
it is not a core dependency and is not the primary model.

Design (kept comparable to the project model). A shared network predicts every
horizon (direct multi-horizon):

* an **LSTM encoder** consumes the recent sequence of (normalized) carbon
  intensity ending at the origin ``t0``;
* a **static head** receives the target-time features (cyclical calendar plus the
  wind / solar / consumption forecast for ``t0+h``) and the horizon, and combines
  them with the encoder state to output the intensity at ``t0+h``.

So the LSTM sees the same information as the tree model (recent signal + exogenous
forecast), which is what makes the comparison fair. Requires ``torch``.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import pandas as pd

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # conda/torch libomp clash
try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("LSTMForecaster needs torch (pip install torch).") from exc

from green_observatory.carbon.climatology import _target_index
from green_observatory.carbon.features import FeatureBuilder
from green_observatory.models import ModelName
from green_observatory.providers.carbon_base import CARBON

_CAL_COLS = ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "is_weekend", "is_holiday"]


class _Net(nn.Module):
    def __init__(self, static_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden + static_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, seq: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(seq)          # h_n: (1, B, hidden)
        return self.head(torch.cat([h_n[-1], static], dim=1)).squeeze(-1)


class LSTMForecaster:
    """Recent-sequence LSTM + exogenous-forecast head, direct multi-horizon."""

    name = ModelName.lstm

    def __init__(
        self,
        feature_builder: FeatureBuilder,
        *,
        horizons: Sequence[int] = (1, 3, 6, 12, 24, 48),
        seq_len: int = 48,
        hidden: int = 64,
        epochs: int = 40,
        batch_size: int = 512,
        lr: float = 1e-3,
        patience: int = 5,
        random_state: int = 42,
        device: str | None = None,
    ) -> None:
        self.fb = feature_builder
        self.horizons = tuple(int(h) for h in horizons)
        self.seq_len = seq_len
        self.hidden = hidden
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.random_state = random_state
        self.device = torch.device(
            device or ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        self.net: _Net | None = None
        self.c_mean = self.c_std = 0.0
        self.stat_mean = self.stat_std = None

    # -- feature construction ------------------------------------------- #
    def _static(self, target_times: pd.DatetimeIndex, horizon: int) -> np.ndarray:
        cal = self.fb.calendar_features(target_times)[_CAL_COLS].to_numpy(dtype=float)
        cols = [cal]
        ff = getattr(self.fb, "forecast_frame", None)
        if ff is not None:
            fc = ff.reindex(target_times)
            wind = pd.to_numeric(fc.get("wind_speed_100m"), errors="coerce").to_numpy()
            solar = pd.to_numeric(fc.get("solar_radiation"), errors="coerce").to_numpy()
            cons = pd.to_numeric(fc.get("consumption_forecast_mw"), errors="coerce").to_numpy()
            if horizon > getattr(self.fb, "forecast_maxlead_h", 24):
                cons = np.zeros_like(cons)
            cols.append(np.stack([wind, solar, cons], axis=1))
        cols.append(np.full((len(target_times), 1), float(horizon)))
        out = np.concatenate(cols, axis=1)
        return np.nan_to_num(out, nan=0.0)

    def _windows(self, carbon_norm: np.ndarray) -> np.ndarray:
        from numpy.lib.stride_tricks import sliding_window_view

        return sliding_window_view(carbon_norm, self.seq_len)  # (N-seq_len+1, seq_len)

    # -- training ------------------------------------------------------- #
    def fit(self, train_frame: pd.DataFrame) -> LSTMForecaster:
        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)

        c = pd.to_numeric(train_frame[CARBON], errors="coerce").to_numpy()
        self.c_mean, self.c_std = float(np.nanmean(c)), float(np.nanstd(c) + 1e-6)
        cn = (c - self.c_mean) / self.c_std
        idx = train_frame.index
        wins = self._windows(cn)  # window w ends at position w + seq_len - 1

        seqs, stats, ys = [], [], []
        n = len(cn)
        for h in self.horizons:
            lo = self.seq_len - 1
            hi = n - 1 - h
            if hi < lo:
                continue
            pos = np.arange(lo, hi + 1)
            valid = np.isfinite(cn[pos]) & np.isfinite(cn[pos + h])
            pos = pos[valid]
            seqs.append(wins[pos - self.seq_len + 1])
            stats.append(self._static(idx[pos + h], h))
            ys.append(cn[pos + h])
        seqs = np.concatenate(seqs).astype(np.float32)
        stats = np.concatenate(stats).astype(np.float32)
        ys = np.concatenate(ys).astype(np.float32)

        self.stat_mean = stats.mean(0)
        self.stat_std = stats.std(0) + 1e-6
        stats = (stats - self.stat_mean) / self.stat_std

        n_ex = len(ys)
        perm = rng.permutation(n_ex)
        n_val = max(1, int(0.1 * n_ex))
        val_i, tr_i = perm[:n_val], perm[n_val:]

        dev = self.device
        S = torch.tensor(seqs, device=dev).unsqueeze(-1)
        X = torch.tensor(stats, device=dev)
        Y = torch.tensor(ys, device=dev)

        net = _Net(stats.shape[1], self.hidden).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        lossf = nn.MSELoss()
        best_val, best_state, bad = np.inf, None, 0
        for _ in range(self.epochs):
            net.train()
            for b in range(0, len(tr_i), self.batch_size):
                bi = tr_i[b : b + self.batch_size]
                opt.zero_grad()
                pred = net(S[bi], X[bi])
                loss = lossf(pred, Y[bi])
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                vloss = float(lossf(net(S[val_i], X[val_i]), Y[val_i]))
            if vloss < best_val - 1e-4:
                best_val, best_state, bad = vloss, {k: v.detach().clone() for k, v in net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        self.net = net
        return self

    # -- prediction ----------------------------------------------------- #
    def predict(
        self, history: pd.DataFrame, origin: pd.Timestamp, horizons_hours: Sequence[float]
    ) -> pd.DataFrame:
        if self.net is None:
            raise RuntimeError("LSTMForecaster.predict called before fit")
        c = pd.to_numeric(history.loc[history.index <= origin, CARBON], errors="coerce")
        cn = ((c - self.c_mean) / self.c_std).to_numpy()
        seq = cn[-self.seq_len :]
        if len(seq) < self.seq_len:
            seq = np.concatenate([np.full(self.seq_len - len(seq), seq[0]), seq])

        seqs, stats = [], []
        for h in horizons_hours:
            tt = pd.DatetimeIndex([origin + pd.Timedelta(hours=int(h))])
            stats.append(self._static(tt, int(h))[0])
            seqs.append(seq)
        S = torch.tensor(np.array(seqs, dtype=np.float32), device=self.device).unsqueeze(-1)
        X = (np.array(stats, dtype=np.float32) - self.stat_mean) / self.stat_std
        X = torch.tensor(X, device=self.device)
        with torch.no_grad():
            out = self.net(S, X).cpu().numpy().ravel()
        preds = np.clip(out * self.c_std + self.c_mean, 0.0, None)
        return pd.DataFrame(
            {"prediction": preds, "lower": np.nan, "upper": np.nan,
             "horizon_hours": list(horizons_hours)},
            index=_target_index(origin, horizons_hours),
        )
