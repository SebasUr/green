"""Learning-to-rank model for 24-hour French green-window selection.

This module is intentionally separate from the point forecasters.  Its output
is an ordering inside each 24-hour forecast query, not an estimate in
gCO2/kWh.  Lower observed carbon receives higher ordinal relevance during
training; at inference time the LightGBM score is converted to a within-day
percentile where zero means *greenest*.

Keeping this distinction explicit prevents a rank score from accidentally
being reported as a carbon-intensity forecast or evaluated with MAPE.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


DEFAULT_HORIZONS = tuple(range(1, 25))


@dataclass(frozen=True)
class RankerSpec:
    """Small, serialisable LightGBM ranking configuration."""

    name: str
    objective: str
    n_estimators: int
    learning_rate: float
    num_leaves: int
    max_depth: int
    min_child_samples: int
    feature_fraction: float = 0.85
    bagging_fraction: float = 0.85
    reg_lambda: float = 1.0
    lambdarank_truncation_level: int = 6
    gain: str = "linear"

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_RANKER_SPECS = (
    RankerSpec(
        name="lambdarank_regular",
        objective="lambdarank",
        n_estimators=320,
        learning_rate=0.035,
        num_leaves=15,
        max_depth=5,
        min_child_samples=96,
        lambdarank_truncation_level=6,
        gain="linear",
    ),
    RankerSpec(
        name="lambdarank_top_heavy",
        objective="lambdarank",
        n_estimators=280,
        learning_rate=0.04,
        num_leaves=31,
        max_depth=6,
        min_child_samples=72,
        lambdarank_truncation_level=3,
        gain="top_heavy",
    ),
    RankerSpec(
        name="rank_xendcg_regular",
        objective="rank_xendcg",
        n_estimators=320,
        learning_rate=0.035,
        num_leaves=15,
        max_depth=5,
        min_child_samples=96,
        gain="linear",
    ),
    RankerSpec(
        name="rank_xendcg_flexible",
        objective="rank_xendcg",
        n_estimators=280,
        learning_rate=0.04,
        num_leaves=31,
        max_depth=6,
        min_child_samples=72,
        gain="top_heavy",
    ),
)


def complete_query_positions(
    meta: pd.DataFrame,
    *,
    mask: pd.Series | np.ndarray | None = None,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> np.ndarray:
    """Return row positions belonging to complete, finite daily queries.

    A time cutoff may split a forecast origin (for example the h24 label at
    midnight).  Such a query is discarded rather than silently training a
    different ranking problem with 23 candidates.
    """

    required = {"origin", "horizon", "actual"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"ranking metadata miss columns: {sorted(missing)}")
    allowed = np.ones(len(meta), dtype=bool)
    if mask is not None:
        allowed &= np.asarray(mask, dtype=bool)
    expected = tuple(int(value) for value in horizons)
    candidates = meta.loc[allowed, ["origin", "horizon", "actual"]].copy()
    candidates["_position"] = np.flatnonzero(allowed)
    positions: list[int] = []
    for _, group in candidates.groupby("origin", sort=True):
        group = group.sort_values("horizon")
        observed = tuple(pd.to_numeric(group["horizon"]).astype(int))
        actual = pd.to_numeric(group["actual"], errors="coerce").to_numpy()
        if observed != expected or not np.isfinite(actual).all():
            continue
        positions.extend(group["_position"].astype(int).tolist())
    return np.asarray(positions, dtype=int)


def ordinal_relevance(
    meta: pd.DataFrame,
    *,
    query_size: int = 24,
) -> np.ndarray:
    """Map lower actual carbon to higher integer relevance per query.

    Equal carbon values receive equal relevance.  The scale is 0..23 for a
    24-hour query, which is accepted by both ``lambdarank`` and
    ``rank_xendcg``.
    """

    work = meta[["origin", "actual"]].copy().reset_index(drop=True)
    work["_position"] = np.arange(len(work), dtype=int)
    labels = np.empty(len(work), dtype=np.int32)
    for _, group in work.groupby("origin", sort=False):
        if len(group) != query_size:
            raise ValueError("ordinal relevance requires complete queries")
        actual = pd.to_numeric(group["actual"], errors="coerce")
        if not np.isfinite(actual.to_numpy()).all():
            raise ValueError("ordinal relevance requires finite actual values")
        ascending_rank = actual.rank(method="min", ascending=True).astype(int)
        relevance = query_size - ascending_rank
        labels[group["_position"].to_numpy(dtype=int)] = relevance.to_numpy(
            dtype=np.int32
        )
    return labels


def within_query_percentile(
    values: pd.Series | np.ndarray,
    origins: pd.Series | np.ndarray,
    *,
    higher_is_greener: bool,
) -> np.ndarray:
    """Return a 0=greenest, 1=dirtiest percentile inside each query."""

    frame = pd.DataFrame(
        {
            "value": np.asarray(values, dtype=float),
            "origin": np.asarray(origins),
            "_position": np.arange(len(values), dtype=int),
        }
    )
    output = np.full(len(frame), np.nan, dtype=float)
    for _, group in frame.groupby("origin", sort=False):
        finite = np.isfinite(group["value"].to_numpy(dtype=float))
        if not finite.all() or len(group) < 2:
            continue
        ranks = group["value"].rank(
            method="average", ascending=not higher_is_greener
        )
        percentiles = (ranks - 1.0) / (len(group) - 1.0)
        output[group["_position"].to_numpy(dtype=int)] = percentiles.to_numpy()
    return output


def blend_query_percentiles(
    ranker_score: pd.Series | np.ndarray,
    reference_intensity: pd.Series | np.ndarray,
    origins: pd.Series | np.ndarray,
    *,
    ranker_weight: float,
) -> np.ndarray:
    """Blend ranker and point-model orderings without mixing their units."""

    weight = float(ranker_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("ranker_weight must lie in [0, 1]")
    ranker_rank = within_query_percentile(
        ranker_score, origins, higher_is_greener=True
    )
    reference_rank = within_query_percentile(
        reference_intensity, origins, higher_is_greener=False
    )
    return weight * ranker_rank + (1.0 - weight) * reference_rank


def _label_gain(name: str, query_size: int = 24) -> list[int]:
    if name == "linear":
        return list(range(query_size))
    if name == "top_heavy":
        # Smoothly increases the importance of the very greenest hours while
        # avoiding the extreme 2**relevance default used by LightGBM.
        return [int(round((value**1.7) * 10.0)) for value in range(query_size)]
    raise ValueError(f"unknown label gain: {name!r}")


class GreenWindowRanker:
    """Thin fitted wrapper around :class:`lightgbm.LGBMRanker`."""

    def __init__(
        self,
        spec: RankerSpec,
        *,
        n_jobs: int = 1,
        random_state: int = 20260719,
    ) -> None:
        self.spec = spec
        self.n_jobs = int(n_jobs)
        self.random_state = int(random_state)
        self.model_ = None
        self.feature_columns_: list[str] = []

    @staticmethod
    def _lightgbm_ranker():
        try:
            from lightgbm import LGBMRanker
        except ImportError as exc:  # pragma: no cover - environment specific
            raise RuntimeError(
                "green-window ranking requires the optional lightgbm dependency"
            ) from exc
        return LGBMRanker

    @staticmethod
    def _numeric_matrix(frame: pd.DataFrame) -> pd.DataFrame:
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        return numeric.replace([np.inf, -np.inf], np.nan).astype(np.float32)

    def fit(self, x: pd.DataFrame, meta: pd.DataFrame) -> "GreenWindowRanker":
        if len(x) != len(meta):
            raise ValueError("features and ranking metadata must have equal length")
        order = meta.sort_values(["origin", "horizon"]).index.to_numpy(dtype=int)
        x_ordered = self._numeric_matrix(x.loc[order].reset_index(drop=True))
        meta_ordered = meta.loc[order].reset_index(drop=True)
        group_sizes = meta_ordered.groupby("origin", sort=False).size().to_numpy(dtype=int)
        if len(group_sizes) == 0 or not np.all(group_sizes == 24):
            raise ValueError("GreenWindowRanker.fit requires complete 24-hour queries")
        useful = x_ordered.columns[x_ordered.notna().any(axis=0)].tolist()
        if not useful:
            raise ValueError("ranker has no usable feature columns")
        self.feature_columns_ = useful
        labels = ordinal_relevance(meta_ordered)
        params = {
            "objective": self.spec.objective,
            "n_estimators": self.spec.n_estimators,
            "learning_rate": self.spec.learning_rate,
            "num_leaves": self.spec.num_leaves,
            "max_depth": self.spec.max_depth,
            "min_child_samples": self.spec.min_child_samples,
            "colsample_bytree": self.spec.feature_fraction,
            "subsample": self.spec.bagging_fraction,
            "subsample_freq": 1,
            "reg_lambda": self.spec.reg_lambda,
            "label_gain": _label_gain(self.spec.gain),
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "verbosity": -1,
            "deterministic": True,
            "force_col_wise": True,
        }
        if self.spec.objective == "lambdarank":
            params["lambdarank_truncation_level"] = (
                self.spec.lambdarank_truncation_level
            )
        LGBMRanker = self._lightgbm_ranker()
        self.model_ = LGBMRanker(**params)
        self.model_.fit(x_ordered[useful], labels, group=group_sizes)
        return self

    def predict_score(self, x: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("GreenWindowRanker must be fitted before prediction")
        matrix = self._numeric_matrix(x.reindex(columns=self.feature_columns_))
        return np.asarray(self.model_.predict(matrix), dtype=float)

    def feature_importance(self, *, importance_type: str = "gain") -> pd.DataFrame:
        """Return sorted fitted feature importance for diagnostics."""

        if self.model_ is None:
            raise RuntimeError("GreenWindowRanker must be fitted before inspection")
        values = self.model_.booster_.feature_importance(
            importance_type=importance_type
        )
        out = pd.DataFrame(
            {"feature": self.feature_columns_, "importance": values.astype(float)}
        )
        total = float(out["importance"].sum())
        out["share"] = out["importance"] / total if total > 0.0 else 0.0
        return out.sort_values(
            ["importance", "feature"], ascending=[False, True]
        ).reset_index(drop=True)

    def save_model(self, path: str | Path) -> None:
        """Persist the native LightGBM booster without pickling Python state."""

        if self.model_ is None:
            raise RuntimeError("GreenWindowRanker must be fitted before saving")
        self.model_.booster_.save_model(str(path))
