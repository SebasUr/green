"""Exhaustive evaluation of contiguous green windows over a dense 24h curve.

The legacy green-window metric asks a useful but narrower question: which one
hour should be selected?  This module evaluates a fixed-duration workload by
enumerating every feasible contiguous start inside horizons 1..24.  It is
deliberately isolated so historical reports keep their original semantics.

Input uses the project's tidy prediction contract::

    model, origin, horizon, target_time, prediction, actual

For each ``(model, origin, duration)`` query, lower mean prediction selects the
window.  The selected window is realized with the actual curve and compared
with perfect foresight, ASAP (the first forecastable slot), and a uniformly
random start.  Durations are never compared to choose a workload: they remain
separate evaluation strata.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


HORIZON_HOURS = 24
REQUIRED_COLUMNS = (
    "model",
    "origin",
    "horizon",
    "target_time",
    "prediction",
    "actual",
)


@dataclass(frozen=True)
class ExhaustiveWindowResult:
    """Three auditable levels of an exhaustive window evaluation."""

    catalog: pd.DataFrame
    decisions: pd.DataFrame
    aggregate: pd.DataFrame


def _validated_horizon_hours(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError("horizon_hours must be a positive integer")
    value = int(value)
    if value < 1:
        raise ValueError("horizon_hours must be a positive integer")
    return value


def _durations(
    values: Sequence[int], *, horizon_hours: int
) -> tuple[int, ...]:
    durations: list[int] = []
    for raw in values:
        if isinstance(raw, (bool, np.bool_)) or not isinstance(raw, (int, np.integer)):
            raise ValueError("window durations must be integer hours")
        value = int(raw)
        if not 1 <= value <= horizon_hours:
            raise ValueError(
                f"window duration must lie in [1, {horizon_hours}], got {value}"
            )
        if value not in durations:
            durations.append(value)
    if not durations:
        raise ValueError("at least one window duration is required")
    return tuple(sorted(durations))


def validate_complete_queries(
    predictions: pd.DataFrame,
    *,
    horizon_hours: int = HORIZON_HOURS,
) -> pd.DataFrame:
    """Validate and normalize complete, common dense model queries.

    Every model must cover exactly the same origins.  Each query must contain
    horizons 1..24 exactly once, target timestamps must equal ``origin+h``, and
    both prediction and actual must be finite.  Actual curves must also agree
    across models, preventing an apparently fair comparison on different
    labels or candidate sets.
    """

    horizon_hours = _validated_horizon_hours(horizon_hours)
    missing = sorted(set(REQUIRED_COLUMNS).difference(predictions.columns))
    if missing:
        raise ValueError(f"window predictions miss columns: {missing}")
    if predictions.empty:
        raise ValueError("window predictions are empty")

    out = predictions.copy()
    out["origin"] = pd.to_datetime(out["origin"], utc=True, errors="raise")
    out["target_time"] = pd.to_datetime(
        out["target_time"], utc=True, errors="raise"
    )
    numeric_horizon = pd.to_numeric(out["horizon"], errors="raise")
    if not np.equal(numeric_horizon, np.floor(numeric_horizon)).all():
        raise ValueError("horizons must be integer hours")
    out["horizon"] = numeric_horizon.astype(int)
    for column in ("prediction", "actual"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
        if not np.isfinite(out[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{column} must be finite in every complete query")

    if out.duplicated(["model", "origin", "horizon"]).any():
        raise ValueError("duplicate model-origin-horizon rows are not allowed")
    expected_horizons = tuple(range(1, horizon_hours + 1))
    for (model, origin), group in out.groupby(["model", "origin"], sort=True):
        group = group.sort_values("horizon")
        horizons = tuple(group["horizon"].tolist())
        if horizons != expected_horizons:
            raise ValueError(
                f"incomplete {horizon_hours}h query for model={model!r}, "
                f"origin={origin}: "
                f"got horizons {horizons}"
            )
        expected_targets = origin + pd.to_timedelta(
            np.arange(1, horizon_hours + 1), unit="h"
        )
        observed_targets = pd.DatetimeIndex(group["target_time"])
        if not observed_targets.equals(expected_targets):
            raise ValueError(
                f"target_time is not origin+horizon for model={model!r}, "
                f"origin={origin}"
            )

    supports = {
        model: frozenset(group["origin"].unique())
        for model, group in out.groupby("model", sort=True)
    }
    reference_model = next(iter(supports))
    reference_support = supports[reference_model]
    for model, support in supports.items():
        if support != reference_support:
            raise ValueError(
                "models must share identical origin support; "
                f"{model!r} differs from {reference_model!r}"
            )

    # The actual curve is model-independent.  Compare it explicitly rather
    # than trusting whichever model happens to appear first in a groupby.
    ordered = out.sort_values(["origin", "model", "horizon"])
    for origin, origin_group in ordered.groupby("origin", sort=True):
        curves = []
        names = []
        for model, group in origin_group.groupby("model", sort=True):
            curves.append(group.sort_values("horizon")["actual"].to_numpy(dtype=float))
            names.append(model)
        reference = curves[0]
        for model, curve in zip(names[1:], curves[1:]):
            if not np.allclose(curve, reference, rtol=0.0, atol=1e-9):
                raise ValueError(
                    f"actual curve differs across models at origin={origin}; "
                    f"model={model!r}"
                )
    return out.sort_values(["model", "origin", "horizon"]).reset_index(drop=True)


def _rolling_means(values: np.ndarray, duration: int) -> np.ndarray:
    """All fixed-duration means in O(H) via one prefix sum."""

    prefix = np.concatenate(([0.0], np.cumsum(np.asarray(values, dtype=float))))
    return (prefix[duration:] - prefix[:-duration]) / float(duration)


def enumerate_contiguous_windows(
    predictions: pd.DataFrame,
    *,
    durations: Sequence[int] | None = None,
    horizon_hours: int = HORIZON_HOURS,
) -> pd.DataFrame:
    """Return every feasible contiguous window for every complete query.

    ``start_horizon`` and ``end_horizon`` are inclusive slot labels.
    ``window_end_time`` is exclusive, matching :class:`GreenWindow`.
    """

    horizon_hours = _validated_horizon_hours(horizon_hours)
    frame = validate_complete_queries(
        predictions, horizon_hours=horizon_hours
    )
    duration_values = _durations(
        range(1, horizon_hours + 1) if durations is None else durations,
        horizon_hours=horizon_hours,
    )
    parts: list[pd.DataFrame] = []
    for (model, origin), group in frame.groupby(["model", "origin"], sort=True):
        group = group.sort_values("horizon")
        prediction = group["prediction"].to_numpy(dtype=float)
        actual = group["actual"].to_numpy(dtype=float)
        target_times = pd.DatetimeIndex(group["target_time"])
        for duration in duration_values:
            predicted_cost = _rolling_means(prediction, duration)
            actual_cost = _rolling_means(actual, duration)
            starts = np.arange(len(predicted_cost), dtype=int)
            start_horizons = starts + 1
            end_horizons = start_horizons + duration - 1
            part = pd.DataFrame(
                {
                    "model": model,
                    "origin": origin,
                    "duration_hours": duration,
                    "start_horizon": start_horizons,
                    "end_horizon": end_horizons,
                    "delay_from_first_slot_hours": starts,
                    "delay_from_origin_hours": start_horizons,
                    "window_start_time": target_times[starts],
                    "window_end_time": target_times[end_horizons - 1]
                    + pd.Timedelta(hours=1),
                    "predicted_window_cost": predicted_cost,
                    "actual_window_cost": actual_cost,
                    "candidate_count": len(starts),
                    "deadline_horizon": horizon_hours,
                }
            )
            parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _earliest_minimum(values: np.ndarray, tolerance: float) -> tuple[int, np.ndarray]:
    minimum = float(np.min(values))
    tied = np.flatnonzero(
        np.isclose(values, minimum, rtol=0.0, atol=float(tolerance))
    )
    return int(tied[0]), tied


def decisions_from_catalog(
    catalog: pd.DataFrame,
    *,
    epsilon_gco2: float = 1.0,
    tie_tolerance: float = 1e-9,
) -> pd.DataFrame:
    """Select one window and score it for each model-origin-duration query."""

    if epsilon_gco2 < 0.0:
        raise ValueError("epsilon_gco2 must be non-negative")
    if tie_tolerance < 0.0:
        raise ValueError("tie_tolerance must be non-negative")
    required = {
        "model",
        "origin",
        "duration_hours",
        "start_horizon",
        "window_start_time",
        "predicted_window_cost",
        "actual_window_cost",
    }
    missing = sorted(required.difference(catalog.columns))
    if missing:
        raise ValueError(f"window catalog miss columns: {missing}")

    rows: list[dict] = []
    for (model, origin, duration), group in catalog.groupby(
        ["model", "origin", "duration_hours"], sort=True
    ):
        group = group.sort_values("start_horizon").reset_index(drop=True)
        prediction = group["predicted_window_cost"].to_numpy(dtype=float)
        actual = group["actual_window_cost"].to_numpy(dtype=float)
        if not np.isfinite(prediction).all() or not np.isfinite(actual).all():
            raise ValueError("window catalog costs must be finite")
        selected_position, predicted_ties = _earliest_minimum(
            prediction, tie_tolerance
        )
        oracle_position, oracle_ties = _earliest_minimum(actual, tie_tolerance)
        selected_cost = float(actual[selected_position])
        oracle_cost = float(actual[oracle_position])
        asap_cost = float(actual[0])
        random_expected = float(np.mean(actual))
        strictly_better = int(
            np.sum(actual < selected_cost - float(tie_tolerance))
        )
        rank_min = 1 + strictly_better
        candidates = len(group)
        rank_percentile = (
            float(rank_min - 1) / float(candidates - 1)
            if candidates > 1
            else 0.0
        )
        top10_cutoff = max(1, int(np.ceil(0.10 * candidates)))
        if (
            candidates > 1
            and np.ptp(prediction) > tie_tolerance
            and np.ptp(actual) > tie_tolerance
        ):
            correlation = float(spearmanr(prediction, actual).statistic)
        else:
            correlation = float("nan")
        selected = group.iloc[selected_position]
        oracle = group.iloc[oracle_position]
        regret = selected_cost - oracle_cost
        savings = asap_cost - selected_cost
        opportunity = asap_cost - oracle_cost
        rows.append(
            {
                "model": model,
                "origin": origin,
                "duration_hours": int(duration),
                "candidate_count": candidates,
                "selected_start_horizon": int(selected["start_horizon"]),
                "selected_start_time": selected["window_start_time"],
                "selected_predicted_cost": float(
                    selected["predicted_window_cost"]
                ),
                "selected_actual_cost": selected_cost,
                "oracle_start_horizon": int(oracle["start_horizon"]),
                "oracle_start_time": oracle["window_start_time"],
                "oracle_actual_cost": oracle_cost,
                "asap_actual_cost": asap_cost,
                "random_expected_actual_cost": random_expected,
                "regret_gco2_kwh": regret,
                "savings_vs_asap_gco2_kwh": savings,
                "oracle_opportunity_gco2_kwh": opportunity,
                "improvement_vs_random_gco2_kwh": random_expected
                - selected_cost,
                "random_regret_gco2_kwh": random_expected - oracle_cost,
                "top1_tie_aware": bool(regret <= tie_tolerance),
                "epsilon_optimal": bool(
                    selected_cost <= oracle_cost + epsilon_gco2 + tie_tolerance
                ),
                "selected_actual_rank_min": rank_min,
                "selected_actual_rank_percentile": rank_percentile,
                "top10pct_hit": bool(rank_min <= top10_cutoff),
                "window_cost_spearman": correlation,
                "predicted_tie_count": int(len(predicted_ties)),
                "oracle_tie_count": int(len(oracle_ties)),
                "epsilon_gco2": float(epsilon_gco2),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["model", "duration_hours", "origin"]
    ).reset_index(drop=True)


def _ratio_of_sums(numerator: pd.Series, denominator: pd.Series) -> float:
    denominator_sum = float(denominator.sum())
    if denominator_sum <= 1e-12:
        return float("nan")
    return 100.0 * float(numerator.sum()) / denominator_sum


def aggregate_window_metrics(decisions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate decision quality per model and duration.

    Oracle potential is deliberately a ratio of sums and is not clipped.  A
    policy worse than ASAP therefore receives a negative score, while a zero-
    opportunity stratum (notably duration 24) receives ``NaN``.
    """

    required = {
        "model",
        "origin",
        "duration_hours",
        "selected_actual_cost",
        "oracle_actual_cost",
        "asap_actual_cost",
        "random_expected_actual_cost",
        "regret_gco2_kwh",
        "savings_vs_asap_gco2_kwh",
        "oracle_opportunity_gco2_kwh",
        "improvement_vs_random_gco2_kwh",
        "random_regret_gco2_kwh",
        "top1_tie_aware",
        "epsilon_optimal",
        "selected_actual_rank_percentile",
        "top10pct_hit",
        "window_cost_spearman",
    }
    missing = sorted(required.difference(decisions.columns))
    if missing:
        raise ValueError(f"window decisions miss columns: {missing}")

    rows: list[dict] = []
    for (model, duration), group in decisions.groupby(
        ["model", "duration_hours"], sort=True
    ):
        opportunity = group["oracle_opportunity_gco2_kwh"]
        rows.append(
            {
                "model": model,
                "duration_hours": int(duration),
                "origins": int(group["origin"].nunique()),
                "mean_selected_actual_cost": float(
                    group["selected_actual_cost"].mean()
                ),
                "mean_oracle_actual_cost": float(
                    group["oracle_actual_cost"].mean()
                ),
                "mean_asap_actual_cost": float(group["asap_actual_cost"].mean()),
                "mean_random_expected_actual_cost": float(
                    group["random_expected_actual_cost"].mean()
                ),
                "mean_regret_gco2_kwh": float(group["regret_gco2_kwh"].mean()),
                "mean_savings_vs_asap_gco2_kwh": float(
                    group["savings_vs_asap_gco2_kwh"].mean()
                ),
                "mean_oracle_opportunity_gco2_kwh": float(opportunity.mean()),
                "mean_improvement_vs_random_gco2_kwh": float(
                    group["improvement_vs_random_gco2_kwh"].mean()
                ),
                "mean_random_regret_gco2_kwh": float(
                    group["random_regret_gco2_kwh"].mean()
                ),
                "oracle_potential_pct": _ratio_of_sums(
                    group["savings_vs_asap_gco2_kwh"], opportunity
                ),
                "random_oracle_potential_pct": _ratio_of_sums(
                    group["asap_actual_cost"]
                    - group["random_expected_actual_cost"],
                    opportunity,
                ),
                "top1_accuracy_tie_aware": float(
                    group["top1_tie_aware"].mean()
                ),
                "epsilon_optimal_rate": float(group["epsilon_optimal"].mean()),
                "mean_selected_actual_rank_percentile": float(
                    group["selected_actual_rank_percentile"].mean()
                ),
                "top10pct_hit_rate": float(group["top10pct_hit"].mean()),
                "mean_window_cost_spearman": float(
                    group["window_cost_spearman"].mean()
                ),
                "opportunity_positive_origins": int(
                    (opportunity > 1e-12).sum()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["duration_hours", "mean_regret_gco2_kwh", "model"]
    ).reset_index(drop=True)


def evaluate_exhaustive_windows(
    predictions: pd.DataFrame,
    *,
    durations: Sequence[int] | None = None,
    horizon_hours: int = HORIZON_HOURS,
    epsilon_gco2: float = 1.0,
    tie_tolerance: float = 1e-9,
) -> ExhaustiveWindowResult:
    """Validate, enumerate, decide and aggregate an exhaustive 24h backtest."""

    catalog = enumerate_contiguous_windows(
        predictions,
        durations=durations,
        horizon_hours=horizon_hours,
    )
    decisions = decisions_from_catalog(
        catalog,
        epsilon_gco2=epsilon_gco2,
        tie_tolerance=tie_tolerance,
    )
    aggregate = aggregate_window_metrics(decisions)
    return ExhaustiveWindowResult(
        catalog=catalog, decisions=decisions, aggregate=aggregate
    )


__all__ = [
    "HORIZON_HOURS",
    "ExhaustiveWindowResult",
    "aggregate_window_metrics",
    "decisions_from_catalog",
    "enumerate_contiguous_windows",
    "evaluate_exhaustive_windows",
    "validate_complete_queries",
]
