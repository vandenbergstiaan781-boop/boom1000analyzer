"""Boom 1000 math engine: pure NumPy, no UI, no network, no API keys.

The engine takes arrays and returns arrays or plain dictionaries, so it can sit
behind a mobile app or a small API later without changes.

Typical use:
    bars = Bars.from_prices(prices)        # or Bars(open, high, low, close)
    result = analyze(bars, Params())
    snapshot = latest_snapshot(result)     # verdict and checklist, JSON-safe
    report = edge_report(bars, result)     # did the score predict anything?

Every series is causal: the value at bar t uses only data up to bar t. That keeps
edge_report honest, because the score cannot peek at future spikes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# Verdict codes, ordered from weakest to strongest.
WAIT, WATCH, HIGH = 0, 1, 2
VERDICT_LABELS = {WAIT: "Wait", WATCH: "Watch Closely", HIGH: "High Probability Setup"}

# Number of completed spike cycles needed before the cycle percentile means anything.
MIN_CYCLES = 5


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Bars:
    """OHLC arrays of equal length. Time is optional (epoch seconds)."""

    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    time: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        sizes = {len(self.open), len(self.high), len(self.low), len(self.close)}
        if len(sizes) != 1:
            raise ValueError("open, high, low and close must have the same length")
        if len(self.close) == 0:
            raise ValueError("no price data supplied")
        if self.time is not None and len(self.time) != len(self.close):
            raise ValueError("time must have the same length as the prices")

    def __len__(self) -> int:
        return len(self.close)

    @classmethod
    def from_prices(cls, prices, time=None) -> "Bars":
        """Build bars from one price per tick. Each tick becomes a flat bar."""
        p = np.asarray(prices, dtype=float)
        return cls(p, p, p, p, None if time is None else np.asarray(time, dtype=float))


@dataclass(frozen=True)
class Params:
    """Every tunable number. The Streamlit sliders map onto these one to one."""

    # Baseline
    ema_period: int = 20
    atr_period: int = 14

    # Spike detection: a bar is a spike when its upward move beats
    # max(min_spike_points, spike_multiplier * median recent bar move).
    spike_multiplier: float = 5.0
    min_spike_points: float = 0.0
    baseline_window: int = 300

    # Pullback
    swing_lookback: int = 30

    # Grind quality
    grind_window: int = 10

    # Targets: a component scores 100% once it reaches its target.
    ema_gap_target_atr: float = 8.0
    pullback_target_atr: float = 25.0
    cycle_percentile_target: float = 0.80
    grind_ratio_target: float = 0.90

    # Weights (normalised, so they do not have to add up to 100).
    weight_ema: float = 30.0
    weight_pullback: float = 30.0
    weight_cycle: float = 25.0
    weight_grind: float = 15.0

    # Verdict cut-offs on the 0-100 score.
    high_threshold: float = 85.0
    watch_threshold: float = 60.0


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------


def ema(values: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average, seeded with an SMA. NaN during warm-up."""
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan)
    if period < 1 or len(values) < period:
        return out
    alpha = 2.0 / (period + 1.0)
    out[period - 1] = values[:period].mean()
    for i in range(period, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def _previous_close(bars: Bars) -> np.ndarray:
    return np.concatenate(([bars.open[0]], bars.close[:-1]))


def _trailing_baseline(moves: np.ndarray, window: int, min_history: int = 20) -> np.ndarray:
    """Typical bar size from past bars only (the current bar is excluded).

    Uses the median so past spikes do not inflate it. Falls back to the mean when
    most bars did not move and the median is zero. NaN until min_history bars exist.
    """
    n = len(moves)
    out = np.full(n, np.nan)
    window = max(window, min_history)

    def typical(sample: np.ndarray) -> np.ndarray:
        median = np.median(sample, axis=-1)
        return np.where(median > 0, median, sample.mean(axis=-1))

    # Short head: the window is still growing.
    for t in range(min_history + 1, min(window + 1, n)):
        out[t] = typical(moves[1:t])
    # Long tail: full windows, vectorised.
    if n > window + 1:
        full = sliding_window_view(moves[1:], window)[: n - window - 1]
        out[window + 1 :] = typical(full)
    return out


def detect_spikes(
    bars: Bars,
    spike_multiplier: float = 5.0,
    min_spike_points: float = 0.0,
    baseline_window: int = 300,
) -> np.ndarray:
    """Boolean mask of upward spike bars, using an adaptive threshold.

    The upward move is high minus previous close. That catches a spike inside a
    candle and a jump between two ticks.
    """
    prev_close = _previous_close(bars)
    up_move = bars.high - prev_close
    abs_moves = np.abs(bars.close - prev_close)
    baseline = _trailing_baseline(abs_moves, baseline_window)

    with np.errstate(invalid="ignore"):
        threshold = np.maximum(min_spike_points, spike_multiplier * baseline)
        return (up_move > threshold) & (up_move > 0)


def grind_atr(bars: Bars, spikes: np.ndarray, period: int) -> np.ndarray:
    """Average true range with spike bars left out, so it measures the grind only.

    All pullback and EMA distances are expressed in this unit, which lets the same
    settings work at any price level or timeframe.
    """
    prev_close = _previous_close(bars)
    true_range = np.maximum.reduce(
        [bars.high - bars.low, np.abs(bars.high - prev_close), np.abs(bars.low - prev_close)]
    )
    valid = ~spikes
    sum_tr = np.concatenate(([0.0], np.cumsum(np.where(valid, true_range, 0.0))))
    count = np.concatenate(([0], np.cumsum(valid)))

    out = np.full(len(bars), np.nan)
    if len(bars) >= period:
        window_sum = sum_tr[period:] - sum_tr[:-period]
        window_count = count[period:] - count[:-period]
        with np.errstate(divide="ignore", invalid="ignore"):
            atr = window_sum / window_count
        atr[(window_count == 0) | (atr <= 0)] = np.nan
        out[period - 1 :] = atr
    return out


def swing_highs(high: np.ndarray, lookback: int) -> np.ndarray:
    """Highest high over the last `lookback` bars, including the current bar."""
    out = np.maximum.accumulate(high)
    if len(high) >= lookback:
        out[lookback - 1 :] = sliding_window_view(high, lookback).max(axis=1)
    return out


def pullback_depth(bars: Bars, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (swing high, drop from swing high to the close in points)."""
    swing = swing_highs(bars.high, lookback)
    return swing, np.maximum(swing - bars.close, 0.0)


def bars_since_spike(spikes: np.ndarray) -> np.ndarray:
    """Bars since the latest spike: 0 on a spike bar, NaN before the first spike."""
    index = np.arange(len(spikes))
    last_spike = np.maximum.accumulate(np.where(spikes, index, -1))
    return np.where(last_spike >= 0, index - last_spike, np.nan).astype(float)


def cycle_percentile(spikes: np.ndarray, min_cycles: int = MIN_CYCLES) -> np.ndarray:
    """Where the current cycle age ranks among cycles already completed (0 to 1).

    0.80 means the current wait is longer than 80% of past completed cycles. It is
    NaN until min_cycles have finished. A high value says the wait is long. It does
    NOT by itself say a spike is due: see cycle_hazard().
    """
    out = np.full(len(spikes), np.nan)
    spike_index = np.flatnonzero(spikes)
    intervals = np.diff(spike_index)
    for k, start in enumerate(spike_index):
        if k < min_cycles:
            continue
        end = spike_index[k + 1] if k + 1 < len(spike_index) else len(spikes)
        past = np.sort(intervals[:k])
        ages = np.arange(end - start)
        out[start:end] = np.searchsorted(past, ages, side="right") / len(past)
    return out


def grind_ratio(close: np.ndarray, window: int) -> np.ndarray:
    """Share of the last `window` bars that closed lower. NaN during warm-up."""
    out = np.full(len(close), np.nan)
    if len(close) > window:
        down = (np.diff(close) < 0).astype(float)
        out[window:] = sliding_window_view(down, window).mean(axis=1)
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _ramp(value: np.ndarray, start: float, target: float) -> np.ndarray:
    """Linear 0-to-1 ramp between start and target. NaN scores 0."""
    span = max(target - start, 1e-12)
    return np.nan_to_num(np.clip((value - start) / span, 0.0, 1.0), nan=0.0)


@dataclass(frozen=True)
class Analysis:
    """Every computed series, one value per bar, plus the settings used."""

    params: Params
    close: np.ndarray
    ema: np.ndarray
    atr: np.ndarray
    spikes: np.ndarray
    age: np.ndarray
    swing_high: np.ndarray
    pullback_points: np.ndarray
    pullback_atr: np.ndarray
    ema_gap_atr: np.ndarray
    percentile: np.ndarray
    grind: np.ndarray
    score_ema: np.ndarray
    score_pullback: np.ndarray
    score_cycle: np.ndarray
    score_grind: np.ndarray
    score: np.ndarray
    verdict: np.ndarray
    ready: np.ndarray  # True once all warm-ups have finished


def analyze(bars: Bars, params: Params = Params()) -> Analysis:
    """Run the full pipeline over every bar."""
    p = params
    spikes = detect_spikes(bars, p.spike_multiplier, p.min_spike_points, p.baseline_window)
    ema_line = ema(bars.close, p.ema_period)
    atr = grind_atr(bars, spikes, p.atr_period)
    swing, pull_pts = pullback_depth(bars, p.swing_lookback)
    age = bars_since_spike(spikes)
    percentile = cycle_percentile(spikes)
    grind = grind_ratio(bars.close, p.grind_window)

    with np.errstate(divide="ignore", invalid="ignore"):
        pull_atr = pull_pts / atr
        gap_atr = (ema_line - bars.close) / atr  # positive means price is below the EMA

    score_ema = _ramp(gap_atr, 0.0, p.ema_gap_target_atr)
    score_pull = _ramp(pull_atr, 0.0, p.pullback_target_atr)
    score_cycle = _ramp(percentile, 0.0, p.cycle_percentile_target)
    # A ratio of 0.5 is a coin flip, so a grind only starts to count above that.
    score_grind = _ramp(grind, 0.5, p.grind_ratio_target)

    weights = np.array([p.weight_ema, p.weight_pullback, p.weight_cycle, p.weight_grind], float)
    total_weight = weights.sum()
    parts = np.vstack([score_ema, score_pull, score_cycle, score_grind])
    if total_weight > 0:
        score = 100.0 * (weights @ parts) / total_weight
    else:
        score = np.zeros(len(bars))

    ready = np.isfinite(ema_line) & np.isfinite(atr) & np.isfinite(percentile) & np.isfinite(grind)
    score = np.where(ready, score, 0.0)

    verdict = np.full(len(bars), WAIT, dtype=int)
    verdict[ready & (score >= p.watch_threshold)] = WATCH
    verdict[ready & (score >= p.high_threshold)] = HIGH

    return Analysis(
        params=p,
        close=bars.close,
        ema=ema_line,
        atr=atr,
        spikes=spikes,
        age=age,
        swing_high=swing,
        pullback_points=pull_pts,
        pullback_atr=pull_atr,
        ema_gap_atr=gap_atr,
        percentile=percentile,
        grind=grind,
        score_ema=score_ema,
        score_pullback=score_pull,
        score_cycle=score_cycle,
        score_grind=score_grind,
        score=score,
        verdict=verdict,
        ready=ready,
    )


def _clean(value) -> Optional[float]:
    """Turn a NumPy number into a JSON-safe float, or None when it is NaN."""
    value = float(value)
    return value if np.isfinite(value) else None


@dataclass(frozen=True)
class ChecklistItem:
    label: str
    passed: bool
    detail: str
    score: float  # 0 to 1


def latest_snapshot(a: Analysis) -> dict:
    """Summarise the newest bar as a plain dictionary for the UI or a mobile client."""
    i = len(a.close) - 1
    p = a.params

    def show(value: float, fmt: str = ".2f") -> str:
        return "n/a" if not np.isfinite(value) else format(value, fmt)

    age_text = "no spike yet" if not np.isfinite(a.age[i]) else f"{int(a.age[i])} bars"
    checklist = [
        ChecklistItem(
            f"Price at least {p.ema_gap_target_atr:g} ATR below the {p.ema_period} EMA",
            bool(a.score_ema[i] >= 1.0),
            (
                f"{show(abs(a.ema_gap_atr[i]))} ATR "
                f"{'above' if a.ema_gap_atr[i] < 0 else 'below'}"
            ),
            float(a.score_ema[i]),
        ),
        ChecklistItem(
            f"Pullback at least {p.pullback_target_atr:g} ATR from the swing high",
            bool(a.score_pullback[i] >= 1.0),
            f"{show(a.pullback_atr[i])} ATR ({show(a.pullback_points[i])} pts)",
            float(a.score_pullback[i]),
        ),
        ChecklistItem(
            f"Cycle older than {p.cycle_percentile_target:.0%} of past cycles",
            bool(a.score_cycle[i] >= 1.0),
            f"{age_text}, percentile {show(a.percentile[i] * 100, '.0f')}",
            float(a.score_cycle[i]),
        ),
        ChecklistItem(
            f"At least {p.grind_ratio_target:.0%} down bars in the last {p.grind_window}",
            bool(a.score_grind[i] >= 1.0),
            f"{show(a.grind[i] * 100, '.0f')}% down",
            float(a.score_grind[i]),
        ),
    ]

    code = int(a.verdict[i])
    return {
        "verdict": VERDICT_LABELS[code],
        "verdict_code": code,
        "score": round(float(a.score[i]), 1),
        "ready": bool(a.ready[i]),
        "price": _clean(a.close[i]),
        "ema": _clean(a.ema[i]),
        "atr": _clean(a.atr[i]),
        "swing_high": _clean(a.swing_high[i]),
        "pullback_points": _clean(a.pullback_points[i]),
        "pullback_atr": _clean(a.pullback_atr[i]),
        "ema_gap_atr": _clean(a.ema_gap_atr[i]),
        "bars_since_spike": _clean(a.age[i]),
        "cycle_percentile": _clean(a.percentile[i]),
        "grind_ratio": _clean(a.grind[i]),
        "spike_count": int(a.spikes.sum()),
        "checklist": [asdict(item) for item in checklist],
    }


# ---------------------------------------------------------------------------
# Edge test: does the score actually predict anything?
# ---------------------------------------------------------------------------


def wilson_interval(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a hit rate."""
    if n == 0:
        return 0.0, 0.0
    rate = hits / n
    denom = 1.0 + z * z / n
    centre = (rate + z * z / (2 * n)) / denom
    margin = z * np.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _separate_entries(index: np.ndarray, gap: int) -> np.ndarray:
    """Keep bars at least `gap` apart, so one long signal counts as one trade."""
    kept, next_allowed = [], -1
    for i in index:
        if i >= next_allowed:
            kept.append(i)
            next_allowed = i + gap
    return np.array(kept, dtype=int)


def edge_report(bars: Bars, a: Analysis, horizon: int = 10) -> dict:
    """Compare each verdict tier against simply buying at a random bar.

    For each entry bar it asks two things over the next `horizon` bars: did a spike
    arrive, and what did price do? Entries are spaced `horizon` bars apart so that
    overlapping windows do not inflate the sample.
    """
    n = len(a.close)
    if n <= horizon + 1:
        return {"horizon": horizon, "baseline_rate": None, "tiers": [], "conclusion": "Not enough data."}

    # spike_ahead[t] is True if any bar in t+1 ... t+horizon is a spike.
    spike_count = np.concatenate(([0], np.cumsum(a.spikes)))
    usable = np.arange(n - horizon)
    spike_ahead = spike_count[usable + horizon + 1] - spike_count[usable + 1] > 0
    forward = a.close[usable + horizon] - a.close[usable]

    ready = a.ready[usable]
    if not ready.any():
        return {"horizon": horizon, "baseline_rate": None, "tiers": [], "conclusion": "Not enough data."}
    baseline_rate = float(spike_ahead[ready].mean())

    tiers = []
    for code in (HIGH, WATCH, WAIT):
        signal = np.flatnonzero(ready & (a.verdict[usable] == code))
        entries = _separate_entries(signal, horizon)
        hits = int(spike_ahead[entries].sum()) if len(entries) else 0
        low, high = wilson_interval(hits, len(entries))
        tiers.append(
            {
                "verdict": VERDICT_LABELS[code],
                "entries": int(len(entries)),
                "spike_rate": hits / len(entries) if len(entries) else None,
                "ci_low": low,
                "ci_high": high,
                "lift": (hits / len(entries)) / baseline_rate if len(entries) and baseline_rate else None,
                "avg_forward_points": float(forward[entries].mean()) if len(entries) else None,
            }
        )

    best = tiers[0]
    if best["entries"] < 30:
        conclusion = f"Only {best['entries']} High Probability entries, so too few to judge. Load more data."
    elif best["ci_low"] > baseline_rate:
        conclusion = "High Probability entries beat the baseline spike rate on this data. Re-test on data you did not tune on."
    else:
        conclusion = "No edge shown: High Probability entries did not clearly beat the baseline spike rate."

    return {"horizon": horizon, "baseline_rate": baseline_rate, "tiers": tiers, "conclusion": conclusion}


def cycle_hazard(a: Analysis, bucket: int = 5, max_age: int = 60) -> dict:
    """Chance of a spike on the next bar, grouped by current cycle age.

    This is the direct test of "a spike is due". A flat line means a long wait does
    not raise the odds. A rising line would mean spikes really do become more likely
    the longer the wait.
    """
    n = len(a.close)
    ages = a.age[:-1]
    next_spike = a.spikes[1:]
    known = np.isfinite(ages)

    starts = np.arange(0, max_age, bucket)
    rates, counts = [], []
    for start in starts:
        in_bucket = known & (ages >= start) & (ages < start + bucket)
        count = int(in_bucket.sum())
        counts.append(count)
        rates.append(float(next_spike[in_bucket].mean()) if count else np.nan)

    overall = float(next_spike[known].mean()) if n > 1 and known.any() else np.nan
    return {
        "age_start": starts.tolist(),
        "bucket": bucket,
        "rate": rates,
        "count": counts,
        "overall_rate": overall,
    }
