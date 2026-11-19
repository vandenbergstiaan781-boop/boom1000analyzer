"""Boom 1000 Analyzer: local Streamlit interface. Run with: streamlit run app.py

This file holds the UI, the live tick counter and the master confluence engine. Bar-level
maths lives in math_engine.py and data loading lives in data_feed.py.
"""

from __future__ import annotations

import html
from pathlib import Path
from dataclasses import replace

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import data_feed
import math_engine as me

# Chart colours (dark theme, set in .streamlit/config.toml).
# Brand palette: deep black, metallic slate cards, silver-white text, neon green, sharp red.
BG, CARD, BORDER, TEXT = "#080B09", "#111714", "#2A332E", "#F0F4F2"
NEON, RED = "#00FF41", "#FF2A2A"
# Chart roles mapped to the brand (names kept so the chart code is unchanged).
BLUE, ORANGE, AQUA = NEON, TEXT, NEON      # up candles / score line, EMA line, spike markers
INK, MUTED, GRID, AXIS = "#C9D1CD", "#8A958F", "#1C2420", BORDER
SURFACE = CARD

# Verdict display: an icon and label always go with the colour.
VERDICT_STYLE = {
    me.HIGH: ("✅", NEON),
    me.WATCH: ("👀", "#fab219"),
    me.WAIT: ("⏳", MUTED),
}

# Logo: the full SpikeVision poster.
LOGO = Path(__file__).with_name("assets") / "logo.jpg"

# Compact dashboard styling: tighter spacing, smaller metric text, card headers.
DASHBOARD_CSS = f"""
<style>
/* Layout spacing */
.block-container {{padding-top: 2.6rem; padding-bottom: 1rem;}}
.dash-head {{display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; margin-bottom: 0.2rem;}}
.dash-head h1 {{font-size: 1.6rem; margin: 0; padding: 0; color: {TEXT}; letter-spacing: 0.01em;}}
.dash-pill {{display: inline-block; font-size: 0.8rem; font-weight: 600; padding: 3px 10px;
            border-radius: 999px; margin-left: 10px; vertical-align: middle;}}
.dash-badge {{font-size: 0.78rem; color: {MUTED}; border: 1px solid {BORDER}; border-radius: 999px;
             padding: 2px 10px; background: {CARD};}}

/* Page and sidebar */
.stApp, [data-testid="stHeader"] {{background: {BG};}}
[data-testid="stSidebar"] {{background: linear-gradient(180deg, #0D120F 0%, {BG} 100%);
                           border-right: 1px solid {BORDER};}}

/* Cards: metallic slate with a faint top sheen */
[class*="st-key-card-"] {{
    background: linear-gradient(180deg, #151C18 0%, {CARD} 60%);
    border: 1px solid {BORDER} !important;
    border-radius: 10px;
    box-shadow: inset 0 1px 0 rgba(240, 244, 242, 0.04), 0 2px 12px rgba(0, 0, 0, 0.45);
}}

/* Metrics */
[data-testid="stMetricValue"] {{font-size: 1.3rem; color: {TEXT}; font-variant-numeric: tabular-nums;}}
[data-testid="stMetricLabel"] p {{font-size: 0.78rem; color: {MUTED}; letter-spacing: 0.02em;}}

/* Progress bars: neon green fill with a soft glow on a dark track */
[data-testid="stProgressBarTrack"] {{background: #1A221E; border-radius: 999px; height: 8px;}}
[data-testid="stProgressBarTrack"] > div {{background: {NEON}; border-radius: 999px;
                                          box-shadow: 0 0 8px rgba(0, 255, 65, 0.55);}}
[data-testid="stProgress"] p {{color: {TEXT};}}

/* Alerts: dark fill, coloured edge */
[data-testid="stAlertContainer"] {{border-radius: 8px; border: 1px solid {BORDER};}}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) {{
    background: rgba(255, 42, 42, 0.10); border-color: rgba(255, 42, 42, 0.55);
    border-left: 4px solid {RED}; box-shadow: 0 0 14px rgba(255, 42, 42, 0.18);
}}
[data-testid="stAlertContentError"], [data-testid="stAlertContentError"] p {{color: #FF6B6B;}}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) {{
    background: rgba(0, 255, 65, 0.08); border-color: rgba(0, 255, 65, 0.45);
    border-left: 4px solid {NEON}; box-shadow: 0 0 14px rgba(0, 255, 65, 0.15);
}}
[data-testid="stAlertContentSuccess"], [data-testid="stAlertContentSuccess"] p {{color: {NEON};}}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) {{
    background: rgba(250, 178, 25, 0.08); border-left: 4px solid #fab219;
}}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) {{
    background: rgba(240, 244, 242, 0.04); border-left: 4px solid {MUTED};
}}
[data-testid="stAlertContentInfo"], [data-testid="stAlertContentInfo"] p {{color: {INK};}}

/* Expanders and tabs */
[data-testid="stExpander"] details {{background: {BG}; border: 1px solid {BORDER}; border-radius: 8px;}}
[data-testid="stTab"][aria-selected="true"] p {{color: {NEON}; text-shadow: 0 0 8px rgba(0, 255, 65, 0.4);}}
</style>
"""

TIMEFRAMES = {"Ticks": 0, "M1": 60, "M5": 300, "M15": 900}
SOURCES = ("Simulated", "Deriv live (no key)", "CSV upload")

# Live tick counter.
SPIKE_THRESHOLD_DEFAULT = 10.0  # an upward tick bigger than this many units is a spike
SIM_SPIKE_SIZE = 25.0           # simulator only: typical spike, kept above the default threshold
SIM_GRIND_PER_TICK = 0.0271     # simulator only: cancels the average spike, so price has no trend
SECONDS_PER_TICK = 1            # 1 tick = 1 second on Boom 1000
TICK_SEED_COUNT = 5000          # ticks loaded on start, to find the last real spike
TICK_POLL_COUNT = 300           # ticks loaded on each refresh after that

# Master confluence engine: weights and the value at which each factor maxes out.
WEIGHT_CYCLE, WEIGHT_PULLBACK, WEIGHT_EMA, WEIGHT_DENSITY = 0.35, 0.30, 0.20, 0.15
CYCLE_BASELINE_TICKS = 1000.0
PULLBACK_EXTREME_ATR = 25.0
EMA_EXTREME_PCT_DEFAULT = 6.0
DENSITY_WINDOW_TICKS = 10
PROBABILITY_FLOOR, PROBABILITY_CEILING = 5.0, 95.0
ACTION_ZONE, BUILDING_ZONE = 85.0, 60.0

# Extended down-trend detector: looks back over past bars, not just the latest one.
DOWNTREND_LOOKBACK_DEFAULT = 40     # bars inspected
DOWNTREND_MIN_CYCLE_DEFAULT = 1000  # ticks since the last spike that count as a long wait
DOWNTREND_BELOW_EMA_SHARE = 0.80    # share of window closes below the 20 EMA
DOWNTREND_DOWN_SHARE = 0.60         # share of window bars that closed lower
DOWNTREND_LOWER_LOW_SHARE = 0.50    # share of window bars that made a lower low
DOWNTREND_MIN_STREAK = 5            # longest run of consecutive down closes

# Down-trend exhaustion score: max points per factor and its zones.
EXHAUSTION_POINTS = {"deceleration": 35.0, "shrinking": 30.0, "convergence": 20.0, "stretch": 15.0}
EXHAUSTION_FULL_SHRINK = 0.50       # candles 50% smaller than earlier in the window = full marks
EXHAUSTION_ZONE, SLOWING_ZONE = 75.0, 50.0

# Score context tags (master score ranges, in %).
TRAP_RANGE = (80.0, 100.0)          # with an active down-trend
CLEAN_RANGE = (60.0, 68.0)          # with no down-trend

# Unified Spike Probability: weights (out of 100) and the down-trend discount.
UNIFIED_WEIGHTS = {"ema": 30.0, "pullback": 30.0, "cycle": 25.0, "exhaustion": 15.0}
UNIFIED_MIN_MULTIPLIER = 0.50       # active down-trend with 0% exhaustion halves the score
UNIFIED_STRONG, UNIFIED_BUILDING = 75.0, 50.0


# ---------------------------------------------------------------------------
# Tick counter and master engine (plain functions, no UI, easy to test)
# ---------------------------------------------------------------------------


def process_ticks(prices, threshold, ticks_since_spike=0, last_price=None) -> dict:
    """Run a batch of new ticks through the spike counter.

    An upward move above `threshold` is a spike and resets the counter to 0. Every
    other tick adds 1, including flat ticks and small upticks, so that the count
    stays equal to the seconds elapsed. Pass the previous batch's last price so a
    spike that lands between two batches is still seen.
    """
    prices = np.asarray(prices, dtype=float)
    n = len(prices)
    if n == 0:
        return {"ticks_since_spike": ticks_since_spike, "processed": 0, "spikes": 0, "largest_small_up": 0.0}

    previous = prices[0] if last_price is None else last_price
    moves = np.diff(prices, prepend=previous)
    is_spike = moves > threshold
    spike_index = np.flatnonzero(is_spike)
    since = n - 1 - int(spike_index[-1]) if len(spike_index) else ticks_since_spike + n
    return {
        "ticks_since_spike": since,
        "processed": n,
        "spikes": len(spike_index),
        "largest_small_up": float(np.maximum(moves[~is_spike], 0.0).max(initial=0.0)),
    }


def down_density_pct(prices, window: int = DENSITY_WINDOW_TICKS) -> float:
    """Share of the last `window` ticks that moved down, as a percentage."""
    prices = np.asarray(prices, dtype=float)
    if len(prices) <= window:
        return float("nan")
    return float((np.diff(prices[-(window + 1) :]) < 0).mean() * 100.0)


def master_probability(
    ticks_since_spike,
    pullback_atr,
    ema_deviation_pct,
    down_density,
    ema_extreme_pct: float = EMA_EXTREME_PCT_DEFAULT,
) -> dict:
    """Weighted confluence of four factors. Accepts single numbers or NumPy arrays.

    Each factor is scaled to 0-1 at its extreme. Missing values count as 0.
    `tension` is the raw 0-100 result and `probability` is the same value clamped to
    5-95. The result is a weighted score: it is not a measured chance of a spike.
    """

    def scaled(value, extreme):
        return np.clip(np.nan_to_num(np.asarray(value, dtype=float), nan=0.0) / extreme, 0.0, 1.0)

    cycle = scaled(ticks_since_spike, CYCLE_BASELINE_TICKS)
    pullback = scaled(pullback_atr, PULLBACK_EXTREME_ATR)
    ema_part = scaled(ema_deviation_pct, ema_extreme_pct)
    density = scaled(down_density, 100.0)

    tension = 100.0 * (
        WEIGHT_CYCLE * cycle + WEIGHT_PULLBACK * pullback + WEIGHT_EMA * ema_part + WEIGHT_DENSITY * density
    )
    return {
        "probability": np.clip(tension, PROBABILITY_FLOOR, PROBABILITY_CEILING),
        "tension": tension,
        "points": {
            "cycle": 100.0 * WEIGHT_CYCLE * cycle,
            "pullback": 100.0 * WEIGHT_PULLBACK * pullback,
            "ema": 100.0 * WEIGHT_EMA * ema_part,
            "density": 100.0 * WEIGHT_DENSITY * density,
        },
    }


def longest_down_streak(close: np.ndarray) -> int:
    """Longest run of consecutive lower closes."""
    best = run = 0
    for falling in np.diff(close) < 0:
        run = run + 1 if falling else 0
        best = max(best, run)
    return best


def extended_downtrend(
    close,
    low,
    ema_line,
    spikes,
    cycle_ticks: float,
    lookback: int = DOWNTREND_LOOKBACK_DEFAULT,
    min_cycle_ticks: float = DOWNTREND_MIN_CYCLE_DEFAULT,
) -> dict:
    """Check the last `lookback` bars for a sustained, spike-free bleed.

    Detected when ALL of these hold:
      - the cycle is long (cycle_ticks >= min_cycle_ticks)
      - no spike inside the window, so there was no valid reset
      - price ended the window lower than it started
      - at least 80% of window closes sat below the 20 EMA
    AND at least 2 of these 3 trend patterns show:
      - 60%+ of bars closed lower
      - 50%+ of bars made a lower low than the bar before
      - a run of 5+ consecutive down closes
    """
    close, low = np.asarray(close, float), np.asarray(low, float)
    ema_line, spikes = np.asarray(ema_line, float), np.asarray(spikes, bool)
    result = {"detected": False, "ready": False, "lookback": lookback, "checks": []}
    if len(close) < lookback + 1:
        return result

    window = slice(len(close) - lookback, None)
    c, l, e = close[window], low[window], ema_line[window]
    if not np.isfinite(e).all():
        return result  # EMA still warming up
    result["ready"] = True

    below_ema = float((c < e).mean())
    down_share = float((np.diff(close[-(lookback + 1):]) < 0).mean())
    lower_lows = float((np.diff(low[-(lookback + 1):]) < 0).mean())
    streak = longest_down_streak(close[-(lookback + 1):])
    net_move = float(c[-1] - close[-(lookback + 1)])
    long_cycle = bool(np.isfinite(cycle_ticks) and cycle_ticks >= min_cycle_ticks)
    no_spike = not bool(spikes[window].any())

    required = [
        ("Long cycle", long_cycle,
         "n/a" if not np.isfinite(cycle_ticks) else f"{cycle_ticks:,.0f} ticks", f">= {min_cycle_ticks:,.0f} ticks"),
        ("No spike reset in window", no_spike,
         f"{int(spikes[window].sum())} spikes", "0 spikes"),
        ("Net drop over window", net_move < 0, f"{net_move:+,.2f} pts", "below 0"),
        ("Closes below 20 EMA", below_ema >= DOWNTREND_BELOW_EMA_SHARE,
         f"{below_ema:.0%}", f">= {DOWNTREND_BELOW_EMA_SHARE:.0%}"),
    ]
    patterns = [
        ("Down closes", down_share >= DOWNTREND_DOWN_SHARE, f"{down_share:.0%}", f">= {DOWNTREND_DOWN_SHARE:.0%}"),
        ("Lower lows", lower_lows >= DOWNTREND_LOWER_LOW_SHARE, f"{lower_lows:.0%}", f">= {DOWNTREND_LOWER_LOW_SHARE:.0%}"),
        ("Down streak", streak >= DOWNTREND_MIN_STREAK, f"{streak} bars", f">= {DOWNTREND_MIN_STREAK} bars"),
    ]
    pattern_hits = sum(passed for _, passed, _, _ in patterns)
    result["detected"] = all(passed for _, passed, _, _ in required) and pattern_hits >= 2
    result["pattern_hits"] = pattern_hits
    result["checks"] = [
        {"Check": name, "Group": group, "Reading": reading, "Needs": needs, "Passed": "✅" if passed else "⬜"}
        for group, items in (("Required", required), ("2 of 3", patterns))
        for name, passed, reading, needs in items
    ]
    return result


def downtrend_exhaustion(close, high, low, ema_line, pullback_atr, lookback: int = DOWNTREND_LOOKBACK_DEFAULT) -> dict:
    """0-100 score for how worn out the recent down-trend looks, from past and current bars.

    The lookback window is split into an earlier and a later half and compared:
      - Grind deceleration (35): price falls more slowly in the later half.
      - Shrinking candles (30): later candles are smaller (full marks at 50% smaller).
      - EMA convergence (20): the gap below the 20 EMA has closed from its widest
        point in the window (full marks once price is back at or above the EMA).
      - Historical stretch (15): the current pullback ranks high against every
        earlier pullback in the loaded history.
    Uses only past and current data. It is a weighted score, not a measured chance.
    """
    close, high, low = np.asarray(close, float), np.asarray(high, float), np.asarray(low, float)
    ema_line, pullback_atr = np.asarray(ema_line, float), np.asarray(pullback_atr, float)
    empty = {"ready": False, "score": 0.0, "points": {k: 0.0 for k in EXHAUSTION_POINTS}, "readings": {}}
    half = lookback // 2
    if len(close) < lookback + 1 or half < 2:
        return empty

    window = slice(len(close) - lookback, None)
    if not np.isfinite(ema_line[window]).all():
        return empty

    # Candle size per bar: true range, so tick data (high == low) still has a size.
    prev_close = close[-(lookback + 1):-1]
    size = np.maximum(high[window] - low[window], np.abs(close[window] - prev_close))
    path = close[-(lookback + 1):]
    early_drop = (path[0] - path[half]) / half                 # points lost per bar, earlier half
    late_drop = (path[half] - path[-1]) / (lookback - half)    # points lost per bar, later half

    deceleration = float(np.clip(1.0 - late_drop / early_drop, 0.0, 1.0)) if early_drop > 0 else 0.0
    early_size, late_size = size[:half].mean(), size[half:].mean()
    shrink = float(1.0 - late_size / early_size) if early_size > 0 else 0.0
    shrinking = float(np.clip(shrink / EXHAUSTION_FULL_SHRINK, 0.0, 1.0))

    gap = ema_line[window] - close[window]                     # positive = below the EMA
    widest = gap.max()
    convergence = float(np.clip(1.0 - max(gap[-1], 0.0) / widest, 0.0, 1.0)) if widest > 0 else 0.0

    current, past = pullback_atr[-1], pullback_atr[:-1]
    past = past[np.isfinite(past)]
    stretch = float((past <= current).mean()) if np.isfinite(current) and len(past) else 0.0

    parts = {"deceleration": deceleration, "shrinking": shrinking, "convergence": convergence, "stretch": stretch}
    points = {k: EXHAUSTION_POINTS[k] * v for k, v in parts.items()}
    return {
        "ready": True,
        "score": float(sum(points.values())),
        "points": points,
        "readings": {
            "deceleration": f"{early_drop:,.3f} -> {late_drop:,.3f} pts/bar",
            "shrinking": f"{shrink:+.0%} smaller",
            "convergence": f"gap {max(gap[-1], 0.0):,.2f} of widest {widest:,.2f} pts",
            "stretch": f"{stretch:.0%} of past pullbacks",
        },
    }


def exhaustion_status(score: float) -> str:
    """Status label for the exhaustion bar."""
    if score >= EXHAUSTION_ZONE:
        return "Exhaustion Zone — Down-Trend Likely Ending"
    if score >= SLOWING_ZONE:
        return "Momentum Slowing — Watch for Stabilization"
    return "Trend Still Strong — Do Not Buy"


def score_context_tag(downtrend_detected: bool, probability: float) -> str | None:
    """'trap', 'clean' or None, from the down-trend flag and the master score."""
    if downtrend_detected and TRAP_RANGE[0] <= probability <= TRAP_RANGE[1]:
        return "trap"
    if not downtrend_detected and CLEAN_RANGE[0] <= probability <= CLEAN_RANGE[1]:
        return "clean"
    return None


def unified_spike_probability(
    ema_gap_atr,
    pullback_atr,
    cycle_ticks,
    exhaustion_score: float,
    downtrend_detected: bool,
    ema_target_atr: float,
    pullback_target_atr: float,
) -> dict:
    """One 0-100 reading from the existing indicators.

    Base score (each factor scaled 0-1 and capped at 1, missing values count as 0):
      - EMA distance (30):  ATRs below the 20 EMA / the bar score EMA gap target (8 ATR)
      - Pullback (30):      ATRs below the swing high / the bar score pullback target (25 ATR)
      - Cycle age (25):     ticks since the last spike / 1,000 ticks
      - Exhaustion (15):    Down-Trend Exhaustion score / 100
    Down-trend discount: when the extended down-trend warning is active, the base score is
    multiplied by 0.50 at 0% exhaustion, rising in a straight line to 1.00 (no discount)
    once exhaustion reaches 75%. With no active down-trend, the multiplier is 1.00.
    It is a weighted score, not a measured chance.
    """

    def scaled(value, target):
        value = float(value) if value is not None else float("nan")
        return float(np.clip(np.nan_to_num(value / target, nan=0.0), 0.0, 1.0)) if target > 0 else 0.0

    exhaustion_score = float(np.nan_to_num(exhaustion_score, nan=0.0))
    parts = {
        "ema": scaled(ema_gap_atr, ema_target_atr),
        "pullback": scaled(pullback_atr, pullback_target_atr),
        "cycle": scaled(cycle_ticks, CYCLE_BASELINE_TICKS),
        "exhaustion": float(np.clip(exhaustion_score / 100.0, 0.0, 1.0)),
    }
    points = {k: UNIFIED_WEIGHTS[k] * v for k, v in parts.items()}
    base = sum(points.values())

    if downtrend_detected:
        eased = float(np.clip(exhaustion_score / EXHAUSTION_ZONE, 0.0, 1.0))
        multiplier = UNIFIED_MIN_MULTIPLIER + (1.0 - UNIFIED_MIN_MULTIPLIER) * eased
    else:
        multiplier = 1.0

    return {
        "score": float(np.clip(base * multiplier, 0.0, 100.0)),
        "base": float(base),
        "multiplier": float(multiplier),
        "points": points,
    }


def unified_status(score: float) -> str:
    """Status label for the unified bar."""
    if score >= UNIFIED_STRONG:
        return "Strong Confluence — Indicators Aligned"
    if score >= UNIFIED_BUILDING:
        return "Building — Partial Alignment"
    return "Weak — Wait"


def alert_level(probability: float) -> str:
    """'action' at 85+, 'building' from 60 to under 85, otherwise 'waiting'."""
    if probability >= ACTION_ZONE:
        return "action"
    return "building" if probability >= BUILDING_ZONE else "waiting"


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

TICK_STATE_DEFAULTS = {
    "ticks_since_spike": 0,
    "total_ticks_recorded": 0,
    "spikes_recorded": 0,
    "spike_seen": False,        # False means the cycle started before our data did
    "largest_small_up": 0.0,    # biggest upward tick that was NOT counted as a spike
    "last_tick_price": None,
    "last_tick_epoch": None,
    "recent_ticks": None,
    "tick_source_key": None,
}


def init_tick_state() -> None:
    """Create any missing tick variables. Existing values survive every rerun."""
    for name, value in TICK_STATE_DEFAULTS.items():
        if name not in st.session_state:
            st.session_state[name] = value


def reset_tick_state(source_key) -> None:
    for name, value in TICK_STATE_DEFAULTS.items():
        st.session_state[name] = value
    st.session_state.tick_source_key = source_key


def update_tick_state(prices, times, threshold: float, source_key) -> None:
    """Feed only ticks not processed yet into the counter held in session state."""
    state = st.session_state
    if state.tick_source_key != source_key:
        reset_tick_state(source_key)  # new source or threshold: rebuild from its history

    if state.last_tick_epoch is not None:
        fresh = times > state.last_tick_epoch
        prices, times = prices[fresh], times[fresh]
        if len(prices) == 0:
            return

    result = process_ticks(prices, threshold, state.ticks_since_spike, state.last_tick_price)
    state.ticks_since_spike = result["ticks_since_spike"]
    state.total_ticks_recorded += result["processed"]
    state.spikes_recorded += result["spikes"]
    state.spike_seen = state.spike_seen or result["spikes"] > 0
    state.largest_small_up = max(state.largest_small_up, result["largest_small_up"])
    state.last_tick_price = float(prices[-1])
    state.last_tick_epoch = float(times[-1])
    # Keep enough recent ticks to work out down-bar density on the next rerun.
    previous = state.recent_ticks if state.recent_ticks is not None else np.array([])
    state.recent_ticks = np.concatenate((previous, prices))[-(DENSITY_WINDOW_TICKS + 1) :]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def sidebar_controls() -> dict:
    """Draw all the controls and return the choices as a dictionary."""
    d = me.Params()  # defaults
    with st.sidebar:
        st.header("Data")
        source = st.radio("Source", SOURCES, label_visibility="collapsed")
        choices = {"source": source, "upload": None}

        if source == "CSV upload":
            choices["upload"] = st.file_uploader(
                "MT5 export or any OHLC / price CSV", type=["csv", "txt", "tsv"]
            )
            choices["timeframe"] = "M1"
        else:
            choices["timeframe"] = st.selectbox("Timeframe", list(TIMEFRAMES), index=1)
            choices["count"] = st.slider("History (bars)", 500, 5000, 5000, 100)

        if source == "Simulated":
            choices["seed"] = st.number_input("Random seed", 0, 99999, 42)
            choices["speed"] = st.slider("Ticks added per refresh", 1, 300, 30)
        if source != "CSV upload":
            choices["auto"] = st.toggle("Auto-refresh", value=source == "Deriv live (no key)")
            choices["refresh"] = st.slider("Refresh every (seconds)", 2, 60, 5)

        with st.expander("Master engine", expanded=True):
            choices["spike_threshold"] = st.slider(
                "Tick spike threshold (units)", 0.5, 100.0, SPIKE_THRESHOLD_DEFAULT, 0.5,
                help="An upward tick bigger than this resets the cycle counter.",
            )
            choices["ema_extreme_pct"] = st.slider(
                "EMA deviation for full marks (%)", 0.01, 10.0, EMA_EXTREME_PCT_DEFAULT, 0.01,
                help="How far below the EMA, in percent of price, earns the full 20% weight.",
            )

        with st.expander("Extended down-trend"):
            choices["downtrend_lookback"] = st.slider(
                "Lookback (bars)", 20, 100, DOWNTREND_LOOKBACK_DEFAULT, 5,
                help="How many past bars are checked for a sustained bleed.",
            )
            choices["downtrend_min_cycle"] = st.slider(
                "Long cycle from (ticks)", 300, 5000, DOWNTREND_MIN_CYCLE_DEFAULT, 100,
                help="Cycle age that counts as a long wait.",
            )

        st.header("Indicators")
        ema_period = st.slider("EMA period", 5, 100, d.ema_period)
        swing_lookback = st.slider("Swing high lookback (bars)", 5, 200, d.swing_lookback)
        grind_window = st.slider("Grind window (bars)", 3, 50, d.grind_window)

        with st.expander("Spike detection"):
            spike_multiplier = st.slider(
                "Spike size (× typical bar)", 2.0, 30.0, d.spike_multiplier, 0.5,
                help="A bar is a spike when its upward move beats this many typical bar moves.",
            )
            min_spike_points = st.slider("Minimum spike (points)", 0.0, 50.0, d.min_spike_points, 0.5)

        with st.expander("Score targets", expanded=True):
            gap_target = st.slider(
                "EMA gap target (ATR)", 0.5, 20.0, d.ema_gap_target_atr, 0.5,
                help="Full marks when price is this far below the EMA.",
            )
            pull_target = st.slider("Pullback target (ATR)", 1.0, 80.0, d.pullback_target_atr, 1.0)
            cycle_target = st.slider("Cycle percentile target", 0.1, 1.0, d.cycle_percentile_target, 0.05)
            grind_target = st.slider("Grind target (share of down bars)", 0.55, 1.0, d.grind_ratio_target, 0.05)

        with st.expander("Score weights"):
            w_ema = st.slider("EMA gap", 0, 100, int(d.weight_ema))
            w_pull = st.slider("Pullback", 0, 100, int(d.weight_pullback))
            w_cycle = st.slider("Cycle", 0, 100, int(d.weight_cycle))
            w_grind = st.slider("Grind", 0, 100, int(d.weight_grind))

        st.header("Verdict")
        watch, high = st.slider(
            "Watch / High cut-offs", 0, 100, (int(d.watch_threshold), int(d.high_threshold))
        )

        st.header("Display")
        choices["window"] = st.slider("Bars on chart", 50, 1000, 200, 50)
        choices["horizon"] = st.slider(
            "Edge test horizon (bars)", 1, 100, 10,
            help="How far ahead the edge test looks for a spike after each entry.",
        )

    choices["params"] = me.Params(
        ema_period=ema_period,
        swing_lookback=swing_lookback,
        grind_window=grind_window,
        spike_multiplier=spike_multiplier,
        min_spike_points=min_spike_points,
        ema_gap_target_atr=gap_target,
        pullback_target_atr=pull_target,
        cycle_percentile_target=cycle_target,
        grind_ratio_target=grind_target,
        weight_ema=w_ema,
        weight_pullback=w_pull,
        weight_cycle=w_cycle,
        weight_grind=w_grind,
        watch_threshold=watch,
        high_threshold=high,
    )
    return choices


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def simulated_bars(choices: dict) -> me.Bars:
    """Keep one simulator in the session, so its history survives each refresh."""
    granularity = TIMEFRAMES[choices["timeframe"]]
    ticks_per_bar = max(granularity, 1)
    key = (choices["seed"], ticks_per_bar)

    if st.session_state.get("sim_key") != key:
        feed = data_feed.SimulatedFeed(
            seed=choices["seed"], ticks_per_bar=ticks_per_bar, spike_size=SIM_SPIKE_SIZE,
            grind_per_tick=SIM_GRIND_PER_TICK,
        )
        feed.backfill(min(choices["count"] * ticks_per_bar, feed.MAX_TICKS - 1))
        st.session_state.sim_feed = feed
        st.session_state.sim_key = key
    else:
        st.session_state.sim_feed.advance(choices["speed"] if choices["auto"] else 0)

    bars = st.session_state.sim_feed.bars()
    return slice_bars(bars, len(bars) - choices["count"])


def slice_bars(bars: me.Bars, start: int) -> me.Bars:
    start = max(start, 0)
    return me.Bars(
        bars.open[start:],
        bars.high[start:],
        bars.low[start:],
        bars.close[start:],
        None if bars.time is None else bars.time[start:],
    )


@st.cache_data(ttl=2, show_spinner=False)
def cached_deriv(granularity: int, count: int) -> me.Bars:
    return data_feed.fetch_deriv("BOOM1000", granularity, count)


@st.cache_data(show_spinner=False)
def cached_csv(raw: bytes) -> me.Bars:
    return data_feed.load_csv(raw)


def live_ticks(source_key) -> tuple[np.ndarray, np.ndarray]:
    """Latest Deriv ticks: a long history on start, then a short poll on each refresh."""
    state = st.session_state
    seeded = state.tick_source_key == source_key and state.last_tick_epoch is not None
    ticks = cached_deriv(0, TICK_POLL_COUNT if seeded else TICK_SEED_COUNT)
    if seeded and ticks.time[0] > state.last_tick_epoch:
        # The poll doesn't reach back to our last tick (the PC slept, say), so ticks
        # were missed. Rebuild the counter from a full history instead of guessing.
        state.tick_source_key = None
        ticks = cached_deriv(0, TICK_SEED_COUNT)
    return ticks.close, ticks.time


def load_market(choices: dict):
    """Return (bars, description, ticks, tick source key). Errors are never replaced with fake data.

    `ticks` is (prices, times), or None when the source only has candles.
    """
    source = choices["source"]
    threshold = choices["spike_threshold"]

    if source == "Simulated":
        bars = simulated_bars(choices)
        feed = st.session_state.sim_feed
        ticks = (feed.prices, feed.first_time + np.arange(len(feed.prices), dtype=float))
        key = ("sim", st.session_state.sim_key, threshold)
        return bars, "SIMULATED data, not the real market", ticks, key

    if source == "CSV upload":
        upload = choices["upload"]
        if upload is None:
            st.info("Upload an MT5 export to analyze it. See the README for how to export one.")
            return None, "", None, None
        bars = cached_csv(upload.getvalue())
        text = f"CSV file: {upload.name}"
        if not np.array_equal(bars.high, bars.low):
            return bars, text, None, None  # candles: individual ticks are not available
        # Tick files are read once, in order. Row numbers stand in for time because
        # exports can repeat a timestamp.
        return bars, text, (bars.close, np.arange(len(bars), dtype=float)), ("csv", upload.name, upload.size, threshold)

    key = ("live", threshold)
    bars = cached_deriv(TIMEFRAMES[choices["timeframe"]], choices["count"])
    return bars, f"Live Deriv BOOM1000, {choices['timeframe']}", live_ticks(key), key


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


def bar_seconds(bars: me.Bars) -> float:
    """Typical seconds per bar, which is also ticks per bar at 1 tick per second."""
    if bars.time is None or len(bars) < 2:
        return float(SECONDS_PER_TICK)
    return max(float(np.median(np.diff(bars.time))), float(SECONDS_PER_TICK))


def cycle_reading(bars: me.Bars, a: me.Analysis, has_ticks: bool) -> dict:
    """Current cycle age and down density, from ticks if we have them, else candles."""
    state = st.session_state
    if has_ticks:
        recent = state.recent_ticks if state.recent_ticks is not None else np.array([])
        return {
            "ticks": state.ticks_since_spike,
            "known": state.spike_seen,
            "density": down_density_pct(recent),
            "estimated": False,
        }
    age = a.age[-1]
    return {
        "ticks": age * bar_seconds(bars) if np.isfinite(age) else float("nan"),
        "known": bool(np.isfinite(age)),
        "density": a.grind[-1] * 100.0,
        "estimated": True,
    }


def show_alert(probability: float) -> None:
    level = alert_level(probability)
    if level == "action":
        st.error(f"ACTION ZONE: master score {probability:.1f}% (85% or higher)", icon="🚨")
    elif level == "building":
        st.warning(f"BUILDING: master score {probability:.1f}% (60% to 84%)", icon="⚠️")
    else:
        st.info(f"WAITING: master score {probability:.1f}% (under 60%)", icon="ℹ️")


def show_downtrend(downtrend: dict) -> None:
    """Red warning when past bars confirm a prolonged grind."""
    if downtrend["detected"]:
        st.error(
            "**⚠️ EXTENDED DOWN-TREND DETECTED**\n\n"
            "Past and current data show a prolonged downward grind. "
            "Do not assume a spike is due just because the wait is long."
        )


def downtrend_checks(downtrend: dict) -> None:
    """Expander with the readings behind the down-trend warning."""
    if downtrend["checks"]:
        label = "Extended down-trend checks" + (" (detected)" if downtrend["detected"] else "")
        with st.expander(f"{label}: last {downtrend['lookback']} bars"):
            st.dataframe(downtrend["checks"], width="stretch", hide_index=True)
            st.caption("Detected when every Required check passes and at least 2 of the 3 trend patterns show.")


def show_context_tag(downtrend_detected: bool, probability: float) -> None:
    """Trap vs Clean Setup tag for the master score."""
    tag = score_context_tag(downtrend_detected, probability)
    if tag == "trap":
        st.error("⚠️ High-Risk Grind Trap — Avoid")
    elif tag == "clean":
        st.success("✅ Clean Setup Zone — No Heavy Down-Trend")


def big_number(kicker: str, value: str, pill: str = "", colour: str = MUTED) -> None:
    """Card header: small label, large value and an optional coloured status pill."""
    pill_html = (
        f'<span class="dash-pill" style="color:{colour};background:{colour}22;'
        f'border:1px solid {colour}66">{html.escape(pill)}</span>'
        if pill else ""
    )
    st.markdown(
        f'<div style="font-size:0.72rem;letter-spacing:0.08em;text-transform:uppercase;color:{MUTED}">'
        f'{html.escape(kicker)}</div>'
        f'<div style="font-size:2.4rem;font-weight:700;line-height:1.15;margin:2px 0 6px">'
        f'{html.escape(value)}{pill_html}</div>',
        unsafe_allow_html=True,
    )


def exhaustion_colour(score: float) -> str:
    if score >= EXHAUSTION_ZONE:
        return NEON
    return "#fab219" if score >= SLOWING_ZONE else RED


def exhaustion_card(exhaustion: dict, downtrend: dict) -> None:
    """Command center, right column: exhaustion bar, percentage and status."""
    with st.container(border=True, key="card-master"):
        if not exhaustion["ready"]:
            big_number("Down-Trend Exhaustion", "n/a")
            st.caption("Waiting for enough bars and a warm 20 EMA.")
            downtrend_checks(downtrend)
            return

        score = exhaustion["score"]
        icon = "🟢" if score >= EXHAUSTION_ZONE else ("🟡" if score >= SLOWING_ZONE else "🔴")
        big_number("Down-Trend Exhaustion", f"{score:.0f}%")
        st.progress(min(max(int(round(score)), 0), 100), text=f"{icon} {exhaustion_status(score)}")

        with st.expander("How the exhaustion score was built"):
            names = {
                "deceleration": "Grind deceleration",
                "shrinking": "Shrinking candles",
                "convergence": "EMA convergence",
                "stretch": "Historical stretch",
            }
            st.dataframe(
                [
                    {"Factor": names[k], "Reading": exhaustion["readings"][k],
                     "Points": f"{exhaustion['points'][k]:.1f} / {EXHAUSTION_POINTS[k]:.0f}"}
                    for k in EXHAUSTION_POINTS
                ],
                width="stretch",
                hide_index=True,
            )
            st.caption("Compares the earlier and later half of the down-trend lookback. Weighted score, not a measured chance.")
        downtrend_checks(downtrend)


def unified_card(unified: dict, readings: dict) -> None:
    """Full-width Unified Spike Probability bar under the command center."""
    score = unified["score"]
    icon = "🟢" if score >= UNIFIED_STRONG else ("🟡" if score >= UNIFIED_BUILDING else "🔴")
    with st.container(border=True, key="card-exhaustion"):
        number_col, bar_col = st.columns([1, 3], vertical_alignment="top")
        with number_col:
            big_number("Unified Spike Probability", f"{score:.0f}%")
        with bar_col:
            discount = (
                f" · down-trend discount ×{unified['multiplier']:.2f}" if unified["multiplier"] < 1.0 else ""
            )
            st.progress(min(max(int(round(score)), 0), 100),
                        text=f"{icon} {unified_status(score)}{discount}")
            with st.expander("How the unified score was built"):
                names = {"ema": "Below 20 EMA", "pullback": "Pullback from swing high",
                         "cycle": "Cycle age", "exhaustion": "Down-trend exhaustion"}
                rows = [
                    {"Factor": names[k], "Reading": readings[k],
                     "Points": f"{unified['points'][k]:.1f} / {UNIFIED_WEIGHTS[k]:.0f}"}
                    for k in UNIFIED_WEIGHTS
                ]
                rows.append({"Factor": "Down-trend discount", "Reading": "active" if unified["multiplier"] < 1.0 else "none",
                             "Points": f"×{unified['multiplier']:.2f} on {unified['base']:.1f}"})
                st.dataframe(rows, width="stretch", hide_index=True)
                st.caption("Weighted score, not a measured chance of a spike.")


def cycle_texts(cycle: dict) -> tuple[str, str]:
    """Cycle age as (ticks, seconds) text, with ~ for estimates and + for 'at least'."""
    if not np.isfinite(cycle["ticks"]):
        return "n/a", "n/a"
    suffix = "" if cycle["known"] else "+"  # no spike seen yet, so the true age is at least this
    prefix = "~" if cycle["estimated"] else ""
    return (
        f"{prefix}{cycle['ticks']:,.0f}{suffix}",
        f"{prefix}{cycle['ticks'] * SECONDS_PER_TICK:,.0f}{suffix} s",
    )


def master_card(cycle: dict, reading: dict, master: dict, downtrend: dict) -> None:
    """Command center, left column: master score, alert, context tag and cycle stats."""
    probability = float(master["probability"])
    tension = float(master["tension"])
    ticks_text, seconds_text = cycle_texts(cycle)

    with st.container(border=True, key="card-unified"):
        big_number("Master Spike Probability", f"{probability:.1f}%")
        show_alert(probability)
        show_context_tag(downtrend["detected"], probability)
        st.progress(min(max(int(round(tension)), 0), 100), text=f"Spring Tension Level: {tension:.0f}/100")

        cols = st.columns(2)
        cols[0].metric("Cycle Age (Ticks)", ticks_text)
        cols[1].metric("Cycle Age (Seconds)", seconds_text)

        state = st.session_state
        if cycle["estimated"]:
            st.caption("Candle CSV: cycle age and density are estimates. Upload a tick export for exact counts.")
        else:
            st.caption(
                f"{state.total_ticks_recorded:,} ticks · {state.spikes_recorded:,} spikes · "
                f"biggest uptick not counted as a spike: {state.largest_small_up:.2f}"
            )

        with st.expander("How this score was built"):
            points = master["points"]
            density_text = "n/a" if not np.isfinite(cycle["density"]) else f"{cycle['density']:.0f}%"
            st.dataframe(
                [
                    {"Factor": "Cycle age", "Reading": ticks_text, "Maxes out at": "1,000 ticks",
                     "Points": f"{float(points['cycle']):.1f} / 35"},
                    {"Factor": "ATR pullback", "Reading": reading["pullback"], "Maxes out at": "25 ATR",
                     "Points": f"{float(points['pullback']):.1f} / 30"},
                    {"Factor": "EMA deviation", "Reading": reading["ema"], "Maxes out at": reading["ema_extreme"],
                     "Points": f"{float(points['ema']):.1f} / 20"},
                    {"Factor": "Down-bar density", "Reading": density_text, "Maxes out at": "100%",
                     "Points": f"{float(points['density']):.1f} / 15"},
                ],
                width="stretch",
                hide_index=True,
            )
            st.caption("Weighted score, not a measured chance of a spike. The Edge test tab compares it to what actually happened.")


def verdict_card(snapshot: dict) -> None:
    """Compact bar score verdict for the stats strip."""
    icon, colour = VERDICT_STYLE[snapshot["verdict_code"]]
    note = "Score label, not a probability" if snapshot["ready"] else "Warming up: needs 5 spike cycles"
    st.markdown(
        f"""
        <div style="border-left:4px solid {colour};padding:2px 0 2px 12px">
          <div style="font-size:0.72rem;letter-spacing:0.08em;text-transform:uppercase;color:{MUTED}">Bar score</div>
          <div style="font-size:1.1rem;font-weight:700;line-height:1.35">{icon} {html.escape(snapshot["verdict"])}</div>
          <div style="font-size:0.85rem;color:{INK}">{snapshot["score"]:.0f}/100</div>
          <div style="font-size:0.75rem;color:{MUTED}">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def fmt(value, spec: str = ",.2f", suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:{spec}}{suffix}"


def stats_strip(s: dict) -> None:
    """Bar verdict plus the secondary metrics in one bordered row."""
    with st.container(border=True, key="card-stats"):
        cols = st.columns([2, 1, 1, 1, 1, 1])
        with cols[0]:
            verdict_card(s)
        cols[1].metric("Price", fmt(s["price"]))
        cols[2].metric("Below EMA", fmt(s["ema_gap_atr"], ".1f", " ATR"))
        cols[3].metric("Pullback", fmt(s["pullback_atr"], ".1f", " ATR"))
        cols[4].metric("Bars since spike", fmt(s["bars_since_spike"], ".0f"))
        percentile = None if s["cycle_percentile"] is None else s["cycle_percentile"] * 100
        cols[5].metric("Cycle percentile", fmt(percentile, ".0f"))


def checklist_panel(snapshot: dict) -> None:
    st.markdown("**Technical checklist**")
    for item in snapshot["checklist"]:
        icon = "✅" if item["passed"] else ("🟡" if item["score"] >= 0.5 else "⬜")
        st.progress(min(max(item["score"], 0.0), 1.0), text=f"{icon} {item['label']} · {item['detail']}")


def x_values(bars: me.Bars) -> np.ndarray:
    if bars.time is None:
        return np.arange(len(bars))
    return (bars.time * 1000).astype("datetime64[ms]")


def base_layout(fig: go.Figure, height: int, y_title: str) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=28, b=8),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(color=INK),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor=GRID, linecolor=AXIS, rangeslider=dict(visible=False)),
        yaxis=dict(gridcolor=GRID, linecolor=AXIS, title=y_title, side="right"),
    )
    return fig


def price_chart(bars: me.Bars, a: me.Analysis, window: int) -> go.Figure:
    """Candles with EMA, swing high and spike markers. One price axis only."""
    start = max(len(bars) - window, 0)
    x = x_values(bars)[start:]
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=x,
            open=bars.open[start:],
            high=bars.high[start:],
            low=bars.low[start:],
            close=bars.close[start:],
            name="Price",
            increasing=dict(line=dict(color=BLUE, width=1), fillcolor=BLUE),
            decreasing=dict(line=dict(color=MUTED, width=1), fillcolor=MUTED),
        )
    )
    fig.add_trace(go.Scatter(x=x, y=a.ema[start:], name=f"EMA {a.params.ema_period}",
                             line=dict(color=ORANGE, width=2)))
    fig.add_trace(go.Scatter(x=x, y=a.swing_high[start:], name="Swing high",
                             line=dict(color=INK, width=1, dash="dot", shape="hv")))
    spike = a.spikes[start:]
    fig.add_trace(
        go.Scatter(
            x=x[spike],
            y=bars.high[start:][spike],
            mode="markers",
            name="Spike",
            marker=dict(symbol="triangle-up", size=11, color=AQUA, line=dict(color=SURFACE, width=2)),
        )
    )
    return base_layout(fig, 380, "Price")


def score_chart(bars: me.Bars, a: me.Analysis, window: int) -> go.Figure:
    start = max(len(bars) - window, 0)
    x = x_values(bars)[start:]
    fig = go.Figure(go.Scatter(x=x, y=a.score[start:], name="Score", line=dict(color=BLUE, width=2)))
    for level, label in ((a.params.high_threshold, "High"), (a.params.watch_threshold, "Watch")):
        fig.add_hline(y=level, line=dict(color=MUTED, width=1, dash="dash"),
                      annotation_text=label, annotation_position="top left",
                      annotation_font_color=INK)
    base_layout(fig, 170, "Score")
    fig.update_yaxes(range=[0, 100])
    fig.update_layout(showlegend=False)
    return fig


def hazard_chart(hazard: dict) -> go.Figure:
    labels = [f"{s}-{s + hazard['bucket'] - 1}" for s in hazard["age_start"]]
    rates = [None if np.isnan(r) else r * 100 for r in hazard["rate"]]
    fig = go.Figure(
        go.Bar(x=labels, y=rates, name="Spike chance", marker_color=BLUE, customdata=hazard["count"],
               hovertemplate="Age %{x} bars<br>%{y:.1f}% next-bar spike<br>%{customdata} samples<extra></extra>")
    )
    if np.isfinite(hazard["overall_rate"]):
        fig.add_hline(y=hazard["overall_rate"] * 100, line=dict(color=ORANGE, width=2),
                      annotation_text="Average", annotation_position="top right",
                      annotation_font_color=INK)
    base_layout(fig, 300, "Next-bar spike chance (%)")
    fig.update_layout(barcornerradius=4, hovermode="closest", showlegend=False)
    fig.update_xaxes(title="Bars since last spike", type="category")
    return fig


def tier_table(report: dict, names: dict | None = None) -> None:
    rows = []
    for tier in report["tiers"]:
        has_data = tier["entries"] > 0
        rows.append(
            {
                "Verdict": (names or {}).get(tier["verdict"], tier["verdict"]),
                "Entries": tier["entries"],
                "Spike rate": f"{tier['spike_rate']:.1%}" if has_data else "n/a",
                "95% range": f"{tier['ci_low']:.1%} to {tier['ci_high']:.1%}" if has_data else "n/a",
                "Vs baseline": f"×{tier['lift']:.2f}" if tier["lift"] is not None else "n/a",
                "Avg move (pts)": f"{tier['avg_forward_points']:+.2f}" if has_data else "n/a",
            }
        )
    st.dataframe(rows, width="stretch", hide_index=True)


def master_edge_panel(bars: me.Bars, a: me.Analysis, horizon: int, ema_extreme_pct: float) -> None:
    """Replay the master formula over past bars and compare each zone with what happened."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ema_deviation = (a.ema - a.close) / a.ema * 100.0
    series = master_probability(
        a.age * bar_seconds(bars), a.pullback_atr, ema_deviation, a.grind * 100.0, ema_extreme_pct
    )["probability"]
    ready = np.isfinite(a.age) & np.isfinite(a.atr) & np.isfinite(a.ema) & np.isfinite(a.grind)
    zones = np.select([series >= ACTION_ZONE, series >= BUILDING_ZONE], [me.HIGH, me.WATCH], me.WAIT)

    report = me.edge_report(bars, replace(a, verdict=zones, ready=ready), horizon)
    st.subheader("Master Spike Probability vs what happened")
    if not report["tiers"]:
        st.info(report["conclusion"])
        return
    if report["tiers"][0]["entries"] == 0:
        st.markdown("**Action Zone never fired on this data. Check the EMA deviation slider under Master engine.**")
    else:
        st.markdown(f"**{report['conclusion'].replace('High Probability', 'Action Zone')}**")
    st.caption(
        f"Random-bar baseline: {report['baseline_rate']:.1%} chance of a spike within {horizon} bars. "
        "Replayed on candles, so cycle age and density are approximate. Costs are not included."
    )
    tier_table(
        report,
        {
            "High Probability Setup": "Action Zone (85%+)",
            "Watch Closely": "Building (60-84%)",
            "Wait": "Waiting (under 60%)",
        },
    )


def edge_panel(bars: me.Bars, a: me.Analysis, horizon: int, ema_extreme_pct: float) -> None:
    master_edge_panel(bars, a, horizon, ema_extreme_pct)

    st.subheader("Bar score verdict vs what happened")
    report = me.edge_report(bars, a, horizon)
    if not report["tiers"]:
        st.info(report["conclusion"])
        return

    st.markdown(f"**{report['conclusion']}**")
    st.caption(
        f"Random-bar baseline: {report['baseline_rate']:.1%} chance of a spike within {horizon} bars. "
        "Spread, slippage and commission are not included."
    )
    tier_table(report)

    st.subheader("Is a spike ever \"due\"?")
    st.caption(
        "A flat row of bars means a long wait does not raise the odds. "
        "A clear rise would support cycle timing on this data. Older ages have fewer samples, "
        "so expect those bars to jump around (hover to see the count)."
    )
    st.plotly_chart(hazard_chart(me.cycle_hazard(a)), width="stretch", key="hazard")


def data_table(bars: me.Bars, a: me.Analysis, rows: int = 50) -> None:
    start = max(len(bars) - rows, 0)
    index = np.arange(len(bars) - 1, start - 1, -1)
    time_values = x_values(bars)
    labels = {me.WAIT: "Wait", me.WATCH: "Watch", me.HIGH: "High"}
    table = {
        "Bar": time_values[index],
        "Close": np.round(bars.close[index], 2),
        "EMA": np.round(a.ema[index], 2),
        "Below EMA (ATR)": np.round(a.ema_gap_atr[index], 2),
        "Pullback (ATR)": np.round(a.pullback_atr[index], 2),
        "Bars since spike": a.age[index],
        "Spike": a.spikes[index],
        "Score": np.round(a.score[index], 1),
        "Verdict": [labels[int(v)] for v in a.verdict[index]],
    }
    st.dataframe(table, width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Main view
# ---------------------------------------------------------------------------


def render(choices: dict) -> None:
    try:
        bars, source_text, ticks, tick_key = load_market(choices)
    except data_feed.FeedError as exc:
        st.error(f"{exc}\n\nNo data is shown, so nothing here is out of date or simulated.")
        return
    if bars is None:
        return

    if ticks is not None:
        update_tick_state(ticks[0], ticks[1], choices["spike_threshold"], tick_key)

    analysis = me.analyze(bars, choices["params"])
    snapshot = me.latest_snapshot(analysis)

    # Master engine inputs, using the EMA and ATR of the selected timeframe.
    cycle = cycle_reading(bars, analysis, ticks is not None)
    close, ema_now = analysis.close[-1], analysis.ema[-1]
    ema_deviation = (ema_now - close) / ema_now * 100.0 if np.isfinite(ema_now) and ema_now else float("nan")
    master = master_probability(
        cycle["ticks"], analysis.pullback_atr[-1], ema_deviation, cycle["density"], choices["ema_extreme_pct"]
    )
    reading = {
        "pullback": fmt(snapshot["pullback_atr"], ".1f", " ATR"),
        "ema": "n/a" if not np.isfinite(ema_deviation) else f"{ema_deviation:.3f}% below",
        "ema_extreme": f"{choices['ema_extreme_pct']:g}% below",
    }

    downtrend = extended_downtrend(
        bars.close, bars.low, analysis.ema, analysis.spikes, cycle["ticks"],
        choices["downtrend_lookback"], choices["downtrend_min_cycle"],
    )

    exhaustion = downtrend_exhaustion(
        bars.close, bars.high, bars.low, analysis.ema, analysis.pullback_atr, choices["downtrend_lookback"]
    )

    # Source badge, then the command center: master score | exhaustion.
    st.markdown(f'<span class="dash-badge">{html.escape(source_text)}</span>', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        master_card(cycle, reading, master, downtrend)
    with right:
        exhaustion_card(exhaustion, downtrend)

    # Unified Spike Probability: combines the existing readings into one bar.
    p = analysis.params
    unified = unified_spike_probability(
        snapshot["ema_gap_atr"], snapshot["pullback_atr"], cycle["ticks"],
        exhaustion["score"] if exhaustion["ready"] else 0.0, downtrend["detected"],
        p.ema_gap_target_atr, p.pullback_target_atr,
    )
    unified_card(
        unified,
        {
            "ema": fmt(snapshot["ema_gap_atr"], ".1f", f" ATR (target {p.ema_gap_target_atr:g})"),
            "pullback": fmt(snapshot["pullback_atr"], ".1f", f" ATR (target {p.pullback_target_atr:g})"),
            "cycle": cycle_texts(cycle)[0] + " ticks (target 1,000)",
            "exhaustion": f"{exhaustion['score']:.0f}%" if exhaustion["ready"] else "n/a",
        },
    )
    show_downtrend(downtrend)
    stats_strip(snapshot)

    tab_live, tab_edge, tab_data = st.tabs(["Breakdown", "Edge test", "Data"])
    with tab_live:
        chart_col, check_col = st.columns([3, 1])
        with chart_col, st.container(border=True, key="card-charts"):
            st.plotly_chart(price_chart(bars, analysis, choices["window"]),
                            width="stretch", key="price")
            st.plotly_chart(score_chart(bars, analysis, choices["window"]),
                            width="stretch", key="score")
        with check_col, st.container(border=True, key="card-checklist"):
            checklist_panel(snapshot)
            st.caption(
                f"{len(bars):,} bars loaded · {snapshot['spike_count']} spikes detected · "
                "newest bar may still be forming"
            )
    with tab_edge, st.container(border=True, key="card-edge"):
        edge_panel(bars, analysis, choices["horizon"], choices["ema_extreme_pct"])
    with tab_data, st.container(border=True, key="card-data"):
        data_table(bars, analysis)


def main() -> None:
    has_logo = LOGO.exists()
    st.set_page_config(
        page_title="Boom 1000 SpikeVision Analyzer",
        page_icon=str(LOGO) if has_logo else "📈",
        layout="wide",
    )
    init_tick_state()
    st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)
    st.markdown('<div class="dash-head"><h1>Boom 1000 SpikeVision Analyzer</h1></div>', unsafe_allow_html=True)
    if has_logo:
        st.sidebar.image(str(LOGO), width="stretch")  # full poster at the top of the sidebar
    choices = sidebar_controls()
    interval = choices["refresh"] if choices.get("auto") else None

    # A fragment reruns on a timer without redrawing the sidebar.
    @st.fragment(run_every=interval)
    def live_view() -> None:
        render(choices)

    live_view()


if __name__ == "__main__":  # Streamlit runs this file as __main__; tests can import it safely
    main()
