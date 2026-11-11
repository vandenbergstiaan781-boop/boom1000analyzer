"""Boom 1000 Analyzer: local Streamlit interface. Run with: streamlit run app.py

This file holds the UI, the live tick counter and the master confluence engine. Bar-level
maths lives in math_engine.py and data loading lives in data_feed.py.
"""

from __future__ import annotations

import html
from dataclasses import replace

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import data_feed
import math_engine as me

# Chart colours (dark theme, set in .streamlit/config.toml).
BLUE, ORANGE, AQUA = "#3987e5", "#d95926", "#199e70"
INK, MUTED, GRID, AXIS = "#c3c2b7", "#898781", "#2c2c2a", "#383835"
SURFACE = "#1a1a19"

# Verdict display: an icon and label always go with the colour.
VERDICT_STYLE = {
    me.HIGH: ("✅", "#0ca30c"),
    me.WATCH: ("👀", "#fab219"),
    me.WAIT: ("⏳", MUTED),
}

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


def master_panel(cycle: dict, reading: dict, master: dict, source_text: str) -> None:
    """Alert banner, cycle and probability cards, tension bar and factor breakdown."""
    st.caption(source_text)
    probability = float(master["probability"])
    tension = float(master["tension"])
    show_alert(probability)

    has_age = np.isfinite(cycle["ticks"])
    suffix = "" if cycle["known"] else "+"  # no spike seen yet, so the true age is at least this
    prefix = "~" if cycle["estimated"] else ""
    ticks_text = f"{prefix}{cycle['ticks']:,.0f}{suffix}" if has_age else "n/a"
    seconds_text = f"{prefix}{cycle['ticks'] * SECONDS_PER_TICK:,.0f}{suffix} s" if has_age else "n/a"

    cols = st.columns(3)
    cols[0].metric("Cycle Age (Ticks)", ticks_text)
    cols[1].metric("Cycle Age (Seconds)", seconds_text)
    cols[2].metric("Master Spike Probability", f"{probability:.1f}%")

    st.progress(min(max(int(round(tension)), 0), 100), text=f"Spring Tension Level: {tension:.0f}/100")

    state = st.session_state
    if cycle["estimated"]:
        st.caption("This CSV has candles, so cycle age and density are estimates. Upload a tick export for exact counts.")
    else:
        st.caption(
            f"{state.total_ticks_recorded:,} ticks recorded · {state.spikes_recorded:,} spikes · "
            f"biggest uptick not counted as a spike: {state.largest_small_up:.2f}"
        )
    st.caption("Weighted score, not a measured chance of a spike. The Edge test tab compares it to what actually happened.")

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


def verdict_card(snapshot: dict, source_text: str) -> None:
    icon, colour = VERDICT_STYLE[snapshot["verdict_code"]]
    if snapshot["ready"]:
        note = "Label comes from the score. It is not a measured probability: check the Edge test tab."
    else:
        note = "Still warming up: needs enough bars and at least 5 completed spike cycles."
    st.markdown(
        f"""
        <div style="border-left:6px solid {colour};background:{SURFACE};border-radius:8px;
                    padding:14px 20px;margin-bottom:12px">
          <div style="font-size:0.85rem;color:{MUTED}">Bar score · {html.escape(source_text)}</div>
          <div style="font-size:1.9rem;font-weight:700;line-height:1.3">
            {icon} {html.escape(snapshot["verdict"])}
            <span style="font-weight:400;color:{INK}">· {snapshot["score"]:.0f}/100</span>
          </div>
          <div style="font-size:0.85rem;color:{INK}">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def fmt(value, spec: str = ",.2f", suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:{spec}}{suffix}"


def metric_row(s: dict) -> None:
    cols = st.columns(5)
    cols[0].metric("Price", fmt(s["price"]))
    cols[1].metric("Below EMA", fmt(s["ema_gap_atr"], ".1f", " ATR"))
    cols[2].metric("Pullback", fmt(s["pullback_atr"], ".1f", " ATR"))
    cols[3].metric("Bars since spike", fmt(s["bars_since_spike"], ".0f"))
    percentile = None if s["cycle_percentile"] is None else s["cycle_percentile"] * 100
    cols[4].metric("Cycle percentile", fmt(percentile, ".0f"))


def checklist_panel(snapshot: dict) -> None:
    st.subheader("Technical checklist")
    for item in snapshot["checklist"]:
        icon = "✅" if item["passed"] else ("🟡" if item["score"] >= 0.5 else "⬜")
        st.markdown(f"{icon} **{item['label']}**  \n{item['detail']}")
        st.progress(min(max(item["score"], 0.0), 1.0))


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
    return base_layout(fig, 460, "Price")


def score_chart(bars: me.Bars, a: me.Analysis, window: int) -> go.Figure:
    start = max(len(bars) - window, 0)
    x = x_values(bars)[start:]
    fig = go.Figure(go.Scatter(x=x, y=a.score[start:], name="Score", line=dict(color=BLUE, width=2)))
    for level, label in ((a.params.high_threshold, "High"), (a.params.watch_threshold, "Watch")):
        fig.add_hline(y=level, line=dict(color=MUTED, width=1, dash="dash"),
                      annotation_text=label, annotation_position="top left",
                      annotation_font_color=INK)
    base_layout(fig, 200, "Score")
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

    master_panel(cycle, reading, master, source_text)
    st.divider()
    verdict_card(snapshot, source_text)
    metric_row(snapshot)

    tab_live, tab_edge, tab_data = st.tabs(["Breakdown", "Edge test", "Data"])
    with tab_live:
        chart_col, check_col = st.columns([3, 1])
        with chart_col:
            st.plotly_chart(price_chart(bars, analysis, choices["window"]),
                            width="stretch", key="price")
            st.plotly_chart(score_chart(bars, analysis, choices["window"]),
                            width="stretch", key="score")
        with check_col:
            checklist_panel(snapshot)
            st.caption(
                f"{len(bars):,} bars loaded · {snapshot['spike_count']} spikes detected · "
                "newest bar may still be forming"
            )
    with tab_edge:
        edge_panel(bars, analysis, choices["horizon"], choices["ema_extreme_pct"])
    with tab_data:
        data_table(bars, analysis)


def main() -> None:
    st.set_page_config(page_title="Boom 1000 Analyzer", page_icon="📈", layout="wide")
    init_tick_state()
    st.title("Boom 1000 Analyzer")
    choices = sidebar_controls()
    interval = choices["refresh"] if choices.get("auto") else None

    # A fragment reruns on a timer without redrawing the sidebar.
    @st.fragment(run_every=interval)
    def live_view() -> None:
        render(choices)

    live_view()


if __name__ == "__main__":  # Streamlit runs this file as __main__; tests can import it safely
    main()
