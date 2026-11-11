"""Tests for the math engine and data feeds. Run with: python -m pytest -q"""

import json
from dataclasses import replace

import numpy as np
import pytest

import data_feed
from math_engine import (
    HIGH,
    MIN_CYCLES,
    WAIT,
    WATCH,
    Bars,
    Params,
    analyze,
    bars_since_spike,
    cycle_hazard,
    cycle_percentile,
    detect_spikes,
    edge_report,
    ema,
    grind_ratio,
    latest_snapshot,
    pullback_depth,
    swing_highs,
    wilson_interval,
)


def sim_bars(seed=7, n_bars=3000, ticks_per_bar=60):
    feed = data_feed.SimulatedFeed(seed=seed, ticks_per_bar=ticks_per_bar)
    feed.advance(n_bars * ticks_per_bar)
    return feed.bars()


# --- Indicators, checked against values worked out by hand ------------------


def test_ema_known_values():
    out = ema(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 3)
    assert np.isnan(out[:2]).all()
    # Seed is the SMA of the first 3 (2.0), then alpha = 0.5.
    assert out[2:] == pytest.approx([2.0, 3.0, 4.0])


def test_ema_too_little_data_is_all_nan():
    assert np.isnan(ema(np.array([1.0, 2.0]), 5)).all()


def test_swing_high_rolls_off_old_highs():
    high = np.array([5.0, 9.0, 3.0, 2.0, 1.0])
    assert swing_highs(high, 3).tolist() == [5.0, 9.0, 9.0, 9.0, 3.0]


def test_pullback_depth_is_drop_from_swing_high():
    bars = Bars.from_prices([10.0, 12.0, 11.0, 9.0])
    swing, depth = pullback_depth(bars, 3)
    assert swing[-1] == 12.0
    assert depth[-1] == 3.0
    assert (depth >= 0).all()


def test_bars_since_spike():
    spikes = np.array([False, False, True, False, False, True, False])
    out = bars_since_spike(spikes)
    assert np.isnan(out[:2]).all()
    assert out[2:].tolist() == [0, 1, 2, 0, 1]


def test_grind_ratio():
    close = np.array([10.0, 9.0, 8.0, 9.0, 8.0])
    out = grind_ratio(close, 4)
    assert np.isnan(out[:4]).all()
    assert out[4] == pytest.approx(0.75)


# --- Spike detection --------------------------------------------------------


def test_detect_spikes_finds_planted_spikes_only():
    prices = 1000.0 - 0.1 * np.arange(200)
    prices[100:] += 10.0
    prices[150:] += 10.0
    found = np.flatnonzero(detect_spikes(Bars.from_prices(prices)))
    assert found.tolist() == [100, 150]


def test_detect_spikes_ignores_downward_jumps():
    prices = 1000.0 - 0.1 * np.arange(200)
    prices[100:] -= 10.0
    assert not detect_spikes(Bars.from_prices(prices)).any()


def test_detect_spikes_matches_simulated_rate():
    # Spikes average one per 1000 ticks, so 60-tick bars give about 16.7 bars each.
    a = analyze(sim_bars(n_bars=5000))
    interval = np.diff(np.flatnonzero(a.spikes)).mean()
    assert 14 < interval < 20


# --- Cycle percentile -------------------------------------------------------


def test_cycle_percentile_needs_enough_history_and_ranks_correctly():
    spike_at = [10, 20, 30, 40, 50, 60]
    spikes = np.zeros(80, dtype=bool)
    spikes[spike_at] = True
    out = cycle_percentile(spikes, min_cycles=5)
    assert np.isnan(out[:60]).all()
    assert out[60 + 5] == 0.0     # younger than every past cycle (all 10 bars)
    assert out[60 + 10] == 1.0    # as old as every past cycle
    assert MIN_CYCLES == 5


# --- The critical property: no peeking at the future ------------------------


@pytest.mark.parametrize("field", ["ema", "atr", "spikes", "age", "percentile", "grind", "score", "verdict"])
def test_series_are_causal(field):
    """Cutting off future bars must not change any past value."""
    bars = sim_bars(n_bars=1500)
    cut = 1000
    short = Bars(bars.open[:cut], bars.high[:cut], bars.low[:cut], bars.close[:cut])
    full_values = getattr(analyze(bars), field)[:cut]
    short_values = getattr(analyze(short), field)
    assert np.array_equal(full_values, short_values, equal_nan=True) or np.allclose(
        full_values.astype(float), short_values.astype(float), equal_nan=True
    )


# --- Scoring ----------------------------------------------------------------


def test_score_range_and_verdict_consistency():
    p = Params()
    a = analyze(sim_bars(), p)
    assert (a.score >= 0).all() and (a.score <= 100).all()
    assert (a.score[a.verdict == HIGH] >= p.high_threshold).all()
    watch = a.score[a.verdict == WATCH]
    assert (watch >= p.watch_threshold).all() and (watch < p.high_threshold).all()
    assert (a.verdict[~a.ready] == WAIT).all()


def test_short_input_says_wait_without_crashing():
    for n in (1, 2, 5, 30):
        snapshot = latest_snapshot(analyze(Bars.from_prices(1000.0 - np.arange(n, dtype=float))))
        assert snapshot["verdict"] == "Wait"
        assert snapshot["ready"] is False


def test_flat_prices_do_not_crash():
    a = analyze(Bars.from_prices(np.full(500, 1000.0)))
    assert not a.spikes.any()
    assert latest_snapshot(a)["verdict"] == "Wait"


def test_zero_weights_do_not_crash():
    p = replace(Params(), weight_ema=0, weight_pullback=0, weight_cycle=0, weight_grind=0)
    assert (analyze(sim_bars(n_bars=500), p).score == 0).all()


def test_snapshot_is_json_safe():
    snapshot = latest_snapshot(analyze(Bars.from_prices([1.0, 2.0, 3.0])))
    json.dumps(snapshot, allow_nan=False)
    snapshot = latest_snapshot(analyze(sim_bars(n_bars=500)))
    json.dumps(snapshot, allow_nan=False)
    assert len(snapshot["checklist"]) == 4


# --- Edge test and hazard ---------------------------------------------------


def test_wilson_interval():
    low, high = wilson_interval(50, 100)
    assert low == pytest.approx(0.404, abs=0.01)
    assert high == pytest.approx(0.596, abs=0.01)
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_edge_report_shape_and_small_data():
    assert edge_report(Bars.from_prices([1.0, 2.0]), analyze(Bars.from_prices([1.0, 2.0])))["tiers"] == []
    bars = sim_bars()
    report = edge_report(bars, analyze(bars), horizon=10)
    assert [t["verdict"] for t in report["tiers"]] == ["High Probability Setup", "Watch Closely", "Wait"]
    assert 0 < report["baseline_rate"] < 1


def test_edge_report_detects_a_real_edge():
    """The test must be able to say yes: plant spikes that follow deep pullbacks."""
    rng = np.random.default_rng(3)
    moves = -0.5 + rng.normal(0, 0.05, 20000)
    i, wait = 0, 0
    # Spikes arrive only after a long, fixed wait, so an old cycle really does predict one.
    while i < len(moves):
        wait += 1
        if wait >= 40:
            moves[i] += 25.0
            wait = 0
        i += 1
    prices = 12000 + np.cumsum(moves)
    bars = Bars.from_prices(prices)
    params = Params(cycle_percentile_target=0.5, high_threshold=70, watch_threshold=40,
                    ema_gap_target_atr=5, pullback_target_atr=15, swing_lookback=60)
    a = analyze(bars, params)
    report = edge_report(bars, a, horizon=5)
    best = report["tiers"][0]
    assert best["entries"] >= 30
    assert best["ci_low"] > report["baseline_rate"]
    # The hazard rises sharply at the planted age.
    hazard = cycle_hazard(a, bucket=5, max_age=40)
    assert hazard["rate"][0] == 0.0     # ages 0-4: spikes never arrive this early
    assert hazard["rate"][7] > 0.15     # ages 35-39: the bucket holding the planted spike


def test_hazard_is_flat_on_memoryless_simulator():
    bars = sim_bars(n_bars=20000)
    hazard = cycle_hazard(analyze(bars))
    rates = np.array(hazard["rate"][:6])
    assert np.all(np.abs(rates - hazard["overall_rate"]) < 0.02)


# --- Data feeds -------------------------------------------------------------


def test_ticks_to_bars_ohlc():
    bars = data_feed.ticks_to_bars(np.array([1.0, 3.0, 2.0, 5.0, 4.0]), 2)
    assert bars.open.tolist() == [1.0, 2.0, 4.0]
    assert bars.high.tolist() == [3.0, 5.0, 4.0]
    assert bars.low.tolist() == [1.0, 2.0, 4.0]
    assert bars.close.tolist() == [3.0, 5.0, 4.0]


def test_simulator_is_reproducible_and_buffer_is_capped():
    a, b = data_feed.SimulatedFeed(seed=1), data_feed.SimulatedFeed(seed=1)
    a.advance(5000); b.advance(5000)
    assert np.array_equal(a.prices, b.prices)
    a.advance(data_feed.SimulatedFeed.MAX_TICKS)
    assert len(a.prices) <= data_feed.SimulatedFeed.MAX_TICKS


def test_simulator_backfill_ends_at_the_current_time():
    feed = data_feed.SimulatedFeed(seed=1, ticks_per_bar=60)
    start = feed.first_time
    feed.backfill(6000)
    assert feed.bars().time[-1] <= start


def test_csv_mt5_candles_tab_separated():
    raw = (
        "<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\n"
        "2026.01.05\t10:00:00\t100.5\t101\t99\t100\t60\n"
        "2026.01.05\t10:01:00\t100\t100.5\t98\t99\t60\n"
    ).encode()
    bars = data_feed.load_csv(raw)
    assert bars.close.tolist() == [100.0, 99.0]
    assert bars.time[1] - bars.time[0] == 60


def test_csv_utf16_export():
    raw = "time,open,high,low,close\n1000,1,2,0.5,1.5\n".encode("utf-16")
    assert data_feed.load_csv(raw).high.tolist() == [2.0]


def test_csv_mt5_ticks_skip_blank_bid():
    raw = (
        "<DATE>\t<TIME>\t<BID>\t<ASK>\n"
        "2026.01.05\t10:00:00.000\t100.1\t100.2\n"
        "2026.01.05\t10:00:01.000\t\t100.3\n"
        "2026.01.05\t10:00:02.000\t100.0\t100.3\n"
    ).encode()
    assert data_feed.load_csv(raw).close.tolist() == [100.1, 100.0]


def test_csv_bad_input_gives_readable_error():
    with pytest.raises(data_feed.FeedError):
        data_feed.load_csv(b"")
    with pytest.raises(data_feed.FeedError, match="price columns"):
        data_feed.load_csv(b"foo,bar\n1,2\n")


def test_fetch_deriv_request_and_fallback(monkeypatch):
    """No network: the first endpoint fails, the second answers with candles."""
    import websocket

    sent, urls = [], []

    class FakeConnection:
        def send(self, message):
            sent.append(json.loads(message))

        def recv(self):
            return json.dumps({"candles": [{"epoch": 60, "open": 2, "high": 3, "low": 1, "close": 2.5}]})

        def close(self):
            pass

    def fake_connect(url, timeout):
        urls.append(url)
        if len(urls) == 1:
            raise OSError("refused")
        return FakeConnection()

    monkeypatch.setattr(websocket, "create_connection", fake_connect)
    bars = data_feed.fetch_deriv("BOOM1000", granularity=60, count=100)
    assert bars.close.tolist() == [2.5]
    assert urls == list(data_feed.DERIV_ENDPOINTS)
    assert sent[0]["ticks_history"] == "BOOM1000"
    assert sent[0]["style"] == "candles" and sent[0]["granularity"] == 60
    assert "authorize" not in sent[0]


def test_fetch_deriv_reports_all_failures(monkeypatch):
    import websocket

    def refuse(url, timeout):
        raise OSError("refused")

    monkeypatch.setattr(websocket, "create_connection", refuse)
    with pytest.raises(data_feed.FeedError, match="refused"):
        data_feed.fetch_deriv()


def test_parse_deriv_candles_and_ticks_and_error():
    candles = {"candles": [{"epoch": 60, "open": "1.5", "high": 2, "low": 1, "close": 1.75}]}
    assert data_feed.parse_deriv_response(candles).open.tolist() == [1.5]
    ticks = {"history": {"prices": [1.0, 2.0], "times": [1, 2]}}
    assert data_feed.parse_deriv_response(ticks).close.tolist() == [1.0, 2.0]
    with pytest.raises(data_feed.FeedError, match="InvalidSymbol"):
        data_feed.parse_deriv_response({"error": {"code": "InvalidSymbol", "message": "nope"}})


# --- Live tick counter and master engine (app.py) ---------------------------


import app as dashboard  # noqa: E402
from pathlib import Path  # noqa: E402


def counted_one_tick_at_a_time(prices, threshold):
    """Plain loop that follows the spec line by line, as a check on process_ticks."""
    since = 0
    for previous, price in zip(prices, prices[1:]):
        since = 0 if price - previous > threshold else since + 1
    return since


def test_process_ticks_resets_on_spike_and_counts_after():
    prices = [100.0, 99.9, 99.8, 112.0, 111.9, 111.8]
    result = dashboard.process_ticks(prices, 10.0)
    assert result["ticks_since_spike"] == 2
    assert result["spikes"] == 1
    assert result["processed"] == 6


def test_process_ticks_threshold_is_strictly_greater():
    assert dashboard.process_ticks([100.0, 110.0], 10.0)["spikes"] == 0
    assert dashboard.process_ticks([100.0, 110.5], 10.0)["spikes"] == 1


def test_small_upticks_still_count_and_are_reported():
    result = dashboard.process_ticks([100.0, 104.0, 103.0], 10.0, ticks_since_spike=7, last_price=100.0)
    assert result["ticks_since_spike"] == 10
    assert result["largest_small_up"] == 4.0


def test_spike_between_two_batches_is_caught():
    first = dashboard.process_ticks([100.0, 99.0], 10.0)
    second = dashboard.process_ticks([120.0, 119.0], 10.0, first["ticks_since_spike"], 99.0)
    assert second["spikes"] == 1
    assert second["ticks_since_spike"] == 1


def test_any_batch_size_gives_the_same_count_as_a_plain_loop():
    feed = data_feed.SimulatedFeed(seed=5)
    feed.advance(20000)
    prices = feed.prices
    since, last = 0, None
    for chunk in np.array_split(prices, 137):
        out = dashboard.process_ticks(chunk, 5.0, since, last)
        since, last = out["ticks_since_spike"], chunk[-1]
    # The plain loop leaves out the first tick, which has no earlier price to compare.
    plain = counted_one_tick_at_a_time(prices.tolist(), 5.0)
    has_spike = (np.diff(prices) > 5.0).any()
    assert since == (plain if has_spike else plain + 1)


def test_master_formula_hand_worked():
    # Every factor is at half of its extreme, so the result is exactly half of 100.
    half = dashboard.master_probability(500, 12.5, 3.0, 50.0)
    assert float(half["tension"]) == pytest.approx(50.0)
    assert float(half["points"]["cycle"]) == pytest.approx(17.5)
    assert float(half["points"]["pullback"]) == pytest.approx(15.0)
    assert float(half["points"]["ema"]) == pytest.approx(10.0)
    assert float(half["points"]["density"]) == pytest.approx(7.5)


def test_master_clamps_and_handles_missing_values():
    top = dashboard.master_probability(5000, 100, 50, 100)
    assert float(top["tension"]) == pytest.approx(100.0)
    assert float(top["probability"]) == 95.0
    low = dashboard.master_probability(float("nan"), float("nan"), -3.0, float("nan"))
    assert float(low["tension"]) == 0.0
    assert float(low["probability"]) == 5.0


def test_alert_levels_at_the_boundaries():
    assert [dashboard.alert_level(p) for p in (5, 59.9, 60, 84.9, 85, 95)] == [
        "waiting", "waiting", "building", "building", "action", "action",
    ]


def test_counter_survives_reruns_and_never_double_counts():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(Path(__file__).with_name("app.py")), default_timeout=60)
    at.run()
    assert not at.exception
    first_total = at.session_state["total_ticks_recorded"]
    assert first_total > 0

    # Auto-refresh off: reruns add no ticks, so nothing may change.
    at.run()
    assert at.session_state["total_ticks_recorded"] == first_total

    # Auto-refresh on: each rerun adds exactly the simulator's new ticks.
    next(t for t in at.sidebar.toggle if t.label == "Auto-refresh").set_value(True).run()
    at.run()
    feed = at.session_state["sim_feed"]
    assert at.session_state["total_ticks_recorded"] == first_total + 60
    assert at.session_state["total_ticks_recorded"] == len(feed.prices)
    assert at.session_state["ticks_since_spike"] == counted_one_tick_at_a_time(feed.prices.tolist(), 10.0)


def test_live_mode_counts_new_ticks_once_and_recovers_from_a_gap(monkeypatch):
    """Runs the real app against a fake Deriv feed that gains ticks between refreshes."""
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    market = {"prices": list(1000.0 - 0.01 * np.arange(6000)), "requests": []}

    def fake_fetch(symbol, granularity, count):
        market["requests"].append((granularity, count))
        prices = np.array(market["prices"], dtype=float)
        times = np.arange(len(prices), dtype=float)
        if granularity == 0:
            return Bars.from_prices(prices[-count:], times[-count:])
        return data_feed.ticks_to_bars(prices, 60, 0.0)

    def add_ticks(moves):
        for move in moves:
            market["prices"].append(market["prices"][-1] + move)

    monkeypatch.setattr(data_feed, "fetch_deriv", fake_fetch)
    st.cache_data.clear()
    at = AppTest.from_file(str(Path(__file__).with_name("app.py")), default_timeout=60)
    at.run()
    at.sidebar.radio[0].set_value("Deriv live (no key)").run()
    assert not at.exception
    assert (0, dashboard.TICK_SEED_COUNT) in market["requests"]
    assert at.session_state["total_ticks_recorded"] == dashboard.TICK_SEED_COUNT
    assert at.session_state["spike_seen"] is False  # flagged as "at least", not a false zero
    age = at.session_state["ticks_since_spike"]

    # 20 new ticks: the counter adds exactly 20, using the short poll.
    add_ticks([-0.01] * 20)
    st.cache_data.clear()
    at.run()
    assert at.session_state["ticks_since_spike"] == age + 20
    assert (0, dashboard.TICK_POLL_COUNT) in market["requests"]

    # A refresh with no new ticks changes nothing.
    st.cache_data.clear()
    at.run()
    assert at.session_state["ticks_since_spike"] == age + 20

    # A spike followed by 3 ticks.
    add_ticks([30.0, -0.01, -0.01, -0.01])
    st.cache_data.clear()
    at.run()
    assert at.session_state["ticks_since_spike"] == 3
    assert at.session_state["spike_seen"] is True
    assert at.session_state["spikes_recorded"] == 1

    # The PC sleeps past the short poll: the counter rebuilds and stays correct.
    add_ticks([-0.01] * (dashboard.TICK_POLL_COUNT + 50))
    st.cache_data.clear()
    at.run()
    assert not at.exception
    assert at.session_state["ticks_since_spike"] == 3 + dashboard.TICK_POLL_COUNT + 50
    assert at.session_state["spike_seen"] is True


@pytest.mark.parametrize("score, element", [(90.0, "error"), (70.0, "warning"), (30.0, "info")])
def test_each_alert_uses_the_right_banner(score, element):
    from streamlit.testing.v1 import AppTest

    def script(value):
        import app as dashboard

        dashboard.show_alert(value)

    at = AppTest.from_function(script, args=(score,), default_timeout=30).run()
    assert not at.exception
    shown = {name: len(getattr(at, name)) for name in ("error", "warning", "info")}
    assert shown == {name: int(name == element) for name in shown}
