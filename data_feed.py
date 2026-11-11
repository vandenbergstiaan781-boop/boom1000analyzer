"""Price sources for the analyzer. Each one returns a math_engine.Bars object.

- SimulatedFeed: offline synthetic Boom-style ticks.
- load_csv:      a CSV exported from MT5, or any file with OHLC or price columns.
- fetch_deriv:   Deriv's public market data over WebSocket. No account or key.

To add a source later, write a function that returns Bars. The engine never needs
to know where the numbers came from.
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from math_engine import Bars

# Public, no-auth gateway first, then the older endpoint with Deriv's public test app id.
DERIV_ENDPOINTS = (
    "wss://api.derivws.com/trading/v1/options/ws/public",
    "wss://ws.derivws.com/websockets/v3?app_id=1089",
)


class FeedError(RuntimeError):
    """Raised with a readable message when a data source fails."""


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


def ticks_to_bars(prices: np.ndarray, ticks_per_bar: int, first_time: float = 0.0) -> Bars:
    """Group ticks into OHLC bars. The last bar may still be forming."""
    if ticks_per_bar <= 1:
        return Bars.from_prices(prices, first_time + np.arange(len(prices), dtype=float))
    n = len(prices)
    starts = np.arange(0, n, ticks_per_bar)
    ends = np.minimum(starts + ticks_per_bar, n) - 1
    return Bars(
        open=prices[starts],
        high=np.maximum.reduceat(prices, starts),
        low=np.minimum.reduceat(prices, starts),
        close=prices[ends],
        time=first_time + starts.astype(float),
    )


class SimulatedFeed:
    """Synthetic Boom 1000: a slow downward grind plus rare random upward spikes.

    Every tick has the same spike chance regardless of the time since the last spike
    (memoryless). The numbers are illustrative and are not fitted to Deriv's feed.
    """

    MAX_TICKS = 400_000

    def __init__(
        self,
        seed: int = 42,
        ticks_per_bar: int = 60,
        spike_every_ticks: float = 1000.0,
        spike_size: float = 10.0,
        grind_per_tick: float = 0.0105,
        noise_per_tick: float = 0.01,
        start_price: float = 12000.0,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.ticks_per_bar = ticks_per_bar
        self.spike_prob = 1.0 / spike_every_ticks
        self.spike_size = spike_size
        self.grind = grind_per_tick
        self.noise = noise_per_tick
        self.prices = np.array([start_price], dtype=float)
        self.first_time = float(int(datetime.now(timezone.utc).timestamp()))

    def advance(self, n_ticks: int) -> None:
        """Generate more ticks and keep the buffer to a fixed size."""
        if n_ticks <= 0:
            return
        moves = -self.grind + self.rng.normal(0.0, self.noise, n_ticks)
        is_spike = self.rng.random(n_ticks) < self.spike_prob
        sizes = self.rng.lognormal(np.log(self.spike_size), 0.4, n_ticks)
        moves = moves + is_spike * sizes
        self.prices = np.concatenate((self.prices, self.prices[-1] + np.cumsum(moves)))

        excess = len(self.prices) - self.MAX_TICKS
        if excess > 0:
            excess += -excess % self.ticks_per_bar  # round up to keep bar boundaries aligned
            self.prices = self.prices[excess:]
            self.first_time += excess

    def backfill(self, n_ticks: int) -> None:
        """Generate history that ends now, so chart times are not in the future."""
        self.advance(n_ticks)
        self.first_time -= n_ticks

    def bars(self) -> Bars:
        return ticks_to_bars(self.prices, self.ticks_per_bar, self.first_time)


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

_TIME_FORMATS = (
    "%Y.%m.%d %H:%M:%S.%f",
    "%Y.%m.%d %H:%M:%S",
    "%Y.%m.%d %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M:%S",
    "%Y.%m.%d",
    "%Y-%m-%d",
)


def _decode(raw: bytes) -> str:
    """MT5 sometimes exports UTF-16, so check the byte-order mark first."""
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    return raw.decode("utf-8-sig", errors="replace")


def _parse_time(text: str) -> Optional[float]:
    text = text.strip()
    if not text:
        return None
    try:
        return float(text)  # already epoch seconds
    except ValueError:
        pass
    for fmt in _TIME_FORMATS:
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    return None


def _to_float(text: str) -> Optional[float]:
    try:
        value = float(text.strip())
    except (ValueError, AttributeError):
        return None
    return value if np.isfinite(value) else None


def load_csv(raw: bytes) -> Bars:
    """Parse MT5 candle exports, MT5 tick exports, or generic CSV files.

    Accepted columns (any case, with or without <>):
      candles: open, high, low, close, with optional date, time or epoch
      ticks:   one of bid, price, quote, last, with optional date, time or epoch
    """
    text = _decode(raw).strip()
    if not text:
        raise FeedError("The file is empty.")

    first_line = text.splitlines()[0]
    delimiter = max("\t,;", key=first_line.count)
    rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))
    header = [cell.strip().strip("<>").lower() for cell in rows[0]]
    column = {name: i for i, name in enumerate(header)}

    def cell(row: list[str], name: str) -> str:
        i = column.get(name)
        return row[i] if i is not None and i < len(row) else ""

    def row_time(row: list[str]) -> Optional[float]:
        if "date" in column and "time" in column:
            return _parse_time(f"{cell(row, 'date')} {cell(row, 'time')}")
        for name in ("time", "datetime", "date", "epoch", "timestamp"):
            if name in column:
                return _parse_time(cell(row, name))
        return None

    if all(name in column for name in ("open", "high", "low", "close")):
        o, h, l, c, t = [], [], [], [], []
        for row in rows[1:]:
            values = [_to_float(cell(row, name)) for name in ("open", "high", "low", "close")]
            if None in values:
                continue
            o.append(values[0]); h.append(values[1]); l.append(values[2]); c.append(values[3])
            t.append(row_time(row))
        if not c:
            raise FeedError("No valid OHLC rows found in the file.")
        times = np.array(t, dtype=float) if all(x is not None for x in t) else None
        return Bars(np.array(o), np.array(h), np.array(l), np.array(c), times)

    price_column = next((n for n in ("bid", "price", "quote", "last", "close") if n in column), None)
    if price_column is None:
        raise FeedError(f"Could not find price columns. Found: {', '.join(header)}")

    prices, t = [], []
    for row in rows[1:]:
        value = _to_float(cell(row, price_column))
        # MT5 tick exports leave Bid blank on rows where only Ask changed: skip those.
        if value is None:
            continue
        prices.append(value)
        t.append(row_time(row))
    if not prices:
        raise FeedError("No valid price rows found in the file.")
    times = np.array(t, dtype=float) if all(x is not None for x in t) else None
    return Bars.from_prices(np.array(prices), times)


# ---------------------------------------------------------------------------
# Live data from Deriv (public, no key)
# ---------------------------------------------------------------------------


def parse_deriv_response(response: dict) -> Bars:
    """Convert a ticks_history response into Bars. Separate so it can be unit tested."""
    if "error" in response:
        error = response["error"]
        raise FeedError(f"{error.get('code', 'Error')}: {error.get('message', 'unknown error')}")
    if "candles" in response:
        candles = response["candles"]
        if not candles:
            raise FeedError("Deriv returned no candles.")
        field = lambda name: np.array([float(c[name]) for c in candles])
        return Bars(field("open"), field("high"), field("low"), field("close"), field("epoch"))
    if "history" in response:
        history = response["history"]
        return Bars.from_prices(
            np.array(history["prices"], dtype=float), np.array(history["times"], dtype=float)
        )
    raise FeedError("Unexpected response from Deriv.")


def fetch_deriv(symbol: str = "BOOM1000", granularity: int = 60, count: int = 5000) -> Bars:
    """Fetch recent history. granularity is seconds per candle, or 0 for raw ticks."""
    try:
        import websocket  # websocket-client; imported here so offline modes never need it
    except ImportError as exc:
        raise FeedError("Run: pip install websocket-client") from exc

    request = {
        "ticks_history": symbol,
        "adjust_start_time": 1,
        "count": count,
        "end": "latest",
        "start": 1,
        "style": "candles" if granularity > 0 else "ticks",
    }
    if granularity > 0:
        request["granularity"] = granularity

    failures = []
    for url in DERIV_ENDPOINTS:
        connection = None
        try:
            connection = websocket.create_connection(url, timeout=10)
            connection.send(json.dumps(request))
            return parse_deriv_response(json.loads(connection.recv()))
        except FeedError as exc:
            failures.append(str(exc))
        except Exception as exc:  # network errors: try the next endpoint
            failures.append(f"{type(exc).__name__}: {exc}")
        finally:
            if connection is not None:
                connection.close()
    raise FeedError("Could not get Deriv data. " + " | ".join(failures))
