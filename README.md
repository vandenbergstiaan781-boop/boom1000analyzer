# Boom 1000 Analyzer

Local data analyzer for the Boom 1000 index. Pure maths: no AI models, no paid services, no API keys.
It analyzes only and never places trades.

## Project layout

```
boom1000_analyzer/
├── app.py              Streamlit interface (display only)
├── math_engine.py      EMA, pullback, cycles, scoring, edge test (NumPy only)
├── data_feed.py        Simulator, CSV loader, Deriv public data
├── test_engine.py      Automated tests
├── requirements.txt
├── README.md
└── .streamlit/
    └── config.toml     Dark theme, usage stats off
```

## Install and run

You need Python 3.11 or newer.

**Windows (PowerShell)**

```powershell
cd boom1000_analyzer
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -q
streamlit run app.py
```

If PowerShell blocks the activate script, run this once, then activate again:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

**macOS / Linux**

```bash
cd boom1000_analyzer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
streamlit run app.py
```

The app opens at http://localhost:8501. If Streamlit asks for an email on first run, leave it blank and
press Enter. Press `Ctrl+C` in the terminal to stop it.

**Open it on your phone:** run the app on a PC with the command below, connect the phone to the same
Wi-Fi, then browse to `http://<your-PC-IP>:8501`. Windows may ask you to allow Python through the firewall.

```bash
streamlit run app.py --server.address 0.0.0.0
```

## Data sources

| Source | What it uses | Notes |
|---|---|---|
| Simulated | Random Boom-style ticks | Works offline. Not real market data. |
| Deriv live | Deriv's public market data | No account or key. Needs internet. |
| CSV upload | Your own export | Handles MT5 candle and tick files. |

**Export from MT5:** press `Ctrl+U` to open Symbols, select Boom 1000 Index, open the Bars or Ticks tab,
request the date range, then choose Export.

## How the score works

| Component | Measures | Full marks at (default) | Weight |
|---|---|---|---|
| EMA gap | Distance below the 20 EMA | 8 ATR | 30 |
| Pullback | Drop from the swing high | 25 ATR | 30 |
| Cycle | Current wait against past waits | 80th percentile | 25 |
| Grind | Share of recent bars closing lower | 90% | 15 |

- **ATR** leaves spike bars out, so it measures the normal grind.
- **Verdict:** 85+ is High Probability Setup, 60+ is Watch Closely, anything lower is Wait.
- **Defaults** were set on simulated M1 data, so that the verdicts spread out. They are not optimized for
  profit. On other timeframes, especially ticks, retune the targets with the sliders.

## Live tick counter and master score

- **Cycle age:** counts ticks since the last upward tick above the spike threshold (default 10 units).
  Every other tick adds 1, and 1 tick counts as 1 second. It reads back through recent history on start,
  so it doesn't begin at 0 mid-cycle. A `+` means no spike has been seen in the data yet.
- **Master Spike Probability:** a weighted score clamped to 5-95%.

| Factor | Weight | Full marks at |
|---|---|---|
| Cycle age | 35% | 1,000 ticks |
| ATR pullback | 30% | 25 ATR |
| EMA deviation | 20% | 6% below the 20 EMA (slider) |
| Down-bar density, last 10 ticks | 15% | 100% |

- **Spring Tension:** the same score before clamping, from 0 to 100.
- **Banners:** 85%+ Action Zone, 60-84% Building, under 60% Waiting.

**Tune two sliders on real data:**

1. **EMA deviation for full marks:** at 6%, this factor adds almost nothing, because price rarely gets that
   far below the 20 EMA. Action Zone then can't fire. On simulated M1 data, price was rarely more than about 0.13% below it, so the banner
   needed roughly 0.15%. Set yours from the real reading under "How this score was built".
2. **Tick spike threshold:** a real spike under this size won't reset the counter. Under the cards, check
   "biggest uptick not counted as a spike". If it looks like a spike, lower the threshold.

## Read this before trading on it

The verdict label comes from the score. It is not a measured probability.

All four components mostly grow with time since the last spike. If spikes arrive at random, as they do in
the simulator, a long wait does not make a spike more likely, and the score has no predictive power.

The **Edge test** tab checks this on whichever data is loaded:

- **Tier table:** compares the spike rate after each verdict with a random bar.
- **"Is a spike ever due?" chart:** shows the next-bar spike chance by cycle age. Flat means no timing edge.

Tuning sliders until the edge test looks good will overfit. Tune on one CSV, then confirm on a later
period that you did not touch. The test also leaves out spread, slippage and commission.

## Extending it

- **New data source:** write a function in `data_feed.py` that returns `Bars`.
- **New score component:** add its series and weight in `analyze()`, a slider in `app.py`, and a test.
- **Mobile app:** `latest_snapshot()` returns a JSON-safe dictionary and the engine has no UI code, so a
  small API (FastAPI, for example) can serve it to a mobile front end.
