# mesa-quant-tests

Pre-registered backtests of popular trading ideas. Each idea gets a written rule for what counts as
"it works" **before** any data is run, and the code then reports PASS or DIES. Most ideas die, and
that is the point: the cheapest place to lose money on a bad idea is a backtest.

No accounts, API keys or real money are used. Everything runs on public data.

## The method

1. Write the hypothesis and its death rule first (minimum sample size, minimum edge after costs,
   minimum t-statistic, robustness checks), in the script's docstring.
2. Pull public data (scripts named `fetch_*.py`).
3. Run the test once. No tuning after seeing the result.
4. Keep the verdict, including the failures.

## Results

| Idea | Script | Verdict | In one line |
| --- | --- | --- | --- |
| Copy US Congress members' stock buys in sectors their committees oversee | `h_congress.py` | **DIES** | +0.47% vs sector over 20 days, t = 0.95; one trade carries most of the mean, trimmed mean +0.06% |
| Fade ARK Invest's trades (buy what they sell, sell what they buy) | `h_ark_fade.py` | **DIES** | -1.53% per week net, t = -1.67, across 15,063 trades; copying ARK's buys also loses |
| 200-day trend filter on hyped themes (dot-com, clean energy, ARKK) | `h_theme_trend.py` | **PASSES 4/5** | Cut the worst drawdowns sharply in 4 of 5 bubbles; caveat: the test windows were chosen around known crashes |
| Weekly moving-average filter on Bitcoin (20/50/100/200 weeks, "3 strikes" exit) | `h_weekly_ma_btc.py` | **DIES 0/8** | 50-week versions beat buy-and-hold since 2021 (+26-27%/yr vs +19.5%, max drop -47% vs -75%) but missed the strict drawdown bar |
| Sell Bitcoin puts with cash-secured rules | `h005_btc_puts.py` | **DIES** | Profitable 93% of the time, but the worst trade lost 15x the average premium |
| Buy prediction-market favourites (90c and up) and hold to settlement, Kalshi | `h004_kalshi_favorites.py` | Survives, barely | +2.5c per contract as a maker (upper bound: assumes every limit order fills) |
| Same idea on Polymarket | `h004_polymarket.py` | **DIES** | -1.6c per share as a maker, t = -3.65 |

Full output for each test is in the `results_*.txt` files.

## Running it

```
pip install pandas numpy requests yfinance pypdf
python fetch_public.py        # BTC daily, Deribit implied volatility, Kalshi settled markets
python fetch_house.py         # US House periodic transaction reports (parsed from the Clerk's PDFs)
python h_congress.py          # then any h_*.py test
```

Downloaded data goes in `data/`, which is not tracked in this repo.

## Author

Randall Bachir · Caracas, Venezuela · built with AI-assisted coding (Claude)
