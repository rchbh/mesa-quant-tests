"""H-005: sell Bitcoin puts with the cash-secured-put rules. Does it survive its own death rule?

Mesa quant's H-005 (docs/hipotesis-semilla.json): "BTC puts with the CSP rules (distance, implied
vs realized volatility, exit at 50%)". Its death rule, written before any data:
    dies if  mean premium / mean costs < 2    or    max loss / mean premium > 3     (N >= 60)

Unlike the csp-scanner test, the premium here comes from REAL implied volatility: Deribit's DVOL
index, the market's 30-day at-the-money implied volatility for BTC, recorded daily. Out-of-the-
money puts trade at a higher implied vol than at-the-money (skew), so pricing them at DVOL
UNDERSTATES the premium a seller would actually receive: a conservative choice.

Pre-registered rules (fixed before the data was downloaded):
  * Sell a 30-calendar-day put, strike 12% below the BTC close (the csp-scanner's 10-14% band).
  * Only sell when DVOL is above BTC's 30-day realized volatility (the CSP "IV vs realized" rule);
    otherwise wait for the next day.
  * Exit: buy back as soon as the daily mark is at or below 50% of the credit; otherwise hold to
    expiry and pay the intrinsic value. Marks use that day's DVOL and BTC close.
  * One position at a time; the next put is sold the day after the previous one closes.
  * Costs: exchange fee 0.03% of the BTC notional per trade capped at 12.5% of the option price
    (Deribit / Binance Options schedule), half-spread 5% of the option price on every trade, and
    0.015% of notional at expiry when the put finishes in the money.
  * Returns are on cash collateral (the strike), as in a cash-secured put.

    python h005_btc_puts.py
"""
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
OTM = 0.12
DAYS = 30
TAKE_PROFIT = 0.5
FEE_PCT_NOTIONAL = 0.0003
FEE_CAP_PCT_OPTION = 0.125
HALF_SPREAD = 0.05
SETTLE_FEE = 0.00015
DEATH_PREMIUM_OVER_COSTS = 2
DEATH_MAXLOSS_OVER_PREMIUM = 3
N_MIN = 60


def N(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_put(S, K, T, sigma):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
    return K * N(-(d1 - sigma * math.sqrt(T))) - S * N(-d1)


def trade_fee(S, option_price):
    return min(FEE_PCT_NOTIONAL * S, FEE_CAP_PCT_OPTION * option_price) + HALF_SPREAD * option_price


def load():
    rows = json.load(gzip.open(DATA / "deribit_dvol_btc_1d.json.gz", "rt"))
    dv = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close"][: len(rows[0])])
    dv["date"] = pd.to_datetime(dv["ts"], unit="ms").dt.strftime("%Y-%m-%d")
    dvol = dv.drop_duplicates("date").set_index("date")["close"].astype(float)
    btc = pd.read_csv(DATA / "btc_usd_1d.csv", index_col=0)["close"].astype(float)
    df = pd.DataFrame({"S": btc, "iv": dvol / 100}).dropna()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    lr = np.log(btc / btc.shift(1))
    hv = (lr.rolling(30).std() * math.sqrt(365)).rename("hv")
    hv.index = pd.to_datetime(hv.index)
    return df.join(hv)


def run(df, iv_filter=True, take_profit=True):
    """Sequential trades. Returns a DataFrame, one row per trade."""
    dates, S, iv, hv = df.index, df["S"].to_numpy(), df["iv"].to_numpy(), df["hv"].to_numpy()
    out, i = [], 0
    while i < len(df):
        if np.isnan(hv[i]) or (iv_filter and not iv[i] > hv[i]):
            i += 1
            continue
        expiry = dates[i] + pd.Timedelta(days=DAYS)
        j_exp = dates.searchsorted(expiry)
        if j_exp >= len(df):
            break
        K = S[i] * (1 - OTM)
        credit = bs_put(S[i], K, DAYS / 365, iv[i])
        costs = trade_fee(S[i], credit)
        exit_j, exit_price, how = None, None, "expiry"
        if take_profit:
            for j in range(i + 1, j_exp):
                T = (expiry - dates[j]).days / 365
                mark = bs_put(S[j], K, T, iv[j])
                if mark <= TAKE_PROFIT * credit:
                    exit_j, exit_price, how = j, mark, "take-profit"
                    costs += trade_fee(S[j], mark)
                    break
        if exit_j is None:
            exit_j = j_exp
            exit_price = max(K - S[j_exp], 0.0)
            if exit_price > 0:
                costs += SETTLE_FEE * S[j_exp]
        pnl = credit - exit_price - costs
        out.append(dict(entry=dates[i], exit=dates[exit_j], S=S[i], K=K, iv=iv[i], hv=hv[i], credit=credit,
                        paid=exit_price, costs=costs, pnl=pnl, ret=pnl / K, how=how,
                        btc_move=S[exit_j] / S[i] - 1, assigned=how == "expiry" and exit_price > 0))
        i = exit_j + 1
    return pd.DataFrame(out)


def summary(t, label):
    eq = (1 + t["ret"]).cumprod()
    years = (t["exit"].iloc[-1] - t["entry"].iloc[0]).days / 365.25
    cagr = eq.iloc[-1] ** (1 / years) - 1
    dd = (eq / eq.cummax() - 1).min()
    worst = t.loc[t["pnl"].idxmin()]
    print(f"\n{label}: {len(t)} trades, {t['entry'].iloc[0]:%Y-%m-%d} .. {t['exit'].iloc[-1]:%Y-%m-%d}")
    print(f"  win rate {(t['pnl'] > 0).mean()*100:.1f}% | take-profit exits {(t['how'] == 'take-profit').mean()*100:.0f}% "
          f"| finished in the money {t['assigned'].mean()*100:.1f}%")
    print(f"  mean credit {t['credit'].mean()/t['K'].mean()*100:.2f}% of strike | mean costs {t['costs'].mean()/t['K'].mean()*100:.2f}% "
          f"| mean net {t['ret'].mean()*100:+.2f}% per trade")
    print(f"  return on collateral: CAGR {cagr*100:+.1f}% | max drawdown {dd*100:.1f}%")
    print(f"  worst trade: {worst['entry']:%Y-%m-%d}, BTC {worst['btc_move']*100:+.1f}%, loss {worst['ret']*100:.1f}% of collateral "
          f"= {-worst['pnl']/t['credit'].mean():.1f}x the mean premium")
    return dict(cagr=cagr, dd=dd)


def main():
    if not (DATA / "deribit_dvol_btc_1d.json.gz").exists():
        sys.exit("No data. Run fetch_public.py first.")
    df = load()
    print(f"BTC + DVOL: {len(df)} days {df.index[0]:%Y-%m-%d} .. {df.index[-1]:%Y-%m-%d} | strike {OTM*100:.0f}% OTM, "
          f"{DAYS} days, take-profit at {TAKE_PROFIT*100:.0f}%, premium at DVOL (no skew: conservative)")
    print(f"average DVOL {df['iv'].mean()*100:.0f}% vs realized 30-day vol {df['hv'].mean()*100:.0f}%")

    t = run(df)
    summary(t, "H-005 AS SPECIFIED (IV > realized filter, 50% take-profit)")
    t.to_csv(Path(__file__).resolve().parent / "results_h005_trades.csv", index=False)

    mean_prem, mean_cost = t["credit"].mean(), t["costs"].mean()
    max_loss = max(0.0, -t["pnl"].min())
    ratio_cost = mean_prem / mean_cost
    ratio_loss = max_loss / mean_prem
    print("\nDEATH RULE (from hipotesis-semilla.json, H-005)")
    checks = [
        (f"N >= {N_MIN} trades", len(t) >= N_MIN, f"{len(t)}"),
        (f"mean premium / mean costs >= {DEATH_PREMIUM_OVER_COSTS}", ratio_cost >= DEATH_PREMIUM_OVER_COSTS, f"{ratio_cost:.1f}"),
        (f"max loss / mean premium <= {DEATH_MAXLOSS_OVER_PREMIUM}", ratio_loss <= DEATH_MAXLOSS_OVER_PREMIUM, f"{ratio_loss:.1f}"),
    ]
    for text, ok, val in checks:
        print(f"   [{'PASS' if ok else 'FAIL'}] {text}   (measured: {val})")
    print(f"   => H-005 {'SURVIVES' if all(ok for _, ok, _ in checks) else 'DIES'}")

    print("\nFor context (not part of the death rule):")
    summary(run(df, iv_filter=False), "without the IV > realized filter")
    summary(run(df, take_profit=False), "without the 50% take-profit (hold to expiry)")
    btc = df["S"]
    yrs = (btc.index[-1] - btc.index[0]).days / 365.25
    print(f"\nBTC buy & hold over the same span: CAGR {((btc.iloc[-1]/btc.iloc[0])**(1/yrs)-1)*100:+.1f}%, "
          f"max drawdown {((btc/btc.cummax())-1).min()*100:.1f}%")


if __name__ == "__main__":
    main()
