"""AI Bot ki poori report: har trade ka date/time/points, sahi-galat, aur kab trade lena hai / kab nahi.
  python report.py           -> reports/report.html + reports/trades.csv   (bot har trade ke baad khud banata hai)
  python report.py --policy  -> AI ne kya seekha (uska behaviour) reports/ai_strategy.json  (roz training ke baad khud)
Website par: http://localhost:8000/report
"""
import glob, json, os, sqlite3, sys
import numpy as np
import pandas as pd

DB, OUT = "ledger.db", os.environ.get("REPORT_DIR", "reports")
MIN_SOLID, MIN_HINT = 100, 30            # kam se kam itne trades ho tabhi pattern par bharosa


def load():
    con = sqlite3.connect(DB)
    try:
        t = pd.read_sql("SELECT * FROM trades ORDER BY id", con)
        a = con.execute("SELECT balance, start FROM account").fetchone()
    except Exception:
        t, a = pd.DataFrame(), (10000.0, 10000.0)
    con.close()
    return t, a


def enrich(t):
    t = t.copy()
    ctx = t["ctx"].map(lambda x: json.loads(x) if isinstance(x, str) and x else {}) if "ctx" in t else pd.Series([{}] * len(t))
    for k in ("rsi", "ema_gap", "atr_pct", "vol_z"):
        t["f_" + k] = ctx.map(lambda c: c.get(k, np.nan)).astype(float)
    t["win"] = (t.pnl > 0).astype(int)
    t["date"], t["in_time"], t["out_time"] = t.t_in.str[:10], t.t_in.str[11:16], t.t_out.str[11:16]
    t["stock"] = t.symbol.str.split(".").str[0]
    t["tf"] = t["tf"].fillna("?") if "tf" in t else "?"
    t["side"] = t.direction
    t["ratio"] = "1:" + t.rr.map(lambda x: f"{x:g}")
    h = pd.to_numeric(t.t_in.str[11:13], errors="coerce").fillna(0).astype(int)
    t["hour"] = h.map(lambda x: f"{x:02d}:00-{x + 1:02d}:00")
    t["rsi_zone"] = pd.cut(t.f_rsi + 0.5, [-1, .3, .7, 2], labels=["RSI<30", "RSI 30-70", "RSI>70"])
    t["trend"] = pd.cut(t.f_ema_gap, [-np.inf, -.05, .05, np.inf], labels=["Downtrend", "Sideways", "Uptrend"])
    t["volatility"] = pd.cut(t.f_atr_pct, [-1, .15, .3, np.inf], labels=["Low vol", "Mid vol", "High vol"])
    t["volume"] = pd.cut(t.f_vol_z, [-np.inf, 1, np.inf], labels=["Normal volume", "Volume spike"])
    return t


def breakdown(t, col):
    g = t.groupby(col, observed=True)
    d = g.agg(trades=("pnl", "size"), sahi=("win", "sum"), avg_R=("r_multiple", "mean"),
              total_R=("r_multiple", "sum"), pnl_Rs=("pnl", "sum"))
    d["galat"] = d.trades - d.sahi
    d["win_%"] = (100 * d.sahi / d.trades).round(1)
    d["bharosa"] = np.where(d.trades >= MIN_SOLID, "pakka", np.where(d.trades >= MIN_HINT, "shuruati", "data kam"))
    return d.round(2)[["trades", "sahi", "galat", "win_%", "avg_R", "total_R", "pnl_Rs", "bharosa"]]


def _cls(v):
    return "up" if v >= 0 else "dn"


def log_html(t):
    rows = "".join(
        f"<tr><td>{r.date}</td><td>{r.in_time} → {r.out_time}</td><td>{r.stock}</td><td>{r.tf}</td><td>{r.side}</td>"
        f"<td>{r.ratio}</td><td class='{_cls(r.r_multiple)}'>{r.r_multiple:+.2f}</td><td class='{_cls(r.pnl)}'>{r.pnl:+.0f}</td>"
        f"<td class='{'up' if r.win else 'dn'}'>{'SAHI' if r.win else 'GALAT'}</td><td>{r.reason}</td></tr>"
        for r in t.iloc[::-1].itertuples())
    return ("<div style='overflow-x:auto'><table><tr><th>Date</th><th>Time (in → out)</th><th>Stock</th><th>TF</th><th>Side</th>"
            "<th>Ratio</th><th>Points (R)</th><th>₹</th><th>Result</th><th>Exit</th></tr>" + rows + "</table></div>")


def strategy_book(t):
    cols = {"tf": "Timeframe", "ratio": "Ratio", "side": "Side", "stock": "Stock", "hour": "Time slot", "rsi_zone": "RSI",
            "trend": "Trend", "volatility": "Volatility", "volume": "Volume"}
    good, bad = [], []
    for c, name in cols.items():
        d = breakdown(t, c)
        for k, r in d[d.trades >= MIN_HINT].iterrows():
            line = f"{name}: <b>{k}</b> → {int(r.trades)} trades, win {r['win_%']:.0f}%, avg {r.avg_R:+.2f} R ({r.bharosa})"
            (good if r.avg_R > 0.1 else bad if r.avg_R < -0.1 else []).append(line) if abs(r.avg_R) > 0.1 else None
    if not good and not bad:
        return (f"<p class='mut'>Abhi {len(t)} trades hain. Kisi pattern par bharosa karne ke liye us pattern ke kam se kam {MIN_HINT} "
                f"trades chahiye (pakka bharosa {MIN_SOLID}+ par). Tab tak neeche wala 'AI ne kya seekha' dekho.</p>")
    li = lambda L: "<ul>" + "".join(f"<li>{x}</li>" for x in L) + "</ul>" if L else "<p class='mut'>koi nahi</p>"
    return f"<div class='box'><b class='up'>✔ TRADE LO (in halat mein paisa bana):</b>{li(good)}<b class='dn'>✘ AVOID (in halat mein paisa gaya):</b>{li(bad)}</div>"


def ai_section():
    p = os.path.join(OUT, "ai_strategy.json")
    if not os.path.exists(p):
        return "<p class='mut'>Abhi nahi bana. Chalao: <code>python report.py --policy</code> (training ke baad bot roz khud banata hai)</p>"
    j = json.load(open(p))
    T = {k: pd.DataFrame(v).set_index("key") for k, v in j["tables"].items()}
    h, z, tr = T["hour"]["trade_%"], T["rsi_zone"], T["trend"]
    pts = [f"Overall AI <b>{j['trade_pct']:.0f}%</b> candles par trade leta hai, baaki par wait karta hai.",
           f"Sabse zyada trade: <b>{h.idxmax()}</b> ({h.max():.0f}%). Sabse kam: <b>{h.idxmin()}</b> ({h.min():.0f}%).",
           "RSI ke hisaab se: " + ", ".join(f"<b>{i}</b> par {r['long_%']:.0f}% LONG" for i, r in z.iterrows()) + ".",
           "Trend ke hisaab se: " + ", ".join(f"<b>{i}</b> par {r['long_%']:.0f}% LONG" for i, r in tr.iterrows()) + "."]
    out = "<div class='box'><ul>" + "".join(f"<li>{x}</li>" for x in pts) + f"</ul><span class='mut'>{j['n']:,} candles par model ka behaviour, {j['made']}. Ye batata hai AI kya karta hai, ye nahi ki wo sahi hai.</span></div>"
    for name, d in T.items():
        out += f"<h3>{name}</h3>" + d.round(1).to_html(border=0)
    return out


CSS = """body{background:#0b0e14;color:#d1d4dc;font:14px/1.5 system-ui,'Segoe UI',sans-serif;margin:auto;padding:24px;max-width:1200px}
h1{margin:0}h2{margin:30px 0 10px;border-bottom:1px solid #2a2e39;padding-bottom:6px}h3{margin:18px 0 6px;color:#9aa0ae;text-transform:capitalize}
.mut{color:#787b86}.cards{display:flex;gap:10px;flex-wrap:wrap;margin-top:14px}.card{background:#131722;border:1px solid #2a2e39;border-radius:8px;padding:10px 16px;min-width:120px}
.card b{display:block;font-size:21px}table{border-collapse:collapse;width:100%;background:#131722;font-size:13px}th,td{padding:6px 10px;border-bottom:1px solid #1c2030;text-align:right}
th:first-child,td:first-child{text-align:left}th{color:#787b86;font-weight:500}.up{color:#26a69a}.dn{color:#ef5350}
.box{background:#131722;border:1px solid #2a2e39;border-radius:8px;padding:12px 18px}ul{margin:6px 0 10px;padding-left:20px}code{background:#1c2030;padding:2px 6px;border-radius:4px}"""


def _save(html):
    for n in ("report.html", "index.html"):            # index.html = GitHub Pages ka mukhya page
        open(os.path.join(OUT, n), "w", encoding="utf-8").write(html)


def page(body):
    return f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>AI Bot Report</title><style>{CSS}</style></head><body>{body}</body></html>"


def build():
    os.makedirs(OUT, exist_ok=True)
    t, (bal, start) = load()
    head = f"<h1>🤖 AI Bot Report</h1><div class='mut'>Update: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</div>"
    if t.empty:
        _save(page(head + "<p>Abhi koi trade nahi hua.</p><h2>AI ne kya seekha</h2>" + ai_section()))
        return
    t = enrich(t)
    t.drop(columns=[c for c in ("win", "ctx") if c in t]).to_csv(os.path.join(OUT, "trades.csv"), index=False)
    n, w = len(t), int(t.win.sum())
    gp, gl = t.pnl[t.pnl > 0].sum(), -t.pnl[t.pnl <= 0].sum()
    eq = np.r_[start, t.balance_after.values]
    dd = ((np.maximum.accumulate(eq) - eq) / np.maximum.accumulate(eq)).max() * 100
    card = lambda k, v, c="": f"<div class='card'><span class='mut'>{k}</span><b class='{c}'>{v}</b></div>"
    cards = "".join([card("Balance", f"₹{bal:,.0f}", _cls(bal - start)), card("Return", f"{(bal / start - 1) * 100:+.2f}%", _cls(bal - start)),
                     card("Trades", n), card("Sahi / Galat", f"<span class='up'>{w}</span> / <span class='dn'>{n - w}</span>"),
                     card("Win rate", f"{100 * w / n:.0f}%"), card("Total points", f"{t.r_multiple.sum():+.2f} R", _cls(t.r_multiple.sum())),
                     card("Avg / trade", f"{t.r_multiple.mean():+.2f} R"), card("Profit factor", f"{gp / gl:.2f}" if gl > 0 else "–"),
                     card("Max drawdown", f"{dd:.1f}%")])
    daily = t.groupby("date").agg(trades=("pnl", "size"), sahi=("win", "sum"), points_R=("r_multiple", "sum"), pnl_Rs=("pnl", "sum"))
    daily["galat"] = daily.trades - daily.sahi
    daily = daily.round(2)[["trades", "sahi", "galat", "points_R", "pnl_Rs"]].iloc[::-1]
    body = [head, f"<div class='cards'>{cards}</div>", "<h2>Roz ka hisaab</h2>", daily.to_html(border=0),
            "<h2>Har trade (date, time, points, sahi/galat)</h2>", log_html(t),
            f"<h2>Kab trade lena hai, kab nahi (asli trades se, {MIN_HINT}+ trades wale pattern)</h2>", strategy_book(t),
            "<h2>AI ne kya seekha (uska behaviour)</h2>", ai_section(), "<h2>Detail: kisme kitna chala</h2>"]
    for c, name in {"tf": "Timeframe", "ratio": "Ratio", "side": "Long / Short", "stock": "Stock", "hour": "Time slot", "rsi_zone": "RSI",
                    "trend": "Trend", "volatility": "Volatility", "volume": "Volume", "reason": "Exit reason"}.items():
        body += [f"<h3>{name}</h3>", breakdown(t, c).to_html(border=0)]
    _save(page("".join(body)))


def policy_profile(model=None):
    """Model ko poore purane data par chalao aur dekho: kab trade leta hai, kab wait, LONG ya SHORT."""
    from numpy.lib.stride_tricks import sliding_window_view
    import trading_ai as ta
    if model is None:
        from stable_baselines3 import PPO
        model = PPO.load("models/ppo_latest")
    try:
        tfs = list(json.load(open("models/tf_scores.json"))["scores"])
    except Exception:
        tfs = ["15m", "30m", "1h"]
    latest = {}
    for p in sorted(glob.glob("data_cache/in_*_*_*.csv")):
        tk, tf, _ = os.path.basename(p)[3:-4].rsplit("_", 2)
        if tf in tfs:
            latest[(tk, tf)] = p                                   # sabse naya cache
    W, parts = ta.WINDOW_DEFAULT, []
    dirs = np.array([0] + [a[0] for a in ta.ACTIONS[1:]])
    rrs = np.array([0] + [a[2] for a in ta.ACTIONS[1:]])
    sms = np.array([0] + [a[1] for a in ta.ACTIONS[1:]])
    for (tk, tf), p in latest.items():
        df = ta.add_features(pd.read_csv(p, index_col=0, parse_dates=True))
        if len(df) < W + 50:
            continue
        F = df[ta.FEATURES].values.astype(np.float32)
        win = sliding_window_view(F, (W, F.shape[1])).reshape(-1, W * F.shape[1])
        obs = np.clip(np.c_[win, np.zeros((len(win), 3)), np.full((len(win), 1), ta.tf_code(tf))], -10, 10).astype(np.float32)
        act = np.asarray(model.predict(obs, deterministic=True)[0]).astype(int)
        d = df.iloc[W - 1:]
        parts.append(pd.DataFrame({"tf": tf, "hour": [f"{h:02d}:00-{h + 1:02d}:00" for h in d.index.hour], "trade": act > 0,
                                   "long": dirs[act] == 1, "rr": rrs[act], "sl": sms[act],
                                   "rsi_zone": pd.cut(d["rsi"].values + 0.5, [-1, .3, .7, 2], labels=["RSI<30", "RSI 30-70", "RSI>70"]),
                                   "trend": pd.cut(d["ema_gap"].values, [-np.inf, -.05, .05, np.inf], labels=["Downtrend", "Sideways", "Uptrend"]),
                                   "volatility": pd.cut(d["atr_pct"].values, [-1, .15, .3, np.inf], labels=["Low vol", "Mid vol", "High vol"])}))
    if not parts:
        sys.exit("Data cache nahi mila (pehle training chalao).")
    A = pd.concat(parts, ignore_index=True)

    def tab(col):
        g = A.groupby(col, observed=True)
        r = pd.DataFrame({"trade_%": g.trade.mean() * 100, "long_%": g.apply(lambda x: x.long[x.trade].mean() * 100 if x.trade.any() else np.nan),
                          "candles": g.size()})
        return r.round(1).reset_index().rename(columns={col: "key"}).to_dict("records")
    tr = A[A.trade]
    tables = {c: tab(c) for c in ("tf", "hour", "rsi_zone", "trend", "volatility")}
    tables["ratio_chuna"] = [{"key": f"1:{k:g}", "trade_%": round(v * 100, 1), "long_%": np.nan, "candles": int((tr.rr == k).sum())} for k, v in tr.rr.value_counts(normalize=True).items()]
    tables["stoploss_chuna"] = [{"key": f"{k:g}x ATR", "trade_%": round(v * 100, 1), "long_%": np.nan, "candles": int((tr.sl == k).sum())} for k, v in tr.sl.value_counts(normalize=True).items()]
    os.makedirs(OUT, exist_ok=True)
    json.dump({"made": str(pd.Timestamp.now().date()), "n": len(A), "trade_pct": float(A.trade.mean() * 100), "tables": tables},
              open(os.path.join(OUT, "ai_strategy.json"), "w"))
    print("AI ka behaviour save:", os.path.join(OUT, "ai_strategy.json"))


if __name__ == "__main__":
    if "--policy" in sys.argv:
        policy_profile()
    build()
    print("Report ban gayi:", os.path.join(OUT, "report.html"))
