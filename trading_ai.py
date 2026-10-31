"""
AI PAPER TRADER - poora project ek hi file mein
================================================
Install (ek baar):
    pip install numpy pandas yfinance ccxt gymnasium stable-baselines3 torch streamlit

Chalane ke tareeke:
    python trading_ai.py --synthetic --steps 20000     # chhota test (nakli data)
    python trading_ai.py                               # BTC/USDT 15m, 2 saal asli data
    python trading_ai.py --days 1095                   # 3 saal
    python trading_ai.py --exchange okx                # Binance na chale to
    python trading_ai.py --market india                # NSE stocks, saare timeframes (1m,2m,5m,15m,30m,1h), din ke end par square-off
    python trading_ai.py --market india --tfs 5m,15m,3m --steps 500000
    python trading_ai.py --market india --tickers TCS,INFY --suffix .BO    # chune hue stocks, BSE
    python trading_ai.py --market india --swing        # raat bhar position rakhne wala
    streamlit run trading_ai.py                        # DASHBOARD kholo
"""
import argparse, datetime, os, sqlite3, sys, time
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

DB = "journal.db"
FEATURES = ["ret1", "ret4", "rsi", "ema_gap", "atr_pct", "vol_z", "range_atr"]
START_BALANCE = 10000.0
WINDOW_DEFAULT = 10


def tf_minutes(tf):                      # "15m" -> 15, "1h" -> 60
    return int(tf[:-1]) * (60 if tf.endswith("h") else 1)


def tf_code(tf):                         # AI ko batane ke liye ki ye kaunsa timeframe hai (0 = 1m ... 1 = 1h)
    return float(np.log(max(tf_minutes(tf), 1)) / np.log(60))


# ============================================================ DATA
def download_yf(ticker="BTC-USD", interval="15m", period="60d"):
    import yfinance as yf
    df = yf.download(ticker, interval=interval, period=period, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()


def download_ccxt(symbol="BTC/USDT", timeframe="15m", days=730, exchange="binance"):
    """Exchange se saalon ka free data. Pehli baar download, phir CSV cache se."""
    os.makedirs("data_cache", exist_ok=True)
    path = f"data_cache/{exchange}_{symbol.replace('/', '')}_{timeframe}_{days}d.csv"
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < 86400:
        print("Cache se data le raha hoon:", path)
        return pd.read_csv(path, index_col=0, parse_dates=True)
    import ccxt
    ex = getattr(ccxt, exchange)({"enableRateLimit": True})
    tf_ms = ex.parse_timeframe(timeframe) * 1000
    since = ex.milliseconds() - days * 86400 * 1000
    rows = []
    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not batch:
            break
        rows += batch
        print(f"\rDownload: {len(rows)} candles", end="", flush=True)
        if batch[-1][0] + 1 <= since or batch[-1][0] >= ex.milliseconds() - tf_ms:
            break
        since = batch[-1][0] + 1
    print()
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"]).drop_duplicates("ts")
    df.index = pd.to_datetime(df.pop("ts"), unit="ms")
    df = df.sort_index().iloc[:-1]
    df.to_csv(path)
    return df


AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
NATIVE = {"1m", "2m", "5m", "15m", "30m", "1h"}


def _yf_range(ticker, interval, period):
    import yfinance as yf
    if interval == "1m":     # Yahoo: 1m sirf 7-7 din ke tukdon mein, pichhle ~30 din tak
        end, parts = datetime.date.today() + datetime.timedelta(days=1), []
        for _ in range(4):
            st = end - datetime.timedelta(days=7)
            parts.append(yf.download(ticker, start=str(st), end=str(end), interval="1m", auto_adjust=True, progress=False))
            end = st
            time.sleep(0.4)
        df = pd.concat([p for p in parts if len(p)])
    else:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
    if df.index.tz is not None:
        df.index = df.index.tz_convert("Asia/Kolkata").tz_localize(None)   # IST wall time
    return df[~df.index.duplicated()].sort_index()


def fetch_india(ticker, tf):
    """Kisi bhi timeframe ka data. Yahoo limits: 1m ~28 din, 2m-30m 60 din, 1h ~2 saal. Roz ka cache."""
    os.makedirs("data_cache", exist_ok=True)
    path = f"data_cache/in_{ticker}_{tf}_{datetime.date.today()}.csv"
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0, parse_dates=True)
    if tf in NATIVE:
        df = _yf_range(ticker, tf, "700d" if tf == "1h" else "60d")
    else:                    # 3m, 10m, 45m jaise custom: chhoti candles jodkar, market 9:15 se align
        n = tf_minutes(tf)
        df = _yf_range(ticker, "5m" if n % 5 == 0 else "1m", "60d")
        df = df.resample(f"{n}min", offset=f"{555 % n}min").agg(AGG).dropna()
    df.to_csv(path)
    return df


def synthetic(n=20000, seed=0):
    """Sirf testing ke liye nakli data."""
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.002, n) + 0.0004 * np.sin(np.arange(n) / 200)
    close = 30000 * np.exp(np.cumsum(r))
    open_ = np.r_[close[0], close[:-1]]
    hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.001, n)))
    lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.001, n)))
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    return pd.DataFrame({"open": open_, "high": hi, "low": lo, "close": close,
                         "volume": rng.lognormal(5, 0.5, n)}, index=idx)


def add_features(df):
    d = df.copy()
    c = d["close"]
    d["ret1"] = c.pct_change() * 100
    d["ret4"] = c.pct_change(4) * 100
    delta = c.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    dn = (-delta.clip(upper=0)).rolling(14).mean()
    d["rsi"] = (100 - 100 / (1 + up / (dn + 1e-12))) / 100 - 0.5
    d["ema_gap"] = (c.ewm(span=10).mean() / c.ewm(span=50).mean() - 1) * 100
    tr = pd.concat([d["high"] - d["low"], (d["high"] - c.shift()).abs(),
                    (d["low"] - c.shift()).abs()], axis=1).max(axis=1)
    d["atr"] = tr.rolling(14).mean()
    d["atr_pct"] = d["atr"] / c * 100
    d["vol_z"] = (d["volume"] - d["volume"].rolling(50).mean()) / (d["volume"].rolling(50).std() + 1e-9)
    d["range_atr"] = (d["high"] - d["low"]) / (d["atr"] + 1e-9)
    d = d.dropna()
    d[FEATURES] = d[FEATURES].clip(-5, 5)
    return d


# ============================================================ PAPER BROKER + ENVIRONMENT
SL_MULTS = [1.0, 2.0]   # stoploss = ATR x ye
RRS = [1.0, 2.0, 3.0]   # target = risk x ye
ACTIONS = [None] + [(d, s, r) for d in (1, -1) for s in SL_MULTS for r in RRS]  # 0 = wait


class TradingEnv(gym.Env):
    """Virtual account. Har trade mein risk = balance ka risk_pct. Reward = R-multiple (fees ke baad)."""

    def __init__(self, df, start_balance=START_BALANCE, risk_pct=0.01, fee=0.0005, slippage=0.0002,
                 window=10, max_hold=48, max_leverage=5.0, random_start=True, episode_len=2000, intraday=False, tf="1h"):
        super().__init__()
        self.df, self.n = df, len(df)
        self.h, self.l, self.c = (df[k].values for k in ("high", "low", "close"))
        self.atr = df["atr"].values
        self.f = df[FEATURES].values.astype(np.float32)
        self.times = df.index.astype(str).values
        days = np.asarray(df.index.strftime("%Y-%m-%d"))
        self.newday = np.r_[False, days[1:] != days[:-1]]
        self.intraday, self.tfc = intraday, tf_code(tf)
        self.start_balance, self.risk_pct, self.fee, self.slip = start_balance, risk_pct, fee, slippage
        self.window, self.max_hold, self.max_lev = window, max_hold, max_leverage
        self.random_start, self.episode_len = random_start, episode_len
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(-10, 10, (window * len(FEATURES) + 4,), np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        lo = self.window
        if self.random_start:
            hi = max(lo + 1, self.n - self.episode_len - 1)
            self.t = int(self.np_random.integers(lo, hi))
            self.end = min(self.n - 1, self.t + self.episode_len)
        else:
            self.t, self.end = lo, self.n - 1
        self.balance = self.start_balance
        self.pos = None
        self.trades, self.equity = [], []
        return self._obs(), {}

    def _unreal(self):
        if not self.pos:
            return 0.0
        p = self.pos
        return p["dir"] * (self.c[self.t] - p["entry"]) * p["qty"]

    def _obs(self):
        w = self.f[self.t - self.window + 1: self.t + 1].flatten()
        p = self.pos
        extra = [1.0 * p["dir"], self._unreal() / p["risk"], p["held"] / self.max_hold] if p else [0, 0, 0]
        return np.clip(np.r_[w, extra, self.tfc], -10, 10).astype(np.float32)

    def _open(self, a):
        d, s, rr = ACTIONS[a]
        entry = self.c[self.t] * (1 + self.slip * d)
        sl_dist = max(self.atr[self.t] * s, entry * 0.0005)
        risk = self.balance * self.risk_pct
        qty = min(risk / sl_dist, self.balance * self.max_lev / entry)
        risk = qty * sl_dist
        self.pos = dict(dir=d, sl_mult=s, rr=rr, entry=entry, qty=qty, risk=risk, held=0,
                        sl=entry - d * sl_dist, tp=entry + d * sl_dist * rr,
                        t_in=self.times[self.t], rsi=float(self.f[self.t][2]) + 0.5,
                        atr_pct=float(self.f[self.t][4]))

    def _close(self, price, reason):
        p = self.pos
        price = price * (1 - self.slip * p["dir"])
        pnl = p["dir"] * (price - p["entry"]) * p["qty"] - self.fee * (p["entry"] + price) * p["qty"]
        self.balance += pnl
        r_mult = pnl / p["risk"]
        self.trades.append(dict(entry_time=p["t_in"], exit_time=self.times[self.t],
                                direction="LONG" if p["dir"] == 1 else "SHORT",
                                sl_mult=p["sl_mult"], rr=p["rr"], entry=p["entry"], exit=price,
                                sl=p["sl"], tp=p["tp"], qty=p["qty"], risk_amt=p["risk"], pnl=pnl,
                                r_multiple=r_mult, reason=reason, balance_after=self.balance,
                                rsi_at_entry=p["rsi"], atr_pct_at_entry=p["atr_pct"]))
        self.pos = None
        return r_mult

    def step(self, action):
        reward = 0.0
        if self.pos is None and action > 0 and not (self.intraday and self.newday[self.t + 1]):
            self._open(action)             # intraday mein din ki aakhri candle par naya trade nahi
        self.t += 1
        if self.pos and self.intraday and self.newday[self.t]:
            self.t -= 1
            reward = self._close(self.c[self.t], "EOD")   # din ke end par square-off (raat bhar position nahi)
            self.t += 1
        if self.pos:
            p = self.pos
            p["held"] += 1
            hi, lo = self.h[self.t], self.l[self.t]
            sl_hit = lo <= p["sl"] if p["dir"] == 1 else hi >= p["sl"]
            tp_hit = hi >= p["tp"] if p["dir"] == 1 else lo <= p["tp"]
            if sl_hit:                      # dono ek candle mein hon to SL maano (conservative)
                reward = self._close(p["sl"], "SL")
            elif tp_hit:
                reward = self._close(p["tp"], "TP")
            elif p["held"] >= self.max_hold:
                reward = self._close(self.c[self.t], "TIMEOUT")
        done = self.t >= self.end or self.balance < self.start_balance * 0.5
        if done and self.pos:
            reward += self._close(self.c[self.t], "END")
        self.equity.append((self.times[self.t], self.balance + self._unreal()))
        return self._obs(), float(reward), bool(done), False, {}


# ============================================================ JOURNAL (SQLite)
def save_run(run, trades, equity, start_balance, db=DB):
    con = sqlite3.connect(db)
    for tbl in ("trades", "equity", "runs"):
        if con.execute("SELECT name FROM sqlite_master WHERE name=?", (tbl,)).fetchone():
            con.execute(f"DELETE FROM {tbl} WHERE run=?", (run,))
    t = pd.DataFrame(trades)
    t.insert(0, "run", run)
    e = pd.DataFrame(equity, columns=["time", "equity"])
    e.insert(0, "run", run)
    t.to_sql("trades", con, if_exists="append", index=False)
    e.to_sql("equity", con, if_exists="append", index=False)
    pd.DataFrame([{"run": run, "start_balance": start_balance}]).to_sql("runs", con, if_exists="append", index=False)
    con.commit()
    con.close()


def list_runs(db=DB):
    try:
        con = sqlite3.connect(db)
        r = pd.read_sql("SELECT DISTINCT run FROM runs", con)["run"].tolist()
        con.close()
        return r
    except Exception:
        return []


def load_run(run, db=DB):
    con = sqlite3.connect(db)
    t = pd.read_sql("SELECT * FROM trades WHERE run=?", con, params=(run,))
    e = pd.read_sql("SELECT * FROM equity WHERE run=?", con, params=(run,))
    s = pd.read_sql("SELECT start_balance FROM runs WHERE run=?", con, params=(run,))
    con.close()
    return t, e, float(s["start_balance"].iloc[0]) if len(s) else START_BALANCE


# ============================================================ TRAINING
INDIA_DEFAULT = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN", "ITC", "LT"]


class MultiEnv(gym.Env):
    """Har episode mein random (stock, timeframe) chunta hai, taaki AI kisi ek ko ratta na maare."""

    def __init__(self, envs):
        super().__init__()
        self.envs = envs
        self.observation_space, self.action_space = envs[0].observation_space, envs[0].action_space

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cur = self.envs[int(self.np_random.integers(len(self.envs)))]
        return self.cur.reset()

    def step(self, a):
        return self.cur.step(a)


def train_main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="crypto", choices=["crypto", "india"])
    ap.add_argument("--tickers", default=",".join(INDIA_DEFAULT))
    ap.add_argument("--tfs", default="1m,2m,5m,15m,30m,1h")          # india: in sab timeframes par seekho (3m, 10m, 45m bhi chalega)
    ap.add_argument("--suffix", default=".NS")                       # .NS = NSE, .BO = BSE
    ap.add_argument("--finetune", action="store_true")               # purana model load karke poore (naye) data par aage seekho
    ap.add_argument("--swing", action="store_true")                  # india: raat bhar position rakhne do
    ap.add_argument("--fee", type=float, default=None)
    ap.add_argument("--source", default="ccxt", choices=["ccxt", "yf"])
    ap.add_argument("--symbol", default="BTC/USDT")
    ap.add_argument("--exchange", default="binance")
    ap.add_argument("--days", type=int, default=730)
    ap.add_argument("--ticker", default="BTC-USD")
    ap.add_argument("--interval", default="15m")
    ap.add_argument("--period", default="60d")
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--balance", type=float, default=START_BALANCE)
    a = ap.parse_args()

    from stable_baselines3 import PPO
    india = a.market == "india"
    fee = a.fee if a.fee is not None else (0.0003 if india else 0.0005)
    intraday = india and not a.swing
    tfs = [x.strip() for x in a.tfs.split(",") if x.strip()]

    if a.synthetic:
        raws = {("SYN1", "15m"): synthetic(seed=0), ("SYN2", "5m"): synthetic(seed=1)} if india else {("SYN", a.interval): synthetic()}
    elif india:
        raws = {}
        for tk in [x.strip().upper() for x in a.tickers.split(",") if x.strip()]:
            for tf in tfs:
                try:
                    raws[(tk, tf)] = fetch_india(tk + a.suffix, tf)
                    print(f"{tk}{a.suffix} {tf}: {len(raws[(tk, tf)])} candles")
                except Exception as ex:
                    print(f"{tk}{a.suffix} {tf}: skip ({ex})")
    elif a.source == "ccxt":
        raws = {(a.symbol, a.interval): download_ccxt(a.symbol, a.interval, a.days, a.exchange)}
    else:
        raws = {(a.ticker, a.interval): download_yf(a.ticker, a.interval, a.period)}

    sets = {}
    for (k, tf), raw in raws.items():
        df = add_features(raw)
        if len(df) < 300:
            print(f"{k} {tf}: data kam hai, skip")
            continue
        cut = len(df) if a.finetune else int(len(df) * 0.7)   # pehle 70% train, baaki 30% unseen test
        sets[(k, tf)] = (df.iloc[:cut], df.iloc[cut:])
    if not sets:
        sys.exit("Koi data nahi mila. Internet ya symbols check karo.")

    mk = lambda d, tf, **kw: TradingEnv(d, start_balance=a.balance, fee=fee, intraday=intraday, tf=tf, **kw)
    env = MultiEnv([mk(tr, tf) for (_, tf), (tr, _) in sets.items()])
    os.makedirs("models", exist_ok=True)
    if a.finetune and os.path.exists("models/ppo_latest.zip"):
        import shutil
        shutil.copy("models/ppo_latest.zip", f"models/ppo_{datetime.date.today()}.zip")   # backup
        model = PPO.load("models/ppo_latest", env=env)
        print("Purana model load kiya, purane + naye data par aage seekh raha hai")
        model.learn(total_timesteps=a.steps, reset_num_timesteps=False)
    else:
        model = PPO("MlpPolicy", env, verbose=0, ent_coef=0.01, learning_rate=3e-4, n_steps=2048, batch_size=256)
        print(f"Training shuru: {a.steps} steps, {len(sets)} (stock, timeframe) combos, intraday={intraday}, fee={fee*100:.3f}% per side")
        model.learn(total_timesteps=a.steps)
    model.save("models/ppo_latest")
    if a.finetune:
        print("Fine-tune ho gaya. Asli test ab website par bot ke live paper trades hain.")
        return

    def evaluate(d, tf, name):
        e = mk(d, tf, random_start=False)
        obs, _ = e.reset()
        done = False
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = e.step(int(act))
        save_run(name, e.trades, e.equity, a.balance)
        print(f"[{name}] trades={len(e.trades)} final balance=Rs {e.balance:,.0f}")
        return e.balance

    if len(sets) == 1:
        (k, tf), (tr, te) = next(iter(sets.items()))
        evaluate(tr, tf, "train_period")
        evaluate(te, tf, "unseen_test")
    else:
        by = {}
        for (k, tf), (_, te) in sets.items():
            by.setdefault(tf, []).append(evaluate(te, tf, f"unseen_{tf}_{k}"))
        scores = {tf: float(np.mean(v)) for tf, v in by.items()}
        print(f"\nTimeframe ke hisaab se average unseen balance (shuru Rs {a.balance:,.0f}):")
        for tf in sorted(scores, key=tf_minutes):
            print(f"  {tf:>4}: Rs {scores[tf]:,.0f}")
        import json
        json.dump({"start": a.balance, "scores": scores}, open("models/tf_scores.json", "w"))   # bot isse apne timeframes chunega
    print("Done. Ab chalao: streamlit run trading_ai.py")


# ============================================================ DASHBOARD
def dashboard_main():
    import streamlit as st
    st.set_page_config(page_title="AI Paper Trader", layout="wide")
    st.title("AI Paper Trader - Virtual Account")

    runs = list_runs()
    if not runs:
        st.warning("Journal khali hai. Pehle `python trading_ai.py` chalao (training).")
        st.stop()
    run = st.sidebar.selectbox("Run chuno", runs, index=runs.index("unseen_test") if "unseen_test" in runs else 0)
    st.sidebar.caption("unseen_test = jo data AI ne training mein nahi dekha. Asli imtihaan yahi hai.")
    t, e, start = load_run(run)
    if t.empty:
        st.info("Is run mein koi trade nahi hua.")
        st.stop()

    bal = t["balance_after"].iloc[-1]
    eq = e["equity"]
    dd = ((eq.cummax() - eq) / eq.cummax()).max() * 100
    c = st.columns(5)
    c[0].metric("Virtual Balance", f"Rs {bal:,.0f}", f"{bal - start:+,.0f}")
    c[1].metric("Total Trades", len(t))
    c[2].metric("Win Rate", f"{(t['pnl'] > 0).mean() * 100:.1f}%")
    c[3].metric("Avg R / trade", f"{t['r_multiple'].mean():.2f}")
    c[4].metric("Max Drawdown", f"{dd:.1f}%")

    st.subheader("Equity curve (account balance)")
    e["time"] = pd.to_datetime(e["time"])
    st.line_chart(e.set_index("time")["equity"])

    l, r = st.columns(2)
    with l:
        st.subheader("Ratio ke hisaab se")
        g = t.groupby(["direction", "rr"]).agg(
            trades=("pnl", "size"), win_pct=("pnl", lambda x: (x > 0).mean() * 100),
            avg_R=("r_multiple", "mean"), total_pnl=("pnl", "sum")).round(2)
        st.dataframe(g)
    with r:
        st.subheader("Exit reason")
        st.bar_chart(t["reason"].value_counts())

    st.subheader("Strategy patterns (100+ trades wale hi maane jayenge)")
    t["rsi_zone"] = pd.cut(t["rsi_at_entry"], [0, .3, .7, 1], labels=["RSI<30", "RSI 30-70", "RSI>70"])
    p = t.groupby(["rsi_zone", "direction"], observed=True).agg(
        trades=("pnl", "size"), avg_R=("r_multiple", "mean"),
        win_pct=("pnl", lambda x: (x > 0).mean() * 100)).round(2)
    p["verdict"] = p.apply(lambda x: "kam data (luck ho sakta)" if x.trades < 100
                           else ("achha" if x.avg_R > 0 else "kharab"), axis=1)
    st.dataframe(p)

    st.subheader("Saare trades (journal)")
    st.dataframe(t.drop(columns=["run", "rsi_zone"]).iloc[::-1], width="stretch")


# ============================================================ ENTRY
def _under_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if _under_streamlit():
    dashboard_main()
elif __name__ == "__main__":
    train_main()
