"""BharatCharts backend. Chalao: pip install flask yfinance pandas  ->  python server.py  ->  http://localhost:8000"""
import json, os, re, sqlite3, threading, time
from datetime import datetime, timezone, timedelta
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder=None)

# timeframe -> (yfinance interval, period). Yahoo limits: 1m=7d, 5m/15m/30m=60d, 1h=730d
TF = {"1m": ("1m", "7d"), "2m": ("2m", "60d"), "5m": ("5m", "60d"), "15m": ("15m", "60d"), "30m": ("30m", "60d"),
      "1h": ("1h", "730d"), "4h": ("1h", "730d"), "1D": ("1d", "max"), "1W": ("1wk", "max"), "1M": ("1mo", "max")}
cache = {}
SUFFIX = os.environ.get("SUFFIX", ".NS")            # .NS = NSE (Yahoo par bharosemand), .BO = BSE (aksar fail hota hai)
BOT_KEY = os.environ.get("BOT_KEY", "bot-secret")   # public karne se pehle badal dena
RISK_PCT, FEE, SLIP, MAX_LEV, MAX_OPEN, START = 0.01, 0.0003, 0.0002, 5.0, 3, 10000.0
DB, lock, beat = "ledger.db", threading.Lock(), {"t": 0, "msg": "bot abhi tak chala nahi"}


def cached(key, ttl, fn):
    """ttl seconds tak yaad rakho. Yahoo fail/khaali de (rate-limit) to purana data hi de do."""
    hit = cache.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    try:
        val = fn()
    except Exception:
        if hit:
            return hit[1]
        raise
    if val is None and hit:
        return hit[1]
    cache[key] = (time.time(), val)
    return val


def sym(s):
    s = s.upper().strip()
    return s if "." in s or s.startswith("^") else s + SUFFIX


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/api/candles")
def candles():
    s, tf = sym(request.args.get("symbol", "RELIANCE")), request.args.get("tf", "15m")
    der = re.fullmatch(r"(\d+)m", tf) if tf not in TF else None   # 3m, 10m, 45m jaise custom timeframe
    n_der = int(der.group(1)) if der else 0
    interval, period = (("5m", "60d") if n_der % 5 == 0 else ("1m", "7d")) if n_der else TF.get(tf, TF["15m"])
    period = request.args.get("period") or period

    def fetch():
        for _ in range(2):                    # Yahoo kabhi khaali deta hai, ek baar dobara try
            df = yf.Ticker(s).history(period=period, interval=interval, auto_adjust=False)
            if not df.empty:
                break
            time.sleep(0.8)
        if df.empty:
            return None
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        if tf == "4h":
            df = df.resample("4h", offset="1h15min").agg(
                {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
        if n_der:
            df = df.resample(f"{n_der}min", offset=f"{555 % n_der}min").agg(
                {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
        daily = tf in ("1D", "1W", "1M")
        out = []
        for ts, r in df.iterrows():
            t = ts.strftime("%Y-%m-%d") if daily else int(ts.timestamp()) + 19800   # IST dikhane ke liye
            out.append({"time": t, "open": round(r.Open, 2), "high": round(r.High, 2),
                        "low": round(r.Low, 2), "close": round(r.Close, 2), "volume": int(r.Volume)})
        return out

    try:
        data = cached(f"c:{s}:{tf}:{period}", 3 if tf in ("1m", "5m") else 15, fetch)
    except Exception as e:
        return jsonify(error=str(e)), 502
    if not data:
        return jsonify(error=f"{s} ka data nahi mila (symbol check karo)"), 404
    return jsonify(candles=data)


@app.route("/api/quotes")
def quotes():
    syms = [sym(x) for x in request.args.get("symbols", "").split(",") if x][:60]

    def fetch():
        df = yf.download(syms, period="5d", interval="1d", progress=False, auto_adjust=False,
                         group_by="ticker", threads=True)
        out = {}
        for s in syms:
            try:
                c = df[s]["Close"].dropna()
                out[s] = {"price": round(float(c.iloc[-1]), 2), "chg": round(float(c.iloc[-1] / c.iloc[-2] - 1) * 100, 2)}
            except Exception:
                pass
        return out

    try:
        return jsonify(cached("q:" + ",".join(syms), 15, fetch))
    except Exception as e:
        return jsonify(error=str(e)), 502


# ======================= BOT LEDGER (website ka hisaab-kitaab) =======================
def now_ist():
    return datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime("%Y-%m-%d %H:%M:%S")


def db():
    con = sqlite3.connect(DB, timeout=10)
    con.row_factory = sqlite3.Row
    con.executescript(f"""
    CREATE TABLE IF NOT EXISTS account(id INTEGER PRIMARY KEY, balance REAL, start REAL);
    CREATE TABLE IF NOT EXISTS positions(symbol TEXT PRIMARY KEY, dir INT, qty REAL, entry REAL, sl REAL, tp REAL,
        risk REAL, rr REAL, sl_mult REAL, t_in TEXT, candle_t TEXT, rsi REAL, atr_pct REAL);
    CREATE TABLE IF NOT EXISTS trades(id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, direction TEXT, qty REAL,
        entry REAL, exit REAL, sl REAL, tp REAL, risk_amt REAL, rr REAL, pnl REAL, r_multiple REAL, result TEXT,
        reason TEXT, balance_after REAL, t_in TEXT, t_out TEXT);
    INSERT OR IGNORE INTO account VALUES(1, {START}, {START});""")
    for tbl, cols in (("positions", ["tf TEXT", "ctx TEXT"]), ("trades", ["tf TEXT", "ctx TEXT", "sl_mult REAL"])):
        for c in cols:                                   # purane database mein naye column jodo
            try:
                con.execute(f"ALTER TABLE {tbl} ADD COLUMN {c}")
            except sqlite3.OperationalError:
                pass
    con.commit()
    return con


def denied():
    return request.headers.get("X-Bot-Key") != BOT_KEY


@app.route("/api/bot/open", methods=["POST"])
def bot_open():
    if denied():
        return jsonify(error="key galat"), 403
    j, err = request.get_json(), None
    s, d = sym(j["symbol"]), int(j["direction"])
    with lock:
        con = db()
        bal = con.execute("SELECT balance FROM account").fetchone()[0]
        if con.execute("SELECT 1 FROM positions WHERE symbol=?", (s,)).fetchone():
            err = "position pehle se open hai"
        elif con.execute("SELECT COUNT(*) FROM positions").fetchone()[0] >= MAX_OPEN:
            err = f"max {MAX_OPEN} positions ek saath"
        if err:
            con.close()
            return jsonify(error=err), 409
        entry = float(j["price"]) * (1 + SLIP * d)
        dist, rr = float(j["sl_dist"]), float(j["rr"])
        qty = min(bal * RISK_PCT / dist, bal * MAX_LEV / entry)
        con.execute("INSERT INTO positions(symbol,dir,qty,entry,sl,tp,risk,rr,sl_mult,t_in,candle_t,rsi,atr_pct,tf,ctx) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (s, d, qty, entry, entry - d * dist, entry + d * dist * rr, qty * dist, rr, float(j.get("sl_mult", 0)),
                     now_ist(), str(j.get("candle_t", "")), float(j.get("rsi", 0)), float(j.get("atr_pct", 0)), j.get("tf", ""), json.dumps(j.get("ctx", {}))))
        con.commit(); con.close()
    return jsonify(ok=True, qty=qty, entry=entry)


@app.route("/api/bot/close", methods=["POST"])
def bot_close():
    if denied():
        return jsonify(error="key galat"), 403
    j = request.get_json()
    s = sym(j["symbol"])
    with lock:
        con = db()
        p = con.execute("SELECT * FROM positions WHERE symbol=?", (s,)).fetchone()
        if not p:
            con.close()
            return jsonify(error="position nahi mili"), 404
        d = p["dir"]
        px = float(j["price"]) * (1 - SLIP * d)
        pnl = d * (px - p["entry"]) * p["qty"] - FEE * (p["entry"] + px) * p["qty"]
        r = pnl / p["risk"]                                   # ratio ke hisaab se points (R)
        bal = con.execute("SELECT balance FROM account").fetchone()[0] + pnl
        con.execute("INSERT INTO trades(symbol,direction,qty,entry,exit,sl,tp,risk_amt,rr,pnl,r_multiple,result,reason,balance_after,t_in,t_out,tf,ctx,sl_mult) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (s, "LONG" if d == 1 else "SHORT", p["qty"], p["entry"], px, p["sl"], p["tp"], p["risk"], p["rr"],
                     pnl, r, "WIN" if pnl > 0 else "LOSS", j.get("reason", ""), bal, p["t_in"], j.get("t_out") or now_ist(), p["tf"], p["ctx"], p["sl_mult"]))
        con.execute("DELETE FROM positions WHERE symbol=?", (s,))
        con.execute("UPDATE account SET balance=?", (bal,))
        con.commit(); con.close()
    return jsonify(ok=True, pnl=pnl, r=r, balance=bal)


@app.route("/api/bot/state")
def bot_state():
    con = db()
    a = con.execute("SELECT * FROM account").fetchone()
    pos = [dict(r) for r in con.execute("SELECT * FROM positions")]
    t = con.execute("SELECT COUNT(*) n, SUM(pnl>0) w, SUM(r_multiple) r FROM trades").fetchone()
    con.close()
    return jsonify(balance=a["balance"], start=a["start"], positions=pos, trades=t["n"], wins=t["w"] or 0,
                   total_r=t["r"] or 0, beat_age=round(time.time() - beat["t"]), beat_msg=beat["msg"])


@app.route("/api/bot/trades")
def bot_trades():
    con = db()
    rows = [dict(r) for r in con.execute("SELECT * FROM trades ORDER BY id DESC LIMIT ?", (int(request.args.get("limit", 100)),))]
    con.close()
    return jsonify(rows)


@app.route("/report")
def report_page():
    import report
    report.build()
    return send_from_directory(report.OUT, "report.html")


@app.route("/trades.csv")
def trades_csv():
    import report
    report.build()
    return send_from_directory(report.OUT, "trades.csv", as_attachment=True)


@app.route("/api/bot/heartbeat", methods=["POST"])
def bot_beat():
    if denied():
        return jsonify(error="key galat"), 403
    beat.update(t=time.time(), msg=str((request.get_json() or {}).get("msg", ""))[:80])
    return jsonify(ok=True)


@app.route("/api/bot/reset", methods=["POST"])
def bot_reset():
    if denied():
        return jsonify(error="key galat"), 403
    with lock:
        con = db()
        con.executescript(f"DELETE FROM positions; DELETE FROM trades; UPDATE account SET balance={START}, start={START};")
        con.close()
    return jsonify(ok=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, threaded=True)
