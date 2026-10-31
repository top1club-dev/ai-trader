"""AI BOT - website ke through paper trade karta hai, kai timeframes par.
Pehle: 1) python trading_ai.py --market india   2) python server.py     Phir: start_bot.bat
Timeframes: models/tf_scores.json se wahi chunta hai jin par AI unseen data par achha tha.
Zabardasti chunne ke liye:  set BOT_TFS=5m,15m,30m
"""
import os, sys, json, time, argparse, subprocess, traceback
from datetime import datetime, timezone, timedelta
import numpy as np, pandas as pd, requests
import trading_ai as ta

SERVER = os.environ.get("SERVER", "http://localhost:8000")
H = {"X-Bot-Key": os.environ.get("BOT_KEY", "bot-secret")}
SYMBOLS = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN", "ITC", "LT"]
SUFFIX, CYCLE, WINDOW, MAX_OPEN = ".NS", 30, 10, 3
IST = timezone(timedelta(hours=5, minutes=30))
MODEL, SCORES, TRAINED = "models/ppo_latest.zip", "models/tf_scores.json", "models/last_train.txt"


def get(path, **p):
    r = requests.get(SERVER + path, params=p, timeout=30)
    r.raise_for_status()
    return r.json()


def post(path, js=None):
    r = requests.post(SERVER + path, json=js or {}, headers=H, timeout=30)
    if not r.ok:
        print("  server:", r.text[:120])
    return r


def pick_tfs():
    if os.environ.get("BOT_TFS"):
        return os.environ["BOT_TFS"].split(",")
    try:
        j = json.load(open(SCORES))
        sc = j["scores"]
    except Exception as e:
        print("tf_scores.json nahi padhi:", e, "-> default 30m")
        return ["30m"]
    good = sorted([t for t, v in sc.items() if v >= j["start"]], key=lambda t: -sc[t])[:3]
    return good or [max(sc, key=sc.get)]              # koi profit mein nahi to sabse kam kharab wala


def market_state(now):
    if now.weekday() >= 5:
        return "closed"
    m = now.hour * 60 + now.minute
    if m < 9 * 60 + 15 or m >= 15 * 60 + 30:
        return "closed"
    return "square" if m >= 15 * 60 + 15 else ("noentry" if m >= 15 * 60 else "open")


def candles(sym, tf, period):
    j = get("/api/candles", symbol=sym + SUFFIX, tf=tf, period=period)["candles"]
    df = pd.DataFrame(j)
    df.index = pd.to_datetime(df.pop("time"), unit="s")      # IST wall time
    return df


def first_hit(p, m):
    """Entry ke baad ki candles chronological dekho: SL ya target mein se pehle kaun chhua. (reason, price, time)"""
    d = p["dir"]
    for ts, r in m[m.index >= pd.to_datetime(p["t_in"]).floor("min")].iterrows():
        sl = r["low"] <= p["sl"] if d == 1 else r["high"] >= p["sl"]
        tp = r["high"] >= p["tp"] if d == 1 else r["low"] <= p["tp"]
        if sl:
            return "SL", p["sl"], ts                             # ek hi candle mein dono hon to SL (training jaisa)
        if tp:
            return "TP", p["tp"], ts
    return None


def close(key, price, reason, ts=None):
    post("/api/bot/close", dict(symbol=key, price=float(price), reason=reason, t_out=str(ts)[:19] if ts is not None else None))
    print(f"  CLOSE {key} @ {price:.2f} ({reason})")
    try:
        import report
        report.build()                                       # har trade ke baad report file taaza
    except Exception as e:
        print("  report nahi bani:", e)


def cycle(st):
    from stable_baselines3 import PPO
    if not os.path.exists(MODEL):
        post("/api/bot/heartbeat", dict(msg="model nahi mila - pehle train karo"))
        return
    sig = (os.path.getmtime(MODEL), os.path.getmtime(SCORES) if os.path.exists(SCORES) else 0)
    if st.get("sig") != sig:                                 # naya model/scores aate hi reload
        st["model"], st["sig"], st["tfs"] = PPO.load(MODEL), sig, pick_tfs()
        print("Model load hua. Timeframes:", st["tfs"])
    now = datetime.now(IST)
    mk = market_state(now)
    pos = {p["symbol"]: p for p in get("/api/bot/state")["positions"]}
    n = 0
    # --- 1) open trades ka exit check: entry ke baad ki saari candles ek-ek karke (pehle kaun chhua)
    for key, p in pos.items():
        try:
            m1 = candles(key.split(".")[0], "1m", "5d")
        except Exception:
            try:
                m1 = candles(key.split(".")[0], "5m", "5d")      # 1m na mile to 5m se
            except Exception as e:
                print(f"{key}: exit check ke liye data nahi ({e})")
                continue
        price, hit = float(m1["close"].iloc[-1]), first_hit(p, m1)
        if hit:
            close(key, hit[1], hit[0], hit[2])                   # SL ya TP, jo pehle chhua, uske asli time ke saath
        elif p["t_in"][:10] != str(now.date()):                  # purani position (bot band tha): us din ke end par EOD
            day = m1[m1.index.strftime("%Y-%m-%d") == p["t_in"][:10]]
            close(key, float(day["close"].iloc[-1]) if len(day) else price, "EOD", day.index[-1] if len(day) else None)
        elif mk in ("square", "closed"):
            close(key, price, "EOD")
    # --- 2) nayi poori candle par decision, har (stock, timeframe) ke liye
    if mk == "open":
        for s in SYMBOLS:
            key = s + SUFFIX
            for tf in st["tfs"]:
                if len(pos) >= MAX_OPEN:
                    break                                    # jagah full, aur scan bekaar
                m = ta.tf_minutes(tf)
                if key in pos or time.time() < st["next"].get((key, tf), 0):
                    continue
                st["next"][(key, tf)] = time.time() + max(30, min(300, m * 20))   # Yahoo ko baar-baar na pareshaan karo
                time.sleep(0.5)                                                   # requests ke beech gap (rate-limit se bachne)
                try:
                    raw = candles(s, tf, "5d" if m < 5 else "10d" if m <= 10 else "60d")
                except Exception as e:
                    print(f"{s} {tf}: data nahi mila ({e})")
                    continue
                n += 1
                done = raw.iloc[:-1]                         # abhi ban rahi candle hata do
                ts = str(done.index[-1]) if len(done) else ""
                if len(done) < 70 or done.index[-1].date() != now.date() or st["seen"].get((key, tf)) == ts:
                    continue
                st["seen"][(key, tf)] = ts
                df = ta.add_features(done)
                obs = np.clip(np.r_[df[ta.FEATURES].values[-WINDOW:].flatten(), 0, 0, 0, ta.tf_code(tf)], -10, 10).astype(np.float32)
                a = int(st["model"].predict(obs, deterministic=True)[0])
                if a == 0:
                    continue
                d, sm, rr = ta.ACTIONS[a]
                price = float(raw["close"].iloc[-1])
                dist = max(float(df["atr"].iloc[-1]) * sm, price * 0.0005)
                print(f"  OPEN {key} [{tf}] {'LONG' if d == 1 else 'SHORT'} @ {price:.2f} SL {sm}xATR target 1:{rr}")
                r = post("/api/bot/open", dict(symbol=key, direction=d, price=price, sl_dist=dist, rr=rr, sl_mult=sm,
                                               candle_t=str(raw.index[-1]), rsi=float(df["rsi"].iloc[-1]) + 0.5,
                                               atr_pct=float(df["atr_pct"].iloc[-1]), tf=tf,
                                               ctx={k: round(float(df[k].iloc[-1]), 4) for k in ta.FEATURES}))
                if r.ok:
                    pos[key] = {}
    # --- 3) roz market band hone ke baad purane + naye data par dobara seekhna
    last = open(TRAINED).read().strip() if os.path.exists(TRAINED) else ""   # file mein yaad: bot restart ho to bhi dobara train na ho
    if mk == "closed" and now.weekday() < 5 and now.hour >= 16 and last != str(now.date()):
        open(TRAINED, "w").write(str(now.date()))
        try:
            tfs = ",".join(json.load(open(SCORES))["scores"].keys())            # wahi timeframes jin par model pehle seekha tha
        except Exception:
            tfs = "15m,30m,1h"
        print("Daily fine-tune shuru:", tfs)
        st["profiled"] = False
        st["train"] = subprocess.Popen([sys.executable, "trading_ai.py", "--market", "india", "--finetune", "--tfs", tfs, "--steps", "100000"])
    tr = st.get("train")
    if tr is not None and tr.poll() is not None and not st.get("profiled"):
        st["profiled"] = True                                # training khatam: AI ka behaviour dobara naapo
        subprocess.Popen([sys.executable, "report.py", "--policy"])
    busy = st.get("train") is not None and st["train"].poll() is None
    post("/api/bot/heartbeat", dict(msg="seekh raha hai (daily retrain)" if busy else f"{mk} | {','.join(st['tfs'])}"))


def is_holiday(now):
    """Market ke time par bhi aaj ki koi candle nahi = exchange holiday."""
    try:
        m = candles("RELIANCE", "5m", "1d")
        return m.index[-1].date() != now.date()
    except Exception:
        return False


def git_push():
    """GitHub Actions mein: ledger aur report repo mein save karo."""
    run = lambda *c: subprocess.run(list(c), check=False).returncode
    run("git", "add", "-A", "ledger.db")                      # alag-alag add: ek fail ho to dusra na ruke
    run("git", "add", "-A", "docs")
    if run("git", "diff", "--cached", "--quiet") != 0:
        run("git", "commit", "-m", "bot update", "--quiet")
        if run("git", "push", "--quiet") != 0:                # remote aage ho to pehle milao
            run("git", "pull", "--rebase", "--autostash", "--quiet")
            run("git", "push", "--quiet")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--stop-at", default=None, help="IST HH:MM par band (GitHub Actions)")
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--push-every", type=float, default=0, help="itne minute mein git push")
    args = ap.parse_args()
    if args.reset:
        post("/api/bot/reset"); print("Account reset: Rs 10,000"); sys.exit()
    if args.stop_at and datetime.now(IST).weekday() >= 5:
        print("Weekend hai, bot band."); sys.exit(0)
    try:
        import report
        report.build()                                       # report page shuru se maujood rahe
    except Exception as e:
        print("report:", e)
    state = {"seen": {}, "next": {}}
    print("Bot chalu. Ctrl+C se band.")
    t0 = last_push = time.time()
    while True:
        try:
            cycle(state)
        except KeyboardInterrupt:
            raise
        except Exception:
            traceback.print_exc()
        now = datetime.now(IST)
        if args.stop_at and now.strftime("%H:%M") >= args.stop_at:
            print("Stop time aa gaya."); break
        if args.max_minutes and (time.time() - t0) / 60 >= args.max_minutes:
            print("Session time poora."); break
        if args.stop_at and "hol" not in state and now.hour * 60 + now.minute >= 9 * 60 + 45:
            state["hol"] = is_holiday(now)
            if state["hol"]:
                print("Aaj market holiday hai, bot band."); break
        if args.push_every and time.time() - last_push > args.push_every * 60:
            git_push(); last_push = time.time()
        time.sleep(CYCLE)
    if args.push_every:
        git_push()
