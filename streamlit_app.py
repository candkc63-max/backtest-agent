import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Backtest Agent", layout="wide")
st.title("Backtest Agent")
st.caption("CSV yükle → stratejiyi test et → süre, CAGR, benchmark ve risk metriklerini zorunlu gör.")

# -----------------------------
# Sabit strateji: Boğa Pençesi v5
# -----------------------------
ADX_MIN = 22.5
RVOL_MIN = 1.3
RSI_MIN = 40
RSI_MAX = 70
BREAKOUT = 15
SWING = 10
ATR_MULT = 1.0
RR = 3.5
RISK_PCT = 0.01
KOMISYON = 0.001
BASLANGIC = 100_000

with st.sidebar:
    st.header("Strateji")
    st.write("Boğa Pençesi v5")
    st.write(f"ADX > {ADX_MIN}")
    st.write(f"RVOL ≥ {RVOL_MIN}")
    st.write(f"RSI {RSI_MIN}-{RSI_MAX}")
    st.write(f"{BREAKOUT} mum kırılım")
    st.write(f"Stop: son {SWING} mum dibi - {ATR_MULT} ATR")
    st.write(f"Hedef: {RR}R")
    st.write(f"İşlem riski: %{RISK_PCT*100:.1f}")

c1, c2 = st.columns(2)
with c1:
    hisse_file = st.file_uploader("Hisse CSV (1 saatlik)", type=["csv"])
with c2:
    xu_file = st.file_uploader("XU100 CSV (4 saatlik)", type=["csv"])


def prepare(df):
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()
    needed = {"time", "open", "high", "low", "close"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Eksik sütunlar: {sorted(missing)}")
    if "volume" not in df.columns:
        raise ValueError("Hisse CSV dosyasında volume sütunu gerekli.")
    df["date"] = pd.to_datetime(df["time"], unit="s", errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return df


def add_indicators(df):
    df = df.copy()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()

    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0), index=df.index)
    atr_s = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_s = plus_dm.ewm(alpha=1/14, adjust=False).mean()
    minus_s = minus_dm.ewm(alpha=1/14, adjust=False).mean()
    df["plus_di"] = 100 * plus_s / atr_s.replace(0, np.nan)
    df["minus_di"] = 100 * minus_s / atr_s.replace(0, np.nan)
    dx = 100 * (df["plus_di"] - df["minus_di"]).abs() / (df["plus_di"] + df["minus_di"]).replace(0, np.nan)
    df["adx"] = dx.ewm(alpha=1/14, adjust=False).mean()

    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["rvol"] = df["volume"] / df["vol_ma20"].replace(0, np.nan)
    df["resistance"] = df["high"].shift(1).rolling(BREAKOUT).max()
    return df


def cagr(start_value, end_value, years):
    if years <= 0 or start_value <= 0 or end_value <= 0:
        return np.nan
    return (end_value / start_value) ** (1 / years) - 1


def run_backtest(stock, xu):
    stock = prepare(stock)
    xu = xu.copy()
    xu.columns = xu.columns.str.lower().str.strip()
    needed_xu = {"time", "close"}
    missing_xu = needed_xu - set(xu.columns)
    if missing_xu:
        raise ValueError(f"XU100 eksik sütunlar: {sorted(missing_xu)}")
    xu["date"] = pd.to_datetime(xu["time"], unit="s", errors="coerce")
    xu = xu.dropna(subset=["date"]).sort_values("date").drop_duplicates("date").reset_index(drop=True)

    stock = add_indicators(stock)
    xu["ema200_xu"] = xu["close"].ewm(span=200, adjust=False).mean()
    xu["market_bull"] = (xu["close"] > xu["ema200_xu"]).shift(1)

    stock = pd.merge_asof(
        stock.sort_values("date"),
        xu[["date", "market_bull"]].sort_values("date"),
        on="date",
        direction="backward",
    )
    stock["market_bull"] = stock["market_bull"].fillna(False).astype(bool)

    stock["signal"] = (
        stock["market_bull"]
        & (stock["close"] > stock["ema200"])
        & (stock["ema20"] > stock["ema50"])
        & (stock["adx"] > ADX_MIN)
        & (stock["plus_di"] > stock["minus_di"])
        & (stock["rvol"] >= RVOL_MIN)
        & (stock["rsi"] >= RSI_MIN)
        & (stock["rsi"] <= RSI_MAX)
        & (stock["close"] > stock["resistance"])
    )

    capital = BASLANGIC
    trades = []
    equity_curve = []
    in_position = False

    for i in range(250, len(stock) - 1):
        row = stock.iloc[i]

        if not in_position and bool(row["signal"]):
            nxt = stock.iloc[i + 1]
            entry = float(nxt["open"])
            entry_date = nxt["date"]
            swing_low = float(stock["low"].iloc[i-SWING+1:i+1].min())
            stop = swing_low - float(row["atr"]) * ATR_MULT
            risk = entry - stop
            if risk <= 0:
                continue
            qty = min((capital * RISK_PCT) / risk, capital / entry)
            target = entry + risk * RR
            in_position = True
            continue

        if in_position:
            exit_price = None
            reason = None
            if float(row["open"]) <= stop:
                exit_price = float(row["open"])
                reason = "Gap Stop"
            elif float(row["low"]) <= stop:
                exit_price = stop
                reason = "Stop"
            elif float(row["high"]) >= target:
                exit_price = target
                reason = f"{RR}R"

            if exit_price is not None:
                gross = (exit_price - entry) * qty
                commission = (entry * qty + exit_price * qty) * KOMISYON
                pnl = gross - commission
                initial_risk_tl = (entry - stop) * qty
                r_mult = pnl / initial_risk_tl if initial_risk_tl > 0 else np.nan
                capital += pnl
                trades.append({
                    "Giriş": entry_date,
                    "Çıkış": row["date"],
                    "Giriş Fiyat": entry,
                    "Çıkış Fiyat": exit_price,
                    "PnL TL": pnl,
                    "R": r_mult,
                    "Neden": reason,
                })
                in_position = False

        if in_position:
            equity_now = capital + (float(row["close"]) - entry) * qty
        else:
            equity_now = capital
        equity_curve.append(equity_now)

    if in_position:
        last = stock.iloc[-1]
        exit_price = float(last["close"])
        gross = (exit_price - entry) * qty
        commission = (entry * qty + exit_price * qty) * KOMISYON
        pnl = gross - commission
        initial_risk_tl = (entry - stop) * qty
        r_mult = pnl / initial_risk_tl if initial_risk_tl > 0 else np.nan
        capital += pnl
        trades.append({
            "Giriş": entry_date,
            "Çıkış": last["date"],
            "Giriş Fiyat": entry,
            "Çıkış Fiyat": exit_price,
            "PnL TL": pnl,
            "R": r_mult,
            "Neden": "Veri Sonu",
        })
        equity_curve.append(capital)

    t = pd.DataFrame(trades)
    if t.empty:
        return stock, t, None

    first_trade = pd.to_datetime(t["Giriş"].min())
    last_trade = pd.to_datetime(t["Çıkış"].max())
    years = (last_trade - first_trade).days / 365.25

    total_return = capital / BASLANGIC - 1
    strat_cagr = cagr(BASLANGIC, capital, years)

    # Buy & Hold benchmark aynı işlem tarih aralığında
    stock_slice = stock[(stock["date"] >= first_trade) & (stock["date"] <= last_trade)]
    bh_start = float(stock_slice.iloc[0]["close"])
    bh_end = float(stock_slice.iloc[-1]["close"])
    bh_cagr = cagr(bh_start, bh_end, years)

    # XU100 benchmark aynı tarih aralığında
    xu_slice = xu[(xu["date"] >= first_trade) & (xu["date"] <= last_trade)]
    xu_cagr = np.nan
    if len(xu_slice) >= 2:
        xu_cagr = cagr(float(xu_slice.iloc[0]["close"]), float(xu_slice.iloc[-1]["close"]), years)

    gp = t.loc[t["PnL TL"] > 0, "PnL TL"].sum()
    gl = abs(t.loc[t["PnL TL"] < 0, "PnL TL"].sum())
    pf = gp / gl if gl > 0 else np.nan
    win_rate = (t["PnL TL"] > 0).mean()

    equity = pd.Series(equity_curve, dtype=float)
    max_dd = (equity / equity.cummax() - 1).min() if len(equity) else np.nan

    metrics = {
        "İlk işlem": first_trade,
        "Son işlem": last_trade,
        "Süre (yıl)": years,
        "Toplam getiri": total_return,
        "CAGR": strat_cagr,
        "Buy & Hold CAGR": bh_cagr,
        "XU100 CAGR": xu_cagr,
        "Hisseye göre fazla getiri": strat_cagr - bh_cagr if pd.notna(bh_cagr) else np.nan,
        "XU100'e göre fazla getiri": strat_cagr - xu_cagr if pd.notna(xu_cagr) else np.nan,
        "Max Drawdown": max_dd,
        "Profit Factor": pf,
        "Kazanma oranı": win_rate,
        "Ortalama R": t["R"].mean(),
        "İşlem sayısı": len(t),
        "İşlem / yıl": len(t) / years if years > 0 else np.nan,
        "Final sermaye": capital,
    }
    return stock, t, metrics


if hisse_file and xu_file:
    if st.button("TEST ET", type="primary", use_container_width=True):
        try:
            stock_raw = pd.read_csv(hisse_file)
            xu_raw = pd.read_csv(xu_file)
            stock, trades, m = run_backtest(stock_raw, xu_raw)

            if m is None:
                st.warning("Bu kurallarla işlem oluşmadı.")
            else:
                st.subheader("Zorunlu Performans Özeti")
                r1 = st.columns(5)
                r1[0].metric("CAGR", f"%{m['CAGR']*100:.2f}")
                r1[1].metric("Buy & Hold CAGR", f"%{m['Buy & Hold CAGR']*100:.2f}")
                r1[2].metric("XU100 CAGR", f"%{m['XU100 CAGR']*100:.2f}" if pd.notna(m['XU100 CAGR']) else "N/A")
                r1[3].metric("Max Drawdown", f"%{m['Max Drawdown']*100:.2f}")
                r1[4].metric("Profit Factor", f"{m['Profit Factor']:.2f}")

                r2 = st.columns(5)
                r2[0].metric("Toplam Getiri", f"%{m['Toplam getiri']*100:.2f}")
                r2[1].metric("Kazanma", f"%{m['Kazanma oranı']*100:.2f}")
                r2[2].metric("Ortalama R", f"{m['Ortalama R']:.2f}")
                r2[3].metric("İşlem", f"{m['İşlem sayısı']}")
                r2[4].metric("İşlem / yıl", f"{m['İşlem / yıl']:.2f}")

                st.write(
                    f"**Test süresi:** {m['İlk işlem'].date()} → {m['Son işlem'].date()} "
                    f"(**{m['Süre (yıl)']:.2f} yıl**)"
                )

                # Sert karar: benchmark/CAGR hesaplanmadan PASS yok
                if pd.isna(m["CAGR"]) or pd.isna(m["Buy & Hold CAGR"]) or pd.isna(m["XU100 CAGR"]):
                    st.error("KARAR: DEĞERLENDİRİLEMEZ — CAGR veya benchmark eksik.")
                elif m["CAGR"] > m["Buy & Hold CAGR"] and m["CAGR"] > m["XU100 CAGR"] and m["Profit Factor"] > 1.2:
                    st.success("KARAR: PASS — Strateji hem hisseyi hem XU100'ü yıllıklandırılmış bazda geçti.")
                else:
                    st.error("KARAR: FAIL — Yıllıklandırılmış performans benchmarkları geçemedi veya PF yetersiz.")

                st.subheader("İşlemler")
                st.dataframe(trades, use_container_width=True)

        except Exception as e:
            st.exception(e)
else:
    st.info("Başlamak için hisse 1H CSV ve XU100 4H CSV yükle.")
