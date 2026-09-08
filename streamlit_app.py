import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Backtest Agent", layout="wide")
st.title("Backtest Agent")
st.caption(
    "CSV yükle → veri kalitesini doğrula → stratejiyi test et → "
    "aynı dönem için CAGR, benchmark ve risk metriklerini gör."
)

# =====================================================
# SABİT STRATEJİ: BOĞA PENÇESİ v5
# =====================================================
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
WARMUP = 250

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
    st.divider()
    st.caption("Not: Bu ekran tek-hisse backtestidir. Portföy CAGR'ı değildir.")

c1, c2 = st.columns(2)
with c1:
    hisse_file = st.file_uploader("Hisse CSV (1 saatlik)", type=["csv"])
with c2:
    xu_file = st.file_uploader("XU100 CSV (4 saatlik)", type=["csv"])


# =====================================================
# VERİ YARDIMCILARI
# =====================================================
def parse_dates(df):
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()

    if "time" not in df.columns:
        raise ValueError("CSV'de 'time' sütunu bulunamadı.")

    # TradingView unix time saniye varsayımı
    if pd.api.types.is_numeric_dtype(df["time"]):
        df["date"] = pd.to_datetime(df["time"], unit="s", errors="coerce")
    else:
        df["date"] = pd.to_datetime(df["time"], errors="coerce")

    df = (
        df.dropna(subset=["date"])
        .sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )
    return df


def infer_bar_minutes(df):
    """Gece/hafta sonu boşluklarını dışlayarak tahmini mum süresini bul."""
    if len(df) < 3:
        return np.nan

    diffs = df["date"].diff().dropna().dt.total_seconds().div(60)
    # BIST intraday barları için 10 dakika - 8 saat aralığı
    diffs = diffs[(diffs >= 10) & (diffs <= 480)]
    if diffs.empty:
        return np.nan
    return float(diffs.median())


def validate_uploads(stock_raw, xu_raw, stock_name, xu_name):
    stock = parse_dates(stock_raw)
    xu = parse_dates(xu_raw)

    required_stock = {"open", "high", "low", "close", "volume"}
    missing_stock = required_stock - set(stock.columns)
    if missing_stock:
        raise ValueError(f"Hisse CSV eksik sütunlar: {sorted(missing_stock)}")

    required_xu = {"open", "high", "low", "close"}
    missing_xu = required_xu - set(xu.columns)
    if missing_xu:
        raise ValueError(f"XU100 CSV eksik sütunlar: {sorted(missing_xu)}")

    stock_min = infer_bar_minutes(stock)
    xu_min = infer_bar_minutes(xu)

    errors = []

    # Dosya adı kontrolü: yanlış benchmark yüklenmesini engelle
    if "XU100" not in str(xu_name).upper():
        errors.append(
            f"Sağdaki dosya XU100 görünmüyor: '{xu_name}'. "
            "BIST_XU100, 240.csv yükle."
        )

    if "XU100" in str(stock_name).upper():
        errors.append("Soldaki dosya hisse olmalı; XU100 yüklenmiş görünüyor.")

    # Zaman dilimi kontrolü
    if pd.notna(stock_min) and not (45 <= stock_min <= 90):
        errors.append(
            f"Hisse verisinin tahmini mum süresi {stock_min:.0f} dk. "
            "1 saatlik veri bekleniyor."
        )

    if pd.notna(xu_min) and not (180 <= xu_min <= 300):
        errors.append(
            f"XU100 verisinin tahmini mum süresi {xu_min:.0f} dk. "
            "4 saatlik / 240 dk veri bekleniyor."
        )

    if errors:
        raise ValueError("\n".join(errors))

    return stock, xu, stock_min, xu_min


def add_indicators(df):
    df = df.copy()

    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

    # RSI 14
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # ATR 14
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()

    # ADX / DI
    up = df["high"].diff()
    down = -df["low"].diff()

    plus_dm = pd.Series(
        np.where((up > down) & (up > 0), up, 0.0),
        index=df.index,
        dtype=float,
    )
    minus_dm = pd.Series(
        np.where((down > up) & (down > 0), down, 0.0),
        index=df.index,
        dtype=float,
    )

    atr_s = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_s = plus_dm.ewm(alpha=1/14, adjust=False).mean()
    minus_s = minus_dm.ewm(alpha=1/14, adjust=False).mean()

    df["plus_di"] = 100 * plus_s / atr_s.replace(0, np.nan)
    df["minus_di"] = 100 * minus_s / atr_s.replace(0, np.nan)

    dx = (
        100
        * (df["plus_di"] - df["minus_di"]).abs()
        / (df["plus_di"] + df["minus_di"]).replace(0, np.nan)
    )
    df["adx"] = dx.ewm(alpha=1/14, adjust=False).mean()

    # RVOL
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["rvol"] = df["volume"] / df["vol_ma20"].replace(0, np.nan)

    # Önceki 15 mum direnci
    df["resistance"] = df["high"].shift(1).rolling(BREAKOUT).max()

    return df


def cagr(start_value, end_value, years):
    if years <= 0 or start_value <= 0 or end_value <= 0:
        return np.nan
    return (end_value / start_value) ** (1 / years) - 1


def price_cagr(df, start_date, end_date):
    sliced = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
    if len(sliced) < 2:
        return np.nan

    years = (end_date - start_date).total_seconds() / (365.25 * 24 * 3600)
    return cagr(
        float(sliced.iloc[0]["close"]),
        float(sliced.iloc[-1]["close"]),
        years,
    )


# =====================================================
# BACKTEST
# =====================================================
def run_backtest(stock, xu):
    stock = add_indicators(stock.copy())
    xu = xu.copy()

    xu["ema200_xu"] = xu["close"].ewm(span=200, adjust=False).mean()

    # Sadece kapanmış 4H mum bilgisi kullanılsın
    xu["market_bull"] = (xu["close"] > xu["ema200_xu"]).shift(1)

    if len(stock) <= WARMUP + 2:
        raise ValueError("Hisse verisi backtest için çok kısa.")

    valid_xu = xu.dropna(subset=["market_bull"])
    if valid_xu.empty:
        raise ValueError("XU100 verisi EMA200 filtresi için çok kısa.")

    # =================================================
    # ORTAK TEST PENCERESİ
    # CAGR artık ilk/son işleme göre DEĞİL,
    # test edilebilir ortak veri penceresine göre hesaplanır.
    # =================================================
    test_start = max(
        stock.iloc[WARMUP]["date"],
        valid_xu.iloc[0]["date"],
    )
    test_end = min(
        stock.iloc[-1]["date"],
        xu.iloc[-1]["date"],
    )

    if test_end <= test_start:
        raise ValueError("Hisse ve XU100 verilerinin ortak test dönemi yok.")

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
    equity_curve = [BASLANGIC]
    in_position = False

    test_bars = 0
    exposed_bars = 0

    for i in range(WARMUP, len(stock) - 1):
        row = stock.iloc[i]
        nxt = stock.iloc[i + 1]

        if row["date"] < test_start:
            continue

        if row["date"] > test_end or nxt["date"] > test_end:
            break

        test_bars += 1

        # -------------------------
        # GİRİŞ
        # -------------------------
        if not in_position and bool(row["signal"]):
            entry = float(nxt["open"])
            entry_date = nxt["date"]

            swing_low = float(
                stock["low"].iloc[i-SWING+1:i+1].min()
            )
            stop = swing_low - float(row["atr"]) * ATR_MULT
            risk = entry - stop

            if not np.isfinite(risk) or risk <= 0:
                continue

            qty_by_risk = (capital * RISK_PCT) / risk
            qty_by_cash = capital / entry
            qty = min(qty_by_risk, qty_by_cash)

            if qty <= 0:
                continue

            target = entry + risk * RR
            in_position = True

            # Giriş bir sonraki mum açılışında.
            # Aynı mum içinde stop/hedef kontrolü sonraki loop'ta başlar.
            continue

        # -------------------------
        # POZİSYON YÖNETİMİ
        # -------------------------
        if in_position:
            exposed_bars += 1

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
                r_mult = (
                    pnl / initial_risk_tl
                    if initial_risk_tl > 0
                    else np.nan
                )

                capital += pnl

                trades.append(
                    {
                        "Giriş": entry_date,
                        "Çıkış": row["date"],
                        "Giriş Fiyat": entry,
                        "Stop": stop,
                        "Hedef": target,
                        "Çıkış Fiyat": exit_price,
                        "PnL TL": pnl,
                        "R": r_mult,
                        "Neden": reason,
                    }
                )

                in_position = False

        # Mark-to-market equity
        if in_position:
            equity_now = capital + (float(row["close"]) - entry) * qty
        else:
            equity_now = capital

        equity_curve.append(equity_now)

    # Test sonunda açık pozisyonu ortak veri penceresinin son fiyatından kapat
    if in_position:
        end_slice = stock[stock["date"] <= test_end]
        last = end_slice.iloc[-1]
        exit_price = float(last["close"])

        gross = (exit_price - entry) * qty
        commission = (entry * qty + exit_price * qty) * KOMISYON
        pnl = gross - commission

        initial_risk_tl = (entry - stop) * qty
        r_mult = (
            pnl / initial_risk_tl
            if initial_risk_tl > 0
            else np.nan
        )

        capital += pnl

        trades.append(
            {
                "Giriş": entry_date,
                "Çıkış": last["date"],
                "Giriş Fiyat": entry,
                "Stop": stop,
                "Hedef": target,
                "Çıkış Fiyat": exit_price,
                "PnL TL": pnl,
                "R": r_mult,
                "Neden": "Veri Sonu",
            }
        )
        equity_curve.append(capital)

    t = pd.DataFrame(trades)

    years = (
        (test_end - test_start).total_seconds()
        / (365.25 * 24 * 3600)
    )

    total_return = capital / BASLANGIC - 1
    strat_cagr = cagr(BASLANGIC, capital, years)

    # Aynı tam test penceresinde benchmarklar
    bh_cagr = price_cagr(stock, test_start, test_end)
    xu_cagr = price_cagr(xu, test_start, test_end)

    if t.empty:
        win_rate = np.nan
        avg_r = np.nan
        pf = np.nan
    else:
        gp = t.loc[t["PnL TL"] > 0, "PnL TL"].sum()
        gl = abs(t.loc[t["PnL TL"] < 0, "PnL TL"].sum())
        pf = gp / gl if gl > 0 else np.nan
        win_rate = (t["PnL TL"] > 0).mean()
        avg_r = t["R"].mean()

    equity = pd.Series(equity_curve, dtype=float)
    max_dd = (
        (equity / equity.cummax() - 1).min()
        if len(equity)
        else np.nan
    )

    exposure = (
        exposed_bars / test_bars
        if test_bars > 0
        else np.nan
    )

    calmar = (
        strat_cagr / abs(max_dd)
        if pd.notna(max_dd) and max_dd < 0
        else np.nan
    )

    metrics = {
        "Test başlangıcı": test_start,
        "Test bitişi": test_end,
        "Süre (yıl)": years,
        "Toplam getiri": total_return,
        "CAGR": strat_cagr,
        "Buy & Hold CAGR": bh_cagr,
        "XU100 CAGR": xu_cagr,
        "Hisseye göre alpha": (
            strat_cagr - bh_cagr
            if pd.notna(bh_cagr)
            else np.nan
        ),
        "XU100'e göre alpha": (
            strat_cagr - xu_cagr
            if pd.notna(xu_cagr)
            else np.nan
        ),
        "Max Drawdown": max_dd,
        "Calmar": calmar,
        "Profit Factor": pf,
        "Kazanma oranı": win_rate,
        "Ortalama R": avg_r,
        "İşlem sayısı": len(t),
        "İşlem / yıl": len(t) / years if years > 0 else np.nan,
        "Piyasada kalma": exposure,
        "Final sermaye": capital,
        "Ham sinyal": int(
            stock[
                (stock["date"] >= test_start)
                & (stock["date"] <= test_end)
            ]["signal"].sum()
        ),
    }

    return stock, t, metrics


# =====================================================
# ARAYÜZ
# =====================================================
if hisse_file and xu_file:
    if st.button("TEST ET", type="primary", use_container_width=True):
        try:
            stock_raw = pd.read_csv(hisse_file)
            xu_raw = pd.read_csv(xu_file)

            stock_ready, xu_ready, stock_min, xu_min = validate_uploads(
                stock_raw,
                xu_raw,
                hisse_file.name,
                xu_file.name,
            )

            st.success(
                f"Veri doğrulandı — Hisse: ~{stock_min:.0f} dk | "
                f"XU100: ~{xu_min:.0f} dk"
            )

            _, trades, m = run_backtest(stock_ready, xu_ready)

            st.subheader("Zorunlu Performans Özeti")

            r1 = st.columns(5)
            r1[0].metric("CAGR", f"%{m['CAGR']*100:.2f}")
            r1[1].metric(
                "Buy & Hold CAGR",
                f"%{m['Buy & Hold CAGR']*100:.2f}"
                if pd.notna(m["Buy & Hold CAGR"])
                else "N/A",
            )
            r1[2].metric(
                "XU100 CAGR",
                f"%{m['XU100 CAGR']*100:.2f}"
                if pd.notna(m["XU100 CAGR"])
                else "N/A",
            )
            r1[3].metric(
                "Max Drawdown",
                f"%{m['Max Drawdown']*100:.2f}"
                if pd.notna(m["Max Drawdown"])
                else "N/A",
            )
            r1[4].metric(
                "Profit Factor",
                f"{m['Profit Factor']:.2f}"
                if pd.notna(m["Profit Factor"])
                else "N/A",
            )

            r2 = st.columns(5)
            r2[0].metric("Toplam Getiri", f"%{m['Toplam getiri']*100:.2f}")
            r2[1].metric(
                "Kazanma",
                f"%{m['Kazanma oranı']*100:.2f}"
                if pd.notna(m["Kazanma oranı"])
                else "N/A",
            )
            r2[2].metric(
                "Ortalama R",
                f"{m['Ortalama R']:.2f}"
                if pd.notna(m["Ortalama R"])
                else "N/A",
            )
            r2[3].metric("İşlem", f"{m['İşlem sayısı']}")
            r2[4].metric("İşlem / yıl", f"{m['İşlem / yıl']:.2f}")

            r3 = st.columns(5)
            xu_alpha = m["XU100'e göre alpha"]
            stock_alpha = m["Hisseye göre alpha"]
            r3[0].metric(
                "XU100 Alpha",
                f"{xu_alpha*100:+.2f} puan"
                if pd.notna(xu_alpha)
                else "N/A",
            )
            r3[1].metric(
                "Hisse Alpha",
                f"{stock_alpha*100:+.2f} puan"
                if pd.notna(stock_alpha)
                else "N/A",
            )
            r3[2].metric(
                "Calmar",
                f"{m['Calmar']:.2f}"
                if pd.notna(m["Calmar"])
                else "N/A",
            )
            r3[3].metric(
                "Piyasada Kalma",
                f"%{m['Piyasada kalma']*100:.1f}"
                if pd.notna(m["Piyasada kalma"])
                else "N/A",
            )
            r3[4].metric("Ham Sinyal", f"{m['Ham sinyal']}")

            st.write(
                f"**Test penceresi:** "
                f"{m['Test başlangıcı'].date()} → {m['Test bitişi'].date()} "
                f"(**{m['Süre (yıl)']:.2f} yıl**)"
            )

            st.caption(
                "CAGR artık ilk işlem ile son işlem arasından değil, "
                "warm-up sonrası ortak veri penceresinin tamamından hesaplanıyor."
            )

            # Karar katmanları
            if (
                pd.isna(m["CAGR"])
                or pd.isna(m["Buy & Hold CAGR"])
                or pd.isna(m["XU100 CAGR"])
            ):
                st.error(
                    "KARAR: DEĞERLENDİRİLEMEZ — CAGR veya benchmark eksik."
                )
            else:
                edge_ok = (
                    pd.notna(m["Profit Factor"])
                    and m["Profit Factor"] >= 1.30
                    and pd.notna(m["Ortalama R"])
                    and m["Ortalama R"] > 0
                )

                alpha_ok = (
                    m["CAGR"] > m["Buy & Hold CAGR"]
                    and m["CAGR"] > m["XU100 CAGR"]
                )

                if edge_ok and alpha_ok:
                    st.success(
                        "KARAR: PASS — İşlem avantajı var ve CAGR aynı dönemde "
                        "hem hisseyi hem XU100'ü geçti."
                    )
                elif edge_ok and not alpha_ok:
                    st.warning(
                        "KARAR: EDGE VAR, SERMAYE VERİMSİZ — PF/ortalama R pozitif; "
                        "ancak CAGR benchmarkları geçmiyor. Tek-hisse 1% risk testi "
                        "portföy seviyesinde ayrıca sınanmalı."
                    )
                else:
                    st.error(
                        "KARAR: FAIL — İşlem avantajı da yeterince güçlü değil."
                    )

            st.subheader("İşlemler")
            if trades.empty:
                st.info("Bu test penceresinde işlem oluşmadı.")
            else:
                st.dataframe(trades, use_container_width=True)

        except Exception as e:
            st.error(str(e))
else:
    st.info("Başlamak için hisse 1H CSV ve XU100 4H CSV yükle.")
