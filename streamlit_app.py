import pandas as pd
import streamlit as st

from engine import prepare_daily, symbol_from_name
from optimizer import run_research
from strategy_schema import StrategyConfig

st.set_page_config(page_title="Backtest Agent v3", layout="wide")
st.title("Backtest Agent v3 — Autonomous Strategy Research")
st.caption(
    "CSV'leri yükle → başlangıç stratejisini ver → AI sonuçlardan öğrenip yeni strateji yapıları üretir → "
    "Train/Validation üzerinde iteratif test eder → OOS'u yalnızca en sonda açar."
)

with st.sidebar:
    st.header("Başlangıç stratejisi")

    entry_type = st.selectbox(
        "Giriş motoru",
        ["donchian", "ema_cross", "momentum", "rsi_breakout"],
        index=0,
    )

    breakout_days = st.number_input("Breakout günü", 10, 100, 20, 1)
    ema_fast = st.number_input("EMA hızlı", 5, 80, 20, 1)
    ema_slow = st.number_input("EMA yavaş", 20, 250, 50, 1)
    momentum_days = st.number_input("Momentum günü", 5, 120, 20, 1)

    st.divider()
    st.subheader("Filtreler")

    use_rvol = st.checkbox("RVOL filtresi", True)
    rvol_lookback = st.number_input("RVOL ortalama günü", 10, 60, 20, 1)
    rvol_min = st.number_input("Minimum RVOL", 1.0, 3.0, 1.5, 0.1)

    use_trend_filter = st.checkbox("Trend MA filtresi", True)
    trend_ma = st.number_input("Trend MA", 50, 250, 200, 10)

    use_rsi_filter = st.checkbox("RSI filtresi", False)
    rsi_period = st.number_input("RSI periyodu", 7, 30, 14, 1)
    rsi_min = st.number_input("RSI min", 20.0, 70.0, 45.0, 1.0)
    rsi_max = st.number_input("RSI max", 50.0, 90.0, 75.0, 1.0)

    use_adx_filter = st.checkbox("ADX filtresi", False)
    adx_period = st.number_input("ADX periyodu", 7, 30, 14, 1)
    adx_min = st.number_input("ADX min", 10.0, 50.0, 20.0, 1.0)

    st.divider()
    st.subheader("Çıkış")

    exit_type = st.selectbox(
        "Çıkış motoru",
        ["donchian", "ma", "atr_trailing", "fixed_r"],
        index=0,
    )
    exit_days = st.number_input("Donchian çıkış günü", 5, 80, 10, 1)
    exit_ma = st.number_input("Çıkış MA", 10, 250, 50, 5)
    atr_period = st.number_input("ATR periyodu", 7, 30, 14, 1)
    atr_mult = st.number_input("ATR çarpanı", 1.0, 6.0, 2.0, 0.25)
    target_r = st.number_input("Sabit hedef R", 1.0, 8.0, 3.0, 0.25)

    use_initial_atr_stop = st.checkbox("İlk ATR stop", False)
    initial_atr_mult = st.number_input("İlk stop ATR", 1.0, 6.0, 2.0, 0.25)

    st.divider()
    st.header("Araştırma")
    max_rounds = st.slider("Maksimum AI turu", 3, 30, 12)
    model = st.selectbox(
        "AI modeli",
        ["gpt-5.6-sol", "gpt-5.6-luna"],
        index=0,
    )

st.subheader("1) Günlük hisse verilerini yükle")
files = st.file_uploader(
    "Birden fazla 1D CSV yükleyebilirsin",
    type=["csv"],
    accept_multiple_files=True,
)

secret_key = ""
try:
    secret_key = st.secrets.get("OPENAI_API_KEY", "")
except Exception:
    secret_key = ""

api_key = secret_key or st.text_input(
    "OpenAI API key",
    type="password",
    help="Anahtar yalnızca API çağrıları için kullanılır; buraya sohbetten göndermeyin.",
)

stock_map = {}
valid_rows = []
errors = []

if files:
    for f in files:
        try:
            raw = pd.read_csv(f)
            prepared = prepare_daily(raw)
            symbol = symbol_from_name(f.name)
            if symbol in stock_map:
                symbol = f"{symbol}_{len(stock_map)+1}"
            stock_map[symbol] = prepared
            valid_rows.append({
                "Hisse": symbol,
                "Satır": len(prepared),
                "Başlangıç": prepared.iloc[0]["date"].date(),
                "Bitiş": prepared.iloc[-1]["date"].date(),
            })
        except Exception as e:
            errors.append({"Dosya": f.name, "Hata": str(e)})

if valid_rows:
    st.success(f"{len(valid_rows)} hisse doğrulandı.")
    st.dataframe(pd.DataFrame(valid_rows), use_container_width=True, hide_index=True)

if errors:
    st.warning("Bazı dosyalar teste alınmadı.")
    st.dataframe(pd.DataFrame(errors), use_container_width=True, hide_index=True)

st.subheader("2) Otonom araştırmayı başlat")
st.info(
    "AI sadece parametre değiştirmez. Donchian / EMA cross / momentum / RSI-breakout girişlerini, "
    "RVOL / trend / RSI / ADX filtrelerini ve Donchian / MA / ATR trailing / fixed-R çıkışlarını "
    "sonuçlara göre kendisi kombinleyebilir. OOS optimizasyon boyunca gizlidir."
)

if len(stock_map) < 5:
    st.warning(
        "Araştırma en az 5 hisseyle açılır. Tercihen 10–20 farklı sektörden hisse yükle. "
        "Az hisse overfit riskini ciddi artırır."
    )

can_run = len(stock_map) >= 5 and bool(api_key)

if st.button(
    "ARAŞTIRMAYI BAŞLAT",
    type="primary",
    use_container_width=True,
    disabled=not can_run,
):
    initial = StrategyConfig(
        entry_type=entry_type,
        breakout_days=int(breakout_days),
        ema_fast=int(ema_fast),
        ema_slow=int(ema_slow),
        momentum_days=int(momentum_days),
        use_rvol=use_rvol,
        rvol_lookback=int(rvol_lookback),
        rvol_min=float(rvol_min),
        use_trend_filter=use_trend_filter,
        trend_ma=int(trend_ma),
        use_rsi_filter=use_rsi_filter,
        rsi_period=int(rsi_period),
        rsi_min=float(rsi_min),
        rsi_max=float(rsi_max),
        use_adx_filter=use_adx_filter,
        adx_period=int(adx_period),
        adx_min=float(adx_min),
        exit_type=exit_type,
        exit_days=int(exit_days),
        exit_ma=int(exit_ma),
        atr_period=int(atr_period),
        atr_mult=float(atr_mult),
        target_r=float(target_r),
        use_initial_atr_stop=use_initial_atr_stop,
        initial_atr_mult=float(initial_atr_mult),
    )

    progress = st.progress(0)
    status = st.empty()
    live = st.empty()

    def callback(round_no, total, row):
        progress.progress(round_no / total)
        status.write(f"Tur {round_no}/{total} tamamlandı")
        vm = row["validation_metrics"]
        live.info(
            f"Validation — Medyan CAGR %{vm.get('median_cagr', 0)*100:.2f} | "
            f"Medyan Alpha {vm.get('median_alpha', 0)*100:+.2f} puan | "
            f"Medyan DD %{vm.get('median_max_dd', 0)*100:.2f} | "
            f"B&H geçen %{vm.get('beat_buyhold_pct', 0)*100:.1f}"
        )

    try:
        result = run_research(
            stock_map=stock_map,
            initial_strategy=initial,
            api_key=api_key,
            model=model,
            max_rounds=int(max_rounds),
            progress_callback=callback,
        )

        progress.progress(1.0)
        status.success("Araştırma tamamlandı.")

        st.subheader("3) Araştırma günlüğü")
        hist_rows = []
        for h in result["history"]:
            s = h["strategy"]
            vm = h["validation_metrics"]
            hist_rows.append({
                "Tur": h["round"],
                "Skor": h["score"],
                "Giriş": s["entry_type"],
                "Çıkış": s["exit_type"],
                "RVOL": "Açık" if s["use_rvol"] else "Kapalı",
                "Trend": "Açık" if s["use_trend_filter"] else "Kapalı",
                "RSI": "Açık" if s["use_rsi_filter"] else "Kapalı",
                "ADX": "Açık" if s["use_adx_filter"] else "Kapalı",
                "Val CAGR %": vm.get("median_cagr", 0) * 100,
                "Val Alpha": vm.get("median_alpha", 0) * 100,
                "Val Max DD %": vm.get("median_max_dd", 0) * 100,
                "Val PF": vm.get("median_pf", 0),
                "B&H geçen %": vm.get("beat_buyhold_pct", 0) * 100,
                "Teşhis": h.get("diagnosis", ""),
                "Hipotez": h.get("hypothesis", ""),
            })
        st.dataframe(pd.DataFrame(hist_rows), use_container_width=True, hide_index=True)

        st.subheader("4) En iyi aday")
        st.json(result["best_strategy"])

        oos = result["oos_metrics"]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("OOS Medyan CAGR", f"%{oos.get('median_cagr', 0)*100:.2f}")
        c2.metric("OOS Medyan Alpha", f"{oos.get('median_alpha', 0)*100:+.2f} puan")
        c3.metric("OOS Medyan Max DD", f"%{oos.get('median_max_dd', 0)*100:.2f}")
        c4.metric("OOS Medyan PF", f"{oos.get('median_pf', 0):.2f}")
        c5.metric("OOS B&H Geçen", f"%{oos.get('beat_buyhold_pct', 0)*100:.1f}")

        if result.get("oos_sample_ok"):
            st.success("OOS örneklem eşiği geçti. Yine de gerçek para öncesi daha geniş evren ve farklı dönem testleri gerekir.")
        else:
            st.warning("OOS YETERSİZ ÖRNEKLEM — sonuç yatırım kararı için kullanılamaz. Daha fazla hisse / işlem gerekli.")

        st.subheader("OOS hisse bazlı sonuçlar")
        oos_table = result["oos_table"].copy()
        for col in ["cagr", "buyhold_cagr", "alpha", "max_dd", "win_rate", "exposure"]:
            if col in oos_table.columns:
                oos_table[col] = oos_table[col] * 100
        st.dataframe(oos_table, use_container_width=True, hide_index=True)

        st.subheader("Tam dönem özeti")
        full = result["full_metrics"]
        st.json({
            "Medyan CAGR %": round(full.get("median_cagr", 0)*100, 2),
            "Medyan B&H CAGR %": round(full.get("median_buyhold_cagr", 0)*100, 2),
            "Medyan Alpha": round(full.get("median_alpha", 0)*100, 2),
            "Medyan Max DD %": round(full.get("median_max_dd", 0)*100, 2),
            "Medyan PF": round(full.get("median_pf", 0), 2),
            "Buy&Hold'u geçen %": round(full.get("beat_buyhold_pct", 0)*100, 1),
            "Toplam işlem": int(full.get("total_trades", 0)),
        })

    except Exception as e:
        st.exception(e)
