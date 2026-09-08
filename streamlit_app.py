import os
import pandas as pd
import streamlit as st

from engine import prepare_daily, symbol_from_name
from optimizer import run_research
from strategy_schema import StrategyConfig

st.set_page_config(page_title="Backtest Agent v2", layout="wide")
st.title("Backtest Agent v2 — Autonomous Research")
st.caption(
    "Günlük hisse CSV'lerini yükle → AI stratejiyi iteratif olarak geliştirir → "
    "Train/Validation ile optimize eder → OOS'u yalnızca en sonda açar."
)

with st.sidebar:
    st.header("Başlangıç Stratejisi")
    breakout_days = st.number_input("Breakout günü", 10, 100, 20, 1)
    exit_days = st.number_input("Çıkış günü", 5, 60, 10, 1)
    use_rvol = st.checkbox("RVOL filtresi", True)
    rvol_lookback = st.number_input("RVOL ortalama günü", 10, 60, 20, 1)
    rvol_min = st.number_input("Minimum RVOL", 1.0, 3.0, 1.5, 0.1)
    use_trend = st.checkbox("Trend filtresi", True)
    trend_ma = st.number_input("Trend MA", 50, 250, 200, 10)
    use_atr_stop = st.checkbox("ATR stop", False)
    atr_period = st.number_input("ATR periyodu", 10, 30, 14, 1)
    atr_mult = st.number_input("ATR çarpanı", 1.0, 5.0, 2.0, 0.25)

    st.divider()
    st.header("Araştırma")
    max_rounds = st.slider("Maksimum AI turu", 2, 30, 10)
    model = st.selectbox(
        "AI modeli",
        ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"],
        index=1,
    )
    st.caption("Terra varsayılan: kalite/maliyet dengesi. Sol daha güçlü, Luna daha ucuz.")

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
    help="Anahtar sadece bu oturumdaki API çağrıları için kullanılır; uygulama tarafından kaydedilmez.",
)

if files:
    valid_rows = []
    stock_map = {}
    errors = []

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

    if len(stock_map) < 3:
        st.warning(
            "Otonom optimizasyon için en az 3, tercihen 10–20 farklı hisse kullan. "
            "Az hisse overfit riskini ciddi artırır."
        )

    initial = StrategyConfig(
        breakout_days=int(breakout_days),
        exit_days=int(exit_days),
        use_rvol=bool(use_rvol),
        rvol_lookback=int(rvol_lookback),
        rvol_min=float(rvol_min),
        use_trend_filter=bool(use_trend),
        trend_ma=int(trend_ma),
        use_atr_stop=bool(use_atr_stop),
        atr_period=int(atr_period),
        atr_mult=float(atr_mult),
    )

    st.subheader("2) Otonom araştırmayı başlat")
    st.info(
        "AI yalnızca Train + Validation sonuçlarını görür. Out-of-sample verisi "
        "araştırma bitene kadar gizlidir ve sadece en iyi aday için bir kez açılır."
    )

    start = st.button(
        "ARAŞTIRMAYI BAŞLAT",
        type="primary",
        use_container_width=True,
        disabled=(not api_key or not stock_map),
    )

    if not api_key:
        st.caption("Araştırmayı başlatmak için API key gerekli.")

    if start:
        progress = st.progress(0)
        status = st.empty()
        live = st.empty()

        def on_progress(round_no, total, row):
            progress.progress(round_no / total)
            status.write(f"Tur {round_no}/{total} tamamlandı")
            v = row["validation_metrics"]
            live.info(
                f"Validation — Medyan CAGR %{v['median_cagr']*100:.2f} | "
                f"Medyan Alpha {v['median_alpha']*100:+.2f} puan | "
                f"Medyan DD %{v['median_max_dd']*100:.2f} | "
                f"B&H geçen %{v['beat_buyhold_pct']*100:.1f}"
            )

        try:
            result = run_research(
                stock_map=stock_map,
                initial_strategy=initial,
                api_key=api_key,
                model=model,
                max_rounds=max_rounds,
                progress_callback=on_progress,
            )

            progress.progress(1.0)
            status.success("Araştırma tamamlandı.")

            st.subheader("3) Leaderboard")
            leaderboard_rows = []
            for h in result["history"]:
                tr = h["train_metrics"]
                va = h["validation_metrics"]
                s = h["strategy"]
                leaderboard_rows.append({
                    "Tur": h["round"],
                    "Skor": h["score"],
                    "Breakout": s["breakout_days"],
                    "Exit": s["exit_days"],
                    "RVOL": s["rvol_min"] if s["use_rvol"] else "Kapalı",
                    "Trend MA": s["trend_ma"] if s["use_trend_filter"] else "Kapalı",
                    "ATR Stop": s["atr_mult"] if s["use_atr_stop"] else "Kapalı",
                    "Train CAGR %": tr["median_cagr"] * 100,
                    "Val CAGR %": va["median_cagr"] * 100,
                    "Val Alpha": va["median_alpha"] * 100,
                    "Val Max DD %": va["median_max_dd"] * 100,
                    "Val PF": va["median_pf"],
                    "Val B&H Geçen %": va["beat_buyhold_pct"] * 100,
                })

            leaderboard = pd.DataFrame(leaderboard_rows).sort_values("Skor", ascending=False)
            st.dataframe(leaderboard.round(2), use_container_width=True, hide_index=True)

            best = result["best_strategy"]
            st.subheader("4) En iyi aday")
            st.json(best)

            oos = result["oos_metrics"]
            full = result["full_metrics"]

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("OOS Medyan CAGR", f"%{oos['median_cagr']*100:.2f}")
            c2.metric("OOS Medyan Alpha", f"{oos['median_alpha']*100:+.2f} puan")
            c3.metric("OOS Medyan Max DD", f"%{oos['median_max_dd']*100:.2f}")
            c4.metric("OOS Medyan PF", f"{oos['median_pf']:.2f}")
            c5.metric("OOS B&H Geçen", f"%{oos['beat_buyhold_pct']*100:.1f}")

            st.caption(
                "OOS sonuçları optimizasyon sırasında AI'ya gösterilmedi. "
                "Bu bölüm en önemli doğrulama katmanıdır."
            )

            st.subheader("OOS hisse bazlı sonuçlar")
            oos_table = result["oos_table"].copy()
            show_cols = [
                "symbol", "years", "cagr", "buyhold_cagr", "alpha", "max_dd",
                "calmar", "pf", "win_rate", "trades", "exposure"
            ]
            oos_table = oos_table[show_cols]
            for c in ["cagr", "buyhold_cagr", "alpha", "max_dd", "win_rate", "exposure"]:
                oos_table[c] = oos_table[c] * 100
            st.dataframe(oos_table.round(2), use_container_width=True, hide_index=True)

            st.subheader("Tam dönem özeti")
            st.write({
                "Medyan CAGR %": round(full["median_cagr"] * 100, 2),
                "Medyan B&H CAGR %": round(full["median_buyhold_cagr"] * 100, 2),
                "Medyan Alpha": round(full["median_alpha"] * 100, 2),
                "Medyan Max DD %": round(full["median_max_dd"] * 100, 2),
                "Medyan PF": round(full["median_pf"], 2),
                "Buy&Hold'u geçen %": round(full["beat_buyhold_pct"] * 100, 1),
            })

            st.warning(
                "Bu sistem araştırma aracıdır. OOS iyi olsa bile gerçek para öncesi "
                "daha geniş evren, farklı dönemler ve işlem maliyeti hassasiyet testi gerekir."
            )

        except Exception as e:
            st.exception(e)
else:
    st.info("Başlamak için birden fazla günlük (1D) hisse CSV'si yükle.")
