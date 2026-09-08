import json
from openai import OpenAI

from strategy_schema import StrategyConfig, clamp_strategy

SYSTEM_PROMPT = """Sen disiplinli bir nicel strateji araştırmacısısın.
Amacın geçmiş veriyi ezberlemek değil, sağlam ve genellenebilir hipotezler üretmek.

Kurallar:
- Backtest motorunu değiştiremezsin.
- Sadece izin verilen strateji parametrelerini değiştirebilirsin.
- Her iterasyonda mümkünse tek ana hipotez değiştir.
- Train ve validation sonuçlarını birlikte değerlendir.
- Out-of-sample sonuçlarını optimizasyon sırasında ASLA isteme veya kullanma.
- Tek hissedeki uçuk sonuca değil medyan metriklere ve hisselerin ne kadarında Buy&Hold'un geçildiğine bak.
- CAGR tek başına yeterli değildir; alpha, Max DD, Calmar, PF, işlem sayısı ve tutarlılık birlikte önemlidir.
- Çok az işlemle gelen yüksek PF/CAGR'ı güvenilir kabul etme.
- Çıktı sadece geçerli JSON olsun. Markdown kullanma.

JSON şeması:
{
  "reason": "kısa hipotez açıklaması",
  "strategy": {
    "breakout_days": 20,
    "exit_days": 10,
    "use_rvol": true,
    "rvol_lookback": 20,
    "rvol_min": 1.5,
    "use_trend_filter": true,
    "trend_ma": 200,
    "use_atr_stop": false,
    "atr_period": 14,
    "atr_mult": 2.0
  }
}
"""


def propose_strategy(
    current: StrategyConfig,
    train_metrics: dict,
    validation_metrics: dict,
    history: list[dict],
    api_key: str,
    model: str = "gpt-5.6"
):
    client = OpenAI(api_key=api_key)

    compact_history = history[-6:]
    user_payload = {
        "current_strategy": current.to_dict(),
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "recent_history": compact_history,
        "allowed_ranges": {
            "breakout_days": [10, 100],
            "exit_days": [5, 60],
            "rvol_lookback": [10, 60],
            "rvol_min": [1.0, 3.0],
            "trend_ma": [50, 250],
            "atr_period": [10, 30],
            "atr_mult": [1.0, 5.0]
        }
    }

    response = client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=json.dumps(user_payload, ensure_ascii=False),
    )

    text = response.output_text.strip()

    # Model zaman zaman code fence döndürürse temizle.
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()

    parsed = json.loads(text)
    reason = str(parsed.get("reason", "AI yeni hipotez önerdi."))
    proposed = parsed.get("strategy", {})
    cfg = clamp_strategy(proposed, current)
    return cfg, reason, parsed
