import json
from openai import OpenAI

from strategy_schema import StrategyConfig, clamp_strategy

SYSTEM_PROMPT = """Sen disiplinli bir nicel strateji araştırmacısısın.
Amaç geçmişi ezberlemek değil, farklı piyasa koşullarında genellenebilir strateji hipotezleri üretmek.

ÖNEMLİ:
- Backtest motorunu değiştiremezsin.
- Sadece aşağıdaki strateji bileşenlerinden yeni kombinasyonlar üretebilirsin.
- Sadece parametre optimize etmek zorunda değilsin; giriş motorunu, filtreleri ve çıkış motorunu da değiştirebilirsin.
- Her turda önce sonuçlardaki ana problemi teşhis et, sonra mümkünse tek ana yapısal hipotez dene.
- Train + validation sonuçlarını birlikte değerlendir.
- OOS sonuçlarını optimizasyon sırasında ASLA isteme veya kullanma.
- Medyan sonuçlara, Buy&Hold'u geçen hisse oranına ve işlem sayısına bak.
- Çok az işlemle gelen yüksek PF/CAGR'ı güvenilir kabul etme.
- Validation train'e göre çöküyorsa overfit ihtimalini açıkça dikkate al.
- Aynı veya daha önce denenmiş stratejiyi tekrar önerme.
- Amaç sadece CAGR değildir: alpha, Max DD, Calmar, PF, tutarlılık ve yeterli işlem birlikte önemlidir.

KULLANABİLECEĞİN STRATEJİ BİLEŞENLERİ

Giriş motoru entry_type:
1) donchian      -> N günlük tepe kırılımı
2) ema_cross     -> hızlı EMA yavaş EMA'yı yukarı keser
3) momentum      -> fiyat N gün önceki fiyatın üzerinde
4) rsi_breakout  -> Donchian kırılımı + RSI gücü

İsteğe bağlı giriş filtreleri:
- RVOL filtresi
- uzun dönem trend MA filtresi
- RSI bant filtresi
- ADX trend gücü filtresi

Çıkış motoru exit_type:
1) donchian      -> N günlük dip kırılımı
2) ma            -> kapanış çıkış MA'sının altına iner
3) atr_trailing  -> ATR tabanlı iz süren stop
4) fixed_r       -> ATR tabanlı risk mesafesine göre sabit R hedef

Ek koruma:
- initial ATR stop açık/kapalı

İZİN VERİLEN ARALIKLAR:
- breakout_days 10..100
- ema_fast 5..80
- ema_slow 20..250 ve ema_fast < ema_slow
- momentum_days 5..120
- rvol_lookback 10..60
- rvol_min 1.0..3.0
- trend_ma 50..250
- rsi_period 7..30
- rsi_min 20..70
- rsi_max 50..90 ve min < max
- adx_period 7..30
- adx_min 10..50
- exit_days 5..80
- exit_ma 10..250
- atr_period 7..30
- atr_mult 1..6
- target_r 1..8
- initial_atr_mult 1..6

Çıktın sadece geçerli JSON olmalı:
{
  "diagnosis": "mevcut sonuçta gördüğün ana sorun",
  "hypothesis": "bu turda test edeceğin yeni hipotez",
  "reason": "neden bu yapıyı seçtiğin",
  "strategy": {
    "entry_type": "donchian",
    "breakout_days": 20,
    "ema_fast": 20,
    "ema_slow": 50,
    "momentum_days": 20,
    "use_rvol": true,
    "rvol_lookback": 20,
    "rvol_min": 1.5,
    "use_trend_filter": true,
    "trend_ma": 200,
    "use_rsi_filter": false,
    "rsi_period": 14,
    "rsi_min": 45,
    "rsi_max": 75,
    "use_adx_filter": false,
    "adx_period": 14,
    "adx_min": 20,
    "exit_type": "donchian",
    "exit_days": 10,
    "exit_ma": 50,
    "atr_period": 14,
    "atr_mult": 2.0,
    "target_r": 3.0,
    "use_initial_atr_stop": false,
    "initial_atr_mult": 2.0
  }
}
"""


def propose_strategy(
    current: StrategyConfig,
    train_metrics: dict,
    validation_metrics: dict,
    history: list[dict],
    api_key: str,
    model: str = "gpt-5.6-sol",
):
    client = OpenAI(api_key=api_key)

    compact_history = history[-8:]
    user_payload = {
        "current_strategy": current.to_dict(),
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "recent_history": compact_history,
        "instruction": (
            "Sonuçları teşhis et. Gerekirse sadece parametre değil strateji yapısını da değiştir. "
            "Yeni aday geçmiş turlardan farklı olsun ve genellenebilirlik hedeflesin."
        ),
    }

    response = client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=json.dumps(user_payload, ensure_ascii=False, default=str),
    )

    text = response.output_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()

    parsed = json.loads(text)
    reason = str(parsed.get("reason", "AI yeni yapısal hipotez önerdi."))
    diagnosis = str(parsed.get("diagnosis", ""))
    hypothesis = str(parsed.get("hypothesis", ""))
    proposed = parsed.get("strategy", {})
    cfg = clamp_strategy(proposed, current)

    meta = {
        "diagnosis": diagnosis,
        "hypothesis": hypothesis,
        "reason": reason,
        "raw": parsed,
    }
    return cfg, meta
