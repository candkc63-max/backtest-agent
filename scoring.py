import math


def safe(x, default=0.0):
    try:
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except Exception:
        return default


def score_metrics(m: dict) -> float:
    """
    Araştırma skoru. Tek başına yatırım kararı değildir.
    Amaç: Buy&Hold'a karşı pozitif alpha, farklı hisselerde tutarlılık,
    makul drawdown ve yeterli işlem örneklemi.
    """
    alpha = safe(m.get('median_alpha'))
    cagr = safe(m.get('median_cagr'))
    max_dd = abs(safe(m.get('median_max_dd')))
    calmar = safe(m.get('median_calmar'))
    pf = safe(m.get('median_pf'))
    beat = safe(m.get('beat_buyhold_pct'))
    trades = safe(m.get('median_trades'))
    total_trades = safe(m.get('total_trades'))

    # Yüzdeler fraction olarak gelir: 0.12 = %12.
    score = 0.0

    # Benchmarkı geçmek ana hedef.
    score += alpha * 100 * 4.0
    score += beat * 35.0

    # Mutlak büyüme de önemli, fakat alpha kadar değil.
    score += cagr * 100 * 0.6

    # Risk ayarlı kalite.
    score += min(max(calmar, 0.0), 3.0) * 10.0
    score += min(max(pf - 1.0, 0.0), 3.0) * 5.0

    # %20 üzeri DD gittikçe daha sert cezalandırılır.
    score -= max(max_dd * 100 - 20.0, 0.0) * 1.0

    # Az işlemle gelen parlak sonuçları cezalandır.
    if trades < 5:
        score -= (5 - trades) * 4.0
    if total_trades < 80:
        score -= (80 - total_trades) * 0.15

    return round(score, 4)


def pass_validation(train: dict, validation: dict) -> bool:
    """
    Güçlü aday kapısı.

    Bir strateji ancak hem train hem validation'da benchmark üstünlüğü gösterirse,
    farklı hisselerde yeterince tutarlıysa ve risk/örneklem şartlarını geçerse
    güçlü aday sayılır.
    """
    return (
        # Train'de de edge olmalı; validation tesadüfü istemiyoruz.
        safe(train.get('median_alpha')) > 0
        and safe(train.get('beat_buyhold_pct')) >= 0.55

        # Validation'da daha sıkı benchmark şartı.
        and safe(validation.get('median_alpha')) >= 0.02       # en az +2 puan yıllık alpha
        and safe(validation.get('beat_buyhold_pct')) >= 0.60  # hisselerin en az %60'ında B&H üstü

        # İşlem kalitesi / risk ayarlı getiri.
        and safe(validation.get('median_pf')) >= 1.50
        and safe(validation.get('median_calmar')) >= 0.80
        and safe(validation.get('median_max_dd')) >= -0.35

        # Örneklem büyüklüğü.
        and safe(validation.get('median_trades')) >= 5
        and safe(validation.get('total_trades')) >= 80
    )
