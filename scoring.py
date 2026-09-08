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
    Amaç: yüksek medyan alpha + tutarlılık + makul DD/Calmar/PF.
    """
    alpha = safe(m.get('median_alpha'))
    cagr = safe(m.get('median_cagr'))
    max_dd = abs(safe(m.get('median_max_dd')))
    calmar = safe(m.get('median_calmar'))
    pf = safe(m.get('median_pf'))
    beat = safe(m.get('beat_buyhold_pct'))
    trades = safe(m.get('median_trades'))

    # Yüzdeler unit fraction olarak gelir: 0.12 = %12.
    score = 0.0
    score += alpha * 100 * 3.0
    score += cagr * 100 * 0.8
    score += beat * 30.0
    score += min(max(calmar, 0.0), 3.0) * 8.0
    score += min(max(pf - 1.0, 0.0), 3.0) * 5.0
    score -= max(max_dd * 100 - 20.0, 0.0) * 0.7

    # Çok az işlemle gelen aşırı sonuçları cezalandır.
    if trades < 10:
        score -= (10 - trades) * 2.0

    return round(score, 4)


def pass_validation(train: dict, validation: dict) -> bool:
    """Validation kapısı: alpha/tutarlılık/risk için kaba minimumlar."""
    return (
        safe(validation.get('median_alpha')) > 0
        and safe(validation.get('beat_buyhold_pct')) >= 0.50
        and safe(validation.get('median_pf')) >= 1.20
        and safe(validation.get('median_max_dd')) > -0.45
        and safe(validation.get('median_trades')) >= 3
    )
