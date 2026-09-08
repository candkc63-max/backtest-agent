from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class StrategyConfig:
    # Giriş motoru
    entry_type: str = "donchian"  # donchian | ema_cross | momentum | rsi_breakout
    breakout_days: int = 20
    ema_fast: int = 20
    ema_slow: int = 50
    momentum_days: int = 20

    # Giriş filtreleri
    use_rvol: bool = True
    rvol_lookback: int = 20
    rvol_min: float = 1.5

    use_trend_filter: bool = True
    trend_ma: int = 200

    use_rsi_filter: bool = False
    rsi_period: int = 14
    rsi_min: float = 45.0
    rsi_max: float = 75.0

    use_adx_filter: bool = False
    adx_period: int = 14
    adx_min: float = 20.0

    # Çıkış motoru
    exit_type: str = "donchian"  # donchian | ma | atr_trailing | fixed_r
    exit_days: int = 10
    exit_ma: int = 50
    atr_period: int = 14
    atr_mult: float = 2.0
    target_r: float = 3.0

    # Koruyucu stop
    use_initial_atr_stop: bool = False
    initial_atr_mult: float = 2.0

    def to_dict(self):
        return asdict(self)


CHOICES = {
    "entry_type": {"donchian", "ema_cross", "momentum", "rsi_breakout"},
    "exit_type": {"donchian", "ma", "atr_trailing", "fixed_r"},
}

BOUNDS = {
    "breakout_days": (10, 100),
    "ema_fast": (5, 80),
    "ema_slow": (20, 250),
    "momentum_days": (5, 120),
    "rvol_lookback": (10, 60),
    "rvol_min": (1.0, 3.0),
    "trend_ma": (50, 250),
    "rsi_period": (7, 30),
    "rsi_min": (20.0, 70.0),
    "rsi_max": (50.0, 90.0),
    "adx_period": (7, 30),
    "adx_min": (10.0, 50.0),
    "exit_days": (5, 80),
    "exit_ma": (10, 250),
    "atr_period": (7, 30),
    "atr_mult": (1.0, 6.0),
    "target_r": (1.0, 8.0),
    "initial_atr_mult": (1.0, 6.0),
}

BOOL_FIELDS = {
    "use_rvol",
    "use_trend_filter",
    "use_rsi_filter",
    "use_adx_filter",
    "use_initial_atr_stop",
}


def clamp_strategy(payload: dict, fallback: StrategyConfig) -> StrategyConfig:
    base = fallback.to_dict()

    for key, value in payload.items():
        if key not in base:
            continue

        if key in CHOICES:
            if isinstance(value, str) and value in CHOICES[key]:
                base[key] = value
            continue

        if key in BOOL_FIELDS:
            if isinstance(value, bool):
                base[key] = value
            continue

        if key in BOUNDS:
            lo, hi = BOUNDS[key]
            try:
                x = float(value)
            except (TypeError, ValueError):
                continue
            x = min(max(x, lo), hi)
            if isinstance(base[key], int):
                x = int(round(x))
            base[key] = x

    # Mantıksal korumalar
    if base["ema_fast"] >= base["ema_slow"]:
        base["ema_fast"] = max(5, base["ema_slow"] // 2)
    if base["rsi_min"] >= base["rsi_max"]:
        base["rsi_min"] = min(60.0, base["rsi_max"] - 5.0)

    return StrategyConfig(**base)


def strategy_key(cfg: StrategyConfig) -> tuple:
    d = cfg.to_dict()
    return tuple((k, d[k]) for k in sorted(d))
