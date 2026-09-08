from dataclasses import dataclass, asdict
from typing import Optional


@dataclass(frozen=True)
class StrategyConfig:
    breakout_days: int = 20
    exit_days: int = 10
    use_rvol: bool = True
    rvol_lookback: int = 20
    rvol_min: float = 1.5
    use_trend_filter: bool = True
    trend_ma: int = 200
    use_atr_stop: bool = False
    atr_period: int = 14
    atr_mult: float = 2.0

    def to_dict(self):
        return asdict(self)


BOUNDS = {
    "breakout_days": (10, 100),
    "exit_days": (5, 60),
    "rvol_lookback": (10, 60),
    "rvol_min": (1.0, 3.0),
    "trend_ma": (50, 250),
    "atr_period": (10, 30),
    "atr_mult": (1.0, 5.0),
}

BOOL_FIELDS = {"use_rvol", "use_trend_filter", "use_atr_stop"}


def clamp_strategy(payload: dict, fallback: StrategyConfig) -> StrategyConfig:
    base = fallback.to_dict()

    for key, value in payload.items():
        if key not in base:
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

    return StrategyConfig(**base)


def strategy_key(cfg: StrategyConfig) -> tuple:
    d = cfg.to_dict()
    return tuple((k, d[k]) for k in sorted(d))
