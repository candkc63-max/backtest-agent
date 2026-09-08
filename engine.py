import math
import os
import pandas as pd
import numpy as np

from strategy_schema import StrategyConfig

START_CAPITAL = 100_000.0
COMMISSION = 0.001


def symbol_from_name(name: str) -> str:
    base = os.path.basename(name).upper().replace('.CSV', '')
    base = base.split(',')[0]
    for prefix in ('BIST_', 'BVC_DLY_', 'BVC_', 'DLY_'):
        base = base.replace(prefix, '')
    return base.strip()


def prepare_daily(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()
    required = {'time', 'open', 'high', 'low', 'close', 'volume'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Eksik sütunlar: {sorted(missing)}")

    if pd.api.types.is_numeric_dtype(df['time']):
        df['date'] = pd.to_datetime(df['time'], unit='s', errors='coerce')
    else:
        df['date'] = pd.to_datetime(df['time'], errors='coerce')

    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    df = (
        df.dropna(subset=['date', 'open', 'high', 'low', 'close', 'volume'])
          .sort_values('date')
          .drop_duplicates('date')
          .reset_index(drop=True)
    )

    if len(df) < 300:
        raise ValueError('En az yaklaşık 300 günlük veri gerekli.')

    diff_hours = df['date'].diff().dropna().dt.total_seconds().div(3600)
    if len(diff_hours) and diff_hours.median() < 20:
        raise ValueError('Veri günlük görünmüyor. 1D CSV gerekli.')

    return df


def add_indicators(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    out = df.copy()
    out['breakout_high'] = out['high'].shift(1).rolling(cfg.breakout_days).max()
    out['exit_low'] = out['low'].shift(1).rolling(cfg.exit_days).min()
    out['trend_ma'] = out['close'].rolling(cfg.trend_ma).mean()
    out['vol_ma'] = out['volume'].shift(1).rolling(cfg.rvol_lookback).mean()
    out['rvol'] = out['volume'] / out['vol_ma'].replace(0, np.nan)

    prev_close = out['close'].shift(1)
    tr = pd.concat([
        out['high'] - out['low'],
        (out['high'] - prev_close).abs(),
        (out['low'] - prev_close).abs(),
    ], axis=1).max(axis=1)
    out['atr'] = tr.ewm(alpha=1 / cfg.atr_period, adjust=False).mean()
    return out


def cagr(start_value: float, end_value: float, years: float) -> float:
    if years <= 0 or start_value <= 0 or end_value <= 0:
        return np.nan
    return (end_value / start_value) ** (1 / years) - 1


def _slice_bounds(df: pd.DataFrame, split: str, warmup: int):
    usable_start = warmup
    usable_n = len(df) - usable_start
    if usable_n < 100:
        raise ValueError('Warm-up sonrası test için yeterli veri yok.')

    train_end = usable_start + int(usable_n * 0.60)
    val_end = usable_start + int(usable_n * 0.80)

    if split == 'train':
        return usable_start, max(train_end, usable_start + 2)
    if split == 'validation':
        return train_end, max(val_end, train_end + 2)
    if split == 'oos':
        return val_end, len(df)
    if split == 'full':
        return usable_start, len(df)
    raise ValueError(f'Bilinmeyen split: {split}')


def backtest_one(df: pd.DataFrame, cfg: StrategyConfig, split: str = 'full') -> dict:
    data = add_indicators(df, cfg)
    warmup = max(cfg.breakout_days, cfg.exit_days, cfg.rvol_lookback, cfg.trend_ma, cfg.atr_period) + 2
    start_i, end_i = _slice_bounds(data, split, warmup)

    if end_i - start_i < 30:
        raise ValueError('Seçilen dönem çok kısa.')

    test_start = data.iloc[start_i]['date']
    test_end = data.iloc[end_i - 1]['date']
    years = (test_end - test_start).total_seconds() / (365.25 * 24 * 3600)
    if years <= 0:
        raise ValueError('Test süresi hesaplanamadı.')

    # Buy & hold: aynı pencere, aynı komisyon.
    bh_entry = float(data.iloc[start_i]['open'])
    bh_exit = float(data.iloc[end_i - 1]['close'])
    bh_qty = START_CAPITAL / (bh_entry * (1 + COMMISSION))
    bh_final = bh_qty * bh_exit * (1 - COMMISSION)
    bh_cagr = cagr(START_CAPITAL, bh_final, years)

    cash = START_CAPITAL
    qty = 0.0
    in_pos = False
    pending_buy = False
    pending_sell = False
    entry_price = entry_cost = entry_date = None
    atr_stop = None
    trades = []
    equity = [START_CAPITAL]
    position_bars = 0
    bars = 0

    for i in range(start_i, end_i):
        row = data.iloc[i]
        op = float(row['open'])
        cl = float(row['close'])
        dt = row['date']
        bars += 1

        if pending_sell and in_pos:
            exit_price = op
            gross = qty * exit_price
            net = gross * (1 - COMMISSION)
            pnl = net - entry_cost
            trades.append({'entry': entry_date, 'exit': dt, 'pnl': pnl, 'return': pnl / entry_cost})
            cash = net
            qty = 0.0
            in_pos = False
            entry_price = entry_cost = entry_date = atr_stop = None
            pending_sell = False

        if pending_buy and not in_pos:
            entry_price = op
            qty = cash / (entry_price * (1 + COMMISSION))
            entry_cost = qty * entry_price * (1 + COMMISSION)
            entry_date = dt
            cash = 0.0
            in_pos = True
            pending_buy = False
            if cfg.use_atr_stop and pd.notna(row['atr']):
                atr_stop = entry_price - float(row['atr']) * cfg.atr_mult

        if not in_pos:
            buy_signal = pd.notna(row['breakout_high']) and cl > float(row['breakout_high'])
            if cfg.use_rvol:
                buy_signal = buy_signal and pd.notna(row['rvol']) and float(row['rvol']) >= cfg.rvol_min
            if cfg.use_trend_filter:
                buy_signal = buy_signal and pd.notna(row['trend_ma']) and cl > float(row['trend_ma'])
            if buy_signal and i + 1 < end_i:
                pending_buy = True
        else:
            exit_signal = pd.notna(row['exit_low']) and cl < float(row['exit_low'])
            if cfg.use_atr_stop and atr_stop is not None and float(row['low']) <= atr_stop:
                # Günlük veride intraday sıralama bilinmediğinden stop kapanışta tetiklenir,
                # gerçek çıkış sonraki açılışta yapılır. Motor look-ahead kullanmaz.
                exit_signal = True
            if exit_signal and i + 1 < end_i:
                pending_sell = True

        if in_pos:
            position_bars += 1
            equity_now = qty * cl * (1 - COMMISSION)
        else:
            equity_now = cash
        equity.append(equity_now)

    if in_pos:
        last = data.iloc[end_i - 1]
        exit_price = float(last['close'])
        net = qty * exit_price * (1 - COMMISSION)
        pnl = net - entry_cost
        trades.append({'entry': entry_date, 'exit': last['date'], 'pnl': pnl, 'return': pnl / entry_cost})
        cash = net
        equity.append(cash)

    final_capital = cash
    strat_cagr = cagr(START_CAPITAL, final_capital, years)
    total_return = final_capital / START_CAPITAL - 1

    t = pd.DataFrame(trades)
    if len(t):
        gp = t.loc[t['pnl'] > 0, 'pnl'].sum()
        gl = abs(t.loc[t['pnl'] < 0, 'pnl'].sum())
        pf = gp / gl if gl > 0 else np.inf
        win_rate = (t['pnl'] > 0).mean()
        avg_trade = t['return'].mean()
    else:
        pf = win_rate = avg_trade = np.nan

    eq = pd.Series(equity, dtype=float)
    max_dd = (eq / eq.cummax() - 1).min() if len(eq) else np.nan
    calmar = strat_cagr / abs(max_dd) if pd.notna(max_dd) and max_dd < 0 else np.nan

    return {
        'start': test_start,
        'end': test_end,
        'years': years,
        'cagr': strat_cagr,
        'buyhold_cagr': bh_cagr,
        'alpha': strat_cagr - bh_cagr,
        'total_return': total_return,
        'max_dd': max_dd,
        'calmar': calmar,
        'pf': pf,
        'win_rate': win_rate,
        'avg_trade': avg_trade,
        'trades': int(len(t)),
        'trades_per_year': len(t) / years if years > 0 else np.nan,
        'exposure': position_bars / bars if bars else np.nan,
        'final_capital': final_capital,
    }


def aggregate_results(per_stock: list[dict]) -> dict:
    if not per_stock:
        raise ValueError('Agregasyon için sonuç yok.')

    frame = pd.DataFrame(per_stock)
    numeric = ['cagr', 'buyhold_cagr', 'alpha', 'max_dd', 'calmar', 'pf', 'win_rate', 'avg_trade', 'trades', 'trades_per_year', 'exposure']
    out = {f'median_{c}': float(frame[c].replace([np.inf, -np.inf], np.nan).median()) for c in numeric}
    out['n_stocks'] = int(len(frame))
    out['beat_buyhold_pct'] = float((frame['alpha'] > 0).mean())
    out['positive_pf_pct'] = float((frame['pf'] > 1).mean())
    out['pf_15_pct'] = float((frame['pf'] >= 1.5).mean())
    out['min_trades'] = int(frame['trades'].min())
    return out


def evaluate_universe(stock_map: dict[str, pd.DataFrame], cfg: StrategyConfig, split: str) -> tuple[dict, pd.DataFrame]:
    rows = []
    for symbol, raw in stock_map.items():
        m = backtest_one(raw, cfg, split=split)
        rows.append({'symbol': symbol, **m})
    return aggregate_results(rows), pd.DataFrame(rows)
