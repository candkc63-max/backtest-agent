from dataclasses import asdict

from ai_strategist import propose_strategy
from engine import evaluate_universe
from scoring import score_metrics
from strategy_schema import StrategyConfig, strategy_key


def run_research(
    stock_map: dict,
    initial_strategy: StrategyConfig,
    api_key: str,
    model: str = "gpt-5.6-sol",
    max_rounds: int = 12,
    progress_callback=None,
):
    current = initial_strategy
    seen = set()
    history = []

    for round_no in range(1, max_rounds + 1):
        key = strategy_key(current)
        if key in seen:
            break
        seen.add(key)

        train_metrics, train_table = evaluate_universe(stock_map, current, 'train')
        val_metrics, val_table = evaluate_universe(stock_map, current, 'validation')

        train_score = score_metrics(train_metrics)
        val_score = score_metrics(val_metrics)
        combined_score = round(train_score * 0.35 + val_score * 0.65, 4)

        row = {
            'round': round_no,
            'strategy': current.to_dict(),
            'train_metrics': train_metrics,
            'validation_metrics': val_metrics,
            'train_score': train_score,
            'validation_score': val_score,
            'score': combined_score,
            'reason': 'Başlangıç stratejisi' if round_no == 1 else history[-1].get('next_reason', ''),
        }
        history.append(row)

        if progress_callback:
            progress_callback(round_no, max_rounds, row)

        if round_no == max_rounds:
            break

        next_cfg, reason, raw = propose_strategy(
            current=current,
            train_metrics=train_metrics,
            validation_metrics=val_metrics,
            history=[
                {
                    'round': h['round'],
                    'strategy': h['strategy'],
                    'train_metrics': h['train_metrics'],
                    'validation_metrics': h['validation_metrics'],
                    'score': h['score'],
                }
                for h in history
            ],
            api_key=api_key,
            model=model,
        )

        row['next_reason'] = reason
        row['ai_raw'] = raw

        if strategy_key(next_cfg) in seen:
            break
        current = next_cfg

    if not history:
        raise ValueError('Araştırma turu oluşmadı.')

    ranked = sorted(history, key=lambda x: x['score'], reverse=True)
    best = ranked[0]
    best_cfg = StrategyConfig(**best['strategy'])

    # OOS sadece araştırma tamamlandıktan sonra ve yalnızca en iyi aday için açılır.
    oos_metrics, oos_table = evaluate_universe(stock_map, best_cfg, 'oos')
    full_metrics, full_table = evaluate_universe(stock_map, best_cfg, 'full')

    return {
        'history': history,
        'best': best,
        'best_strategy': best_cfg.to_dict(),
        'oos_metrics': oos_metrics,
        'oos_table': oos_table,
        'full_metrics': full_metrics,
        'full_table': full_table,
    }
