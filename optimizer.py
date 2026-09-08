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

        train_metrics, _ = evaluate_universe(stock_map, current, 'train')
        val_metrics, _ = evaluate_universe(stock_map, current, 'validation')

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
            'diagnosis': '',
            'hypothesis': 'Başlangıç stratejisi' if round_no == 1 else '',
            'reason': 'Başlangıç stratejisi' if round_no == 1 else '',
        }
        history.append(row)

        if progress_callback:
            progress_callback(round_no, max_rounds, row)

        if round_no == max_rounds:
            break

        next_cfg, meta = propose_strategy(
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
                    'diagnosis': h.get('diagnosis', ''),
                    'hypothesis': h.get('hypothesis', ''),
                }
                for h in history
            ],
            api_key=api_key,
            model=model,
        )

        row['diagnosis'] = meta.get('diagnosis', '')
        row['hypothesis'] = meta.get('hypothesis', '')
        row['reason'] = meta.get('reason', '')
        row['ai_raw'] = meta.get('raw', {})

        if strategy_key(next_cfg) in seen:
            # AI aynı fikre döndüyse araştırmayı erken bitir.
            break

        current = next_cfg

    if not history:
        raise ValueError('Araştırma turu oluşmadı.')

    ranked = sorted(history, key=lambda x: x['score'], reverse=True)
    best = ranked[0]
    best_cfg = StrategyConfig(**best['strategy'])

    # OOS yalnızca araştırma tamamlandıktan sonra açılır.
    oos_metrics, oos_table = evaluate_universe(stock_map, best_cfg, 'oos')
    full_metrics, full_table = evaluate_universe(stock_map, best_cfg, 'full')

    # OOS örneklem kalitesi ayrı raporlanır; optimizer bunu görmez.
    oos_sample_ok = (
        oos_metrics.get('total_trades', 0) >= max(15, len(stock_map) * 2)
        and oos_metrics.get('n_stocks', 0) >= 3
    )

    return {
        'history': history,
        'best': best,
        'best_strategy': best_cfg.to_dict(),
        'oos_metrics': oos_metrics,
        'oos_table': oos_table,
        'oos_sample_ok': oos_sample_ok,
        'full_metrics': full_metrics,
        'full_table': full_table,
    }
