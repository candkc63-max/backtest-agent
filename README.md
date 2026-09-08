# Backtest Agent v2

Autonomous strategy research app for daily stock CSV files.

## What it does

1. Loads multiple daily stock CSV files.
2. Runs a fixed, non-editable backtest engine.
3. Splits each stock chronologically into Train / Validation / Out-of-Sample.
4. Sends only Train + Validation metrics to the AI strategist.
5. Lets the AI change only bounded strategy parameters.
6. Repeats backtest → analysis → new strategy for N rounds.
7. Opens OOS only once, after research, for the best candidate.

## Required CSV columns

- time
- open
- high
- low
- close
- volume

Data must be daily (1D).

## Streamlit Cloud setup

Add this secret in Streamlit app settings if you do not want to paste the API key each session:

```toml
OPENAI_API_KEY="your-key-here"
```

The app also supports entering the key in a password field for the current session.

## Files

- `streamlit_app.py`: UI
- `engine.py`: locked backtest engine
- `strategy_schema.py`: allowed strategy parameters and bounds
- `scoring.py`: robust multi-metric ranking
- `ai_strategist.py`: OpenAI-based hypothesis generator
- `optimizer.py`: autonomous research loop
- `requirements.txt`: dependencies

## Important

This is a research tool, not an execution engine. A strong in-sample result is not enough. Out-of-sample performance, transaction-cost sensitivity, broader universes and later paper/live testing are still required.
