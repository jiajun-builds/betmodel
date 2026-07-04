# LigaMX Betting Tool

Command-line tool for finding potential positive EV Liga MX spread bets using historical match data, xG reconciliation, precomputed simulations, and Pinnacle odds from The Odds API.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill `.env` with:

```bash
API_FOOTBALL_API_KEY=...
THE_ODDS_API_KEY=...
```

## Usage

```bash
python main.py
python main.py --date 2026-04-18
python main.py --no-update
```

Generated CSV output is written to `output/`.

## Project Structure

```text
src/
  odds_client.py
  sofascore_scraper.py
  xg_calculator.py
  prediction_model.py
  ev_calculator.py
  cli_output.py
  data_updater.py
main.py
DC_MEX.py
data/
models/
tests/
```
