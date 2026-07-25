# LigaMX Terminal

Local workflow for finding potential positive-EV Liga MX bets from historical match
data, SofaScore xG reconciliation, a Dixon-Coles style goals model, and Pinnacle
odds via The Odds API. Outputs a static dashboard published to GitHub Pages.

See `EV Calculation Logic.md` for the EV/Asian-handicap formulas.

## Setup

This project uses **conda**, not a venv — the environment is defined in
`environment.yml` and the scripts activate it for you.

```bash
conda env create -f environment.yml   # creates the `ligamx-workflows` env
cp .env.local.example .env.local
```

Fill `.env.local` with your key:

```bash
THE_ODDS_API_KEY=...
```

`requirements.txt` mirrors `environment.yml` for pip-only contexts (CI); locally,
prefer the conda env.

## Usage

Everything goes through `scripts/ligamx.sh`, which activates the conda env and
sets `PYTHONPATH=src` before running anything.

```bash
./scripts/ligamx.sh all         # full workflow, including the odds fetch
./scripts/ligamx.sh help        # list all commands
```

| Command | What it does | Network |
| --- | --- | --- |
| `update` | Fetch fixtures/results/xG from SofaScore, recompute ExpG+ | SofaScore |
| `recompute` | Recompute ExpG+ and fix dates after hand-editing the CSV | — |
| `verify-xg` | Audit stored xG/scores/Round/Season against SofaScore; read-only unless `--fix` | SofaScore |
| `model` | Run the goals model export | — |
| `odds` | Fetch Pinnacle 1X2 odds (needs `THE_ODDS_API_KEY`) | Odds API |
| `publish` | Rebuild market comparison + dashboard + `site/` | — |
| `all` | `update` → `model` → `odds` → `publish` | both |

`publish` is fully offline, so it is the command to use when you only need to
regenerate output without burning Odds API credit.

Note that `publish` rebuilds from the *model output* (`models/MEX_team_stats.csv`
and `..._match_simulations.csv`), not from `MEX_ligamx.csv` directly. Editing the
match CSV and running `publish` alone will only refresh timestamps — the full
sequence after a hand edit is `recompute` → `model` → `publish`.

### `update` does not backfill — run `verify-xg`

`update` is incremental and forward-only: it skips every match at or before the
newest date already in `MEX_ligamx.csv`. If SofaScore has not published a
match's xG at fetch time, that match is skipped, and once a later match advances
the watermark it is never reconsidered. `update` now prints a `[WARN]` listing
any match it skipped for this reason.

`verify-xg` is the only command that re-checks rows already written. It
enumerates every season overlapping the CSV and reports `MISSING`, `NO_XG`,
`XG_DIFF`, `SCORE` and `META` discrepancies:

```bash
./scripts/ligamx.sh verify-xg                              # read-only audit
./scripts/ligamx.sh verify-xg --fix                        # add MISSING, fill blank meta
./scripts/ligamx.sh verify-xg --fix --fix-xg-diffs --fix-meta   # full alignment
```

### Analysis entry points

The backtest/calibration modules are not wrapped by `ligamx.sh`. Run them
directly with `PYTHONPATH=src` inside the env, e.g.
`python -m ligamx.eval.rps_backtest`, `python -m ligamx.eval.draw_calibration`,
`python -m ligamx.odds.capture_odds`.

## Project structure

```text
src/ligamx/
  config.py           # env/config loading
  paths.py            # canonical data paths
  date_utils.py
  sofascore_client.py
  fixtures/           # fixture + xG data updates
  xg/
  models/             # goals model (python -m ligamx.models.dc)
  odds/               # Pinnacle fetch, EV calc, market comparison
  dashboard/          # CSV/JSON exports for the dashboard
  eval/               # backtests + calibration (run directly, see above)
scripts/
  ligamx.sh           # single entry point for every command
  common.sh           # conda activation + .env loading
  build_dashboard_site.sh   # assembles site/; also called by the Pages workflow
dashboard/            # dashboard front-end source
site/                 # generated static site (GitHub Pages)
data/                 # match data, odds snapshots, exports
models/
tests/
```

## Deploy

`.github/workflows/deploy-pages.yml` publishes `site/` to GitHub Pages, calling
`scripts/build_dashboard_site.sh` itself. Locally, `publish` (or `all`) is what
regenerates `site/`.
