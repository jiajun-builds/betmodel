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
sets `PYTHONPATH=src` before running anything. Run with no argument for an
interactive menu.

```bash
./scripts/ligamx.sh all         # full workflow, including the odds fetch
./scripts/ligamx.sh help        # list all commands
```

| Command | What it does |
| --- | --- |
| `update` | Run the fixtures/xG/expg data update pipeline |
| `recompute` | Recompute HExpG+/AExpG+ and fix dates after hand-editing the CSV |
| `verify-xg` | Audit stored xG/scores/Round/Season against SofaScore; read-only unless `--fix` |
| `model` | Run the goals model export |
| `odds` | Fetch Pinnacle odds and export the market comparison |
| `dashboard` | Export dashboard CSV and JSON |
| `publish` | Rebuild dashboard exports and `site/` |
| `republish` | Rebuild market comparison + dashboard + `site/` without fetching odds |
| `all` | The full local workflow |

`republish` exists to avoid burning Odds API credit when you only need to
regenerate output.

The underlying entry points are runnable directly with `PYTHONPATH=src`, e.g.
`python -m ligamx.odds.fetch_pinnacle_h2h`, but going through the script is
preferred since it handles env activation and loads `.env.local`.

## Project structure

```text
src/ligamx/
  config.py           # env/config loading
  paths.py            # canonical data paths
  date_utils.py
  sofascore_client.py
  fixtures/           # fixture + xG data updates
  xg/
  models/             # goals model (see DC_MEX.py wrapper)
  odds/               # Pinnacle fetch, EV calc, market comparison
  dashboard/          # CSV/JSON exports for the dashboard
  eval/
scripts/              # ligamx.sh entry point + helpers
dashboard/            # dashboard front-end source
site/                 # generated static site (GitHub Pages)
data/                 # match data, odds snapshots, exports
models/
tests/
DC_MEX.py             # thin wrapper over the goals model
```

## Deploy

`.github/workflows/deploy-pages.yml` publishes `site/` to GitHub Pages. The
local `publish`/`republish` commands are what regenerate `site/`.
