# Chinese Super League Prediction

Pipeline that pulls Chinese Super League (CSL) match data, derives xG features,
runs a Dixon-Coles model to predict match outcomes and handicaps, compares those
predictions against Pinnacle spreads, and publishes a static dashboard to GitHub
Pages.

This README is a maintainer's operating manual: what the commands do, what they
read and write, and how the automated refresh/deploy works.

## What it produces

- 1X2 probabilities (home win / draw / away win) plus Asian-handicap
  probabilities (Home/Away -1 and -2) per upcoming fixture.
- Team strength rankings from the fitted model.
- A model-vs-market comparison table against Pinnacle handicap lines.
- A static dashboard (`site/`) served from GitHub Pages.

## Setup

```bash
conda env create -f environment.yml   # creates the `csl-workflows` env (Python 3.11)
conda activate csl-workflows

cp .env.local.example .env.local       # then fill in the keys below
```

`.env.local` (local-only, never committed) defines:

- `THE_ODDS_API_KEY` — The Odds API, required by `odds`.

(xG is fetched from the official SofaScore API — no key needed.)

Optional overrides also live in `.env.local` (see `.env.local.example`):
`PYTHON`, `CSL_CONDA_SH`, `CSL_ENV_NAME`.

The wrapper scripts auto-load `.env.local`, activate Conda, and set
`PYTHONPATH=src`, so no manual `export` is needed.

## Everyday usage

Single entry point — run with no argument for an interactive menu:

```bash
./scripts/csl.sh
```

| Command | Does | Needs keys |
|---------|------|-----------|
| `./scripts/csl.sh update`    | Run the data pipeline (fixtures → xG → merge → expg) | — |
| `./scripts/csl.sh model`     | Fit Dixon-Coles and export model CSVs | — |
| `./scripts/csl.sh dashboard` | Export dashboard CSV + JSON | — |
| `./scripts/csl.sh odds`      | Fetch Pinnacle spreads + export market comparison | `THE_ODDS_API_KEY` |
| `./scripts/csl.sh publish`   | Rebuild dashboard exports, then build `site/` | — |
| `./scripts/csl.sh all`       | Full workflow, steps 1–6 below | both |
| `./scripts/csl.sh help`      | Show usage | — |

Common shortcuts:

- Refresh everything: `./scripts/csl.sh all`
- Data only: `./scripts/csl.sh update`
- Rebuild the site without re-fetching or re-modelling: `./scripts/csl.sh publish`

## Full workflow (`all`)

Runs in order, timing each phase:

1. **Data update** — `./scripts/run_csl_update.sh`
2. **Model export** — `./scripts/csl-model.sh`
3. **Odds fetch** — `python -m csl.odds.fetch_pinnacle_spreads`
4. **Market comparison** — `python -m csl.odds.export_upcoming_market_comparison`
5. **Dashboard export** — `csl.dashboard.export_dashboard_csv` + `..._json`
6. **Site build** — `./scripts/build_dashboard_site.sh` → `site/`

`all` requires both API keys and leaves `site/` ready for a GitHub Pages deploy.

## Module reference

Underlying Python entry points (all runnable as `python -m ...` with `PYTHONPATH=src`):

- **Data update**
  - `csl.fixtures.chn_fixture_v5` — results + upcoming fixtures (TheSportsDB)
  - `csl.xg.xg_pipeline` — fetch xG (official SofaScore API, no key)
  - `csl.xg.chn_merge` — merge xG back into the main table
  - `csl.xg.compute_expg` — compute `HExpG+` / `AExpG+`
- **Model** — `python DC_CHN.py` (thin wrapper over `src/csl/models/dc.py`)
- **Odds** — `csl.odds.fetch_pinnacle_spreads`, `csl.odds.export_upcoming_market_comparison`
- **Dashboard** — `csl.dashboard.export_dashboard_csv`, `csl.dashboard.export_dashboard_json`

## The model (brief)

`src/csl/models/dc.py` fits a Dixon-Coles / zero-inflated Poisson goals model
(via `penaltyblog`) on the most recent **18 months** of matches, using
time-decay weights so recent games count more. It is trained on expected-goal
targets (`HExpG+` / `AExpG+`) rather than raw scores; fixtures without a complete
xG pair are dropped from training. Exported probability columns:

- `Home Win Probability`, `Draw Probability`, `Away Win Probability`
- `Home -1 Handicap`, `Home -2 Handicap`, `Away -1 Handicap`, `Away -2 Handicap`

`model comparison/` holds standalone experiments (Poisson, negative binomial,
bivariate, Weibull, zero-inflated, time-decay) used to compare candidate models.

## Data layout

| Path | Contents |
|------|----------|
| `data/raw_data/`      | Main league CSV, xG, fixture exports, Pinnacle spreads, backups |
| `data/output_data/`   | Team-name mapping, Dixon-Coles team stats, match simulations, market comparison |
| `data/dashboard/csv/` | CSVs the dashboard/exporters read |
| `data/dashboard/json/`| JSON the dashboard page loads |

## Dashboard

Static front end in `dashboard/` (`index.html`, `app.js`, `styles.css`, `assets/`)
with a left nav to switch between views (rankings, predictions, fixtures, market
comparison). `build_dashboard_site.sh` assembles `site/` by copying `dashboard/`
and the JSON in `data/dashboard/json/` into `site/data/`. It fails fast if any
required JSON file is missing.

## Automation (CI)

Two GitHub Actions workflows in `.github/workflows/`:

- **`csl-refresh.yml`** — scheduled daily (09:17 Europe/London) plus manual
  dispatch. Sets up the Conda env and runs the refresh pipeline, committing
  updated data back to `main`.
- **`deploy-pages.yml`** — builds `site/` and deploys to GitHub Pages. Triggers
  on push to `main`, manual dispatch, and on completion of **CSL Refresh**
  (`workflow_run`). The `workflow_run` chain exists because the refresh commit
  is made with `GITHUB_TOKEN`, which by design does not fire `push` workflows;
  it always builds the current tip of `main`.
