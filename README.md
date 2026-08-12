# LigaMX Terminal

Workflow for finding potential positive-EV Liga MX bets from historical match
data, SofaScore xG reconciliation, a Dixon-Coles style goals model, and bookmaker
odds. Outputs a static dashboard published to GitHub Pages.

See `EV Calculation Logic.md` for the EV/Asian-handicap formulas, and
`SETUP_CAPTURE.md` for the odds-capture loop.

## How the pipeline runs

Half of it is automatic and half of it has to run on your machine. The split is
not a matter of taste — see "Why the refresh is local" below.

```text
CLOUD (automatic, nothing to do)
├─ Cloudflare Worker ──every 5/15 min──▶ GitHub Actions "Capture Odds"
│                                        ├─ opening lines  Betano UK + Duel
│                                        └─ closing lines  Pinnacle + Betfair + Matchbook
│                                              ↓ commits to main by itself
│                                        data/MEX_odds_capture_history.csv
│
└─ GitHub Actions "Deploy Pages" ◀──on push to data/dashboard/json/**
                                         ↓
                                   jiajun-builds.github.io/ligamxterminal

YOUR MACHINE (manual, roughly once per matchday)
└─ ./scripts/ligamx.sh all
   ├─ update    SofaScore results/xG  +  fold captured odds into MEX_ligamx.csv
   ├─ model     fit the goals model
   ├─ odds      fetch the current Pinnacle line
   └─ publish   rebuild dashboard JSON + site/
```

### Day to day

On a non-matchday there is nothing to do; odds accumulate on their own.

After a matchday:

```bash
git pull
./scripts/ligamx.sh all
git add data models && git commit -m "Refresh data and model through matchday N" && git push
```

The push triggers the Pages deploy; the site updates a few minutes later.

**`git pull` first is not optional.** The capture workflow commits to `main` on
its own whenever it banks a new price, so your local `main` is routinely behind.
Skipping the pull means a rejected push, and possibly a conflict in
`MEX_odds_capture_history.csv` — both sides append to the end of the same file.

### Why the refresh is local

SofaScore fingerprints the TLS handshake behind Cloudflare, so plain `requests`
gets a 403 from any datacenter IP. `sofascore_client.py` uses `curl_cffi` to
impersonate a real browser, which works from a home connection but not from a
GitHub Actions runner. So results and xG can only be refreshed from your machine.

Odds capture is unaffected — it goes through two ordinary APIs that do not care
where the request comes from — which is why that half could be automated.

## Setup

This project uses **conda**, not a venv — the environment is defined in
`environment.yml` and the scripts activate it for you.

```bash
conda env create -f environment.yml   # creates the `ligamx-workflows` env
cp .env.local.example .env.local
```

Fill `.env.local` with your keys:

```bash
THE_ODDS_API_KEY=...     # The Odds API — Pinnacle lines and closing capture
ODDS_API_IO_KEY=...      # odds-api.io — opening lines (Betano UK + Duel)
```

Both also exist as GitHub Actions secrets, which is what the capture workflow
uses; the local copies are for running anything by hand. See `SETUP_CAPTURE.md`
for the bookmaker-entitlement quirks and the budget arithmetic.

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
| `update` | Fetch fixtures/results/xG from SofaScore, recompute ExpG+, fold captured odds into the newly-played rows | SofaScore |
| `recompute` | Recompute ExpG+ and fix dates after hand-editing the CSV | — |
| `verify-xg` | Audit stored xG/scores/Round/Season against SofaScore; read-only unless `--fix` | SofaScore |
| `model` | Run the goals model export | — |
| `odds` | Fetch Pinnacle 1X2 odds (needs `THE_ODDS_API_KEY`) | Odds API |
| `capture` | One odds-capture tick — normally CI's job, see below | both APIs |
| `publish` | Rebuild market comparison + dashboard + `site/` | — |
| `all` | `update` → `model` → `odds` → `publish` | both |

`publish` is fully offline, so it is the command to use when you only need to
regenerate output without burning Odds API credit.

`capture` is mostly useful as `capture --dry-run`, which reports what the next CI
tick would spend without spending it. The Worker is already running the real
thing every few minutes, and both captures self-gate, so running it by hand is
safe but usually a no-op.

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
  odds/               # odds capture, EV calc, market comparison
    capture_store.py            # append-only capture history (tracked in git)
    fetch_oddsapiio_opens.py    # opening lines, odds-api.io  [CI, every 15 min]
    capture_close.py            # closing lines, The Odds API [CI, near kickoff]
    reduce_capture_history.py   # history -> MEX_ligamx.csv   [part of `update`]
  dashboard/          # CSV/JSON exports for the dashboard
  eval/               # backtests + calibration (run directly, see above)
scripts/
  ligamx.sh           # single entry point for every command
  common.sh           # conda activation + .env loading
  build_dashboard_site.sh   # assembles site/; also called by the Pages workflow
tools/capture-timer/  # Cloudflare Worker that drives the capture ticks
dashboard/            # dashboard front-end source
site/                 # generated static site (GitHub Pages), gitignored
data/                 # match data, odds snapshots, exports
models/
tests/
```

## Deploy

`.github/workflows/deploy-pages.yml` publishes `site/` to GitHub Pages, calling
`scripts/build_dashboard_site.sh` itself. It fires on pushes touching
`dashboard/**` or `data/dashboard/json/**`, so committing the output of `publish`
(or `all`) is what triggers a redeploy. `site/` itself is gitignored and rebuilt
in CI.

`.github/workflows/capture-odds.yml` is the odds-capture tick. It commits to
`main` on its own and deliberately does **not** trigger a Pages deploy — captured
prices only reach the dashboard after the next local `update` → `publish`. Full
setup, budgets and failure modes are in `SETUP_CAPTURE.md`.
