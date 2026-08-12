# AGENTS.md

## Project Snapshot
- Project purpose: update Chinese Super League match data, compute xG-derived features, run a Dixon-Coles model, and export dashboard- and market-comparison datasets.
- Main Python package: `src/csl/`
- Repository root entry points:
  - `./scripts/csl.sh`
  - `./scripts/run_csl_update.sh`
  - `./scripts/csl-model.sh`
  - `python DC_CHN.py`

## Environment
- Conda environment: `csl-workflows`
- Python: `3.11`
- Core packages from `environment.yml`: `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `requests`
- Pip package: `penaltyblog`

## Setup
```bash
conda env create -f environment.yml
conda activate csl-workflows
cp .env.local.example .env.local
```

## Execution Conventions
- Prefer running commands from the repository root: `/Users/jordan/Developer/python/cslmonitor`
- The preferred local workflow entry point is `./scripts/csl.sh`
- `./scripts/csl.sh`, `./scripts/run_csl_update.sh`, and `./scripts/csl-model.sh` activate Conda, load `.env.local`, and set `PYTHONPATH` automatically
- `.env.local` is local-only and should define:
  - `THE_ODDS_API_KEY`  (xG uses the official SofaScore API — no key needed)
- Conda initialization defaults to `~/anaconda3/etc/profile.d/conda.sh`

## Primary Workflows

### 1. Full Local Workflow
Run:
```bash
./scripts/csl.sh all
```

This command performs these steps, in order:
1. Data update pipeline
2. Model export
3. Pinnacle odds fetch
4. market comparison export
5. dashboard CSV / JSON export
6. GitHub Pages `site/` build

### 2. Data Update Pipeline
Run:
```bash
./scripts/csl.sh update
```

This maps to:
- `./scripts/run_csl_update.sh`

### 3. Run the Prediction Model
Run either:
```bash
./scripts/csl.sh model
```

or:
```bash
python DC_CHN.py
```

Outputs:
- `data/output_data/CHN_team_stats.csv`
- `data/output_data/CHN_team_stats_match_simulations.csv`

Implementation notes:
- The model entry point is `DC_CHN.py`
- Core model code is in `src/csl/models/dc.py`
- `DC_CHN.py` currently uses absolute paths for its input and output CSVs

### 4. Export Dashboard Files
Run:
```bash
./scripts/csl.sh dashboard
```

CSV outputs:
- `data/dashboard/csv/dashboard_meta.csv`
- `data/dashboard/csv/upcoming_fixtures.csv`
- `data/dashboard/csv/match_predictions.csv`
- `data/dashboard/csv/team_strength_rankings.csv` — **`attack_rating`/`defense_rating` are
  goals per match against an average opponent** (`exp(Const + coef)` times the home/away
  venue factor), not the raw coefficients, which are carried alongside as
  `attack_coef`/`defense_coef`. **`overall_rating` is the calibrated strength rating** from
  `src/csl/models/strength.py` — it is deliberately NOT `attack_rating - defense_rating`;
  see that module for why the naive difference misranks clubs. Also carries `low_sample`
  and `in_current_season`, which the panel uses to mark and grey rows.
- `data/dashboard/csv/upcoming_market_comparison.csv` if market comparison has been generated

JSON outputs:
- `data/dashboard/json/dashboard_meta.json`
- `data/dashboard/json/upcoming_fixtures.json`
- `data/dashboard/json/match_predictions.json`
- `data/dashboard/json/team_strength_rankings.json`
- `data/dashboard/json/upcoming_market_comparison.json` if the source CSV exists

### 5. Fetch Pinnacle Odds and Market Comparison
Requires local `.env.local` or exported shell vars for:
```bash
THE_ODDS_API_KEY=...
```

Run:
```bash
./scripts/csl.sh odds
```

Default output:
- `data/raw_data/CHN_pinnacle_now.csv`
- `data/output_data/CHN_upcoming_market_comparison.csv`
- `data/dashboard/csv/upcoming_market_comparison.csv`

Notes:
- Uses The Odds API
- Fetch target is `soccer_china_superleague`
- Bookmaker is fixed to `pinnacle`
- Market is fixed to `spreads`
- Team-name normalization depends on `data/output_data/CHN_team_name_mapping.csv`

### 6. Build GitHub Pages Site
Run:
```bash
./scripts/build_dashboard_site.sh
```

Outputs:
- `site/index.html`
- `site/app.js`
- `site/styles.css`
- `site/assets/`
- `site/data/*.json`

### 7. Rebuild Publish Artifacts
Run:
```bash
./scripts/csl.sh publish
```

This rebuilds:
- dashboard CSV / JSON
- GitHub Pages `site/`

## Automation (GitHub Actions)
Three workflows in `.github/workflows/` (scheduled workflows only run from `main`):
- **`csl-refresh.yml` (`name: CSL Refresh`) — dual-mode.** Mode is resolved from the
  trigger (cron string / `workflow_dispatch` `mode` input):
  - `full` — daily `17 9` Europe/London cron → `./scripts/csl.sh all` (data + model +
    odds + dashboard + site). Runs the model, so it (re)writes `CHN_model_meta.json`.
  - `odds` — every-12h `0 */12` UTC cron → `./scripts/csl.sh odds && ./scripts/csl.sh publish`
    (re-fetch the "Now" line + rebuild the site). Has a pre-spend `/sports` quota guard
    (skips if remaining < 50) and does not write the model sidecar. It **does** write the
    history CSV, via the zero-quota `backfill_open` fallback.
    **Cadence 3h → 12h (2026-08-02)**, freeing ~180 req/month for `pinnacle_close`. Three
    of the four things this mode did had become redundant — the Now line is not on the
    page since dashboard v2.7, `run_oddsapiio_opens` is now done every ~10min by
    `capture-odds.yml`, and the site rebuild is covered by that workflow's gated `publish`
    plus the daily `full`. Only `backfill_open` is load-bearing (safety net for a Pinnacle
    open whose 12h capture window closed unfilled — the model's λ anchor), and two
    runs/day is ample for a window that wide.
    ⚠️ The "Resolve mode" step matches on the **literal cron string**, so changing this
    schedule requires changing it there too or the odds cron silently resolves to `full`.
  Uses a cached conda env (`use-mamba` + `actions/cache` on the pkgs dir); kept
  `conda-incubator/setup-miniconda` because `scripts/common.sh` needs the `conda` command.
- **`capture-odds.yml`** — every-10-min capture tick; independent concurrency group.
  **Three capture jobs plus a gated `publish`** (restructured 2026-08-02), each capture
  running lightweight pandas+requests and exposing an `appended` output:

  | # | Job | Line | Book | Provider | Budget | Gate |
  | - | --- | ---- | ---- | -------- | ------ | ---- |
  | 1 | `oddsapiio_opens` | opening | **1xBet + Duel** | odds-api.io | ~500/**day** | none — every tick, capped by `--max-requests 2` |
  | 2 | `pinnacle_open` | opening | Pinnacle (+recon) | The Odds API | ~500/**month** | predicted open window `[anchor, anchor+12h]` |
  | 3 | `pinnacle_close` | **closing** | Pinnacle (+recon) | The Odds API | ~500/**month** | pre-kickoff `[KO-60m, KO)`, target `T-5m` |

  The three are **chained sequentially** (`needs:` + `if: always()`), not parallel: all
  three append to the end of the same history CSV, and two same-tick appends at EOF do
  **not** auto-merge, so parallel jobs would hit genuine rebase conflicts on push. The
  chain gives one writer at a time while keeping per-capture logs and failure isolation
  (`always()` means a failed or `only`-skipped job never blocks the next). Runner minutes
  are free on this public repo, so the extra checkouts cost nothing. The shared
  commit/rebase/retry block lives in the local composite action
  `.github/actions/capture-commit`.

  `publish` runs **only when job 1 or 2 appended** — it sets up the conda env, runs
  `./scripts/csl.sh republish` (rebuild comparison + site from the existing Now-line +
  updated history, **no `/odds` spend**), commits the dashboard artifacts, and deploys
  Pages itself (job-level `concurrency: pages` to serialize with `deploy-pages.yml`).
  `pinnacle_close` is deliberately **not** in that gate: closing lines are CLV archive
  data and do not surface on the dashboard, so a close-only tick commits and stops rather
  than triggering a no-op redeploy.

  Manual `workflow_dispatch` takes an `only` input (`all` / `oddsapiio-opens` /
  `pinnacle-open` / `pinnacle-close`) to run a single leg, plus `dry_run`.
- **`deploy-pages.yml`** — builds + deploys Pages; `push` is path-filtered, and it chains
  off the `CSL Refresh` `workflow_run` (so both `full` and `odds` runs redeploy). The
  capture-driven redeploy above is done inside `capture-odds.yml`, not here.

All writer workflows push with a rebase+retry loop to survive the push race between the
12h refresh, the daily refresh, and the 10-min capture tick.

Free Odds-API budget ≈ 150–210 of 500 requests/month (2026-08-02): 30 daily `full` + 60
for the 12h `odds` cron + ~20–40 `pinnacle_open` + ~40–80 `pinnacle_close`. Was ~330–390
before the 3h→12h cut; the freed ~180 is what pays for the new closing-line capture, and
headroom went from ~25% to ~60%.

### Dashboard refresh behaviour (two independent update streams)
The page updates via **two streams** with different cadences/triggers — reason about them
separately. The **Now** stream is independent of opening windows (always runs on schedule);
the **Open** stream only writes when a fixture is inside its capture window, not yet captured,
and present in the Odds API feed.

| Stream   | Page columns it drives                         | Driven by      | Cadence / trigger                                   | Spends `/odds`?          |
| -------- | ---------------------------------------------- | -------------- | --------------------------------------------------- | ------------------------ |
| **Now**  | *none since v2.7 — archive + `backfill_open` only* | `CSL Refresh`  | odds every 12h (UTC `0 */12`) + daily `full` 09:17 LDN | 1 per run             |
| **Open** | "Open" line/odds (the opening line)            | `capture-odds` | 10-min tick; in-window + uncaptured + present-in-feed | 1 only when it captures |
| **Close**| *none — CLV archive only, not on the page*     | `capture-odds` | 10-min tick; `[KO-60m, KO)` until a close lands at `T-5m` | 1 per tick with a pending fixture |

Scenario matrix (gated `publish` job + 12h capture window + 12h odds cron, as of 2026-08-02):

| Situation                          | `capture-odds` tick                                   | `CSL Refresh`             | What the page shows                                      |
| ---------------------------------- | ----------------------------------------------------- | ------------------------- | ------------------------------------------------------- |
| **Outside any window**             | 1xBet job may still capture; Pinnacle jobs idle (0 req) | odds refresh every 12h  | static unless a 1xBet open lands                        |
| **In open window, feed has it**    | captures → append → gated `publish` rebuild + deploy  | 12h refresh continues     | Open/bet-price cols appear within ~1 tick                |
| **In open window, feed lacks it**  | nothing this tick; retries each tick (12h window)     | 12h refresh continues     | cols blank until the feed lists it (arrives in waves)    |
| **Fixture already captured**       | skipped (an `open` row exists)                        | 12h refresh continues     | Open locked to the true opening line                     |
| **In close window `[KO-60m, KO)`** | `pinnacle_close` captures each tick until `T-5m` lands | 12h refresh continues    | **nothing** — close is archive data, not a page column   |
| **Kickoff passes**                 | fixture drops out of every pending set                | —                         | fixture leaves the board on the next rebuild (`is_upcoming`, cadence-independent since 2026-08-02) |
| **Quota < 50 remaining**           | Pinnacle jobs abort (`min-remaining`); 1xBet unaffected (other provider) | odds refresh skips fetch | 1xBet signal keeps updating; Pinnacle streams pause until the monthly reset |
| **Manual dispatch**                | `Capture Odds` with `only` + `dry_run` inputs         | `CSL Refresh` `mode=full`/`odds` | Forces the corresponding refresh                  |

## Key Source Modules
- Fixtures/results ingestion: `src/csl/fixtures/chn_fixture_v5.py`
- xG pipeline: `src/csl/xg/xg_pipeline.py`
- xG merge: `src/csl/xg/chn_merge.py`
- xG staleness alert (`all` STEP 1b — xG is fetched off-CI on a home Mac and the merge is
  no-erase, so a dead fetcher is otherwise silent): `src/csl/xg/check_freshness.py`
- expected-goals-plus calculation: `src/csl/xg/compute_expg.py`
- Dixon-Coles model: `src/csl/models/dc.py`
- dashboard CSV export: `src/csl/dashboard/export_dashboard_csv.py` (emits `updated_at`
  = export time AND `model_updated_at` = last model-fit time, read from the
  `CHN_model_meta.json` sidecar via `paths.model_meta_json()`)
- dashboard JSON export: `src/csl/dashboard/export_dashboard_json.py`
- Pinnacle fetch (single "current" snapshot; **1X2/moneyline since roadmap #10** — module
  keeps its legacy "spreads" name; `CAPTURE_BOOKMAKERS` lists the books the capture path
  stores): `src/csl/odds/fetch_pinnacle_spreads.py`
- bookmaker survey (roadmap #8 recon — who quotes CSL 1X2 and at what overround; costs 1
  credit per region): `src/csl/odds/survey_bookmakers.py` (`python -m csl.odds.survey_bookmakers --dry-run`)
- market comparison export (now + captured-open 1X2, hybrid λ/δ de-biased probs, per-outcome
  EV): `src/csl/odds/export_upcoming_market_comparison.py`
- Pinnacle opening-time calendar: `src/csl/odds/opening_calendar.py` (`python -m csl.odds.opening_calendar`; `build_open_windows()` returns tz-aware windows for the scheduler)
- odds-capture history store (append-only): `src/csl/odds/snapshot_store.py`
- single-shot snapshot capture: `src/csl/odds/capture_snapshot.py` (`python -m csl.odds.capture_snapshot`)
- scheduler tick (captures **Pinnacle** opening lines in-window; 1xBet moved off it 2026-08-02): `src/csl/odds/capture_scheduler.py` (`python -m csl.odds.capture_scheduler`)
- fallback open backfill (zero-quota safety net in the 12h refresh — records a missed open from the current Now line; Pinnacle only since 1xBet left The Odds API): `src/csl/odds/backfill_open.py` (`python -m csl.odds.backfill_open --dry-run`)
- odds-api.io client (base URL, key, rate-limit headers, row mapping, and the `Book`
  registry / `CAPTURE_BOOKS`): `src/csl/odds/oddsapi_io.py`
- **Opening-line capture, no predicted window**: `src/csl/odds/fetch_oddsapiio_opens.py`
  (`python -m csl.odds.fetch_oddsapiio_opens --dry-run`). Polls odds-api.io and records
  the first 1X2 price it sees per **(fixture, book)** — 1xBet and Duel, equal priority,
  same tick, one request. See "Two odds providers" below for why.
  - **Pending is per (fixture, book) and must stay that way.** A fixture is requested
    while *any* book still owes an open, but only the books in `PendingFixture.missing`
    may write one. Collapse that back to per-fixture and a fixture with a banked 1xBet
    open but no Duel open would overwrite the banked opening line with 1xBet's *current*
    price — silently, since nothing downstream validates it.
  - Pending is ordered soonest-kickoff-first so a set larger than the 10-fixture batch
    always covers the next round (8 fixtures) in full.
- canonical path helpers: `src/csl/paths.py`

## Important Data Paths

### Raw Inputs
- Main match table: `data/raw_data/CHN_Super League.csv`
- fresh fixture/schedule pull: `data/raw_data/chinese_super_league_data.csv`
- upcoming fixtures for dashboard/export: `data/raw_data/chn_upcoming_fixtures.csv`
- xG data: `data/raw_data/xg_data.csv`
- Pinnacle 1X2 odds (single current snapshot, overwritten each run; legacy "spreads"
  filename): `data/raw_data/CHN_pinnacle_now.csv`
- Pinnacle 1X2 capture history (append-only, tracked in git so the GitHub capture
  workflow can persist it; `market=moneyline` since roadmap #10, old spreads rows replaced
  by the user's manual opening-1X2 backfill): `data/raw_data/CHN_pinnacle_spreads_history.csv`
- backups: `data/raw_data/backups/`

### Model / Processed Outputs
- team name mapping: `data/output_data/CHN_team_name_mapping.csv`
- team stats: `data/output_data/CHN_team_stats.csv` — `Team,Attack,Defense,Const,HomeAdv,
  Matches,WeightedMatches,Date`. `Attack`/`Defense` are the raw mean-centred log
  coefficients; `Const`/`HomeAdv` are league-wide scalars repeated per row, needed to turn
  a coefficient into goals per match; `WeightedMatches` is the summed Dixon-Coles weight
  behind each club's fit (promoted sides land near 0.45 of the median)
- match simulations: `data/output_data/CHN_team_stats_match_simulations.csv`
- market comparison: `data/output_data/CHN_upcoming_market_comparison.csv`
- opening-time calendar (predicted Pinnacle open windows): `data/output_data/CHN_opening_time_calendar.csv`
- model-fit timestamp sidecar (written by `DC_CHN.py`, read by the dashboard meta
  export; NOT touched by odds-only refreshes so it stays pinned to the last model run):
  `data/output_data/CHN_model_meta.json`

### Dashboard Assets
- CSV directory: `data/dashboard/csv/`
- JSON directory: `data/dashboard/json/`
- static frontend: `dashboard/`

## External Dependencies
- `csl.fixtures.chn_fixture_v5` depends on TheSportsDB
- `csl.xg.xg_pipeline` depends on the official SofaScore API via `curl_cffi` browser impersonation (no key); the merge lets fresh values win (xG tracks SofaScore's latest) but a blank scrape never erases an xG already in the cache
- `csl.odds.fetch_pinnacle_spreads` depends on The Odds API (`THE_ODDS_API_KEY`)
- `csl.odds.fetch_oddsapiio_opens` depends on odds-api.io (`ODDS_API_IO_KEY`)

### Two odds providers, and why (2026-08-02, books updated 2026-08-08)

| | The Odds API | odds-api.io |
|---|---|---|
| books | Pinnacle (λ anchor) + betfair/matchbook recon | **1xBet + Duel** (free tier = recreational books; sharp books are paid) |
| free quota | ~500 requests / **month** | ~500 / **day**, 100/hour |
| capture style | window-gated (`[anchor, anchor+12h]`) | **no window** — poll until a price appears |
| league/book keys | `soccer_china_superleague`, `pinnacle`/`onexbet` | `china-chinese-super-league`, `1xbet`/`Duel` |

The split exists because the window-gated design **loses opening lines**. Verified case:
Shandong Taishan vs Tianjin Jinmen Tiger (round 22). Its window ran 2026-08-01 11:35→23:35
UTC; the scheduler polled correctly throughout and The Odds API never listed the fixture
(`None of the in-window fixtures were present in the odds response`). The line posted ~25h
after the anchor and the open was gone for good. The window cannot simply be widened — at
1 request per 10-min tick, a monthly budget of ~500 makes a 48h window cost ~288 requests
on one stubborn fixture. odds-api.io's *daily* budget removes that constraint entirely.

**Which books odds-api.io will actually serve (re-probed 2026-08-08).** The plan allows
**two**, and they are `1xbet` + `Duel`. `bookmakers=1xbet,Duel` returns both in ONE
`/odds/multi` request, so capturing the second book costs zero extra quota and runs on
the same tick at the same priority — neither is an anchor or a fallback for the other.
- ⚠️ **`/bookmakers/selected` is stale and lies.** It still reports
  `{"bookmakers":["1xbet"],"count":1}` while both books answer with data. The
  authoritative entitlement is the 403 body from `/odds/multi`:
  `"Access denied. You're allowed max 2 bookmakers. Allowed: 1xbet, Duel"`. Never
  conclude from `/selected` that a book is unavailable.
- Sharp books are not merely paid, they are not valid names here: `Pinnacle`, `Betfair`
  and `Matchbook` return **400** (unknown bookmaker), not 403. `FanDuel` exists in the
  274-book catalogue and is a *different book from `Duel`*; it 403s on the 2-book cap.
- Duel screened at **3.00% mean CSL overround** vs 1xBet's 5.41% (Now lines, 6/16 vs
  7/16 fixtures priced, 2026-08-08) — the cheapest traditional book this project has
  measured. That is a **Now** snapshot: opening overrounds are wider, so do not quote
  3.00% as an opening figure, and Duel pricing fewer fixtures may mean it posts later.
  Both need confirming from captured opens before Duel displaces 1xBet as the bet price.

Notes for anyone touching this:
- 1xBet rows are written with `bookmaker="onexbet"` — The Odds API's key, not `"1xbet"` —
  precisely so `export_upcoming_market_comparison` and every downstream consumer need no
  change. Duel has no The-Odds-API key to inherit (that feed carries only the unrelated
  `fanduel`), so it stores as `"duel"`. The provider is recorded in `regions`
  (`oddsapiio` vs `us`), and `event_id` is namespaced `oddsapiio:<id>` — **shared by both
  books**, which is why `bookmaker` is part of the store's dedup key.
- **Duel became a live bet book on 2026-08-08** (dashboard v2.8) — see "Two bet books"
  below. It is no longer inert; it enters EV, the BET signal and the Telegram alert.
- **The history therefore contains `onexbet` opens from both providers.** Rows captured
  before 2026-08-02 came from The Odds API; filter on `regions` if a backtest needs to
  distinguish them. `load_open_snapshots` takes the earliest `fetched_at` per fixture, so
  pre-cutover fixtures keep their original open.
- `/v3/odds/movements` would have been better still (it returns a real `opening` object
  with a timestamp, making opens retrievable retroactively). Probed 2026-08-02 across four
  market/line combinations: **404 "No data found"** every time, not 403 — the movements
  store has no data for 1xbet on this league. Re-probe if the plan is ever upgraded.
- `/v3/historical/odds` **is** free-tier accessible and returns 1xBet prices for settled
  matches, but its `updatedAt` sits ~1 minute before kickoff — that is a **closing** line,
  not an opening one. Useless for backfilling opens; potentially the missing input for
  roadmap #3 (close/CLV), which has never had a data source.

### Two bet books, and why it is "best price" not "the cheaper book" (2026-08-08, v2.8)

The registry is `src/csl/odds/books.py` (`BetBook`, `BET_BOOKS`, `BOOK_BY_KEY`) — a
**stdlib-only** module so `signal_alert` can import it without dragging in pandas and
the Dixon-Coles fitter. `export_upcoming_market_comparison` re-exports it.

**EV is scored against `max(odds)` across books, per outcome.** Choosing one book by its
headline overround would be wrong, and measurably so. On the three fixtures both books
quoted the day Duel was wired in, Duel's overround was ~2.4pp lower on *every* fixture,
yet:

| side | best price wins |
| ---- | --------------- |
| home | Duel 2 – 1xBet 1 |
| **draw** | **Duel 3 – 1xBet 0** |
| away | Duel 1 – 1xBet 2 |

Duel's cheapness is almost entirely in the **draw**, which `SIGNAL_ALLOW_DRAW = False`
means we never bet. On home/away it is 3–3. Per-outcome best price was worth **+2.06%
mean** (max +5.66%) on those sides — near breakeven, roughly +2pp of EV against the
2.61pp vig wall.

Invariants worth not breaking:
- **Per-book columns are retained** (`onexbet_open_*`, `duel_open_*`) alongside the
  `best_open_*` layer, so 1xBet-only performance stays reconstructible by column
  selection. ⚠️ `SIGNAL_EV_MIN = 0.20` was calibrated on 1xBet **alone** (backtest.md
  §13.4) and a max over books is upward-biased, so best-of-two fires strictly more
  signals at the same threshold. Re-derive it from those columns before trusting the
  new firing rate — there is no Duel backtest at all.
- **Tie-break is `BET_BOOKS` order** (1xBet first, strict `>` while scanning). It must
  stay deterministic: `signal_book` is part of the Telegram dedup key, so a tie-break
  that could flip would re-alert every run.
- **`signal_books` is emitted only when `signal_state == "bet"`.** State is decided on
  the best price but `signal_books` per book, so they can disagree; suppressing it on a
  greyed row keeps "every logo shown is a bet you should place" true. Accepted cost:
  adding a book can *remove* a bet 1xBet alone would have fired (best price over the
  odds cap while the other book's is under it). Rare — 1 of 68 captured home/away opens
  ever exceeded odds 7. Pinned by `tests/test_two_book_ev.py`.
- **Logo assets are `dashboard/assets/{book.key}.png`, lowercase.** macOS resolves any
  case locally but GitHub Pages serves from Linux — a capitalised stem 404s **in
  production only**. Verify with `git ls-files dashboard/assets`, never `ls`.
- **`signal_alert`'s dedup baseline is `git show HEAD:<csv>`**, not a state file. Rows
  from a baseline written before `signal_book` existed are treated as a wildcard ("any
  book already alerted"), which is what stops the migration itself from blasting every
  firing signal. Same guard protects any future key change.

## Validation Guidance
- The repository has a small test suite under `tests/` (no pytest required — each file is
  runnable directly, e.g. `python tests/test_oddsapi_io.py`).
- Practical validation is usually done by running the relevant entry point and checking the expected CSV/JSON outputs.
- For model experimentation, use:
  - `DC_CHN.py`
  - `model comparison/`

## Strategy Context & Findings

### What the project is ultimately for
The dashboard/market-comparison output feeds a **CLV-based betting strategy**: find fixtures
where the model diverges from the market and bet +EV lines at aggregator books.
- The thesis is **not** "beat Pinnacle closing" (closing is assumed efficient). It is
  "beat Pinnacle **opening**" — get down early at soft/aggregator books at prices better
  than even Pinnacle, before the market corrects.
- Success metric is long-run **+CLV** (closing line value vs Pinnacle close), not per-bet
  wins. "Bet early ⇒ +CLV" is an *assumption* whose direction depends on model quality.
- **Biggest gap:** opening/closing lines are not captured automatically, so CLV is measured
  manually today (selection-bias risk) and the edge is unvalidated. Closing that loop is the
  roadmap below.
- **IMPORTANT UPDATE (2026-07-13, extended 2026-07-15):** the "model finds +EV vs Pinnacle's
  opening line" half of this thesis has been tested at length. **Asian handicap: falsified
  outright** (2026-07-13, winner's curse). **1X2: the strategy as specified is dead — 61% of its
  stake sits on a draw-probability bug — but the direction survives** (2026-07-15; drop the draw
  and a baseline-adjusted +CLV holds in all three seasons, though it is still short of the vig
  bar). Read "Betting-edge investigation — conclusions" below and `backtest/backtest.md`
  §11.3 + §11.7 **before quoting any CLV number** — an "always bet home" coin beats this model on
  raw CLV, and breakeven needs CLV > 2.61pp. **The draw de-bias was then built and tested
  (2026-07-15, `backtest.md` §12): it fixes the model (excess CLV doubles) but the strategy still
  fails the vig bar in 2024/2025 — betting Pinnacle's open is closed.** The one live direction is
  the *earliest/cheapest-opening book* (roadmap #8): the de-biased model's +1.2–2.5pp excess CLV
  would clear breakeven at a ≤5%-overround book.

### Model
- **v2.8 (2026-07-26) — `src/csl/models/dc.py` fits `ContinuousPoissonGoalModel`**
  (`src/csl/models/continuous_poisson.py`) on **xG targets** (`HExpG+`/`AExpG+`),
  18-month window, `xi=0.001`, Dixon-Coles time-decay weights, via the single entry
  point `fit_production_model()`. **Do not go back to a penaltyblog family here:**
  penaltyblog's `BaseGoalsModel.__init__` coerces goal arrays to int before the
  likelihood, so it silently fit `floor(HExpG+)` and every λ was **27% too low**
  (score equation `Σwλ/Σwy` = 0.733 instead of 1.0). Affects all families — it is in
  the base class. Full write-up: `backtest/backtest.md` **§15**.
- **The draw de-bias δ is RETIRED** (`DRAW_DELTA_SHRINK = 0.0`). It was a patch over
  the truncation bug; post-fix the residual draw bias *changes sign* by season
  (2023 +0.81pp, 2025 −4.37pp), so no diagonal scale fixes it, and applying the
  fitted δ made out-of-sample calibration worse (draw −1.97pp → +2.19pp). δ is still
  fitted and logged as a diagnostic only. `backtest.md` §15.3.
- **Every hyperparameter tuned before 2026-07-26 was tuned on the broken fit.**
  Re-swept in §15.4: `SIGNAL_EV_MIN=0.20` and `w_xG=0.7` survive; `xi`/lookback is
  boundary-limited and inconclusive, left at 0.001/18mo.
- **`DEBIAS_LAMBDA = 1.0`** (was 0.75, which became worst-of-grid). λ=1.0 takes the
  anchor book's no-vig draw **outright** — the model only splits home/away. Shipped at
  1.0 rather than the measured argmax 1.25 because λ>1 gives `p_D` a *negative* weight
  on the model's own draw, whose bias changes sign by season (§15.3) — the same
  fragility that retired δ. §15.4b.
- **Draws cannot fire as bets** (`SIGNAL_ALLOW_DRAW = False`). At λ≥1.0 a draw signal
  would only mean "1xBet's draw price beats Pinnacle's fair draw", a cross-book price
  gap rather than a model view; excluding them measured free (gap +3.58 → +3.60). The
  draw is still modelled and displayed. §15.4c.
- **Never accept RPS as evidence about this model's calibration.** The 27% λ error
  moved RPS by 0.0031 and log-loss by 0.0102, which is why years of RPS-driven sweeps
  missed it. Rank on per-outcome bias, AH-ladder bias and log-loss.
- Guard rail: `fit_production_model` raises if the score equation drifts from 1.0, and
  `tests/test_continuous_poisson.py` locks it. `backtest/verify_truncation_fix.py`
  re-checks the four pre-registered acceptance criteria.
- Since **v2.6 (2026-07-16, roadmap #10)** the de-bias is **hybrid**: the market-comparison
  surface uses the §12-validated **market-anchored shrink** (λ = 0.75 toward the no-vig
  *captured opening* draw prob, applied to the raw un-δ'd grid via
  `DrawCalibratedModel.predict_raw` — never stacked on δ) for fixtures with a captured
  open, falling back to δ otherwise; the all-pairs `match_predictions` surface stays
  δ-based (no anchor can exist there). Same fixture may therefore show slightly different
  draw probs on the two surfaces (~1pp) — intentional, user-confirmed. The
  `debias_method` column in the comparison CSVs records which path produced each row.
- **History:** production previously fit `ZeroInflatedPoissonGoalsModel` (ZIP), whose
  zero-inflation parameter sat at its ~1e-6 floor in 100% of refits (diagnostic
  `model comparison/zip_zero_inflation_param_test.py`) — ZIP had collapsed to Poisson,
  so the swap changed accuracy only via NegBinom's over-dispersion.
- The model is fit **twice** per full run (STEP 2 model export + STEP 4 market comparison),
  on identical inputs — redundant but cheap (seconds; small single-league data). Left as-is.
  Watch-out: `xi=0.001` is hardcoded in two places (`dc.py`/`DC_CHN.py` and
  `export_upcoming_market_comparison.MODEL_XI`); if they ever diverge the two exports would
  silently use different models.

### Timezone (important data quirk)
- Source CSV `Time` columns (`chinese_super_league_data.csv`, `chn_upcoming_fixtures.csv`)
  are **UTC (GMT / UK time WITHOUT daylight saving)**, *not* UK local wall-clock.
- Always parse as UTC and convert to `Europe/London` so summer (BST) fixtures get +1h.
  Treating raw values as already-local makes summer times 1h early. Handled in
  `export_dashboard_csv.py` and `opening_calendar.py`.

### Pinnacle opening-time pattern (validated 2026-07-03)
- Pinnacle opens a match's line within **~1h after the later of the two teams' most-recent
  (current-round) matches has kicked off** (kickoff start, not full-time).
- `src/csl/odds/opening_calendar.py` predicts these windows from prior-round kickoffs.
  Field-validated: round-17 predicted windows matched the actual Pinnacle open times.
- This lets us catch the true opening (and closing) line on the **free** Odds-API plan
  (no historical-odds endpoint) by scheduling narrow captures.

### Betting-edge investigation — conclusions (2026-07-13)
A full test of whether the model produces a tradeable edge at the opening line.
**Bottom line: it does not, and calibration/distribution changes do not create one.**
The only surviving hypothesis is line *timing* (bet the earliest, softest line before
Pinnacle forms it). Full detail + numbers in `backtest/backtest.md` §9–§10; analysis
scripts in `backtest/` and `model comparison/distribution_comparison.py`.

- **Opening-line AH backtest (826 bets, 4 seasons):** no EV threshold beats zero; realized
  ROI −4% to −8% and *worse* the more selective; model overstates its own EV by ~20%/unit at
  **t=6**, replicated in all four seasons; highest-EV bets are the worst.
- **The overstatement is winner's curse (selection bias), NOT a distribution defect.**
  Symmetric home-cover calibration is ~0 (unbiased); the overconfidence appears *only* once
  you condition on "the side the model likes most". Proven by simulation: an unbiased model +
  efficient market + "bet the +EV side" reproduces ~+14% overstatement from pure noise. So
  only a model genuinely *more accurate than the market* removes it — reshaping a distribution
  cannot.
- **Calibration doesn't fix it.** Walk-forward temperature scaling (T≈1.5) barely dents the
  overstatement (+19.3%→+18.4%). 1X2 is well-calibrated (ECE 0.032); handicap-cover is not
  (ECE 0.086, worst on big lines) because Poisson/ZIP under-disperses the goal-difference
  (margin) distribution.
- **No distribution helps the betting.** NegBinom is the most accurate 1X2 predictor (best RPS,
  ~1.5% better log-loss than ZIP — over-dispersion genuinely helps prediction) but bets
  *slightly worse*; all six distributions overstate EV +17–23%. **A `ZIP→NegBinom` swap in
  `dc.py` is justified for accuracy only, not betting edge.** (ZIP == Poisson exactly; still
  collapsed — see prior finding above.)

> ⚠️ **VOID — every model-family and distribution comparison above and below this line was
> run on truncated targets.** penaltyblog floors non-integer goal targets before the
> likelihood, so all six "distributions" were fit to `floor(HExpG+)` with λ 27% too low, and
> NegBinom's over-dispersion parameter was pinned at its optimizer bound (i.e. it *was*
> Poisson). Ranking them against each other measured nothing. The under-dispersed
> goal-difference / handicap-cover finding (ECE 0.086) is largely this bug: mean |AH ladder
> bias| went 2.61pp → 1.29pp on the corrected fit. See `backtest/backtest.md` **§15**.
> Production is now `ContinuousPoissonGoalModel`. Re-run before citing any number here.
- **Line-magnitude filter doesn't rescue it.** Big lines (>2) catastrophic (−29% ROI); small
  lines (≤0.5) less bad (−3.6%) but still overstate EV +17% (t=3.66). Only takeaway: avoid
  big-favourite lines.
- **CLV (open→close, 2023–24 only — 2025 close lines empty):** overround compresses open 6.1%
  → close 4.0% (~1pp/side vig headwind); no naive rule gets significant +CLV; model picks
  +0.69pp (t=1.9) but weakens with EV threshold (noise-like) and is net-negative after vig.

#### 1X2 opening line — strategy dead, direction alive (2026-07-15, `backtest/backtest.md` §11)
The AH result prompted a switch to Pinnacle **1X2** open+close. The user backfilled 2024–25
(611 gradeable matches, 2024 R1–2026 R18; 2023 unusable — no training history). Betting the
highest-EV outcome **loses** (EV>0.10: ROI −4.8%, t=−0.57; 2024 alone −23.8%, t=−1.98; full
Kelly → 0). But unlike AH the failure is **one fixable defect**, not the whole idea:
- **THE DRAW BUG (the defect).** Model draw prob pinned at ~0.279 vs market 0.234 ≈ actual
  0.242 — high by ~4pp in *every* season and *every* match type (structural: independent-Poisson
  piles mass at goal-diff 0). Lethal interaction with the EV rule: at draw prob 0.28, EV>0.10
  fires whenever the draw is priced > 1.10/0.28 = **3.93**, and the CSL median opening draw
  price is **3.79** — so every above-median draw becomes a bet. **61% of all stake sits on the
  bug, carrying ZERO CLV** (+0.03pp, t=0.22). Worst bucket: draws priced 4.5–6, model says 28%,
  reality 8% (n=40, ROI −62%). **Drop the draw → CLV triples (+0.66 → +2.15pp), survives the
  baseline adjustment (+1.73pp, t=2.51), positive in all 3 seasons.** Still not profitable
  (ROI doesn't replicate: 2024 −5.5% / 2025 −16.9% / 2026 +53.9%).
- **METHODOLOGY — two rules that must be applied to any future CLV claim:**
  1. **Always compute the model-free baseline (§11.3).** This market drifts toward the home team
     every season (+0.91pp overall). **"Always bet home" scores +0.91pp CLV (t=2.84) — better
     than the model's +0.42pp.** The model bets home 165/away 71 and inherits that drift free.
     Report **excess CLV** (model − same-outcome/same-season drift), never raw. The 2026-only
     "signal" that motivated the whole 1X2 thread was largely this artifact.
  2. **The vig wall (§11.7): EV > 0 ⟺ CLV > p × R.** With p≈0.344 and Pinnacle's **opening
     overround 7.55%**, breakeven needs **CLV > 2.61pp**. This is why `always home` earns +0.91pp
     CLV and still returns −4.8%. Everything the model knows is worth ~2–3pp; the vig costs 2.61.
- **Data quality:** an overround<0 sweep found exactly 1 bad cell in 611 (a `368.00` typo for
  `3.68`, fixed). Opening overround median 7.56%, closing 4.72% — the odds data is sound.
- **NegBinom** changes none of this and does **not** fix the draw (0.276 vs ZIP 0.279).

**Reframe / where the edge could still be:** AH is dead outright; 1X2 as-specified is dead but
its *direction* survives a draw fix (above). Either way the user bets via **Sportmarket** (a
sharp-book aggregator/brokerage) on *newly-opened* CSL lines, and the strongest remaining play is
**catching the earliest-opening book before Pinnacle** — the earliest line is softest, and if it
converges toward Pinnacle's close you capture +CLV *without* model edge and *without* winner's
curse. Line-timing/microstructure, not prediction. **The vig wall makes this more valuable, not
less:** everything the model knows is worth ~2–3pp of CLV while Pinnacle's opening vig alone
costs 2.61pp, so the same +2.15pp CLV loses into a 7.55% open and wins into a 4% book. **Paying
less beats predicting better.** See roadmap #8.

## Roadmap / Open Tasks
1. **Verify the dashboard TZ fix at runtime** — run `python -m csl.dashboard.export_dashboard_csv`
   on the `csl-workflows` env and confirm a summer `kickoff_at` shows the London offset
   (`+01:00`) and metadata `timezone` reads `Europe/London`. (Fix is logic-checked, not yet
   run end-to-end.)
2. **Scheduled odds-capture pipeline — DONE (open side; close deferred to #3).**
   Delivered as four modules + a GitHub Actions workflow:
   - `snapshot_store.py` — append-only history CSV (`CHN_pinnacle_spreads_history.csv`),
     schema = `fetch_pinnacle_spreads.OUTPUT_COLUMNS` + `snapshot_type`/`target_round`/
     `capture_reason`; dedup key `(event_id, last_update, snapshot_type)`.
   - `capture_snapshot.py` — single-shot capture with a pre-spend quota guard (reads the
     free `/sports` endpoint first) and `--dry-run`.
   - `capture_scheduler.py` — "tick" run every ~10 min: captures a fixture's opening line
     only while it is inside its predicted open window and not yet captured; one `/odds`
     call covers the whole slate, non-in-window fixtures are discarded.
   - `.github/workflows/capture-odds.yml` — runs the tick on GitHub cron (UTC), commits new
     rows back to `main`. Only fires from the default branch; GitHub cron delay is tolerated
     because windows are ~1h and each fixture is captured at most once.
   The dashboard market-comparison now shows an **Open** and a **Now** group per fixture
   (line @ price + model EV each) plus a **Move** arrow; open EV is recomputed at the
   captured opening line. **Close/CLV columns are intentionally NOT built** — see #3.
   Free-plan quota: 500 requests/month; one `/odds` call = 1 request, `/sports` = 0.

   **Intraday extension — DONE (PR #17).** `csl-refresh.yml` is now dual-mode: the daily
   cron rebuilds the model, and a new every-3h `odds` cron refreshes only the "Now" line +
   site (see Automation section). Because odds-only publishes bump the dashboard export time
   every 3h, a persisted `model_updated_at` (from the `CHN_model_meta.json` sidecar) now
   travels through the meta export so the EV-panel footer shows **model-update time vs
   odds-fetch time** separately (`Model … · Odds fetched …`).
3. **Close the CLV loop:** join the user's bet-tracker fills to the captured closing lines →
   automated, auditable, per-segment CLV. Replaces manual CLV computation.
   **Update (2026-07-13):** an open→close CLV analysis was run on the manually-entered lines
   (2023–24 only; 2025 close lines empty) — **no rule beats the close** once the ~1pp/side
   open-vs-close vig headwind is counted; model picks +0.69pp CLV but net-negative. See the
   2026-07-13 findings above and `backtest/backtest.md` §9.6. Superseded in priority by #8.

   **Capture side — DONE (2026-08-02), on user go-ahead.** `csl.odds.capture_close` is the
   third job in `capture-odds.yml`; before this the only close lines on file were manually
   entered ones (2023–24 AH), so no live source produced the CLV benchmark. Design:
   - **Target `T-5m`, window `[KO-60m, KO)`.** The user asked for the line 5 minutes before
     kickoff. A literal 5-minute window does not survive this repo's tick cadence — measured
     over 299 runs the inter-tick gap is median 10min but p75 74min, **p90 137min**, max
     232min — and a missed close is unrecoverable (the fixture leaves the pre-match feed at
     kickoff). So the window opens an hour out and **re-captures every tick, keeping the
     latest price**; once a Pinnacle close is on file with `fetched_at >= KO - 5m` the
     fixture goes **final** and stops spending. Healthy cadence ⇒ the stored close is the
     `T-5m` line; degraded cadence ⇒ still a usable close instead of none.
   - **Read side:** a fixture accrues several `close` rows as the line moves. The closing
     line is the row with the **latest** `fetched_at` (mirroring open = earliest);
     `capture_close.latest_close_rows()` is the canonical reader — do not re-derive it.
   - **Cost** is per tick, not per fixture (one `/odds` covers the slate; the `bookmakers`
     filter is free so recon books ride along): ~1–6 credits per kickoff wave, ~40–80/month.
     Guarded by the same `--min-remaining` pre-spend check.
   - **Not wired to the dashboard.** Closing lines are archive/CLV data; `pinnacle_close` is
     excluded from the `publish` gate. Building the actual CLV join (the original #3) is
     still open — it now has real input data for fixtures going forward.

   **3h → 12h `odds` cron — DONE (2026-08-02).** Funded the close capture above. Since
   dashboard v2.7 the Now line is not on the page at all (`DASHBOARD_COLUMNS` has no
   Pinnacle Now odds, no Move) and `capture-odds.yml` runs the open capture itself every
   ~10min, so of the four things that cron did only `backfill_open` is load-bearing. Freed
   ~180 req/month. Shipped together with the board-staleness fix below, which it required.

   **Board staleness fix (same change).** `build_base_frame` used to keep any fixture with
   a Now-line `event_id` *without* the `is_upcoming` gate, justified by "Now-line fixtures
   come from the live feed and are inherently upcoming". That premise holds only at *fetch*
   time (the request sends `commenceTimeFrom=now`); `CHN_pinnacle_now.csv` is a disk
   snapshot whose "inherently upcoming" property decays with exactly the refresh interval,
   and `upcoming_fixtures.csv` is not kickoff-filtered either — so this gate was the only
   thing trimming finished matches. At 3h the bug was capped at ~3h of lingering; at 12h it
   would have been half a day of kicked-off matches on the board. Now `keep = (has_now |
   has_pin_open | has_bet_open) & is_upcoming`, gating on the repo's own `kickoff_at`, so
   **the board no longer depends on the odds-fetch cadence at all**. Verified a no-op on
   current data (byte-identical export) and correct on synthetic stale-Now input.
4. **Validation ladder for the edge** (before trusting it): paired Wilcoxon on per-fixture
   RPS (ZIP vs Poisson), and per-segment calibration / reliability diagrams (by handicap
   line, favourite vs underdog) — bet only in well-calibrated segments.
   **Update (2026-07-13) — DONE, negative.** The reliability diagrams were built
   (`backtest/calibration_diagnostic.py`) and calibration was attempted (temperature scaling,
   `backtest/backtest_open_ah_calibrated.py`). Calibration does **not** create an edge: the EV
   overstatement is winner's curse, not a fixable miscalibration (see the 2026-07-13 findings).
   Do not re-attempt "calibrate then bet the opening line" — it is a closed dead end.
5. **Optional simplification: swap production ZIP → `PoissonGoalsModel` — SUPERSEDED.**
   Production moved ZIP → `NegBinomialGoalModel` + draw de-bias instead (v2.5, roadmap #9).
6. **Capture-loop hardening — DONE (two gaps found 2026-07-04, field-observed on round 18):**
   - **Monitor lag after capture — FIXED.** `capture-odds.yml` used to write the history CSV
     and stop, so a freshly captured opening line only surfaced at the next 3-hourly
     `csl-refresh odds` run (up to ~3h later). Now a gated `publish` job runs only when the
     tick appended rows: it runs `./scripts/csl.sh republish` (new command = rebuild
     `upcoming_market_comparison` + dashboard + site from the existing Now-line and updated
     history, **no `/odds` spend**) and deploys Pages. Idle ticks skip it. See Automation.
   - **1h-window feed-lag miss — FIXED.** The Odds API lists fixtures in waves; a fixture whose
     feed entry (or Pinnacle line) appeared only AFTER its predicted `[anchor, anchor+1h]`
     window closed was never captured (`pending_open_fixtures` requires `now ∈ window`) — on
     round 18, `Shanghai Port vs Dalian Yingbo` was at risk of a permanent miss. The scheduler
     now uses a wider **capture** window `capture_scheduler.DEFAULT_CAPTURE_WINDOW_HOURS` (6h),
     separate from the validated ~1h **display** window (`opening_calendar.DEFAULT_WINDOW_HOURS`,
     unchanged, still shown in the calendar). A still-uncaptured fixture is grabbed on first
     feed availability after its window, bounded so a long-open line isn't mislabeled `open`.
   - **Open-only fixtures now shown — FIXED.** The comparison used to keep only fixtures with a
     current **Now** line (`build_base_frame` filtered on `event_id.notna()`), so a fixture
     captured *before* it appeared in a Now-line fetch (e.g. round-18 `Shenzhen vs Qingdao West
     Coast`, captured at 12:45 while the 12:04 Now line lacked it) stayed invisible until the
     next `odds` refresh — even though its opening line was in the history. `build_base_frame`
     now keeps a fixture with a Now line **or** a captured open line (open-only rows render Now
     columns as `--`), gated to a **future kickoff** so already-kicked-off matches don't linger
     once the feed drops them. Now-side probs/EV are left NaN for open-only rows and
     `validate_market_probabilities` skips them; `getBestBet` in `app.js` treats a null Now EV
     as NaN so an open-only fixture is never chosen as the best bet.
   - **Open published outside the capture window — FIXED (2026-07-16, user-reported double
     fix).** Even at 6h the capture window still missed opens that Pinnacle posted later
     (or that the 10-min GitHub cron skipped, or fixtures with no schedulable anchor), which
     surfaced as "dashboard shows current odds but no opening odds." Two layers now:
     (a) *Widened + kickoff-capped window* — `DEFAULT_CAPTURE_WINDOW_HOURS` 6h → 12h, and
     `opening_calendar.build_open_windows` caps `open_to = min(anchor+window, kickoff)` so a
     never-captured fixture stops being "pending" at kickoff instead of burning a credit per
     tick afterward. Widening is ~free normally (a fixture is captured ~1h after anchor and
     drops out; the wider bound only keeps polling genuinely-late opens).
     (b) *Zero-quota safety net* — `csl.odds.backfill_open`, run inside the every-12h Now-line
     refresh (`scripts/csl.sh odds`/`all`, reusing the fetch already made), records the
     current Pinnacle line as a fallback `open` for any fixture with a Now line but no
     captured open **whose capture window has already closed** (or that has no anchor). The
     window-closed guard lets the 10-min capture keep first crack at a fresher open; only real
     misses are backfilled (`capture_reason="now-refresh fallback (open window missed)"`).
     This makes "Now line without an open" impossible: anything the 10-min path misses is
     caught at 3h granularity for free. Consequence: the odds-mode refresh now *can* append to
     the history CSV (append-only + dedup, rebases cleanly against `capture-odds.yml`), so
     `csl-refresh.yml` stages the history CSV in odds mode — it is no longer capture-only.

7. **Date-parse bug in `model comparison/` scripts — FIXED (2026-07-12).** A naive
   `pd.to_datetime(df["Date"], errors="coerce")` is correct on ISO `YYYY-MM-DD` but on
   `DD/MM/YYYY` it coerces every day>12 row to `NaT` and month/day-swaps the rest, corrupting
   the walk-forward training windows. The bug was **dormant** (the committed CSV was ISO) until
   a manual spreadsheet re-save — made while adding the Pinnacle opening lines — rewrote the
   working-tree CSV to `DD/MM/YYYY` and activated it. Production was never affected:
   `src/csl/models/dc.py` already uses `csl.date_utils.parse_date_only_series` (handles both
   formats). Fix: (a) the three active scripts (`xi_lookback_grid_test.py`,
   `zip_zero_inflation_param_test.py`, `poisson_vs_zip_18mo_test.py`) now use
   `parse_date_only_series`; the re-run reproduces the original correct grid, so those findings
   STAND (production `xi=0.001`/18mo ranks within noise of the optimum). (b) `chn_merge.py` now
   **canonicalizes the `Date` column to ISO on write** via `format_date_only_series` (defensive:
   only overwrites cleanly-parsed rows), so any future manual `DD/MM/YYYY` re-save self-heals on
   the next pipeline run instead of silently reactivating locale-dependent parsing downstream.
   The opening-line AH backtest built on this data lives in `backtest/` (see `backtest/backtest.md`).

8. **Earliest-opening-line edge — the strongest live direction (NEW 2026-07-13, promoted
   2026-07-15).** AH is closed outright; 1X2 as-specified is closed too (though a draw fix may
   revive it — see #9). The untested, winner's-curse-free hypothesis: the user bets via
   **Sportmarket** (sharp-book aggregator) on newly-opened lines; if some book opens a CSL line
   *before* Pinnacle, that earliest line is the softest and may be exploitable before the market
   sharpens.
   - **Why this got MORE valuable (the vig wall, `backtest/backtest.md` §11.7):** EV > 0 ⟺
     CLV > p × R. Everything the model knows is worth ~2–3pp of CLV, but Pinnacle's **7.55%
     opening overround alone costs 2.61pp** — so the model's best strategy (+2.15pp CLV) *loses*
     into Pinnacle's open and would *win* into a 4% book. **Paying less beats predicting
     better.** This is now a stronger lever than any model work.
   - **Blocked on reconnaissance (user in progress):** identify which book opens CSL lines
     earlier than Pinnacle, by how much, whether it is exposed by name in The Odds API /
     Sportmarket, and how its early line compares to Pinnacle's open→close.
   - **Then:** if The Odds API carries that book, widen `fetch_pinnacle_spreads` beyond the
     hardcoded `pinnacle`/`spreads` to capture its opening line + timestamp; measure whether
     the earliest line moves toward Pinnacle's close (→ +CLV, exploitable) using the CLV logic
     from `backtest/backtest.md` §9.6. If it already ≈ close, this door is closed too.
   - **Measure it correctly:** any candidate book must be scored with **excess CLV over the
     model-free baseline** (§11.3 — this market drifts +0.91pp/season toward the home team, so
     raw CLV lies) and against the **p × R** bar (§11.7). Its overround matters as much as its
     line.
   - **Data gaps:** no soft-book odds anywhere (the blocker — every line on file is Pinnacle);
     close AH only 2023–24. Pinnacle 1X2 open+close is now complete for 2024–26 (2023 has only
     56 opens and no usable training history) — useful here as the *benchmark* an earlier book's
     line gets measured against.
   - **SURVEY DONE (2026-07-16, `src/csl/odds/survey_bookmakers.py`, 3 credits).** One h2h call,
     `regions=us,eu,uk`, no bookmakers filter: **40 books** quote CSL 1X2 across 8 fixtures.
     Cheapest first (mean overround, Now line):

     | book | overround | events | note |
     | --- | ---: | ---: | --- |
     | `matchbook` | **2.43%** | 4/8 | exchange — commission not in the price |
     | `betfair_ex_eu` / `betfair_ex_uk` | **3.95%** | 8/8 | exchange — commission not in the price |
     | `onexbet` (1xBet) | **4.76%** | 8/8 | **the only cheap *traditional* book** |
     | `gtbets` | 5.64% | 7/8 | |
     | `coolbet` | 5.96% | 8/8 | |
     | **`pinnacle`** (reference) | **6.57%** | 8/8 | vs 7.55% at *open* (§11.7) |
     | the other 33 | 6.7–16.0% | — | William Hill 9.5%, Paddy Power 9.9%, Winamax FR 16.0% |

     **Reading it — three caveats that decide what this means:**
     1. **Exchanges price differently.** Matchbook/Betfair's 2.4–4.0% excludes commission on net
        winnings (~2% Matchbook, 2–5% Betfair), so the p×R bar does not apply to them unmodified —
        their true cost depends on strike rate, not just the quoted overround.
     2. **This is a Now line, not an open.** Pinnacle sits at 6.57% here vs its 7.55% opening
        (§11.7) — books widen at open and tighten toward kickoff. So the candidates' *opening*
        overrounds are **wider than the table**; 1xBet's 4.76% Now might be ~6% at open, which
        would put it back above the bar. **Do not treat these numbers as opening overrounds.**
     3. Matchbook covers only 4/8 fixtures (liquidity).
   - **Forward capture of the candidates — DONE, zero quota (2026-07-16).** `capture_scheduler`
     now stores `fetch_pinnacle_spreads.CAPTURE_BOOKMAKERS` = pinnacle + the four sub-5%
     candidates at every open window. **Measured, not assumed:** The Odds API bills
     `markets × regions`, counts each 10 bookmakers as 1 region, and `bookmakers` takes
     precedence over `regions` — so `regions=us` + this 5-book list costs the **same 1 credit**
     as the old Pinnacle-only call *and* reaches the eu/uk-only books. Fire/pending stays keyed
     on Pinnacle alone (`REFERENCE_BOOKMAKER`), so an early-opening rival can never stop the
     ticks before Pinnacle's anchor line is stored. This settles caveat 2 for free: after a
     round or two the history holds each candidate's **true opening** overround, and a book
     already showing a price when Pinnacle opens is a book that opened earlier.
   - **Historical backfill = 1xBet, done by hand (no API historical endpoint on the free plan).**
     The user manually backfilled 1xBet opening **and** Pinnacle closing 1X2 into
     `CHN_Super League.csv` (`onexbet_open_*` / `onexbet_close_*`) for **2024, 2025 and 2026** (240 +
     238 + 140 rows). All vetted for entry errors with a **logarithmic overround check** (`ln(Σ1/oᵢ)`;
     post-correction median 4.75–4.76%, sd 0.11–0.12, no impossible values — lines are sound). 2023
     still empty (an oddsportal scrape of 2025 was reverse-engineered but rejected as not
     robust/accurate, and 1xBet is geo-hidden there — see [[onexbet-historical-source-attempts]]).
   - **BACKTEST RESULT — first cross-season +EV cell in the project (2026-07-16, extended to 3
     seasons 2026-07-17; `backtest.md` §13.1, `backtest/backtest_1xbet.py`).** Betting the model's EV
     on 1xBet's **opening** 1X2, CLV graded vs Pinnacle's **close**; n=611 over 2024+2025+2026. 1xBet
     opening overround 4.87% → vig bar ~1.71pp. **At thr>0.20 the gap (CLV − bar) is positive in ALL
     THREE seasons independently** for every model variant (production δ no-draw: 2024 gap +2.79 /
     exCLV +4.32 t=3.0; 2025 gap +2.53 / exCLV +4.19 t=2.7; 2026 gap +1.07 / exCLV +2.46 t=1.8;
     pooled +2.28 / +3.82 t=4.4) — the §11.1 single-season caveat is fully discharged. Caveats that
     keep it a lead, not a green light: (a) per-bet ROI is high-variance and **2025 realized −24% ROI
     on that very cell** despite its +2.53 gap / +4.19pp exCLV — the clean proof to read exCLV/gap,
     not ROI; (b) **thr must be 0.20** — at thr>0.10 the 2026 gap flips negative; (c) raw vs δ vs λ
     are the **same edge** — raw scores highest only because its un-repaired draw-deflated H/A probs
     act as an implicit tighter EV filter (raw@0.20 ≈ δ@0.25; converge exactly at thr>0.50). Real
     knob = selectivity (EV threshold), not model; use δ (production) and raise the threshold. Unlike
     §9's AH, here tighter = monotonically better exCLV (a healthy, non-winner's-curse signature).
   - **Next (needs the user):** add **2023** 1xBet opens (and optionally Betfair, if the commission
     math is modelled) to tighten; score with the two rules above (excess CLV vs §11.3 baseline; p×R
     bar §11.7). Before any real staking, confirm the edge survives as *realized* ROI across seasons
     (2025's −24% is the warning), and prefer captured live opens for the honest opening overround.

9. **Draw de-bias (+ ZIP→NegBinom) — TESTED, bar not cleared (2026-07-15, backtest phase
   DONE).** The backtest verdict is in `backtest/backtest.md` §12; `backtest/backtest_1x2.py`
   now carries the full variant grid. Background: the model's draw probability is pinned at
   ~0.279 vs the market's 0.234 and an actual 0.242 — structural (independent-Poisson mass at
   goal-diff 0), 61% of the §11 strategy's stake sat on it.
   - **Mechanism (user-chosen): market-anchored shrink**, per match, no leakage —
     `p'_D = (1−λ)·p_D + λ·m_D` with `m_D` = no-vig *opening* draw prob, freed mass returned
     to home/away pro-rata. λ grid {0.25, 0.5, 0.75, 1.0}, not walk-forward-optimised.
   - **As a model fix it works:** draw prob repaired (0.245 at λ=0.75 vs actual 0.242), stake
     migrates off the bug (draw picks 63%→0%), **excess CLV roughly doubles** (+0.60 →
     +1.42pp at thr>0.10, t=3.2; +2.0–2.5pp at thr>0.20), 0.25-Kelly goes from −59% to +20%.
   - **As a betting strategy it still fails the success bar:** per-season gap (CLV − p×R) is
     **negative in 2024 and 2025 at every λ** (2024 λ=1: exCLV −0.10pp — no signal there at
     all); only 2026 clears, the same one-season shape as the §11.1 false signal. ZIP+λ=1 ≈
     NegBinom+λ=1: the distribution swap contributes ~nothing to CLV, it's all the de-bias.
   - **Consequences:** (a) do NOT build a 1X2 betting pipeline against Pinnacle's open;
     (b) strongest quantified case yet for roadmap #8 — the surviving +1.2–2.5pp excess CLV
     clears the breakeven bar at a ≤5%-overround book (bar ≈ 1.4–1.75pp) while losing into
     Pinnacle's 7.55% open.
   - **Production deployment (v2.5, 2026-07-15, user-approved):** the market-anchored λ
     needs a 1X2 anchor production doesn't have (adding h2h would double /odds quota cost),
     so the deployed mechanism is the **market-free δ calibration** (`backtest.md` §12.4):
     fit walk-forward-validated first (no degradation; repairs ~half the bias, draw
     0.276 → 0.255), then shipped in `dc.py` as `NegativeBinomialGoalModel` +
     `DrawCalibratedModel` (δ ≈ 0.91 on the current fit). Dashboard bumped to **v2.5**
     with model name "Negative Binomial with Dixon-Coles Time Decay". For *accuracy*,
     not betting — the betting verdict above stands. Upgrade path to the full
     market-anchored λ: roadmap #10.

10. **Market-anchored λ de-bias in production — IMPLEMENTED (v2.6, 2026-07-16).**
    Replaces the δ-only de-bias with the §12-validated market-anchored shrink wherever an
    anchor exists; δ stays as the fallback. Draw prob on the comparison surface ≈ 0.245–0.25
    (λ=0.75) instead of δ's 0.255, per fixture, self-adapting by match type (§11.4).

    **Design: h2h REPLACES spreads — AH is retired.** The AH betting route is falsified
    (§9–§10) and the user bets 1X2 going forward. `fetch_pinnacle_spreads` now requests
    `markets=h2h` (parses home/draw/away prices; module and CSV filenames keep their legacy
    "spreads" names). Cost per /odds call unchanged (1 market × 1 region = 1 credit) →
    **zero quota impact** (~290–310/500). The λ anchor is the *captured opening* 1X2
    (exactly what §12 validated); the Now-line refresh continues on the 3h cadence.

    **User decisions that shaped the final schema (2026-07-16):**
    - The user hand-backfilled 23 opening-1X2 rows (rounds 18–19) directly into
      `CHN_pinnacle_spreads_history.csv` with a **17-column schema of their own**: spread
      columns REMOVED (not left empty), `draw_odds` inserted between `home_odds` and
      `away_odds`, and **`market=moneyline`** (not the API's "h2h" key). The code adopts
      this as canonical: `OUTPUT_COLUMNS` matches it and the fetch writes
      `MARKET_LABEL="moneyline"` while requesting `markets=h2h`. `DEDUP_KEY` unchanged
      (no `market` component needed — the file is single-market again).
    - Historical spreads rows were REPLACED by the backfill, not preserved (supersedes the
      earlier "kept untouched" plan). The old AH history survives in git history and in the
      backtest section of `CHN_Super League.csv` (open/close AH columns) — nothing analytic
      was lost.
    - Confirmed: λ = 0.75; hybrid semantics (comparison anchored / predictions δ — see
      "Model" section above); 3-rows-per-match (H/D/A) dashboard layout; full spreads stop
      (Now line included).

    **What shipped (branch `docs/roadmap-10-market-anchored-debias`):**
    - Fetch/capture: h2h parsing in `fetch_pinnacle_spreads.extract_rows` (Draw outcome
      required), `snapshot_store`/`capture_snapshot`/`capture_scheduler` unchanged in logic
      (schema derives from `OUTPUT_COLUMNS`). `CHN_pinnacle_now.csv` reset to a
      new-schema header — repopulated by the first post-merge 3h refresh.
    - Export: `export_upcoming_market_comparison` rewritten — 1X2 columns, hybrid λ/δ
      probs (`DrawCalibratedModel.predict_raw` added to `dc.py` so λ starts from the
      un-δ'd grid), per-outcome open/now EV, `debias_method` audit column; the simulations
      CSV is no longer an input (probs come straight from the model fit).
    - Dashboard: comparison panel = 3 rows per match (home/Draw/away) × Open Odds/EV, Now
      Odds/EV, odds-direction Move arrows; tooltip shows the full H/D/A snapshot triplet;
      Best Bet metric now includes the draw. JSON whitelist updated. Version v2.6.
    - **Validation:** λ math + EV consistency + schema round-trip + dedup re-checked
      locally against the user's backfilled CSV (8 round-19 fixtures, all
      `market_anchor`, anchored draw mean 0.249). A separate backtest "hybrid" variant is
      unnecessary: every backtest row has an opening 1X2, so hybrid ≡ the λ=0.75 variant
      already validated in §12 (δ fallback validated separately in §12.4).

    **Still open:**
    - First live end-to-end observation: next 3h refresh repopulates the Now line (doubles
      as the h2h recon call); next open-window capture (round 20, expected from ~2026-07-17)
      writes the first automated `moneyline` open row. Optional user field-check of that
      first capture against Pinnacle at open (assist item e).
    - The betting verdict is unchanged: this is an accuracy upgrade; do NOT bet Pinnacle's
      1X2 open (§12 gap negative 2024/2025). The surviving +1.2–2.5pp excess CLV is the
      case for roadmap #8 (cheaper book).

11. **Model & strategy status after #9/#10 + next direction — DECIDED (2026-07-16,
    user-confirmed).** The one-stop snapshot for future sessions; numbers are from
    `backtest/backtest.md` §12 (walk-forward, 611 fixtures 2024–26) unless noted.
    - **Model (v2.6 — SUPERSEDED by v2.8, see §15):** NegBinom on xG + DC decay (xi=0.001,
      18mo), δ=0.908 market-free calibration everywhere, λ=0.75 market-anchored shrink on the
      comparison/EV surface. Accuracy: best of six distributions — RPS 0.1971 / log-loss
      0.9755 (~1.5% better than the retired ZIP, which had collapsed to Poisson).
      **All three numbers are artefacts of the truncation bug**: the fit was against
      `floor(HExpG+)`, δ=0.908 was suppressing a bug-induced draw excess (now retired), and
      λ=0.75 is no longer the optimum (~1.25 is; 0.75 is worst-of-grid). Current: v2.8
      `ContinuousPoissonGoalModel`, δ off, λ pending — `backtest/backtest.md` §15.
    - **What the de-bias bought (before → δ → λ=0.75):** OOS draw prob 0.276 → 0.255 →
      **0.245** vs actual 0.242 (bias eliminated); draw share of picks 63% → 45% → 27%
      (stake off the bug); excess CLV thr>0.10 +0.60 → +0.93 → **+1.39–1.42pp (t=3.2)**,
      thr>0.20 up to +2.5pp; 0.25-Kelly three-season end 40.8 (−59%) → **120.1 (+20%)**,
      max drawdown 86% → 50%. The distribution swap itself contributes ~0 CLV — all edge
      movement is the de-bias.
    - **What it did NOT change — the vig wall:** per-season gap (CLV − p×R) is still
      negative in 2024 (−1.78pp) and 2025 (−1.22pp) at every λ; only 2026 clears (+0.91pp).
      Replicable signal ~1.2–1.4pp vs a ~2.2–2.5pp Pinnacle-open bar. **Betting Pinnacle's
      1X2 open stays CLOSED.** The signal clears a ≤5%-overround book (bar ≈ 1.4–1.75pp).
    - **Model work → maintenance only** (§12.3: "cheaper prices, not further model work"):
      watch the δ refit drift (current 0.908; per-round history 0.82–1.00) and the anchored
      draw prob vs actual as 2026 accumulates. No further model changes planned.
    - **Next step (user decision 2026-07-16): the #8 bookmaker survey ONLY** — DONE the same
      day, see #8 for the 40-book table and its three caveats. Headline: 1xBet 4.76% is the
      only cheap traditional book, the two exchanges are cheaper but carry commission, and
      **all of it is Now-line data — opening overrounds are wider and are now being captured
      for free**. Awaiting the user's pick of which book to backfill historically.
      Explicitly NOT chosen (proposed and deferred): the #3 close-capture piggyback
      (persisting the last pre-kickoff 3h Now snapshot as `snapshot_type=close` for a live
      excess-CLV tracker, zero quota) — do not build it without a fresh user go-ahead.


## Agent Tips
- Prefer `./scripts/csl.sh` over direct module execution for local workflow tasks.
- If a task is only "update the data", use `./scripts/csl.sh update`.
- If a task needs a fresh public dashboard bundle, `./scripts/csl.sh all` is the primary end-to-end command.
- If a task touches the dashboard data but not the raw pipeline, `./scripts/csl.sh publish` is the fastest rebuild path.
