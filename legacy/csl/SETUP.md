# Environment Setup

## Prerequisites
Make sure you have [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.

---

## Create the environment

```bash
conda env create -f environment.yml
```

## Activate it

```bash
conda activate csl-workflows
```

## Local secrets

Create a local environment file:

```bash
cp .env.local.example .env.local
```

Fill in:

- `THE_ODDS_API_KEY`

(xG is fetched from the official SofaScore API — no key required.)

The unified workflow script loads `.env.local` automatically.

## Verify everything installed

```bash
python -c "import pandas, numpy, scipy, matplotlib, sklearn, penaltyblog; print('All packages OK')"
```

## Verify the unified workflow entry point

Run these from the repository root after activating `csl-workflows`:

```bash
./scripts/csl.sh help
./scripts/csl.sh dashboard
./scripts/csl.sh publish
```

What each command does:

- `./scripts/csl.sh help`: shows the supported subcommands
- `./scripts/csl.sh dashboard`: generates dashboard-facing CSV and JSON files
- `./scripts/csl.sh publish`: rebuilds dashboard exports and the GitHub Pages-ready `site/` directory

---

## Useful commands

| Task | Command |
|------|---------|
| Show menu | `./scripts/csl.sh` |
| Full local workflow | `./scripts/csl.sh all` |
| Update data | `./scripts/csl.sh update` |
| Run model | `./scripts/csl.sh model` |
| Export dashboard CSV/JSON | `./scripts/csl.sh dashboard` |
| Fetch odds + market comparison | `./scripts/csl.sh odds` |
| Rebuild publish artifacts | `./scripts/csl.sh publish` |
| Deactivate | `conda deactivate` |
| List packages | `conda list` |
| Remove env | `conda env remove -n football-analytics` |
| Export env | `conda env export > environment.yml` |

---

## Notes
- `penaltyblog` is installed via `pip` (not on conda-forge), so it's handled in the `pip:` section of `environment.yml`
- All other packages are pulled from `conda-forge` for better compatibility on both Windows and macOS/Linux
- Live data grabbing depends on external APIs:
  - `csl.fixtures.chn_fixture_v5` uses TheSportsDB
  - `csl.xg.xg_pipeline` uses the official SofaScore API (via `curl_cffi` browser impersonation); no key required
  - `csl.odds.fetch_pinnacle_spreads` uses The Odds API and requires `THE_ODDS_API_KEY`
