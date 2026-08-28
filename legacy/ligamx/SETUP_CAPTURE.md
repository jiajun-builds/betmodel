# Odds capture — setup

The capture loop collects two prices this repo could not otherwise get:

| | source | why it matters |
|---|---|---|
| **Betano UK + Duel openers** | odds-api.io | the Betano opener is the only positive-EV result the project has (n=355, EV>10% → +2.56pp, t=4.64), and all 461 existing rows were entered by hand |
| **Pinnacle closes** | The Odds API | the CLV benchmark every edge claim is graded against; football-data.co.uk stopped publishing `PSC*` for Liga MX after Oct-2025 |

Everything lands in `data/MEX_odds_capture_history.csv` (tracked in git — the
capture runs in CI and has to commit what it collects) and reaches
`MEX_ligamx.csv` via `reduce_capture_history`, which runs as a step of
`./scripts/ligamx.sh update`.

**Nothing has to be run by hand for a captured price to become usable.** The
reduce step lives in `update` rather than in the capture tick because it writes
into `MEX_ligamx.csv`, and that file has no row for a fixture until the fixture
has been played — so captured prices wait in the history until the post-matchday
`update` creates their row.

---

## 1. Repository secrets

`Repo → Settings → Secrets and variables → Actions → New repository secret`:

| Secret | Where to get it | Used by |
|---|---|---|
| `ODDS_API_IO_KEY` | odds-api.io dashboard | job 1 (openers) |
| `THE_ODDS_API_KEY` | the-odds-api.com dashboard | job 2 (closes) |

If either is absent its job logs and skips — it never fails the run.

### Bookmaker entitlement (odds-api.io)

The free plan allows **at most 2 recreational bookmakers per account**, and the
selection is per-key. This account is entitled to **`Betano UK` and `Duel`**.

Two traps:

- The provider name is **`Betano UK`**, not `Betano`. The latter returns 403.
- `/bookmakers/selected` is documented as stale. The **403 body of `/odds/multi`
  is the source of truth**, and it names the allowed books:

  ```
  Access denied. You're allowed max 2 bookmakers. Allowed: Betano UK, Duel.
  ```

To check the current entitlement:

```bash
curl -s "https://api.odds-api.io/v3/odds/multi?apiKey=$ODDS_API_IO_KEY\
&eventIds=1&bookmakers=Betano%20UK,Duel" | head -c 400
```

Changing the selection is done on the odds-api.io dashboard; `CAPTURE_BOOKS` in
`src/ligamx/odds/oddsapi_io.py` then has to match it.

---

## 2. The timer (Cloudflare Worker)

GitHub's own `schedule:` cron **cannot** drive this. Measured on the sister repo
over 299 runs, a requested `*/10` landed at median 10 min but **p90 137 min, worst
232 min** — wider than the whole 20-minute closing window. A missed close is
unrecoverable: the fixture leaves the pre-match feed at kickoff and no provider
sells it afterwards. The workflow keeps a `*/30` cron purely as a fallback so that
if the Worker dies, openers still get captured.

### 2.1 Create the PAT

`GitHub → Settings → Developer settings → Fine-grained tokens → Generate new token`

- **Repository access**: only `jiajun-builds/ligamxterminal`
- **Repository permissions → Contents: Read and write** — this is what authorises
  `repository_dispatch` (Metadata: read is added automatically)

### 2.2 Deploy

```bash
cd tools/capture-timer
npx wrangler secret put GITHUB_PAT     # paste the token
npx wrangler deploy
```

Two crons are registered: `*/15` fires `open-tick`, `*/5` fires `close-tick` —
but only when a fixture is actually within 25 minutes of kickoff. That gate is
what keeps the Actions log readable: unconditional 5-minute firing is 288 runs a
day, ~250 of which would do nothing.

### 2.3 Verify

```bash
# Fire by hand (the Worker also answers HTTP for exactly this):
curl "https://ligamx-capture-timer.<your-subdomain>.workers.dev/?type=open-tick"

# Or straight to GitHub — 204 No Content means it fired:
curl -sS -o /dev/null -w '%{http_code}\n' -X POST \
  -H "Authorization: Bearer <PAT>" \
  -H "Accept: application/vnd.github+json" \
  https://api.github.com/repos/jiajun-builds/ligamxterminal/dispatches \
  -d '{"event_type":"open-tick"}'

npx wrangler tail        # watch the cron decisions live
```

Then check `Actions → Capture Odds`.

**Alternative host.** An old Mac running the same `curl` on a `launchd`
`StartInterval` works, but it loses every tick the machine is asleep or off the
network — fine for openers, not for closes. Prefer the Worker.

---

## 3. Budgets

Both providers are on free tiers, and the design is shaped by their very
different shapes.

| | odds-api.io | The Odds API |
|---|---|---|
| Allowance | ~500/**day**, 100/hour | **500/month** |
| Tick spend | ≤3 (1 `/events` + 2 `/odds/multi`) | 1 per in-window tick |
| Worst case | 96 runs/day × 3 = **288/day** | 39 fixtures × 4 ticks = **~156/month** |
| Idle tick | 0 once every fixture in the lookahead has opened | 0 — the window check reads local CSVs, and the quota probe (`/v4/sports`) is free |

Two levers control the spend, and neither should be changed casually:

- **`--lookahead-days` (default 14)** in `fetch_oddsapiio_opens`. Liga MX
  publishes its whole season, so without this all 126 future fixtures stay
  pending forever and every tick spends the cap on matches nobody has priced. 14
  days sits just outside Betano's measured median open of T-293h.
- **`--window-minutes` (default 20)** in `capture_close`. At 39 fixtures/month a
  60-minute window costs ~468 of 500 credits; 20 minutes costs ~156. Naming
  Betfair and Matchbook alongside Pinnacle is free — The Odds API bills
  markets × regions and counts 10 bookmakers as one region.

---

## 4. Running it by hand

```bash
./scripts/ligamx.sh capture            # both captures, respecting every gate
./scripts/ligamx.sh capture --dry-run  # decide only: spend nothing, write nothing

# or individually
PYTHONPATH=src python -m ligamx.odds.fetch_oddsapiio_opens --dry-run
PYTHONPATH=src python -m ligamx.odds.capture_close --dry-run
PYTHONPATH=src python -m ligamx.odds.reduce_capture_history --dry-run
```

`capture` is only useful by hand with `--dry-run`, to see what the next CI tick
would spend — the Worker is already running the real thing. `reduce_capture_history`
likewise runs inside `update`; invoke it directly only to inspect what it would do.

---

## 5. Which captured openers can be trusted

Not all of them, and this is the part most likely to cause quiet damage.

A captured "open" is a real opening line only if we were watching **before** the
book priced it. `reduce_capture_history` decides that from the data rather than
guessing:

```
trustworthy  ⇔  (kickoff − lookahead_days)  ≥  the first capture we ever made
```

A fixture becomes pending at `kickoff − lookahead`. If that moment came after
capture began, no price can have escaped us and the first price we saw is the
first price posted. If it came earlier, the book may have opened unobserved and
the row is a mid-market price wearing an opener's label.

This self-calibrates — it needs no assumption about when any book opens, and it
widens automatically as the history ages. On day one it correctly rejects
everything. Held-back rows stay in the history (they are perfectly good price
observations); they just never reach `MEX_ligamx.csv`.

`reduce_capture_history` also **never overwrites** a non-blank cell:
`pinnacle_close_*` holds 856 values spliced from three sources and repaired by
hand, and `betano_open_*` holds 461 hand-entered rows.

### Still to validate

**Is "Betano UK" the same book as the Betano behind those 461 hand-collected
openers?** The entitlement is specifically the UK-facing site (the API's own
`urls` field points at `betano.co.uk`). If the hand-collected rows came from
Betano MX or .br, margins and prices may differ and the +2.56pp result would not
automatically transfer. This cannot be settled retroactively — odds-api.io sells
no opener history at any tier. Once ~2 rounds of Betano UK opens are banked,
compare their overround and price levels against the hand-collected distribution
before treating the two as one series.
