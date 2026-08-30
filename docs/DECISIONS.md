# Design decisions taken during the merge

Each entry records a decision that changes behaviour rather than structure, so
that a difference against the frozen pre-merge output can be told apart from a
regression. Anything not listed here must reproduce exactly.

---

## D1 — The Now line is retired. Only the Pinnacle open anchors the model.

**Decided 2026-08-28.**

Three prices exist for a fixture and they answer different questions. The
**open** is the price before the market has digested the information. The
**close** is the price after it has. The **Now line** is whatever it is at this
instant.

Only the open has a decision role. It is the anchor the model's draw probability
is calibrated against, and it is the price a signal is bet at. The Now line
informs nothing: by the time it is fetched, the bet has either fired on an
opening price or it has not.

The two pipelines had drifted into using it differently, and neither use
survives scrutiny:

* Chinese Super League never published it. It fed the fixture-inclusion gate,
  some archive columns nobody reads, and `backfill_open` — the path that wrote a
  Now line into an open slot and produced the 20 contaminated rows.
* Liga MX published it as `pinnacle_*_odds`, shown on the board as a reference
  price. That was an oversight. Comparing a bet book's *opening* price against
  Pinnacle's *current* price puts the two on different points of the market's
  life, which is not a comparison.

**Consequences**

1. The separate Now-line fetch is deleted from both leagues. It was a billed
   request that duplicated one the open poll already makes: the same `/odds`
   call with `bookmakers=pinnacle` returns the current price, and whether that
   price is "the open" or "the Now line" is decided by whether an open is
   already stored, not by making a second request.
2. Liga MX's published `pinnacle_*_odds` fields are filled from the Pinnacle
   **open** instead. The fields stay populated so the downstream board keeps
   working, and the comparison becomes like-for-like.
3. `backfill_open` goes with it, which is what stops new contaminated rows.

**Accepted difference against the golden baseline**

Liga MX `public/legacy/ligamx/upcoming_market_comparison.json`, fields
`pinnacle_home_odds`, `pinnacle_draw_odds`, `pinnacle_away_odds`,
`pinnacle_last_update`, `pinnacle_fetched_at`. These will differ from the frozen
output because their source changed from the current price to the opening price.
Every other field in every other file must still match exactly.

**Deliberately NOT done yet**

Enabling market-anchored de-bias for Liga MX. The intent is that both leagues
calibrate the draw against the Pinnacle open, but Liga MX has zero Pinnacle opens
captured today, so there is nothing to anchor to until polling accumulates them.
Turning it on also changes which bets fire, which is a strategy decision that
needs a backtest, not a side effect of a merge. Revisit once opens exist and the
gates have passed.

---

## D2 — One scoreline grid for every league: 0 to 9.

**Decided 2026-08-28.**

The two fitters used grids of different sizes, and worse, read a field of the
same name differently: one built `arange(max_goals)` and the other
`arange(max_goals + 1)`, so the same number meant a 10-cell grid in one and an
11-cell grid in the other. `max_goals` now means the highest scoreline modelled,
unambiguously, and both leagues use 9.

**Measured cost of the reduction**, at Liga MX's scoring rates (mean targets
1.587 home, 1.197 away):

| case | probability mass outside a 10-cell grid |
|---|---|
| typical fixture | 7.2e-06 |
| a fixture at lambda 3.0 | 1.1e-03 |
| a fixture at lambda 3.5 | 3.3e-03 |

Orders of magnitude below the model's own uncertainty, and the 1X2 aggregation
renormalises, so the residual effect is second order.

**Accepted difference against the golden baseline**

Liga MX model outputs shift slightly. **Measured**: at most 1.1e-04 on a 1X2
probability across all 342 team pairings, which is about one hundredth of a
percentage point against a signal threshold of ten percent. Chinese Super League
is unaffected, since its grid was already 10 cells.

**How gate G1 handles this**

Two separate checks, so a porting bug cannot hide behind a deliberate change:

1. **Faithful port.** Fit each league with its pre-merge optimiser settings and
   grid size, and require the parameters to reproduce the frozen output tightly.
   This answers "is the merged fitter the same mathematics".
2. **Applied change.** Fit again with the unified settings and report the delta.
   This answers "how much did the decisions above move the numbers", and the
   answer has to be consistent with the table above.

---

## D3 — One optimiser setting, the tighter one.

**Decided 2026-08-28.**

The two fitters used different starting points and different convergence
tolerances. Liga MX set `ftol=1e-12, gtol=1e-10`; Chinese Super League used
scipy's defaults, which are far looser. The objective is convex, so both reach
the same optimum in principle, but "in principle" is not a claim to make about a
number that reaches a bet, so it was measured.

Liga MX's settings are adopted for both. Measured on the CSL training window:

| setting | log-likelihood | max gradient | score equation |
|---|---|---|---|
| scipy defaults | -509.7319474717 | 2.30e-03 | (0.999996, 1.000007) |
| ftol 1e-12, gtol 1e-10 | -509.7319470621 | 5.59e-04 | (0.999999, 1.000000) |

The tighter setting is closer to the optimum on every axis: higher likelihood, a
gradient four times smaller, and a score equation an order of magnitude nearer
1.0. The frozen CSL output is slightly under-converged.

**Accepted difference against the golden baseline**

CSL attack and defence coefficients move by at most 9.3e-05, which is a relative
2e-04 on a coefficient of order 0.5 and about 0.01% on a fitted scoring rate. Liga
MX is unaffected: its pre-merge settings were already these.

**Why this does not weaken gate G1**

G1 keeps both settings available and runs two checks. Fitting with each league's
pre-merge settings must reproduce its frozen coefficients to floating-point
precision, which proves the merged fitter is the same mathematics. Fitting with
the unified settings then reports the delta, which measures the decision. A
porting bug cannot hide inside the second number, because the first would fail.

Measured faithful-port result: max coefficient difference 5.6e-16 for CSL and
3.0e-15 for Liga MX. That is floating-point noise.

---

## D4 — 1X2 probabilities are read directly, not recovered from a zero handicap.

**Decided 2026-08-28.**

Both pre-merge pipelines produced their 1X2 figures like this:

```python
home = probs.asian_handicap("home", 0)
away = probs.asian_handicap("away", 0)
draw = 1 - home - away
```

It gives the right answer, and it is worth being precise about why. The library
defines a zero-line "win" as push-excluded, so at line 0 it is exactly the
outright win probability. Its own docstring calls that method
backward-compatible and directs anything careful to `asian_handicap_probs`.

**The cost is fragility, not accuracy.** The draw is a residual of two numbers
that come from a market where the draw is a push. If that zero-line convention
ever became push-adjusted, the standard reading of an Asian 0, then home and away
would sum to one and the draw would silently become zero. Nothing would raise.

It is also redundant. The same call already returns the draw:

```
asian_handicap_probs('home', 0) -> {'win': 0.798, 'push': 0.133, 'lose': 0.069}
```

The push component **is** the draw. It was being discarded and reconstructed by
subtraction.

A third route existed as well, in the signal engine, summing the scoreline grid
by goal difference and renormalising. So three pieces of code computed the same
three numbers three ways.

**All three are now one call**, `model.outcome_probabilities(home, away)`, which
reads the grid's own 1X2. Measured agreement before collapsing them, over the
sampled pairings:

| route | max difference from the direct read |
|---|---|
| zero-handicap detour | 3.9e-16 |
| raw grid sums | 2.2e-16 |

**Accepted difference against the golden baseline**: none. This is a rewrite of
the expression, not of the value. A test pins the equivalence, so a library
upgrade that changes the zero-line convention is caught rather than shipped.

---

## D5 — Chinese Super League gains a reducer, and the master tables get the same odds columns.

**Decided 2026-08-28.**

Only one of the two pipelines ever reduced its capture history into its match
table. Liga MX did; Chinese Super League did not, and read opening prices
straight out of the history at signal time instead. Its match table does carry
odds columns, heavily populated, but those are historical imports rather than
anything the capture loop wrote.

**What changes, and what does not.** CSL's published signals do not move at all:
the signal engine reads the history, not the match table, so this touches only
what the research and CLV layers see. Replaying the history writes 210 values
into blanks that were never filled, and overwrites nothing.

**Two facts found while wiring it up.**

The provenance filter earns its place immediately. Of 160 usable CSL open rows,
20 are excluded as not being opening prices at all, on `onexbet` and `pinnacle`.
Without that filter a reducer would have written mid-market placeholders into the
opening-line columns, which is the exact corruption the opening-line series
cannot survive. Liga MX loses none.

CSL has no `duel_*` or `betfair_*` columns, so 147 captured values had nowhere to
go. Captured data with no column is silent waste, so `missing_columns()` now
surfaces it rather than counting it, and `merge(create_columns=True)` adds them.
Column creation is off by default: growing a tracked file's schema should be a
deliberate migration step, never a side effect of a routine capture tick.

**How gate G2 reads**

Replaying the history must be a no-op for the league that was already reduced.
Measured: Liga MX writes 0 and keeps 36, so every value the reducer would produce
is already present and identical. For CSL there is nothing to replay against, so
the gate is that every write lands in a blank and none overwrites.

---

## D6 — One unambiguous kickoff column, required rather than derived.

**Decided 2026-08-28.**

Three stages decide *when* to act: the open poller decides what is still pending,
the close poller decides what is inside its window, and the signal engine decides
what is still upcoming. All three are wrong in the same way if the kickoff is
wrong, and none of them would raise.

One pipeline had already learned this. Its match table's `Time` column mixes
local and UTC, the ambiguity fabricated a result once, and it responded by
writing an explicit `kickoff_utc` and never reading `Date`/`Time` again. The
other pipeline was still parsing the pair and calling the result UTC, which is
correct for that league today and is exactly the assumption that failed
elsewhere.

The loader now requires `kickoff_utc` and refuses to derive one. CSL's file has
been backfilled from its documented-UTC pair as a one-time migration, and the
fixtures stage writes the column from here on.

**Accepted difference against the golden baseline**: none. The derived values are
identical to what the old parse produced for that league; what changed is that
the assumption is now stated in the data instead of held in a comment.

---

## D7 — odds-api.io credentials are per league.

**Decided 2026-08-28.**

A key on this provider is entitled to a fixed, small number of bookmakers, and
the refusal is explicit: *"You're allowed max 2 bookmakers. Allowed: 1xbet,
Duel."* The two leagues bet three distinct books between them, so they cannot
share a key, and the split is now part of each league's config rather than a
global environment variable.

The same refusal exposed a second problem. The client described the 403 body as
"the authoritative entitlement list, unlike the catalogue endpoint which goes
stale" and then discarded it, which turned a self-explaining error into a guess
and cost real time. Refusals now carry a redacted excerpt of the body, and name
which credential and which environment variable were refused.

**Not a production issue.** A local `.env` in one source repo holds a stale copy
of the other league's key; the repository secret is correct and the live capture
loop is unaffected.

---

## D8 — `price_quality` is deferred with Polymarket.

Its only input is the Polymarket snapshot store, and that provider is deferred to
the evaluation phase because no production path reads it. Porting a module with
no inputs would move dead code rather than merge anything. It travels with the
evaluation layer.

---

## D9 — A fixture is named by its local matchday.

**Decided 2026-08-28.**

`fixture_id` is the join key across every published file, and it embeds a date.
That date is now the kickoff converted to the league's own timezone, not UTC.

A 19:00 kickoff in a UTC-6 league falls on the next UTC day, so naming the
fixture by the UTC date names a day nobody played on. The bug hides completely in
a league whose kickoffs sit in the middle of the UTC day and appears the instant
one does not, which is the worst way for it to appear: it looked like the engine
had failed to produce seven of nine Liga MX fixtures, when it had produced all
nine under names differing by one day.

**Accepted difference against the golden baseline**: none. This restores the
identifiers both pre-merge exporters already produced; the UTC reading was
introduced by the merge and caught by replaying the baseline.

---

## D10 — Model probabilities are read, not refitted, inside the signal path.

One pipeline refitted the model inside its exporter. The other read the fitted
simulations. Refitting means a published probability can differ from the one the
same model wrote minutes earlier, for no reason visible to a reader, and it puts
a several-second fit inside a path that runs on every capture tick.

The engine now reads `simulations.csv`. Verified by replay: probabilities match
the frozen baseline to 3.3e-16 for the league that refitted, and exactly for the
one that read.

**Accepted difference against the golden baseline.** The refitting league's
`match_predictions.json` published probabilities from that in-exporter fit, which
differ from the frozen simulations in the seventh decimal. On one of eight rows
that is enough to move a sixth-decimal rounding, so `away_win_prob` and the fair
odds derived from it differ by one unit in the last published place. Every other
field of every other row matches exactly, and the other league matches
completely.

---

## D11 — What the two legacy shapes disagreed about.

Rendering both pre-merge payloads from one engine turned up three field-level
inversions that had never been visible because each pipeline only ever saw its
own. All three are reproduced exactly by the compatibility exporter and all three
are gone from the canonical contract.

**`kickoff_at` and `match_time` mean opposite things.** One shape puts
league-local time with an offset in `kickoff_at` and UTC time-of-day in
`match_time`; the other puts UTC in `kickoff_at` and local time-of-day in
`match_time`. Both parse to the right instant, so nothing downstream ever broke,
but a consumer reading them as one contract is reading two.

**The envelope timestamp is stamped in different zones**, league-local in one and
UTC in the other. Same inversion, one level up.

**"The book" means two things in one row.** `price_proof`, `price_captured_at`
and `price_age_h` describe the price the row was *judged* on, which is the
best-expected-value side whether or not it cleared the bar. `bookmaker` and
`last_update` describe the price to actually *bet*, which is empty when nothing
fires. Reproducing this needed both notions kept separate; conflating them put
provenance on seven rows that should have had none, or stripped it from seven
that should.

The canonical contract has one UTC field, one zone, and one explicit distinction
between the judged price and the bettable one.

---

## D12 — The match history gains an explicit kickoff, because its date column means three different things.

**Decided 2026-08-28, after a dry run refused to write.**

Classifying every row of one league's match history against provider truth:

| season | league-local | UK-local | UTC | unresolved |
|---|---|---|---|---|
| Apertura 2026 | 27 | 0 | 0 | 0 |
| Clausura 2026 | 38 | 41 | 1 | 80 |

The UK-local block is the stretch backfilled from a public archive that stamps
kickoffs in UK time. So the `Date` and `Time` columns do not carry a timezone,
they carry whichever timezone the source that wrote each row happened to use.

**This is not only a merge-key problem.** The published `fixture_id` embeds a
date, and the results file derives its date from this column while the signals
file derives it from the provider's kickoff. A row stamped in UK time yields an
identifier one day off the one the signal used, and the join between a signal and
its outcome breaks silently, which is the whole reason results are published.

**Three changes.**

The fixture sync keys on the team pairing within a two-day tolerance instead of
on the date. A pairing does not recur within two days, which makes it unambiguous
where the date is not. Keying on the date reported 98 additions where the answer
was none; writing them would have duplicated a third of a season.

The sync writes `kickoff_utc` onto every row it identifies, so the ambiguity
stops propagating. 212 rows stamped on the first run.

The results publisher prefers that column and falls back to `Date` only where it
is absent.

**`Date` and `Time` are left alone.** Several seasons of research reference them,
and rewriting them would silently move rows other work is keyed to. They are
superseded, not corrected.

---

## D13 — Three layers use time, and only one of them is about a reader.

**Decided 2026-08-28.**

**Storage and computation: UTC, without exception.** Every stored timestamp ends
in `Z`, every comparison happens in UTC. The bug that prompted this was a match
history whose date column violated it, carrying whichever timezone the source
that wrote each row happened to use.

**Identity: the league's own local matchday.** Not display. A match belongs to
the day it was played in its own country, and naming a 19:00 kickoff in a UTC-6
league by the next UTC date names a day nobody played on. Frozen, because
published identifiers must keep matching the ones already issued.

**Display: one timezone, and it is the reader's.** Previously each league was
shown in its own zone, which put two timezones in one inbox and answered a
question nobody asked. A Liga MX kickoff shown as "19:00" sounds like an evening;
for a reader in London it is two in the morning. Only the second tells them
whether to be at a screen. `BETMODEL_DISPLAY_TZ` overrides, defaulting to
Europe/London.

The distinction matters because these three pull in different directions, and the
pre-merge pipelines conflated them: a display convenience, local kickoff time,
had leaked into the published contract, where two shapes then disagreed about
which field carried it.

---

## D14 — xG arrives twice, and the first figure is sometimes a zero.

**Decided 2026-08-29, after measuring.**

The provider publishes a figure within minutes of full time and revises it after
review. Measured across the two leagues on the fourteen most recent matches
each: **8 of 14 revised for one league, 3 of 14 for the other**, by as much as
0.41 on a mean around 1.5. A pipeline that fetches once and never looks again
trains on the provisional number permanently.

Worse, in the minutes after full time the provider serves xG as **0.0** before
computing it. Zero is a plausible-looking number that a fitter accepts as a real
observation. The frozen pre-merge history already contained three such rows, all
inside the training window, including a match that finished 3-2 recorded as
having created no chances at all. Under that league's blend that match entered
the fit at 0.9 instead of roughly 3.1.

**Three changes.** Every played match inside a seven-day review window is
refetched regardless of what is stored, and written only when the value actually
differs. An all-zero pair is treated as not-yet-published rather than as a
measurement. And an all-zero pair already stored is treated as missing, so it is
refetched and cleared if still unpublished.

Applied: one league had six placeholder rows, three of which resolved to real
values and three of which were cleared to blank. Its xG coverage fell from 910 to
907, which is the correct direction: three rows now say they have no xG instead
of claiming zero.

**The limit, stated.** A revision landing after seven days is missed. Revisions
were measured to settle within a few days, so the window has margin, but that is
an assumption rather than a guarantee. A full-history audit would close it and is
deferred with the evaluation layer.

---

## D15 — The staleness monitor runs whatever happened before it.

The xG merge never erases, so a feed that stops leaves the previous values in
place and every downstream stage rebuilds green on stale data. One fetcher sat
wedged for ten days with nothing reporting a problem.

The check compares the xG frontier against the results frontier, not against the
xG feed's own state: when a fetcher dies that side freezes whole, so any
self-consistency check on it alone sees nothing wrong. Two conditions are both
required, a gap in days and a number of stranded played matches, because a
provider does not cover every fixture and alerting on a single gap trains the
reader to ignore the channel.

It runs as the last step of the refresh workflow under `if: always()`. A monitor
that only runs when the thing it monitors succeeded is silent exactly when it is
needed: if the xG stage fails or never runs, that is the condition to alert on.

---

## D16 — What the acceptance test found.

**Run 2026-08-29.** The claim under test: adding a league costs one
`leagues/<id>.yml` plus one team-name mapping CSV, and nothing under `src/`.

The claim did not hold on the first attempt. Three things had to change, and each
was a real gap rather than a concession:

**A new league had no match table, and every stage assumed one existed.** The
fixtures stage is what creates it, but it read the file before merging into it.
So onboarding meant hand-building a CSV with exactly the right columns, including
odds columns derived from that league's own book list. The stage now bootstraps
one, deriving those columns from the same source the reducer uses so the two
cannot disagree.

**Two tests enumerated the leagues**, including one named "available leagues is
discovered not hardcoded" which asserted a two-element tuple. It broke the moment
a third league arrived, which is precisely the failure it was written to prevent.
Both now assert the mechanism: what is on disk is what comes back.

**The contract gate assumed every discovered league had a full data set.** A
league being onboarded legitimately has no model and no history, so the gate
failed for it and adding a league broke CI. It now skips a league with no fitted
model, states that it did, and separately requires such a league to be marked
unvalidated and to carry a caveat, so onboarding cannot put an untuned signal in
front of a reader.

**A fourth, found on the second pass.** A gate asserted every league had at
least one quote. A league with no captured odds legitimately has none, so it now
skips such a league but requires at least one to have quotes, or the assertion
would be vacuous. Same family as the other two: a test that assumed every league
looks like the leagues that already existed.

**What the test confirmed.** With those fixed, the league loads, is discovered by
the CLI and the manifest, fetches its own fixtures, and is refused by the model
with a legible reason until it has a training target. No league identifier
appears anywhere in `src/`.

**What the test does not prove.** The parameters in the new league's YAML are
borrowed from the closest existing league, not fitted. Adding a league is cheap;
making its signals mean anything is not, and the config says so with
`validated: false` and a caveat.

## D17 — Existing in the repository and being in production are two different things.

The acceptance test added a league, and publishing it enlisted it in production.
That was not an accident of the test: the capture timer discovers which leagues
to dispatch captures for by reading `public/index.json`, so appearing in the
manifest *is* the enlistment. A league added to prove the abstraction was
therefore dispatched for real captures against credentials nobody had created
for it, and every opening tick for it failed:

```
RuntimeError: no odds-api.io key: set ODDS_API_IO_KEY
```

Four of the last twenty capture runs were this and nothing else.

**No quota was burned.** The providers run in the order `("oddsapiio",
"theoddsapi")`, and the first raised, so the stage aborted before reaching the
account that mattered — the one with 82 of 500 requests left until it resets.
The cost was a red workflow, not lost data. It could easily have been the other
way round with the order reversed.

**The decision.** `publish.published` gates whether a league appears in the
manifest, whether its canonical files are written, and whether its compatibility
payload is written. It defaults to true, so no existing league changes meaning,
and it is deliberately separate from `validated`: that one warns a reader about
the numbers, this one decides whether there is a reader at all.

**Why not simply give the new league a credential.** Because the general problem
is not this league. Any league added from now on is dispatched for captures the
moment it is published, and the natural order of work — add the league, look at
the fixtures, tune the parameters, then arrange the accounts — has publication
happening before credentials exist. The flag makes that order legal.

**What did not change.** The producer still names no league anywhere. The
withholding is a field on the league, read by the manifest builder, so G4 still
proves the list is discovered rather than written.

## D18 — One format for a stored timestamp, and one column deliberately exempt.

Every stamp was written through a bare `isoformat()`, which keeps whatever
precision its source carried. A provider's value arrives to the second; this
pipeline's own clock arrives to the microsecond. Both were written to the same
column, so `fetched_at` held two formats at once and a strict parse of it
raised — which is how it was found, while reading the column for something else.

**It is a regression, not an inheritance.** Both pre-merge pipelines published
seconds; the frozen baselines show `2026-08-25T12:25:16Z`. The merge introduced
the microseconds and shipped them downstream: the compatibility payload the board
reads carried `2026-08-29T11:14:59.055821Z`. Nothing broke, because the board
parses with `new Date()`, which is forgiving. G3 never caught it because it
excludes generation-time fields, correctly — they differ on every run by design.

`dates.stamp()` is now the one way a timestamp is written to storage, matching
`publish.contract.utc` on the publishing side. Eighteen already-written rows were
normalised.

**`last_update` is exempt, and the reason is not cosmetic.** It is the book's own
clock and part of the dedup key. Normalising it would re-identify the row: two
genuinely distinct ticks a fraction of a second apart would collapse into one,
and every affected row — 56 in one league, 35 in the other — would look new to
the upstream sync and be appended a second time. It is stored exactly as the
provider sent it, whatever precision that is. The first attempt at this change
did truncate it, along with rewriting 680 lines of historical `capture_reason`
prose, before the diff was read.

**`capture_reason` is exempt too, for a different reason.** It is free text
describing what a run did, some of it written by pipelines that no longer exist.
Its wording is a record, not a field to reformat.

## D19 — An unreachable provider is not an empty result.

The first refresh run in CI reported success having fetched nothing. Every
SofaScore call had failed at the TCP level — the residential proxy was refusing
connections — and each stage read the empty answer as "the provider has nothing
new for you", logged a warning, and returned cleanly:

```
WARNING no seasons for tournament 11621
WARNING ligamx: the fixture provider returned nothing; leaving files alone
INFO    ligamx: xg {'missing': 0, 'in_review': 15, 'fetched': 0, 'written': 0}
```

Three leagues, nothing fetched, green tick. The `allow_empty_upcoming` guard —
which exists precisely to refuse an empty fixture list — never fired, because the
stage had already returned two hundred lines earlier on the "provider returned
nothing" path.

**The distinction is made where it is known.** `_get(strict=True)` already
existed and already said, in its own docstring, that it exists "so a caller
cannot mistake it for an empty result". It was used on one deep call and not on
`seasons()`, which is the first call every SofaScore-backed stage makes and the
one that gates all the others. It is strict now.

**A 404 stays an answer.** A tournament id that does not exist is a real reply,
and escalating it would make a typo in a league YAML look like a proxy outage.

**Why this matters more than the outage that exposed it.** A refresh that fails
is visible within minutes. A refresh that fetches nothing and says it succeeded
is discovered days later, by someone reading a number that stopped moving. The
leagues stay independent — one league's outage still does not stop another's —
but the run's exit code carries the failure, which it already did.

## D19b — The same swallow, one layer up.

Fixing the provider layer (D19) made the stage exit 1, and the run still went
green:

```
ERROR   ligamx fixtures failed: SofaScore GET /unique-tournament/11621/seasons failed
##[error]Process completed with exit code 1
...
6. Fixtures: success
```

Both fetch steps carried a bare `continue-on-error: true`. The intent is sound
and worth keeping: a provider outage should still leave the model refitted and
published from the data already on disk, because yesterday's numbers beat no
numbers. What was missing is that nothing afterwards remembered it had happened.

The steps now carry ids, and a final always-run step fails the job if either
fetch failed. Everything still runs; the run just stops claiming it worked.

**The general shape.** Two independent layers each turned a failure into a
success, and each looked locally reasonable — one read an empty list as an empty
answer, the other let a step fail so later steps could run. It took an actual
outage to reveal either. Any place that degrades rather than stops needs
somewhere downstream that still knows it degraded.

## D20 — The staleness monitor cannot double as the failure channel.

With D19 and D19b in place a broken refresh finally goes red. The question that
followed — would anyone be told — has a worse answer than expected.

Three channels existed, and none of them reports a failed run promptly:

| channel | what it actually reports |
|---|---|
| GitHub email on a failed scheduled run | the run, but only if a per-account notification setting is on |
| Telegram, from the xG staleness check | data going stale, at a 3-day threshold |
| Telegram, from the capture timer Worker | whether a `repository_dispatch` was accepted, not what it ran |

The staleness monitor is the one that looks like coverage and is not. It is
deliberately `if: always()`, precisely so it still runs when the fetch above it
failed — good design for what it measures. But it measures the age of the data,
not the health of the run, and at three days a refresh broken on Monday is
silent until Thursday. Two different events, two different latencies; using one
as the alarm for the other buys three days of quiet.

The failing step now sends its own alert with the run URL. Two details are
deliberate:

**It uses `curl`, not the package.** The alert has to arrive when the failure is
the package refusing to import, which is exactly when the CLI cannot be asked to
report anything. Routing it through `notify.telegram` would make the alerting
depend on the thing most likely to be broken.

**The send is best-effort.** A failed alert logs and is swallowed, so it can
never replace the failure it exists to report.

It also catches `job.status == 'failure'`, so a step failing anywhere in the
job — not only the two fetches — reports through the same place.

**What is still not covered.** The capture ticks have no such alert. They run
every five minutes, so alerting per failure would be noise, and the right shape
there is a digest or a dead-man's switch rather than a per-run message. Left
undone deliberately, not overlooked: a missed capture is unrecoverable, so it
deserves a better answer than the one that fits here.

## D21 — A stored event id follows its own provider's convention, not a tidy rule.

`event_id` is part of `DEDUP_KEY`, so the form of the id *is* the row's identity.
Both pre-merge pipelines wrote odds-api.io ids as `oddsapiio:72055150` and The
Odds API ids bare. The merged capture wrote `str(event["id"])` for everything, so
the first three odds-api.io opens it captured went in bare — the same captures
upstream had, under a different identity.

**What it did not break.** The opener gate keys on `(home, away, book)`, not on
the id, so no fixture was ever re-opened and no later price was recorded as an
opening line. That is the failure this would have caused if the gate had been
built the obvious way.

**What it would have broken.** The reconciliation due before the old
repositories are archived: all three rows would have been imported a second time,
as duplicate captures of the same quote. Repairing them dropped the Liga MX
delta from 40 rows to 37.

**Why the asymmetry stays.** Namespacing everything is the tidier rule and it is
the wrong one here. 677 stored rows carry bare The Odds API ids; restyling them
would re-identify every one, exactly as truncating `last_update` would have
(D18). The rule is "write what this provider's history already contains", and a
test asserts every row on disk satisfies it.

**Third time this shape has appeared.** `last_update`, `fetched_at`, and now
`event_id`: a column whose format looks cosmetic but is load-bearing because
something downstream keys on it. The distinction that matters each time is
whether the field is part of an identity or merely describes one.

## D22 — Two trees, two failure modes, one commit.

Committing was the last place that treated everything alike. `git pull --rebase`
was applied to `data/` and `public/` together, described in a comment as
rebasing "this append-only change" — true of the capture histories, false of the
published JSON, which is regenerated in full every run. Two whole generated
documents have no meaningful diff to carry, so any concurrent publish conflicted,
and the retry loop replayed the same conflict three times and gave up. It cost
two failed runs in one afternoon.

**Append-only merges; regenerated is resolved.** `.gitattributes` gives the two
capture histories Git's built-in `union` driver, so two writers appending
different rows now merge losslessly instead of conflicting. Everything else is a
function of that data, so a `derived: true` commit resolves what is left in
favour of this run and continues, rather than failing a push and losing the
data rows along with it.

**Rehearsed, not reasoned about.** Three things were wrong on the first attempt
and a scratch-repo simulation of a real race found all three:

1. `--ours` during a rebase is the branch being landed on, and `--theirs` is the
   commit being replayed. The intuitive reading is backwards, so the resolution
   would have kept the wrong side.
2. Splitting into two commits — clean-looking, since it classifies each tree
   separately — leaves the other tree unstaged, and `git pull --rebase` refuses
   to run on a dirty tree.
3. `--autostash`, the obvious fix for (2), is worse than the problem: the stash
   reapplies over the rebased tree, conflicts, and leaves `<<<<<<<` markers
   inside the published JSON, which then gets committed and served to the board.

So it stays one commit, which keeps the tree clean, and the classification lives
in `.gitattributes` and the resolver rather than in the commit boundaries.

**A guard where the reasoning could rot.** If a conflict ever reaches the
resolver in one of the two append-only histories, the union driver did not apply
— a shallow checkout, a stale action ref — and picking a side would discard
opening lines that cannot be recaptured. That case fails loudly instead.

## D23 — A dead-man's switch for the captures, not a failure alert.

D20 left the capture ticks deliberately uncovered, and this is the shape that
fits them.

Alerting per failed run is noise at a five-minute cadence. Alerting on the data
is worse than noise, it is wrong: an idle tick with no fixture in its window
correctly appends nothing, so a healthy quiet stretch and a total outage look
identical from the rows. The tracker already carries this trap in a comment about
its own staleness tolerance.

What is never normal is the workflow not running at all, or running and never
once succeeding in 24 hours. Both mean down rather than idle. `scripts/
check_capture_health.py` checks exactly those two and is run once a day from the
refresh, which already holds the Telegram credentials.

**Cancelled runs count as neither.** Concurrency cancels an overlapping tick by
design. Counting those as failures would make a busy matchday — precisely when
captures matter most — look like an outage.

**Deliberately forgiving.** One success in the window is enough to stay quiet. A
threshold tight enough to catch partial degradation would fire on the ordinary
transient failures this pipeline sees daily, and an alarm that cries wolf is
worse than the gap it filled. Measured against the live repo when written: 88
succeeded, 5 failed.

**It reports without failing the refresh.** Failing its host would conflate a
capture outage with a refresh outage, and the refresh has its own alert for its
own failures.
