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
