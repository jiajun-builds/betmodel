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
