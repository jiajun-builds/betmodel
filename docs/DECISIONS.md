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
