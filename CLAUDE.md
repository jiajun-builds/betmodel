# CLAUDE.md

Read this before changing anything. It says what the tool is *for*, and most bad
changes here come from optimising a part without knowing what the whole is doing.

## What this tool does

Two layers, and they meet in one place.

**The model layer.** On a schedule it fetches xG and match results, fits the
league's model, and produces team strengths and 1X2 outcome probabilities.

**The odds monitoring layer.** It watches Pinnacle and the betting platforms.
Pinnacle's **opening** price calibrates the model's output. The calibrated
probabilities are then compared against the betting platforms' prices, and a
comparison that clears the league's EV threshold is a signal: it goes to Telegram
and onto the board.

When new data changes the model and a signal is invalidated or no longer clears
the bar, that is also worth a message — saying what changed and why the signal is
withdrawn.

## The three captures, and why each exists

A fixture's prices are captured three times, for three unrelated reasons. Confusing
them is the most expensive mistake available in this repo.

| Capture | Why | Timing |
|---|---|---|
| **Pinnacle opening price** | calibrates the model. The model is worst at the draw and the market is best at it, so `debias.apply` replaces the model's draw with the anchor's no-vig draw | Pinnacle publishes first — measured median 6.1 days before kickoff, max 7.0 |
| **Betting platform opening prices** (1xBet, Duel) | the price you actually bet. Soft books open slowly and move slowly, and the edge lives in the gap between the calibrated model and that slow opener | measured median ~6 days before kickoff |
| **Pinnacle closing price** | after the fact, for CLV. The closing line is the market's final word and the yardstick for whether the strategy has edge | a 15-minute window before kickoff |

**Nothing between open and close is captured.** It is neither bet into nor
calibrated against.

## The principles

**A signal must be calibrated with Pinnacle's TRUE opening price, or it is not
published.** Not to Telegram, not to the board. This is not a preference, it is
what keeps the tool inside the evidence that validates it: the backtest that
establishes long-run +CLV on the soft books was run on true opening prices. A
signal calibrated on a later Pinnacle price is not a weaker version of the
strategy, it is an untested variant, and nothing says the two are equivalent.

**An uncalibrated signal is a fake signal.** It is noise. It does not get a
warning message, a hint, or a tentative alert — pushing it only adds confusion to
a decision. Silence is correct.

**A better opening price is worth a second message.** When 1xBet opens above Duel
on the side already signalled, or the reverse, that is real new information and
deserves a push. It is still subject to the rule above.

**The board and Telegram must agree.** They read the same `state` field from the
same published payload, and the reason they agree is that the engine decides
once. Do not add a filter to one side.

## Why an EV threshold rather than EV > 0

EV above zero is not an edge. The model has error and the vig is a wall, so a
threshold (CSL 0.20, Liga MX 0.10) selects edges large enough to survive both.
The number is a backtest result, not a taste, which is also why calibrating
differently invalidates it.

## Things that will bite you

**The capture history is the irreplaceable asset, not the model.** An opening
line exists only while the bookmaker shows it and no provider sells opener
history at any tier, so `data/<league>/odds_capture_history.csv` is append-only
and committed. A model can be refitted; a missed price cannot be bought back.

**The anchor's provenance is computed but not yet checked.** `reduce.py` derives a
`proof` for every opening price via `capture_watch.opener_proof`, and
`capture_watch.csv` records when a (fixture, book) was seen *unpriced* — which is
what proves a later price is genuinely the book's first. But `build_signals` reads
the anchor with a plain `opens.get(...)` and never looks at that proof, and
`require_price_proof` covers only the bet price. So the tool can tell whether an
anchor is a real opener and does not ask.

**A quota refusal is not an idle tick, and used to look like one.** Both return no
rows and no requests, and the workflow step exits green either way. The CSL Odds
API account sat under its floor and stopped anchoring the league — twice, once for
three days. `QuotaRefused` now makes the two distinguishable; keep it that way.

**The two providers have opposite economics.** odds-api.io bills 500 per *day* per
league; each Odds API account bills 500 per *month*. Polling them at one rate
either starves the cheap one or drains the expensive one. See D25 in
`docs/DECISIONS.md`.

**The open lookahead is shared by every book, and it should not be.** It is 21
days for CSL and 14 for Liga MX, while Pinnacle does not price a fixture until
about 7 days out. Every anchor slot therefore spends a request asking about
fixtures that provably cannot be priced yet, which is where the monthly allowance
goes.

**Alert dedup keys on `(fixture_id, side, book)`** and the baseline is the
previously committed signals file, read with `git show HEAD:`. A run whose commit
never lands re-alerts next time.

## Layout

```
leagues/<id>.yml        every league-specific parameter; the only file a new league needs
src/betmodel/           the engine
  config/ providers/ fixtures/ xg/ models/ odds/ signals/ publish/ notify/
data/<league>/          committed CSV/JSON; git history IS the database
public/                 the published contract (docs/CONTRACT.md); the board reads this
docs/DECISIONS.md       why things are the way they are; read before changing a number
```

## Commands

```bash
betmodel <league> <stage>        # fixtures xg model reduce signals publish notify
betmodel <league> capture-opens  # opening lines
betmodel <league> capture-anchor # the anchor, when an edge is waiting on one
betmodel <league> capture-closes # closing lines
betmodel index                   # rebuild public/index.json
python -m pytest -q              # the whole suite; conda env `betmodel`
```
