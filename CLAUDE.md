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
| **Pinnacle opening price** | calibrates the model. The model is worst at the draw and the market is best at it, so `debias.apply` replaces the model's draw with the anchor's no-vig draw (logarithmic-function devig, D32) | Pinnacle publishes first — measured median 6.1 days before kickoff, max 7.0 |
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

**The anchor's proof is checked, and it is the strictest gate in the tool.**
`reduce.py` derives a `proof` for every opening price via
`capture_watch.opener_proof`, from what `capture_watch.csv` recorded about seeing
a (fixture, book) *unpriced* beforehand. `build_signals` then refuses any anchor
that is not `observed` — the strong proof — for anything captured after
`ANCHOR_PROOF_CUTOVER`, and an anchor refused means the fixture publishes
`unanchored` and cannot fire.

Expect that gate to be where a missing signal turns out to live. On 2026-09-06 a
Necaxa v Puebla edge of +15.5% went unpublished because fourteen rows of
`capture_watch.csv` were written with microsecond timestamps and a reader that
inferred one format from the first row dropped every one of them. The proof was
withheld, the real opener was treated as absent, and everything downstream
behaved exactly as designed. When a signal is missing, read the proof first.

**The bet price is gated far more weakly**, and knowing the difference matters.
`require_price_proof` accepts any non-empty proof, so `window` passes — and
`window` says only that the fixture entered the lookahead after capture began,
which for a pipeline running since July is very nearly a tautology. The polled
soft books have earned `observed` nought times in 117 opening prices.

**A tick that could not reach a provider is not an idle tick, and they keep
looking alike.** Both return no rows, and the workflow step exits green either
way. Three cases have now had to be told apart by hand, so assume a fourth. The
CSL Odds API account sat under its floor and stopped anchoring the league —
twice, once for three days; `QuotaRefused` made that one legible. Then
odds-api.io began returning a listing with none of our fixtures in it and
`_capture_oddsapiio` returned at `if not matched` saying nothing, six days of it.
Then a 429 escaped as an unhandled error after fifteen minutes of blind backoff,
which on a concurrency group shared with the closes cancels the ticks queued
behind it. Each now says what it is. **Keep it that way: a path that spends a
request and records nothing must log why, and nothing may outlive its
five-minute tick.**

**The two providers have opposite economics.** odds-api.io bills 500 per *day* per
league; each Odds API account bills 500 per *month*. Polling them at one rate
either starves the cheap one or drains the expensive one. See D25 in
`docs/DECISIONS.md`.

**The open lookahead is per book now.** The league-wide value (21 days for CSL,
14 for Liga MX) is the default, and Pinnacle overrides it to 8 in both — it does
not price a fixture until about 7 days out, so a longer window spent the monthly
allowance asking questions whose answer was known. A book's own
`lookahead_days` beats the league's.

**A fixture is a pairing on a matchday, never a pairing.** Two clubs meet twice a
season and again whenever a postponed match is replayed, so every key that
identifies a fixture carries `dates.local_matchday`. It is a *day* and not a
kickoff on purpose: providers nudge kickoff times by minutes, and a key that
treats drift as a new fixture sends the capture to write today's mid-market line
into an open slot. Keyed on the pair alone, four fixtures were being served an
earlier meeting's opening price, one of them from a match six weeks past.

**The golden baseline is G1's, and only G1's.** `tests/golden/<league>/inputs`
and `model` are the frozen pre-merge match history and coefficients, and G1 asks
whether the merged fitter reproduces them to floating point. It passes with no
exemptions and the evidence cannot be recreated -- the source repositories are
archived -- so do not regenerate those files. G3, which asked the same question
of the published *decisions*, was retired on 2026-08-31: the changes since the
merge have been deliberate ones it could only absorb as exemptions, both its
firing rows were bets the engine now correctly refuses, and the shape it compared
against had no consumer left. What replaced it is the unit tests that state the
rules directly -- a signal must be calibrated, a club the model has barely seen is
not bettable -- which say why a signal fires rather than that one August fixture
came out a particular way.

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
