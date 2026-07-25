# Opening-Line Asian-Handicap Backtest

An unbiased, full-slate test of the project's CLV thesis — *can the model beat
Pinnacle's **opening** Asian-handicap line?* Everything here settles bets against
real results at the real opening prices, so the output is realized ROI at open
(the ground truth that CLV is only a proxy for), not a simulated edge.

Files in this folder:

| File | What it is |
| --- | --- |
| `backtest_open_ah.py` | The walk-forward backtest (self-contained, no API spend). |
| `backtest_open_ah_bets.csv` | Per-bet detail (one row per gradeable match). |
| `backtest_open_ah_summary.csv` | Per-threshold ROI / SE / t / profit / win-rate. |
| `calibration_diagnostic.py` | Reliability diagrams + implied temperature (§9.1; analysis only). |
| `backtest_open_ah_calibrated.py` | Walk-forward temperature scaling vs uncalibrated (§9.2). |
| `backtest_open_ah_calibrated_bets.csv` | Per-bet detail from the calibrated run. |
| `../model comparison/distribution_comparison.py` | Six goal distributions head-to-head (§9.4). |
| `backtest_1x2.py` | Walk-forward **1X2** opening-line backtest + CLV, ZIP vs NegBinom (§11), now with the market-anchored draw de-bias λ grid (§12). |
| `backtest_1x2_zip_bets.csv` / `backtest_1x2_negbinom_bets.csv` | Per-bet detail from §11 (λ=0). |
| `backtest_1x2_negbinom_lam{25,50,75,100}_bets.csv` / `backtest_1x2_zip_lam100_bets.csv` | Per-bet detail from the §12 de-bias variants. |
| `backtest_1x2_negbinom_delta_bets.csv` | Per-bet detail from the §12.4 market-free δ calibration (the deployed mechanism). |
| `backtest_1xbet.py` | Bets 1xBet's **opening** 1X2, grades CLV vs Pinnacle's **close** — the roadmap #8 cheaper-book test (§13.1). |
| `anchor_test.py` | Is Pinnacle or 1xBet the better de-bias anchor? (§13.2). |
| `plot_bankroll.py` | 0.25-Kelly bankroll chart for the recommended config (§13.4). |
| `bankroll_recommended.png` | Output of the above — bankroll $100 → $445, 3 seasons. |
| `staking_compare.py` / `staking_compare.png` | Kelly vs flat vs compound-fixed-fraction staking (§13.5). |
| `backtest.md` | This document. |

> **§9–§12 are the current handoff.** §1–§8 are the original opening-line AH
> backtest. §9–§10 (2026-07-13) explain why the AH/model route is dead and where the
> edge could still be. **§11 (2026-07-15) is the 1X2 result**: the strategy as
> specified is dead — 61% of its stake sits on a draw-probability bug — but unlike AH
> the direction survives: drop the draw and a baseline-adjusted +CLV holds in all
> three seasons. It is still not profitable, for a reason worth internalising.
> **§12 (2026-07-15) is the draw de-bias test (roadmap #9)**: the fix works as a
> *model* fix — excess CLV roughly doubles — but the strategy still fails the vig
> bar in 2024 and 2025 at every shrink strength. Betting Pinnacle's open stays
> closed; the surviving signal (~+1.2–2.5pp excess CLV) would clear the bar at a
> ≤5%-overround book, which is roadmap #8's case, quantified.
> **§13 (2026-07-16, extended 2026-07-17) is that case's data**: betting 1xBet's
> cheaper opening line (4.87% overround) the gap turns positive at thr>0.20 — the vig
> wall is finally cleared — and with 2024 **and 2025** now backfilled (n=611,
> 2024+2025+2026) it clears in **all three seasons independently**, discharging the
> §11.1 single-season caveat for good. The catch that makes the point: **2025 realized
> −24% ROI on a set whose excess CLV was +4.2pp (t=2.7) and gap +2.5** — the closing
> line says those bets had value, the season's dice said otherwise. Read excess CLV /
> gap, not ROI. **§13.3 is a scoreboard of every strategy ever tested here, defined.**
>
> **If you read only two things, read §11.3 and §11.7.** §11.3: never quote a CLV
> number without the model-free baseline (an "always bet home" coin beats this model
> on raw CLV). §11.7: **EV > 0 ⟺ CLV > p × R** — Pinnacle's 7.55% opening vig means
> you need **+2.61pp** CLV just to break even, which is why +CLV strategies here still
> lose money. Then read §12 (the de-bias verdict) and §10 (roadmap #8).

## 1. Data

- **Source:** `data/raw_data/CHN_Super League.csv`, columns
  `pinnacle_open_ah` (home handicap; negative = home favoured),
  `pinnacle_open_ah_h`, `pinnacle_open_ah_a` (decimal prices).
- These opening lines are **maintained manually** and are *not* part of the
  automation-refreshed production schema. A fresh checkout will have those
  columns empty/absent until they are filled in, and the backtest will have
  nothing to grade until then.
- **Current run: all four seasons 2023–2026**, 856 gradeable matches (open line +
  result), **826 bettable** after the walk-forward warm-up (30 skipped for short
  history or an unseen promoted team — almost all in early 2023, which has no
  prior data to train on).
- The xG feeding the model was re-scraped from the official SofaScore API and
  corrected for 2023–2025 (see the historical-xG-backfill work), so the model
  inputs here are the same the production model would have used.
- The opening prices embed an average overround of **~5.3%** — the hurdle any
  edge has to clear.

## 2. Strategy under test

The production betting recipe, applied honestly out-of-sample:

1. **Refit the production model per round**, walk-forward, on data strictly
   *before* that round — `ZeroInflatedPoissonGoalsModel` on xG targets
   (`HExpG+`/`AExpG+`), 18-month trailing window, `xi=0.001`, Dixon-Coles
   time-decay weights. (Mirrors `src/csl/models/dc.py`.)
2. For each match, turn the model's 15×15 scoreline grid into an **exact
   expected return** for backing the home side and for backing the away side at
   that match's opening line and prices.
3. **Back the higher-EV side** if its model EV per unit stake clears a
   threshold. Thresholds tested: `0.00, 0.02, 0.05, 0.10`.
4. Flat 1-unit stakes; P/L accumulated across all seasons.

## 3. Backtest logic (why it is trustworthy)

- **No leakage.** The training window is strictly `Date < match_date` and
  trimmed to the trailing 18 months; a team unseen in that window (e.g. a
  promoted side) is skipped rather than guessed.
- **Exact handicap settlement from the raw grid**, not
  `FootballProbabilityGrid.asian_handicap()`. penaltyblog's helper mishandles
  quarter-line push mass (returns identical values for `0.0 / -0.25 / -0.5`), so
  we collapse the 15×15 grid to `P(goal difference)` and settle every line
  ourselves: **quarter lines** (`4·line` odd, e.g. `-0.25`, `-0.75`) split the
  stake half/half across the two neighbouring half-lines; **integer lines** push
  and refund the stake; otherwise win pays `odds−1`, loss pays `−1`.
- **Away side** is the home computation with the goal difference and the line
  both flipped, priced at the away odds.
- **Dates** are parsed with `csl.date_utils.parse_date_only_series` (handles both
  ISO and legacy `DD/MM/YYYY`); a naive `to_datetime()` would drop every day>12
  row and month/day-swap the rest, corrupting the walk-forward ordering.

## 4. Metrics reported

- **Baselines** — always-home and always-away ROI at the opening line, so the
  strategy is judged against naïve flat betting, not against zero.
- **Threshold table** — for each EV threshold: `n_bets`, ROI (mean per-bet P/L),
  `SE = sd/√n`, `t = ROI/SE`, total profit, win-rate, and mean predicted EV.
  Per-bet P/L has sd ≈ 1, so SE ≈ 0.035 at n ≈ 700; **|t| > ~2 is the bar for a
  real edge.**
- **Model-EV honesty check** — mean *predicted* EV vs *realized* ROI on the same
  bets, plus a paired t-test of `(predicted EV − realized P/L)`. A large positive
  gap means the model's handicap-cover probabilities are overconfident, not
  merely unlucky; the t-test says whether that gap is real or sampling noise.
- **Cross-season replication** — the same `+EV` result broken out per season.
  Each season is an independent sample; a finding that repeats across all four is
  a model property, not one anomalous year.
- **EV calibration buckets** — `+EV` bets split into quartiles by predicted EV;
  if the model is useful for staking, higher predicted EV should map to higher
  realized ROI (monotonic).
- **Per-round P/L** — to eyeball variance.

## 5. Results (current run — 826 bets, 2023–2026)

**Baselines:** always-home ROI **−3.04%**, always-away ROI **−6.87%** (n=826).

| EV threshold | n_bets | ROI | SE | t | win% | avg predicted EV |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ≥ 0.00 | 674 | **−4.19%** | 0.033 | −1.27 | 47.0% | +16.0% |
| ≥ 0.02 | 602 | −4.51% | 0.035 | −1.29 | 46.7% | +17.8% |
| ≥ 0.05 | 519 | −4.95% | 0.038 | −1.32 | 46.1% | +20.1% |
| ≥ 0.10 | 392 | **−8.27%** | 0.043 | −1.93 | 44.1% | +24.2% |

**How to read it:**

- **No threshold beats zero — and the tighter you filter, the worse it gets.**
  Raising the EV bar from 0 to 10% takes ROI from −4.2% to −8.3% (t = −1.93):
  the bets the model is *most* confident in are its *worst*. That is the opposite
  of a usable signal.
- **Model-EV honesty check:** on the 674 `+EV` bets the model predicted a mean EV
  of **+16.0%/unit** but realized **−4.2%/unit** — it **overstates its own edge
  by +20.2%/unit**. Paired t-test of `(predicted − realized)`: **t = +6.01**.
  This is not variance; the model's cover probabilities are systematically
  overconfident.

### Cross-season replication (the reason the history was backfilled)

| Season | +EV bets | predicted EV | realized ROI | t(ROI) | overstatement |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2023 | 174 | +21.5% | −5.9% | −0.94 | **+27.5%** |
| 2024 | 192 | +14.2% | −2.0% | −0.32 | +16.2% |
| 2025 | 192 | +13.5% | −4.7% | −0.77 | +18.2% |
| 2026 | 116 | +14.8% | −4.3% | −0.53 | +19.1% |

**Every season, independently, overstates EV by 16–28%/unit and realizes a
negative ROI.** The overconfidence is not a 2026 artifact — it is a stable
property of the model, confirmed across four separate seasons (pooled t = 6.0).

### EV calibration (quartiles of predicted EV)

| predicted-EV bucket | n | mean predicted EV | realized ROI |
| --- | ---: | ---: | ---: |
| lowest `(−0.001, 0.054]` | 169 | +2.6% | −2.0% |
| `(0.054, 0.124]` | 168 | +8.8% | +2.0% |
| `(0.124, 0.215]` | 168 | +16.4% | −4.9% |
| highest `(0.215, 0.952]` | 169 | +36.1% | −11.9% |

Predicted EV does **not** sort realized ROI. If anything it sorts it *backwards*:
the top quartile (model says +36%) realizes **−11.9%**. A staking rule keyed on
model EV would concentrate stake exactly where the model is most wrong.

## 6. Interpretation

- At opening lines, the model as-is has **no tradeable edge**, and its EV is
  badly miscalibrated in a way that is now statistically nailed down (t = 6 for
  the overstatement, replicated across four seasons). Betting on raw model EV
  loses to the overround, and loses *more* the more selective you are.
- This measures **realized ROI at open, not CLV**. But ROI at open is exactly
  what +CLV is supposed to predict, so a model with no ROI-at-open signal has no
  demonstrated CLV edge either. (Closing lines are now also in the CSV
  (`pinnacle_close_ah*`), so a direct open→close CLV study is now possible — see
  roadmap #3.)
- The companion grid search (`model comparison/xi_lookback_grid_test.py`)
  independently shows production `xi=0.001` / 18-month is within sampling noise
  of the grid optimum — so **the lever is not `xi`/lookback tuning, it is
  probability calibration.**

## 7. Recommendations

1. **Do not stake on raw model EV at opening lines.** There is no measured edge,
   the EV is inflated ~20%/unit (t = 6), and the highest-EV bets are the worst.
2. **Calibrate before betting (roadmap #4).** Build per-segment reliability
   diagrams (by handicap line, favourite vs underdog) and shrink the model's
   probabilities (e.g. temperature scaling) until predicted EV ≈ realized ROI.
   Only then consider betting, and only in segments that are actually
   well-calibrated.
3. **Stop tuning `xi`/lookback for this** — the grid already rules that out as
   the lever. Spend the effort on calibration/shrinkage instead.
4. **Use the captured closing lines (roadmap #3)** to measure true open→close
   CLV now that `pinnacle_close_ah*` is populated, and check whether *any*
   segment shows positive CLV even where raw ROI is flat.

## 8. Reproducing

```bash
# From the repo root, in the csl-workflows env (or any env with penaltyblog).
# Requires the pinnacle_open_ah* columns to be present in the main CSV.
PYTHONPATH=src python backtest/backtest_open_ah.py
```

Outputs `backtest_open_ah_bets.csv` and `backtest_open_ah_summary.csv` next to
the script.

## 9. Follow-up: can the model be fixed? (investigation 2026-07-13)

§1–§8 established that the model overstates its opening-line EV by ~20%/unit.
This section records the investigation into *why*, and whether it is fixable.
**Short answer: the overstatement is winner's curse (selection bias), not a
fixable model defect — and no calibration or distribution change removes it.**

### 9.1 Calibration diagnostic (`calibration_diagnostic.py`)

Walk-forward, records probabilities instead of bets.

- **1X2 is well-calibrated (ECE 0.032). Handicap-cover is not (ECE 0.086)**, and
  cover overconfidence grows with line size: `|line|<0.5` +0.066, `0.5–1.0`
  +0.074, `1.25–2.0` +0.124, `>2` **+0.200**. On the backed side, model
  P(cover) 0.592 vs realized 0.506.
- Mechanism: the cover depends on the **goal-difference (margin) distribution**,
  which Poisson/ZIP under-disperses (worst for big favourites); the win/draw/loss
  split it gets roughly right.
- Implied temperature (in-sample 1X2 log-loss minimiser): **T\* ≈ 1.5**.

### 9.2 Temperature scaling (`backtest_open_ah_calibrated.py`)

Walk-forward T fit strictly on prior fixtures, applied per round, EVs recomputed.

- Fit on **1X2 log-loss** → T median 1.54. Overstatement +19.3% → **+18.4%**
  (barely moved, still t≈5); realized ROI unchanged (~−5% to −7%); EV buckets
  still don't sort.
- Fit on **cover Brier** → T rails to the upper bound (→∞): the optimiser sets
  every cover prob to 0.5, i.e. "the cover signal is noise, switch it off."
- **Temperature scaling does not fix it.** Shrinking a model toward its own
  centre can't help when that centre disagrees with the market only via noise.
  Only shrinking toward the *market* removes the overstatement — and that leaves
  ~0 EV (no bets).

### 9.3 Why it's winner's curse, not a shape defect

- **Symmetric home-cover calibration is ~0** (unbiased across all fixtures). The
  +0.086 overconfidence appears **only after conditioning on "the side the model
  likes most"**. That conditioning *is* the bias.
- **Simulation proof:** a deliberately unbiased model (`E[p̂−p]=0`) + an efficient
  market + "bet the +EV side" reproduces a ~+14% EV overstatement **from pure
  noise**; the selected side's mean estimation error is +0.072 even though the
  global mean error is 0. Classic oil-field winner's curse: you only bet when
  your noise is favourable, so you systematically overpay.
- **Implication:** the overstatement can only be removed by a model genuinely
  *more accurate than the market* — reshaping one model's distribution cannot.

### 9.4 Distribution comparison (`../model comparison/distribution_comparison.py`)

605 fixtures, 2024–26, six penaltyblog distributions on the same walk-forward
harness.

- **1X2 accuracy:** NegBinom best (RPS 0.19705, log-loss 0.97550 — ~1.5% better
  than ZIP 0.19764 / 0.99013). Over-dispersion genuinely helps *prediction*.
  Weibull worst; Bivariate no help; **ZIP == Poisson exactly** (ZIP still
  collapsed).
- **Betting:** all six overstate EV **+17–23%**; NegBinom bets *slightly worse*
  (realized −5.0% vs ZIP −3.6%). Symmetric cover bias ~0 for all — confirming
  §9.3.
- **Takeaway:** a `ZIP→NegBinom` swap in `dc.py` is justified for **prediction
  accuracy** (better dashboard probabilities) but **not** for betting edge.

### 9.5 Line-magnitude filter (does small-line-only rescue it?)

No. Big lines (`>2`) are catastrophic (−29% ROI); small lines (`≤0.5`) are less
bad (−3.6%) but **still overstate EV +17.2% (t=3.66)** and lose at every
threshold (1 of 4 seasons positive = noise). Only practical rule: **avoid
big-favourite lines** — damage control, not edge.

### 9.6 CLV analysis (open→close, 2023–24 only)

Close lines exist for **2023–24 only** (475 matches; 2025 `pinnacle_close_ah` is
empty). Computed inline (no committed script).

- Lines move 66% of the time; overround compresses **open 6.1% → close 4.0%**
  (**~1pp/side vig headwind** for betting at open).
- No naive rule gets significant +CLV (always-home +0.39pp, t=1.1). Model picks
  get **+0.69pp (t=1.9)** but it *weakens* as the EV threshold rises (noise
  signature) and is net-negative after the vig headwind.
- Market moves toward the eventual winner **59.6%** of the time (mildly
  efficient; late money is informed).
- **Betting at open on any static rule does not beat the close** — the extra vig
  at open exceeds any line movement captured.

## 10. Where the edge could still be — handoff (2026-07-13)

Every *model-based* route to beating **Pinnacle's opening line** is closed: no
ROI edge, no CLV edge, calibration can't fix it, no distribution fixes it, and
the miscalibration is winner's curse (structural, not a bug). **A better model
makes better predictions, not betting edge, because Pinnacle's opening line is
already efficient.**

The one hypothesis not yet tested — and the only one that avoids winner's curse —
is **line timing**:

> The user bets via **Sportmarket** (a sharp-book aggregator/brokerage) on
> newly-opened CSL lines. If some book opens a line *earlier* than Pinnacle, that
> earliest line is the softest (least information incorporated). Betting value on
> it **before** Pinnacle forms/sharpens the market captures +CLV **without needing
> model edge and without winner's curse** — you are not selecting on
> model-vs-market disagreement, you are exploiting a not-yet-efficient market.

**What to get to test it (paired, timestamped):** for upcoming CSL matches, the
**earliest-opening book's open line/price + timestamp**, then Pinnacle's **open**
and **close**. Measure: does the earliest line move *toward* Pinnacle's close
(→ +CLV, soft = exploitable) or already ≈ close (→ no edge)?

**Data reality / gaps for whoever picks this up:**

- Current CSV (`pinnacle_open_ah*` + `pinnacle_close_ah*`) has **no 1X2 odds** and
  **no soft-book odds**; close AH only 2023–24.
- Git history (pre-PR#23, 28-col schema) has `AvgBookmakerCH/CD/CA` =
  average-bookmaker **1X2** for 701 matches (2023–25, overround ~7%, single
  near-close snapshot) — usable only for a rough "model vs average book" 1X2
  check, **not** CLV (no open/close).
- Production odds capture (`csl.odds.fetch_pinnacle_spreads`, The Odds API) is
  **fixed to `pinnacle` / `spreads`**. To capture other books' opening lines,
  widen the bookmaker list — *if* The Odds API even carries the earliest book
  (Asian books like SBO/IBC may not be exposed).

**Immediate next step (user in progress):** verify **which book opens CSL lines
earlier than Pinnacle**, by how much, whether The Odds API / Sportmarket exposes
it by name, and how its early line compares to Pinnacle's open→close. That result
decides whether to extend the capture pipeline to that book and run the
earliest-line→close CLV test with the existing CLV logic.

## 11. Pinnacle 1X2 opening line + CLV — the draw is the problem (2026-07-15)

After the AH dead-end the user added Pinnacle **1X2** (home/draw/away) opening +
closing odds, hoping to dodge the big-line bias of §9.5. Read this section as: **the
strategy as originally specified (bet the highest-EV outcome) is dead, but 1X2 is
*not* dead the way AH is** — the loss is concentrated in one fixable model defect,
and once it is removed a real, baseline-adjusted CLV signal survives. It is still not
yet tradeable, and §11.7 explains precisely why (the vig, not the model).

**Method:** same walk-forward production recipe, scoring 1X2 instead of AH. Bet the
highest-EV outcome above a threshold at the opening 1X2 price; settle vs result;
CLV = no-vig *closing* prob − no-vig *opening* prob of the pick. ZIP vs NegBinom
compared. Reproduce: `PYTHONPATH=src python backtest/backtest_1x2.py` (writes
`backtest_1x2_{zip,negbinom}_bets.csv`).

**Data:** 611 gradeable matches, 2024 R1 – 2026 R18 (2024: 240, 2025: 238, 2026: 137
with full open+close+result). **2023 is excluded by `MIN_TRAIN=100`** — the dataset
starts 2023-04-15, so 2023 rows have a median of 26 training fixtures and the model
there is noise; only 56 of 2023's 240 rows carry opening lines anyway.

**Data quality.** A physical check — a real book can never post an overround below
zero — swept all 611 matches and found exactly **one** bad cell (2024-05-05 Wuhan
Three Towns v Qingdao Hainiu, opening draw `368.00`, a decimal typo for `3.68`; it
implied a *−18.9%* overround). The user corrected it; every number in §11 is
post-fix. It barely moved ROI (2024 −24.3% → −23.8%) but it had inflated the EV-
overstatement metric (+0.406 → +0.230). Closing odds: zero errors. Opening overround
sits at median 7.56% (p1–p99: 3.97–9.58%), closing at 4.72% — the odds data is sound.

### 11.1 The 2026-only result that started this (superseded)

At 12 rounds (94 matches) the +EV set showed **+14.6%** ROI; at 18 rounds it fell to
**+2.1% (t=0.16)**. The CLV looked strong and confidence-monotone (EV>0.10 +1.87pp
t=2.76). §11 previously called this a candidate edge and set the promotion test: *get
more seasons and check the +CLV persists.* The seasons arrived and it did not — see
11.2. Two methodological lessons came out of the post-mortem, 11.3 and 11.7; **they
are the most reusable content in this whole document.**

### 11.2 Three-season replication of the as-specified strategy (ZIP, thr>0.10)

| Season | n | realized ROI | CLV (no-vig, pp) |
| --- | ---: | ---: | ---: |
| 2024 | 152 | **−23.8% (t=−1.98)** | +0.24 (t=0.63) |
| 2025 | 142 | +5.0% (t=0.33) | +0.40 (t=0.94) |
| 2026 | 82 | +13.7% (t=0.80) | **+1.87 (t=2.76)** |
| *pooled* | *376* | *−4.8% (t=−0.57)* | *+0.66 (t=2.43)* |

The pooled CLV clears significance but **that is one season**, and the season with the
most bets loses money at t=−2. NegBinom is near-identical, so this is not a
distribution artifact. Full Kelly over three seasons goes to **zero** (100% drawdown);
quarter-Kelly ends at 40.8 (86% drawdown).

### 11.3 **The baseline control — never read a CLV number without it**

*This is the single most important methodological point in the document.* CLV is a
property of the **market**, not only of the model. Before crediting a model for +CLV,
ask what a **model-free** rule betting the same outcome mix would have earned. In the
CSL 1X2 market the no-vig probability drifts systematically **toward the home team**
from open to close — every season:

| Season | home drift | draw drift | away drift |
| --- | ---: | ---: | ---: |
| 2024 | +0.35pp | −0.12pp | −0.23pp |
| 2025 | +1.26pp | −0.45pp | −0.81pp |
| 2026 | +1.29pp | +0.26pp | −1.54pp |
| **All** | **+0.91pp** | **−0.16pp** | **−0.75pp** |

So the model-free baselines are:

| Strategy | CLV | t |
| --- | ---: | ---: |
| **always home** | **+0.91pp** | **+2.84** |
| always draw | −0.16pp | −1.39 |
| always away | −0.75pp | −2.62 |
| random pick | +0.11pp | +0.41 |
| **the model** | **+0.42pp** | **+1.93** |

**A coin that always says "home" beats the model on CLV.** The model bets home 165 /
away 71, so it inherits the home drift for free. Any "the model has +CLV" claim must
therefore be reported as **excess CLV** = model CLV − the mean drift of that season ×
that picked outcome. The 2026 "signal" of 11.1 was substantially this effect plus
noise. **Always compute the baseline first.**

### 11.4 Why the draw destroys the strategy (the actual defect)

The model's draw probability is pinned near 0.28 and is too high **in every
season and every match type**:

| Home's opening win prob | n | model draw | market draw | actual draw | model error |
| --- | ---: | ---: | ---: | ---: | ---: |
| weak home <30% | 141 | 0.285 | 0.230 | 0.213 | **+0.072** |
| even 30–45% | 182 | 0.324 | 0.274 | 0.280 | +0.044 |
| home-ish 45–60% | 153 | 0.288 | 0.247 | 0.294 | −0.006 |
| big favourite >60% | 135 | 0.200 | 0.170 | 0.163 | +0.037 |

The market is near-exact at every level; the model is high by ~4pp. This is the
structural consequence of conditionally-independent Poisson scoring — too much mass
piles up at goal-difference 0. **It is not a tuning problem.**

**The lethal part is the interaction with the EV rule.** With draw prob ≈ 0.28, the
EV>0.10 filter fires whenever the draw price exceeds 1.10/0.28 ≈ **3.93**. The CSL
median opening draw price is **3.79**. So *any match whose draw is priced above the
median automatically becomes a draw bet* — which is why **375/611 = 61% of all bets
are draws**, and those bets carry **zero CLV (+0.03pp, t=0.22)**. The strategy puts
61% of its stake on a model bug carrying no information.

### 11.5 Where 2024's −23.8% actually came from

| pick | n | ROI | t |
| --- | ---: | ---: | ---: |
| home | 15 | −47.0% | −1.99 |
| **draw** | **125** | **−26.6%** | **−2.00** |
| away | 12 | +34.2% | +0.60 |

Almost entirely the draw. Slicing the draw bets by price locates the damage in one
bucket:

| opening draw price | n | ROI | actual draw rate | model said |
| --- | ---: | ---: | ---: | ---: |
| 0–3.9 | 151 | −5.4% | 27% | 37% |
| 3.9–4.5 | 54 | +15.4% | 28% | 30% |
| **4.5–6** | **40** | **−62.0% (t=−2.88)** | **8%** | **28%** |
| 6+ | 15 | +15.4% | 13% | 19% |

In matches with a clear favourite (draw priced 4.5–6) the model claims 28% draws and
reality delivers **8%**. That single bucket is the crater.

### 11.6 Dropping the draw — what actually survives

Restricting the pick to the best of {home, away} (draw never backed):

| EV thr | n | ROI | t | CLV | t | **excess CLV** | t |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| > 0.00 | 280 | +7.4% | 0.95 | +1.45pp | 3.14 | +1.08pp | 2.34 |
| > 0.10 | 138 | +10.0% | 0.79 | **+2.15pp** | **3.14** | **+1.73pp** | **2.51** |
| > 0.20 | 51 | +34.3% | 1.39 | **+3.30pp** | 2.92 | **+2.96pp** | **2.69** |

CLV roughly triples (+0.66 → +2.15pp) and — the part that matters — **it survives the
11.3 baseline adjustment** (+1.73pp, t=2.51). That is a genuine model contribution,
not inherited home drift. CLV is positive in all three seasons (+0.69 / +2.15 /
+3.18pp).

**But do not call this an edge yet.** Realized ROI does **not** replicate: 2024
−5.5%, 2025 **−16.9%**, 2026 **+53.9% (t=2.10)**. The entire positive ROI is again one
season — the same shape as the false signal in 11.1. Per-season excess CLV is
individually significant only in 2026 (+2.91pp, t=2.55; 2024 +0.62, 2025 +1.39).

### 11.7 **The vig wall — why +CLV still loses money**

The second reusable result. Betting at opening price `O`, with no-vig opening
probability `p` and opening overround `R`, if the closing no-vig probability is
`p + CLV` then:

> **EV > 0  ⟺  CLV > p × R**

With the selected side averaging `p = 0.344` and Pinnacle's **opening overround
R = 7.55%**, breakeven needs **CLV > 2.61pp**. Measured against that bar:

| Strategy | CLV | needed | gap |
| --- | ---: | ---: | ---: |
| max-EV, EV>0.10 | +0.66pp | +2.21pp | **−1.55** |
| no-draw, EV>0.10 | +2.15pp | +2.95pp | **−0.80** |
| no-draw, EV>0.20 | +3.30pp | +2.56pp | **+0.74** ✅ (n=51 only) |

This explains everything else in one line: `always home` earns +0.91pp CLV and still
returns −4.8% ROI, because 0.91 < 2.61. Dropping the draw moves the model from −1.55pp
short to −0.80pp short; only the most aggressive cell clears the bar, by a thin
+0.74pp on n=51.

**The wall is the 7.55% opening vig, not model accuracy.** Everything the model knows
is worth ~2–3pp of CLV; the vig alone costs 2.61pp. **This raises, not lowers, the
value of roadmap #8:** the same +2.15pp CLV is a loser into Pinnacle's 7.55% open and a
winner into a 4% book. Cheaper/softer prices beat a smarter model.

### 11.8 On the model choice (ZIP vs NegBinom)

NegBinom is a modest, legitimate upgrade for *accuracy and EV honesty* (best RPS,
~1.5% better log-loss; ZIP has collapsed to Poisson, so its extra parameter is dead).
Across three seasons it changes **none** of the conclusions above, and it does **not**
fix the draw (0.276 vs ZIP 0.279). Adopt it on accuracy grounds, not as an edge play.

### 11.9 Verdict and what to do next

- **The strategy as specified (max-EV incl. draws) is dead.** 61% of stake on a model
  bug; −4.8% ROI over three seasons; full Kelly → 0.
- **The 1X2 *direction* is not dead** the way AH is. After the draw is removed there is
  a baseline-adjusted CLV of +1.73pp (t=2.51) that is positive in all three seasons.
  It is real but **not yet profitable**, and ROI does not replicate.
- **Do not bet this.** The only cell that clears the vig wall is n=51 and clears it by
  0.74pp.

**Priority order (revised 2026-07-15):**
1. **Draw de-bias + NegBinom — promoted from "nice accuracy fix" to the main lever.**
   Shrink the model's draw probability toward the market, re-run, and test whether
   excess CLV can hold above the **2.61pp** bar of 11.7 across all three seasons. This
   is the only quantified lever available (CLV +0.66 → +2.15pp just by *not* betting
   the draw; de-biasing should also repair the home/away split it distorts).
   **→ DONE, see §12. The de-bias works as a model fix but the bar is still not
   cleared in 2024/2025 — betting Pinnacle's open stays closed.**
2. **Roadmap #8 (earliest / cheapest line) — arguably higher value.** 11.7 shows vig
   dominates: reducing the price paid beats improving the model.
3. Re-test betting only after (1), against the 11.3 baseline and the 11.7 bar. **Any
   future CLV claim must report excess CLV over the model-free baseline and be measured
   against `p × R`** — the 2026 false signal is what happens when it is not.

## 12. Draw de-bias (roadmap #9) — model fixed, strategy still short of the bar (2026-07-15)

§11.9's priority 1, executed. The de-bias mechanism chosen (user decision) is a
**market-anchored shrink**, applied after prediction and before EV, per match:

```
m_D  = no-vig OPENING draw prob = (1/oD) / (1/oH + 1/oD + 1/oA)
p'_D = (1−λ)·p_D + λ·m_D
p'_H = p_H · (1−p'_D)/(1−p_D)      # freed mass returned to home/away pro-rata
p'_A = p_A · (1−p'_D)/(1−p_D)
```

No leakage: the opening price is known at bet time. Unlike §11.6's "never back the
draw" rule, the shrink also scales home/away *up* by ~(1−m_D)/(1−p_D) ≈ 1.06, so it
changes which side is picked, not just whether draws are bet. λ was **not** optimised
walk-forward (one parameter, three seasons = overfit bait); the full grid
λ ∈ {0.25, 0.5, 0.75, 1.0} is reported and only a *region* of λ clearing the bar in
all three seasons would count as success. Variants: NegBinom × λ grid (the roadmap-#9
candidate) plus ZIP+λ=1.0 to separate the de-bias effect from the distribution swap.
Reproduce: `PYTHONPATH=src python backtest/backtest_1x2.py` (regression-checked: the
ZIP λ=0 rows reproduce §11.2 exactly, and the §11.3 baselines reproduce to the digit).

### 12.1 The de-bias does exactly what it was designed to do

- **Draw probability is repaired.** NegBinom raw 0.276 → λ=0.75 shrunk **0.245** vs
  actual 0.242 (market 0.234). The 4pp structural bias is gone at λ ≈ 0.75–1.0.
- **The stake migrates off the bug.** Draw share of picks: 63% (384/611) at λ=0 →
  46% at λ=0.25 → 27% at λ=0.75 → 0% at λ=1. Residual draw picks at intermediate λ
  still carry ~0 CLV, confirming they are leftover bug, not signal.
- **Excess CLV roughly doubles.** At thr>0.10: +0.60pp (raw) → **+1.42pp (λ=0.5,
  t=3.20)** / +1.39pp (λ=0.75) / +1.21pp (λ=1.0). At thr>0.20 the pooled numbers
  reach **+2.07 to +2.51pp (t≈3)** — the strongest baseline-adjusted CLV measured in
  this whole investigation.
- **Kelly no longer ruins.** 0.25-Kelly over three seasons: 40.8 (raw, −59%) →
  **120.1 (λ=0.75, +20%)**, max drawdown 86% → 50%. Still a horror ride, but the
  bug-driven death spiral is gone.

### 12.2 …and the strategy still fails the success bar

Success criterion (roadmap #9): per-season **gap = CLV − mean(p×R) > 0 in all three
seasons**, over a λ region. Per-season gap (pp) at thr>0.10:

| λ | 2024 | 2025 | 2026 |
| ---: | ---: | ---: | ---: |
| 0 (NB raw) | −1.92 | −1.95 | −0.32 |
| 0.25 | −1.89 | −1.63 | **+0.03** |
| 0.50 | −1.28 | −1.32 | **+0.88** |
| 0.75 | −1.78 | −1.22 | **+0.91** |
| 1.00 | −2.38 | −0.91 | +0.86 |

**Only 2026 ever clears the bar.** 2024 and 2025 are negative at every λ; 2024 barely
responds to the de-bias at all (λ=1: exCLV −0.10pp — the model simply has no CLV
signal there). Pooled thr>0.20 cells touch the bar (gap +0.03 to +0.50 on n=77–93)
but that is the same one-good-season shape as §11.1's false signal, and per-bet ROI
t-stats never exceed ~1. ZIP+λ=1 ≈ NegBinom+λ=1 (exCLV +1.17 vs +1.21pp): **the
distribution swap contributes ~nothing to CLV; all improvement is the de-bias** —
consistent with §11.8 (adopt NegBinom for accuracy, not edge).

### 12.3 Verdict

- **As a model fix: success.** Draw prob 0.245 vs actual 0.242, stake off the bug,
  excess CLV doubled, Kelly survivable. λ ≈ 0.75 is the accuracy pick.
- **As a betting strategy into Pinnacle's open: still closed.** The §11.7 vig wall
  stands: ~2.2–2.5pp of bar vs ~1.2–1.4pp of season-replicable excess CLV. Do not
  deploy a 1X2 betting pipeline against Pinnacle's opening price on this evidence.
- **The strongest quantitative argument yet for roadmap #8.** The de-biased model's
  excess CLV (+1.2pp replicable, +2.0–2.5pp at high thresholds) clears the breakeven
  bar at a book with **≤5% overround** (bar ≈ 0.35 × 5% = 1.75pp) and comfortably at
  4% (≈1.4pp). The signal exists; Pinnacle's 7.55% opening vig is what eats it.
  Cheaper prices, not further model work, remain the live direction.
- Production note: the market-anchored shrink itself is **not deployable** — it needs
  a market 1X2 anchor, which production doesn't fetch (spreads only, and adding h2h
  to the /odds calls would double their quota cost and bust the free budget). The
  deployable mechanism is the market-free δ calibration of §12.4, which was
  validated here and then shipped.

### 12.4 The market-free δ calibration — validated and deployed (v2.5)

Production (`dc.py` / the dashboard) has no 1X2 odds to anchor on, so the deployable
de-bias is **market-free**: after fitting, find the scale **δ for the scoreline-grid
diagonal** that maximizes the Dixon-Coles-weighted 1X2 log-likelihood of the
*training* fixtures, and apply it (diagonal × δ, renormalize) to every prediction.
Validated walk-forward here first (`NegBinom + delta-cal` variant, per-round δ fit on
each training window — no leakage), then deployed.

- **Fitted δ:** mean 0.894 across rounds (range 0.82–1.00) — the model's diagonal is
  ~11% too heavy, consistently.
- **Out-of-sample draw prob: 0.276 → 0.255** (actual 0.242, market 0.234). It repairs
  **roughly half** the bias — honest MLE on the training window is more conservative
  than anchoring on the market (λ=0.75 reaches 0.245 but needs the anchor). Draw share
  of picks 63% → 45%; residual draw picks still carry ~0 CLV.
- **No metric degrades:** exCLV +0.60 → **+0.93pp** (thr>0.10, t=2.74), 0.25-Kelly
  end 46 → 62. Per-season gaps still all negative (−1.87 / −1.70 / −0.06) — the
  betting verdict of §12.2–12.3 is unchanged, as expected.
- **Deployed (dashboard v2.5, 2026-07-15):** `src/csl/models/dc.py` now fits
  `NegativeBinomialGoalModel` (accuracy grounds, §9.4/§11.8) and returns a
  `DrawCalibratedModel` wrapper applying the training-window δ to every predicted
  grid (`fit_draw_delta` mirrors this section's math exactly). On the current full
  dataset the production fit lands at **δ = 0.908**, mean simulated draw prob 0.259.
  Both consumers verified end-to-end (`DC_CHN.py` and
  `export_upcoming_market_comparison`). The de-bias also thins the diagonal of the
  AH settlement grid, which is directionally correct (§9.1: the margin distribution
  was under-dispersed).

## 13. Cheaper book (1xBet) + strategy scoreboard (2026-07-16, extended 2026-07-17)

Roadmap #8's thesis — *pay less rather than predict better* (§11.7) — gets its data.
The user hand-backfilled 1xBet's opening **and** Pinnacle's closing 1X2 for **2024,
2025 and 2026** into `CHN_Super League.csv` (`onexbet_open_*`, `pinnacle_close_*`; 2023
still empty). All three were vetted for data-entry errors with a **logarithmic overround
check** — `ln(Σ 1/oᵢ)`, which after correction sits at median 4.75–4.76% with sd 0.11–
0.12 and no impossible values, so the lines are sound. `backtest/backtest_1xbet.py` bets
the model's EV on **1xBet's opening** line and grades CLV against **Pinnacle's closing**
line (the sharp fair-value reference).

### 13.1 1xBet opening line — vig wall cleared in ALL THREE seasons (2024 + 2025 + 2026)

n=611 fixtures across **three full seasons** (2024: 240, 2025: 238, 2026: 133 gradeable).
The first cut of this section was 2026-only (a lead); 2024 made it two seasons; **2025
now makes it three, and the gap still clears in every season on its own** — the §11.1
one-good-season caveat is fully discharged.

- **1xBet's opening overround is 4.87%** (vs Pinnacle's 7.55% open). The §11.7 vig wall
  drops from **2.61pp to ~1.71pp** — the whole point of a cheaper book.
- Model-free baseline (§11.3) on the 1xBet-open→Pinnacle-close basis: `always home`
  **+0.62pp** (t=1.94). Small, so the model's edge below stands clearly *above* baseline
  (which is what excess CLV isolates).
- **The gap (CLV − bar) is positive at thr>0.20 for every variant, in all three seasons
  independently.** Production (δ) and the λ=0.75 anchor, split by season:

  | model · rule | season | n | ROI | excess CLV (t) | **gap** |
  | --- | --- | ---: | ---: | ---: | ---: |
  | NegBinom + δ · no-draw | 2024 | 46 | +28.7% | +4.32 (t=3.0) | **+2.79** ✅ |
  | | 2025 | 36 | **−24.0%** | +4.19 (t=2.7) | **+2.53** ✅ |
  | | 2026 | 27 | +43.6% | +2.46 (t=1.8) | **+1.07** ✅ |
  | | *pooled* | *109* | *+15.0%* | *+3.82 (t=4.4)* | *+2.28* |
  | NegBinom + λ=0.75 · with-draw | 2024 | 46 | +27.1% | +3.42 (t=2.9) | **+2.11** ✅ |
  | | 2025 | 42 | **−18.4%** | +4.29 (t=3.6) | **+2.81** ✅ |
  | | 2026 | 26 | +66.7% | +2.48 (t=1.7) | **+1.18** ✅ |
  | | *pooled* | *114* | *+19.4%* | *+3.52 (t=4.8)* | *+2.16* |
  | NegBinom raw · no-draw | 2024 | 38 | +41.1% | +5.82 (t=4.0) | **+4.36** ✅ |
  | | 2025 | 29 | **−15.5%** | +5.11 (t=2.9) | **+3.47** ✅ |
  | | 2026 | 21 | +42.0% | +3.02 (t=1.8) | **+1.72** ✅ |

- **The 2025 season is the whole methodological lesson in one row.** 2025 realized
  **negative ROI** (δ no-draw −24.0%, λ −18.4%, raw −15.5%) while posting the *strongest*
  excess CLV of the three seasons (+4.2 to +5.1pp, t=2.7–3.6) and a large positive gap
  (+2.5 to +3.5). The closing line is unambiguous that those bets had value; that season's
  results simply did not pay out. Per-season ROI t-stats sit at ≈1 (and go negative);
  excess-CLV t-stats are 2.7–4.0 in *every* season. **Judge the strategy by the low-
  variance signal (excess CLV / gap), never by one season's ROI** — the §11.1 / §11.7
  lesson, now demonstrated cleanly in the direction that matters (a +value set can lose
  money short-run).

- **thr must be 0.20, not 0.10.** At thr>0.10 the model's edge is still real (exCLV
  significant, gap positive in 2024 and 2025) but it **flips negative in 2026** (δ −0.43,
  raw −0.44). Only the tighter thr>0.20 cell clears the wall in all three seasons —
  selectivity is load-bearing. (§13.2's finding that raw scores highest is the same
  effect: raw's un-repaired draw-deflated home/away probs act as an implicit tighter
  filter — raw@0.20 ≈ δ@0.25, converging exactly at thr>0.50. The knob is the EV
  threshold, not the model; use production δ and raise the threshold to be more selective.)

- **Draw vs no-draw** (the user's ask): with the **raw** model, no-draw beats with-draw
  on money (it stops raw dumping ~half its stake on over-priced draws). With the
  **de-biased** model the two converge — the de-bias already stops the model betting
  draws. **De-bias ≈ an automatic, smarter "no-draw":** it drops the junk draws but keeps
  the few genuinely +EV ones and returns freed mass to home/away.

### 13.2 Which opening line is the better anchor — Pinnacle or 1xBet?

`backtest/anchor_test.py`, now n=611 (2024+2025+2026, the seasons with both books'
opens), only the anchor book varies. **It barely matters.** The two books' no-vig opening
*draw* probs are nearly identical (per-fixture mean |diff| small), so the λ-anchored
models are statistically indistinguishable (log-loss 0.9603 @ Pinnacle vs 0.9599 @ 1xBet
— a 0.0004 gap = noise; both beat raw 0.9737 and δ 0.9730). The reason: **the λ de-bias
transplants only the draw and keeps the model's own home/away split**, so Pinnacle's
sharper H/A pricing (market-only log-loss 0.9522 vs 1xBet 0.9592) never enters the model.
Practical
split: **anchor on Pinnacle (or either — no accuracy cost), bet on 1xBet (the cheaper
price).**

### 13.3 Strategy scoreboard — everything tested, and what runs now

**Vocabulary** (every strategy is a choice on these axes):

- **Market** — *AH* = Asian handicap (spreads, home/away vs a goal line); *1X2* =
  home/draw/away moneyline.
- **Price / book** — the line we actually bet at: *Pinnacle open*, or *1xBet open*
  (cheaper). CLV is always graded against a **closing** line (Pinnacle close).
- **Selection rule** — how the bet is chosen from the model:
  - *max-EV* — back the single outcome whose model EV = `p·odds − 1` is highest, if it
    clears the EV threshold. Draws eligible.
  - *no-draw* — max-EV over {home, away} only; the draw is never backed.
  - *+ de-bias (λ / δ)* — first correct the model's structural draw over-pricing
    (market-anchored λ, or market-free δ), then apply max-EV. This mostly removes draw
    bets on its own (§13.1).
- **Metrics** — *ROI* = realized return per unit at the bet price (the ground truth).
  *CLV* = no-vig probability drift from the bet line to the closing line, in pp.
  *excess CLV* = CLV minus the §11.3 model-free same-outcome baseline drift (a coin that
  always bets home already earns +CLV — this strips that out). *bar* = `p × R`, the
  §11.7 breakeven CLV set by the book's overround. *gap* = CLV − bar; **gap > 0 ⟺ the
  bet is +EV net of vig.** ROI t-stats are the money truth; a positive gap without a
  significant multi-season ROI is a *lead*, not an edge.

**Scoreboard** (headline cell per strategy; "single-season" = one year carries it,
§11.1):

| # | Strategy (market · price · rule) | § | Sample | ROI | excess CLV | gap vs vig wall | Verdict |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| 0 | *baseline* — always-home, 1X2 · Pinn open | 11.3 | 3 seasons | −4.8% | — (this **is** the baseline, +0.91pp raw CLV) | −1.70 | The control every claim is measured against |
| 1 | **AH · Pinn open · max-EV** | 5–9 | 826 bets, 2023–26 | −4.2% → −8.3% | — | — | **DEAD** — winner's curse; EV overstated +20%/unit (t=6); tighter filter = worse |
| 2 | AH · Pinn open · small-line only | 9.5 | 2023–26 | −3.6% | — | — | **DEAD** — still overstates EV +17% (t=3.7) |
| 3 | **1X2 · Pinn open · max-EV (with draw)** | 11.2 | 376 bets, thr>0.10 | −4.8% | ~0 (61% of stake on draws, 0 CLV) | −1.55 | **DEAD** — the draw-probability bug |
| 4 | 1X2 · Pinn open · no-draw | 11.6 | 138 bets, thr>0.10 | +10.0% (t=0.79, ns) | +1.73pp (t=2.5) | −0.80 | **Direction alive, not profitable** — ROI is one season; below the vig wall except n=51 |
| 5 | 1X2 · Pinn open · **+ λ=0.75 de-bias** | 12 | 611 fixtures | — | +1.42pp (thr>0.10) | neg in 2024 & 2025 | **Model fixed, betting CLOSED** — draw repaired (0.245), exCLV doubled, but gap<0 two of three seasons |
| 6 | 1X2 · Pinn open · **+ δ (market-free)** | 12.4 | 611 fixtures | — | +0.93pp (t=2.7) | neg 2024/25 | **DEPLOYED (v2.5/2.6) for ACCURACY**, not betting — needs no market anchor |
| 7 | **1X2 · 1xBet open · + de-bias** | 13.1 | 611, **2024+2025+2026** | +15% (δ, thr>0.20) | +3.82pp (t=4.4) | **+2.28 ✅ (all 3 seasons)** | **Cross-season positive** — gap clears the vig wall in 2024, 2025 *and* 2026 independently at thr>0.20; only cell in this doc to do so. 2025 lost −24% ROI on +2.5 gap / +4.2pp exCLV → read exCLV/gap, not ROI. Add 2023 to widen further |

**What runs now (2026-07-17).** Nothing is bet live. The production model (v2.6:
NegBinom on xG + Dixon-Coles decay, market-free **δ** everywhere, market-anchored
**λ=0.75** on the dashboard's EV-comparison surface where a Pinnacle open was captured)
is deployed **for forecast accuracy** — it repairs the structural draw over-pricing
(rows 5–6) — **not** as a betting signal. Betting Pinnacle's open stays closed (§12.3).
The one open, positive result is **row 7**: 1xBet's cheaper opening line clears the vig
wall at thr>0.20 in **all three backfilled seasons independently** (2024/2025/2026 gap
+2.79 / +2.53 / +1.07, excess CLV significant each season) — it has now passed the §11.1
promotion test that every earlier "signal" here failed, on three seasons rather than one.
It is still **not** a green light to bet real money: per-bet ROI stays high-variance
(2025 realized −24% on this very cell), only the selective thr>0.20 cell clears
(thr>0.10 flips negative in 2026), and it is graded against Pinnacle's close, not against
realized settlement at a real 1xBet account. The live task remains roadmap #8: add 2023
(and optionally Betfair) to tighten, holding the two guardrails — **excess CLV over the
§11.3 baseline**, measured against the **§11.7 `p × R` bar** — and, before any real
staking, confirm the edge survives as *realized* ROI over more than one season.

### 13.4 Deep-dive: the strategy grid, the odds cap, and the recommended config (2026-07-17)

§13.1–13.3 established row 7 as the one live result. This section drills into *which
cell* of it to actually bet, by sweeping the full grid (6 model variants × with/no-draw
× EV thr ∈ {0.10…0.40}) on the cached walk-forward predictions and stress-testing the
survivors with a per-season bootstrap and an odds slice. Three findings.

**(a) thr>0.20 is the usable floor, not the peak — gap rises monotonically with
selectivity.** For λ=0.75 (with-draw), pooled gap and the *worst* single season:

| EV thr | n | pooled gap | min-season gap | bootstrap P(gap>0 all 3 seasons) |
| ---: | ---: | ---: | ---: | ---: |
| >0.10 | 235 | +0.91 | **−0.68** (2026 flips) | — |
| >0.20 | 114 | +2.16 | +1.18 | 75% |
| >0.25 | 77 | +2.67 | +0.53 | 56% (a noise dip — 2026 thins) |
| >0.30 | 59 | +4.14 | +3.36 | **92%** |

thr>0.30 is cleaner per-season but n=59 (2026 only 10 bets), too thin to lean on; thr>0.20
is the balance of "still cross-season-significant" and "enough bets to trust." The 0.25
dip confirms the boundary itself is noisy — **do not treat 0.20 as a fitted magic
number**; it is a region.

**(b) An odds≤7 cap is a free improvement.** At λ=0.75 · thr>0.20, 14 of the 114 picks sit
at odds ≥ 7. That long-shot tail is the worst slice in the whole book (win 7%, ROI −37%,
exCLV +1.35, gap +0.75 — the least edge). Dropping it (n=114→100):

| | n | exCLV (t) | gap | 2024 / 2025 / 2026 gap | 2025 flat ROI |
| --- | ---: | ---: | ---: | --- | ---: |
| no cap | 114 | +3.52 (4.8) | +2.16 | +2.11 / +2.81 / +1.18 | −18.4% |
| **odds≤7** | 100 | **+3.83 (4.7)** | **+2.35** | +2.43 / +3.09 / +1.18 | **−9.8%** |

Every season's gap improves and 2025's realized loss nearly halves — the cap strips
variance, not edge. The signal lives in the odds 1.5–7 mainstream, not the tail.

**(c) with-draw ≈ no-draw once de-biased.** Under λ=0.75 · thr>0.20 the two rules differ by
**2 bets** out of 114 (both are genuine +EV draws that survive the de-bias, CLV +1.9 and
+6.8pp). And 72 of raw·no-draw's 88 picks are the identical fixture+pick as λ=0.75's — one
edge, three filters (§13.1). So the draw axis is no longer a real choice: **keep draws
eligible and let the de-biased model decide** (or use no-draw — indistinguishable here).

**Recommended live config (if/when betting):** 1xBet open · NegBinom + **λ=0.75** (or
production **δ**) · max-EV, draws eligible · **EV > 0.20** · **pick-odds ≤ 7** · **≤0.25-
Kelly**. On this walk-forward that is n=100, exCLV +3.83pp (t=4.7), gap +2.35pp, positive
in every season; a $100 bankroll compounds to **$445** at 0.25-Kelly with a −28% max
drawdown (`backtest/bankroll_recommended.png`, from `backtest/plot_bankroll.py`).

![0.25-Kelly bankroll, recommended config](bankroll_recommended.png)

**The 2025 caveat, restated because it is the point.** 2025 realized **−9.8% flat per-bet
ROI** (the odds cap lifts it from row 7's −24%) on the set with the *strongest* excess CLV
of the three seasons (+4.6pp, t≈3) — the Kelly path only ends 2025 up (+18% compounded)
because of bet ordering, which is luck, not edge. Per-bet ROI t-stats stay ≈1 and go
negative; excess-CLV / gap t-stats are 2.7–4.7 every season. **Judge and size by exCLV /
gap, never by a season's ROI or a bankroll curve** — the curve is one lucky ordering of a
high-variance process, and the grid on real settlement (not Pinnacle's close) is still
unmeasured. The chart is a variance illustration, not a promise.

### 13.5 Staking-scheme comparison — Kelly vs flat vs compound fixed-fraction (2026-07-17)

Same 100 bets, same edge, `$100` start; only the staking rule varies. "Week" = one CSL
matchday `(Season, Round)`; 100 bets span 56 weeks. maxDD is peak-to-trough on the
after-bet equity curve (`backtest/staking_compare.py`).

| Scheme | rule | final | multiple | max drawdown |
| --- | --- | ---: | ---: | ---: |
| **M1 fractional Kelly** | stake = 0.25·Kelly·bankroll, per bet | **$445** | ×4.45 | **28%** |
| M3 compound fixed-fraction | 5% of bankroll, rebalanced each week | $297 | ×2.97 | **42%** |
| M2 flat unit | $5 fixed (5% of the *initial* bank), never recompounded | $236 | ×2.36 | 27% |
| — 0.50-Kelly (aggression ref) | stake = 0.50·Kelly·bankroll, per bet | $1184 | ×11.8 | ~51% |

![Staking schemes compared](staking_compare.png)

Three points:

- **0.5-Kelly doubles the money and doubles the pain.** $445 → $1184, but max drawdown
  28% → ~51% (both drawdown figures move ±2pp with within-day bet ordering). Half-Kelly is
  growth-optimal in theory; on a 100-bet sample graded against a *proxy* close it is a way
  to get wiped by variance you can't yet trust. Not recommended over 0.25 here.
- **M1 (Kelly) dominates M2/M3 on risk-adjusted terms.** It ends highest *and* its drawdown
  (28%) is no worse than flat betting's (27%), because Kelly sizes **down** on low-edge /
  long-odds bets that the fixed 5% over-bets. M3's flat 5% fraction ignores edge size, so it
  over-bets the marginal bets and rides a **deeper 42% drawdown to a lower $297** — betting
  above the Kelly-optimal fraction is the classic "more risk, less growth" trap.
- **M2 (flat $5) is the floor-risk option.** Because the stake never grows with the bank, it
  compounds linearly, ends lowest ($236), but its late drawdowns are trivial in percentage
  terms — the ride is the smoothest. It is the right choice only if you value a bounded
  dollar stake over growth (e.g. while validating live execution).

All three ride the same 2025 dip (bets ~55–75) — the drawdowns are the same losing streak
scaled by aggression, not three different risks. The staking rule sets the *risk/growth
trade-off*; it does **not** change the edge, and none of it substitutes for confirming that
edge as realized ROI (§13.4). Recommended: **0.25-Kelly** (or M2 flat $5 during a live
trial), never above half-Kelly, on the §13.4 bet config.

### 13.6 Which book anchors the draw de-bias? Pinnacle vs 1xBet — pre-registered (2026-07-17)

Production (dashboard v2.7) bets the 1xBet open but anchors the λ=0.75 draw shrink on
**Pinnacle's** open, not 1xBet's — "de-bias against the sharp reference, bet the cheap
line." §13.1–13.4 validated only the 1xBet-self-anchor. This section runs the missing
cell as a **pre-registered** experiment (`backtest_1xbet.py`, variant
`pinnacle_anchor`, `compare_anchors`): both variants differ *only* in the anchor book —
same bet price (1xBet open), same CLV reference (Pinnacle close), same n. Coverage is
100% (every bettable 1xBet-open fixture also has a Pinnacle open), so the two run on the
identical batch with no sample confound.

**Head-to-head, λ=0.75, with-draw, thr>0.20 (n=611 fixtures):**

| anchor | bets | draws | exCLV (t) | gap | 2024 / 2025 / 2026 gap | boot P(all +) |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| 1xBet-open | 114 | 2 | +3.52 (4.83) | +2.16 | +2.11 / +2.81 / +1.18 | 0.77 |
| **Pinnacle-open** | 125 | 9 | **+3.35 (4.83)** | **+2.02** | +2.40 / +2.48 / +0.68 | 0.68 |

**Pre-registered acceptance criteria (all PASS → adopt Pinnacle anchor):**

- **A. Main bar** — exCLV t ≥ 2.0 and gap>0 every season. Pinnacle: t=**4.83**, per-season
  gap **+2.40 / +2.48 / +0.68** (all positive). **PASS.**
- **B. Non-inferior to 1xBet anchor** — Pinnacle gap ≥ 1xBet gap − 0.5pp. +2.02 ≥
  2.16 − 0.5 = 1.66. **PASS** (the two are statistically the same gap).
- **C. Extra draws aren't noise** — on the disagreement subset (n=77, ≈all draws) the
  Pinnacle pick's exCLV must be ≥ 0: **+0.04pp** (the 1xBet anchor's alternative pick
  there is **−1.42pp**). Sharper, bet-level: the 9 draws Pinnacle actually bets carry
  exCLV **+1.70pp** (t=1.25) / gap **+0.74pp**. **PASS.**
- **D. Beats model-free** — exCLV is CLV over the §11.3 baseline by construction;
  +3.35pp (t=4.83) > 0. **PASS.**
- **E. Coverage** — Pinnacle-open availability ≥ 1xBet-open. 100% = 100%. **PASS.**

**Why the two are near-identical, confirmed by the pick agreement:** the anchors agree on
**100% of H/A picks (384/384)** — H/A is ~98% of bets and *all* of the edge, and it is
byte-identical between anchors. The only difference is the draw: the Pinnacle anchor,
comparing a sharp reference draw against 1xBet's price (an honest cross-book comparison),
releases **9 draw bets vs 2** (168 vs 209 draws at the argmax level). Those extra draws
are collectively **neutral-to-mildly-positive** (bet-level exCLV +1.70pp, wide CI) — not a
new edge, but not noise either, matching §13.4's "with-draw ≈ no-draw" exactly. This is the
mechanism prediction confirmed: swapping the anchor moves nothing on H/A and only lets a
few honestly-de-biased draws through.

**Honest caveat:** the Pinnacle anchor's edge concentrates its weakest season in 2026
(gap +0.68, boot 0.68 vs the 1xBet anchor's +1.18 / 0.77) — the extra draws dilute the
thin (30-bet) 2026 season slightly. It still clears criterion A (all seasons +), but the
2026 margin is slim. If one wanted to drop the unvalidated draws entirely, the Pinnacle
anchor **no-draw** is the cleanest cut (pooled gap +2.11, per-season +2.50 / +2.52 / +0.82,
exCLV t=4.71) — indistinguishable from with-draw, per criterion C.

**Verdict:** adopt the **Pinnacle-open anchor** (A·B·C·D·E all pass) — the config already
shipped in v2.7. The choice is vindicated on quality grounds (sharper reference, one
coherent probability across the prediction surface, honest cross-book draw comparison, and
equal data coverage) while being *measurably* non-inferior on the betting bar. The odds≤7
cap (§13.4b) is orthogonal — it applies identically to both anchors — and stays in the
production config.

---

## 14. Polymarket (roadmap: new source) — open is an artifact, close is efficient (2026-07-25)

Added Polymarket as a third CSL odds source (`src/csl/odds/fetch_polymarket.py`; key-free
Gamma + CLOB, decimal odds = 1/Yes-price). Coverage runs 2025-10-24 → 2026-08-02 (184 main
1X2 events); 164 CSV rows filled (`polymarket_open_*`, `polymarket_close_*`) — the 2025
season tail (16) plus all 2026 rounds 1–19 (148). Two questions were pre-registered: does
betting the Polymarket **open** beat 1xBet's open (§14.1), and can the ~0-vig Polymarket
**close** be used as a cheap execution venue (§14.2)? Both answers are **no**, for opposite
reasons. Overrounds frame the whole section:

| price | mean overround R | note |
|---|---|---|
| polymarket_open  | **44.3 %** | 3 independent binaries each seeded ~0.5 → sum ~1.44 (synthetic) |
| onexbet_open     | 4.87 % | a real posted line |
| pinnacle_close   | 4.87 % | the sharp reference |
| polymarket_close | **0.14 %** | near vig-free |

### 14.1 Betting Polymarket OPEN — a flat-seed artifact, not alpha

Same shipped strategy (NegBinom, λ=0.75 Pinnacle-open anchor, EV≥thr, CLV vs Pinnacle
close), 1xBet-open vs Polymarket-open on the **identical 153 fixtures**. `backtest_polymarket.py`.
Headline cell (prod variant, WITH draw, thr>0.20):

| bet price | n | ROI | ROI t | CLV | exCLV | bar | gap |
|---|---|---|---|---|---|---|---|
| 1xBet open      | 35 | +0.297 | +1.00 | +2.54pp | +2.14pp | 1.64 | **+0.90** |
| Polymarket open | 57 | +0.272 | +2.01 | **+28.18pp** | **+17.65pp** | 15.00 | **+13.18** |

Every framework metric screams that Polymarket wins — and **every one of them is a
measurement artifact**. The smoking gun is the no-vig OPEN probability of the picked
outcome: for Polymarket it ranges only **0.306–0.371** (a flat ~0.333 coin) versus 1xBet's
**0.115–0.640** (a live, matchup-varying line). Polymarket's open carries *zero* matchup
information — it is the market-maker seed. So the +28pp "CLV" is mechanical:
`true close (~0.61 for a favourite) − flat seed (0.33)`. The exCLV home-drift baseline
cannot neutralise it because the flat seed misprices *every* favourite; the 44% "overround"
is likewise synthetic (three binaries seeded near 0.5), not a cost a single-outcome bet
pays. The only honest, comparable number — realized ROI — is **the same as 1xBet**
(+0.27 vs +0.30 pooled; 2026: +0.36 @ t=2.48 vs +0.51 @ t=1.55), and it lives or dies on
getting filled at the thin seed price. Sample is ~90% 2026, so not cross-season.

**Verdict:** do **not** read the giant Polymarket-open CLV/gap as a signal. The open is not
a tradeable line; the apparent edge is "measuring favourites against a coin-flip."

### 14.2 Betting Polymarket CLOSE — the ~0-vig venue is efficient (the second idea)

If the open is an artifact, is the near-vig-free **close** a cheap execution venue? Decouple
the probability source from the execution price and bet *at* the Polymarket close (bar≈0, no
wall). The judge is realized ROI + calibration, not CLV. `backtest_polymarket_close.py`,
same 153 fixtures. **Calibration decides it** (Brier / log-loss vs actual, lower better;
uniform 1/3 → Brier 0.667):

| probability source | Brier | log-loss |
|---|---|---|
| model (prod de-biased) | 0.6402 | 1.0627 |
| Pinnacle close no-vig | 0.6268 | 1.0394 |
| **Polymarket close no-vig** | **0.6228** | **1.0332** |

Polymarket's close is **as sharp as Pinnacle's** (marginally better here; the two agree to a
mean 2.3pp per outcome) — cheap **and** efficient, resolving the paradox toward the
pessimistic branch. Neither bet variant survives:

* **(a) Model vs Poly-close** — no-draw pooled ROI is flat/negative (thr>0.10 +0.037 t=0.23;
  thr>0.20 +0.169 t=0.71). The positive WITH-draw cells are pure longshot variance
  (avg odds 34, t<2), leaning on the 9-fixture 2025 tail. Consistent with calibration: the
  model is *worse* than the close it is betting into.
* **(b) Pinnacle-close vs Poly-close (no model)** — the cleanest test of the venue. No-draw
  2026 ROI +0.539 (t=1.42) at thr>0.10, but driven by avg odds 50–150 tail disagreements,
  t never clears 2, and 2025 flips to −100%. Betting the noise between two equally-sharp
  closes.

**Verdict:** the vig wall vanishes at the Polymarket close, but so does the edge — you face
a price as sharp as Pinnacle's. This is the textbook efficient-market outcome and a clean
negative result.

### 14.3 What Polymarket does and does not change

Polymarket opens **no new edge for CSL**: the open is untradeable (flat seed), the close is
efficient (unbeatable). Two side-findings worth carrying forward:

1. **The model barely beats a coin on 1X2** (Brier 0.640 vs uniform 0.667) and is *worse*
   than both market closes — a standing reminder that the edge is thin, which is exactly why
   it dies at any vig wall (§11.7).
2. **One door left ajar:** the longshot/draw tail cells (avg odds 30–150) are persistently
   *nominally* positive across both variants — consistent with the known favourite-longshot
   bias of prediction markets — but t<2 and almost entirely single-season (2026). Not
   actionable now; would need multi-season Polymarket data and a pre-registered "tail-only"
   test to separate a real inefficiency from variance.

Reproduce: `PYTHONPATH=src python backtest/backtest_polymarket.py` and
`.../backtest_polymarket_close.py`.
