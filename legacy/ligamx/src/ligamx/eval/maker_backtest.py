"""
Backtest resting maker quotes on Polymarket, measuring adverse selection.

THE IDEA UNDER TEST. As a taker at T-150h you cross a ~3c spread on each leg:
Sum(ask) ~= 1.046, so you start 4.6pp behind and need +1.4 to +2.1pp of genuine
edge just to break even -- more than the model has ([[polymarket-price-quality]]).
Quoting instead of crossing flips the sign: Sum(bid) ~= 0.975, so a filled maker
starts 2.5pp ahead. That is a ~7pp swing, and it does not require the model to
beat the market -- only to be unbiased.

WHAT MAKES IT NOT FREE. A resting order fills when someone chooses to cross it,
which is disproportionately when the price is moving against you. Earned spread
minus adverse selection is the whole game, so measuring the second term is the
point of this module.

METHOD. Post a bid on every outcome at a chosen lead time, where anchor is either the
model's fair probability or the market mid, and the quote sits below it by either a
fixed number of cents (--offset-mode cents) or a proportional EV cushion
(--offset-mode ev, quote = anchor / (1 + margin), so a filled quote is worth exactly
that much EV against the anchor). Fill if the market later reaches the quote before
kickoff; the fill price is our own limit, not the market's low. Settle on the real
result.

TWO FILL RULES, and the difference between them is the point:

  path  the hourly price curve dips to the quote. This is also the honest model of a
        broker pending-order (Sportmarket), which fires when the price reaches your
        number regardless of who is on the other side.
  tick  an actual print crossed the quote -- a Yes-space aggressor SELL at or below
        it (see odds/polymarket_trades). The curve is hourly, so it only registers
        moves that persist; the transient dips that revert inside the hour are the
        fills that pay a maker, and only the tape can see them.

  primary   (close - quote) on fills -- did the market end up agreeing with the
            price we got? Low variance, so it actually resolves at n~250.
  secondary realized P/L per share. Correct but noisy: per-share SD ~0.45 means
            se ~2.8pp at 250 fills, against a spread edge of 1.5-3pp. Underpowered
            by ~2x on its own -- read it as a consistency check, not the verdict.

KNOWN BIASES. Under `path`, fills are UNDER-detected (intraday lows we never see
would have filled us) and the fills we do detect are the larger moves, which
over-weights adverse selection -- that is exactly what `tick` exists to measure.
Under `tick`, queue position is unknown: a print STRICTLY below the quote swept our
level and would have filled us whatever our place in the queue, while a print AT the
quote is queue-dependent, so strict/--touch bracket the truth rather than pinpoint it.

    python -m ligamx.eval.maker_backtest [--test-start 2025-10-01] [--lead 72]
        [--offset-mode ev --ev-margin 0.20] [--fill-rule tick]
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from ligamx.eval.clv_backtest import _load, _model_probs
from ligamx.eval.rps_backtest import RESULT_IDX
from ligamx.odds import polymarket_trades, price_quality
from ligamx.odds.price_quality import match_row, row_index

warnings.filterwarnings("ignore")

OUTCOMES = ("home", "draw", "away")
ODDS_COLS = ("home_odds", "draw_odds", "away_odds")

# Quote offsets below the anchor, in cents. 0 = quote at fair (cross nothing, earn
# nothing); 3c is roughly the full spread at this lead time.
OFFSETS_C = (0.0, 1.0, 2.0, 3.0, 4.0)

# Proportional cushions for --offset-mode ev: quote = anchor / (1 + margin), so a
# fill is worth exactly this much EV against the anchor. 0.20 is the live proposal.
EV_MARGINS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30)

# Stand-in for a leg the tape does not cover. The is_sell slot must be bool or the
# boolean masks below fail on dtype rather than simply yielding nothing.
NO_PRINTS = (np.empty(0), np.empty(0), np.empty(0, dtype=bool), np.empty(0))


def tape_index(tape: pd.DataFrame) -> dict:
    """{(event_id, outcome): (lead_h, yes_price, is_sell)} over pre-kickoff prints.

    Which prints imply a fill depends on where they land relative to our bid at q:

      price < q   ANY side implies a fill. A resting bid at q outranks every price
                  below it, so for the book to print lower it must have traded
                  through us first -- a BUY printing under our bid would mean the ask
                  was below the bid, i.e. a crossed book.
      price == q  only a Yes-space aggressor SELL can reach us; a BUY at q lifted
                  somebody's ask and never touched the bid side.

    Hence two lows per leg (see build_quotes), matching the strict/--touch bracket.
    See odds/polymarket_trades for why the raw side column cannot be used unfolded.
    """
    pre = tape[tape["lead_h"] > 0]
    return {k: (g["lead_h"].to_numpy(), g["yes_price"].to_numpy(),
                (g["yes_side"] == "SELL").to_numpy(), g["size"].to_numpy())
            for k, g in pre.groupby(["event_id", "outcome"], sort=False)}


def fill_capacity(q: pd.DataFrame, quote: pd.Series, tape_idx: dict, strict: bool):
    """Size of the FIRST print to cross our quote -- what the fill is actually worth.

    A detected fill only means *a* print crossed us. If that print was 2 shares, the
    fill is real but worthless, so the size is the whole difference between a
    tradeable edge and a rounding artifact.

    Deliberately not the sum of every print below the quote: a resting order fills
    once. Once the market trades through us we are filled and holding, and the volume
    that keeps printing lower is not capacity we could have taken -- it is the market
    moving away from us. Summing it overstates the ticket by ~200x here.
    """
    out = np.full(len(q), np.nan)
    for i, (ev, name, post_lead) in enumerate(zip(q["event_id"], q["outcome"], q["post_lead_h"])):
        leads, px, is_sell, size = tape_idx.get((ev, name), NO_PRINTS)
        if not px.size:
            continue
        live = leads < post_lead
        hit = live & ((px < quote.iat[i]) if strict else (px <= quote.iat[i]) & is_sell)
        if not hit.any():
            continue
        out[i] = size[hit][np.argmax(leads[hit])]  # largest lead = earliest crossing
    return pd.Series(out, index=q.index)


def build_quotes(df, probs, snaps, lead_h: float, index=None, tape_idx: dict | None = None):
    """One record per (fixture, outcome): quote inputs, the future lows, result.

    ``path_low`` is the minimum implied probability for that outcome between the
    posting time and kickoff -- the deepest a resting bid could have been reached on
    the hourly curve. ``tick_low`` is the same thing measured on the trade tape: the
    lowest price any bid-hitting print actually crossed. tick_low <= path_low wherever
    the curve smoothed a dip away, and that gap is the whole question.
    """
    index = row_index(df) if index is None else index
    # Fixture-level tape coverage. A leg with no tape is NOT an unfilled leg -- it is
    # an unobserved one, and scoring it as unfilled would compare the two rules on
    # different samples and manufacture the entire result.
    tape_events = {e for e, _ in tape_idx} if tape_idx is not None else set()
    recs = []
    for (event_id, ko), g in snaps.groupby(["event_id", "commence_time"], sort=False):
        traded = g[g["traded"]]
        if traded.empty:
            continue
        post = traded.loc[(traded["lead_h"] - lead_h).abs().idxmin()]
        future = traded[traded["lead_h"] < post["lead_h"]]
        if future.empty:
            continue
        close = future.loc[future["lead_h"].idxmin()]
        idx = match_row(index, post["home_std"], post["away_std"], ko.strftime("%Y/%m/%d"))
        if idx is None or idx not in probs:
            continue
        for i, (name, col) in enumerate(zip(OUTCOMES, ODDS_COLS)):
            rec = {
                "event_id": event_id, "outcome": name,
                "post_lead_h": post["lead_h"],
                "mid": 1.0 / post[col],
                "model": probs[idx][i],
                "path_low": (1.0 / future[col]).min(),
                "close": 1.0 / close[col],
                "won": 1.0 if RESULT_IDX[df.loc[idx, "Res"]] == i else 0.0,
            }
            if tape_idx is not None:
                leads, px, is_sell, _sz = tape_idx.get((event_id, name), NO_PRINTS)
                live = leads < post["lead_h"]  # prints after we posted, before KO
                any_px, sell_px = px[live], px[live & is_sell]
                rec["tick_low"] = any_px.min() if any_px.size else np.nan
                rec["tick_sell_low"] = sell_px.min() if sell_px.size else np.nan
                rec["n_prints"] = int(any_px.size)
                rec["has_tape"] = event_id in tape_events
            recs.append(rec)
    return pd.DataFrame(recs)


def quote_price(q: pd.DataFrame, anchor: str, offset: float, mode: str) -> pd.Series:
    """The limit we post: anchor minus fixed cents, or anchor discounted for EV."""
    if mode == "ev":
        return q[anchor] / (1.0 + offset)
    return q[anchor] - offset / 100.0


def simulate(q: pd.DataFrame, anchor: str, offset: float, strict: bool = True,
             force_fill: bool = False, mode: str = "cents",
             low_col: str = "path_low") -> dict:
    """Fill every quote the market reached, settle it, and summarise.

    ``strict`` requires the market to trade strictly through the quote (a proxy for
    actually being consumed); otherwise a touch counts. Under the tick rule this is
    also the queue bracket: a print strictly below fills us from any queue position,
    a print exactly at the quote does not necessarily reach us.

    ``force_fill`` is the placebo: award every quote at its limit price whether or
    not the market ever reached it. That strips out the choice of who fills us and
    leaves only drift, so it isolates how much of the result is adverse selection
    rather than an artifact of conditioning on the path.
    """
    quote = quote_price(q, anchor, offset, mode)
    ok = (quote > 0.01) & (quote < 0.99)
    if force_fill:
        filled = ok
    else:
        low = q[low_col]
        filled = ok & ((low < quote) if strict else (low <= quote)).fillna(False)
    n_q, n_f = int(ok.sum()), int(filled.sum())
    if n_f < 2:
        return {"offset_c": offset, "n_quotes": n_q, "n_fills": n_f}

    px = quote[filled]
    adverse = (q["close"][filled] - px) * 100.0          # primary, low variance
    pnl = (q["won"][filled] - px) * 100.0                # secondary, high variance
    roi = (q["won"][filled] - px) / px

    def _t(a):
        se = a.std(ddof=1) / np.sqrt(len(a))
        return a.mean(), (a.mean() / se if se > 0 else np.nan)

    adv_m, adv_t = _t(adverse)
    pnl_m, pnl_t = _t(pnl)
    return {
        "offset_c": offset, "n_quotes": n_q, "n_fills": n_f,
        "fill_rate": 100.0 * n_f / n_q, "avg_px": px.mean(),
        "adverse_pp": adv_m, "adverse_t": adv_t,
        "pnl_pp": pnl_m, "pnl_t": pnl_t, "roi_pct": 100.0 * roi.mean(),
    }


def report(q: pd.DataFrame, anchor: str, label: str, strict: bool,
           mode: str = "cents", low_col: str = "path_low"):
    grid = EV_MARGINS if mode == "ev" else OFFSETS_C
    fmt = (lambda o: f"{o:>6.0%}") if mode == "ev" else (lambda o: f"{o:>6.0f}c")
    print(f"\n{'-'*104}")
    print(f"ANCHOR: {label}")
    print(f"{'-'*104}")
    print(f"{'offset':>7}{'quotes':>8}{'fills':>7}{'fill%':>7}{'avg px':>8}"
          f"{'(close-fill)':>14}{'t':>7}{'placebo':>10}{'selection':>11}"
          f"{'P/L pp':>9}{'t':>7}")
    for off in grid:
        r = simulate(q, anchor, off, strict, mode=mode, low_col=low_col)
        pl = simulate(q, anchor, off, strict, force_fill=True, mode=mode, low_col=low_col)
        if r.get("n_fills", 0) < 2:
            print(f"{fmt(off)} {r['n_quotes']:>7}{r.get('n_fills', 0):>7}{'-':>7}")
            continue
        star = "*" if abs(r["adverse_t"]) > 2 else " "
        pstar = "*" if abs(r["pnl_t"]) > 2 else " "
        # What conditioning on "someone chose to fill us" costs, over pure drift.
        selection = r["adverse_pp"] - pl["adverse_pp"]
        print(f"{fmt(off)} {r['n_quotes']:>7}{r['n_fills']:>7}{r['fill_rate']:>6.0f}%"
              f"{r['avg_px']:>8.3f}{r['adverse_pp']:>+13.2f}{star}{r['adverse_t']:>+6.2f}"
              f"{pl['adverse_pp']:>+10.2f}{selection:>+11.2f}"
              f"{r['pnl_pp']:>+9.2f}{pstar}{r['pnl_t']:>+6.2f}")


def report_fill_rules(q: pd.DataFrame, anchor: str, offset: float, mode: str, strict: bool,
                      tape_idx: dict | None = None):
    """The decisive comparison: does the trade tape find fills the curve missed?

    The pre-registered bar is arithmetic. Curve fills sit at some deficit below zero;
    any extra fills the tape reveals are drawn from the currently-unfilled quotes,
    whose mean (close - quote) is much better. Enough of them at that quality drags
    the average to breakeven -- so the required count is a number, not a judgement.
    """
    # Both rules must see the same legs, so drop fixtures the tape does not cover.
    if "has_tape" in q.columns:
        q = q[q["has_tape"]].reset_index(drop=True)
    quote = quote_price(q, anchor, offset, mode)
    ok = (quote > 0.01) & (quote < 0.99)
    adv = (q["close"] - quote) * 100.0
    off_s = f"{offset:.0%}" if mode == "ev" else f"{offset:.0f}c"
    print(f"\n{'='*104}")
    print(f"FILL-RULE COMPARISON  |  anchor={anchor}  offset={off_s}  "
          f"|  {q['event_id'].nunique()} tape-covered fixtures, {len(q)} legs")
    print(f"{'='*104}")
    print(f"{'rule':>12}{'fills':>8}{'fill%':>8}{'(close-fill)':>15}{'t':>8}{'vs path':>10}")
    base = None
    tick_col = "tick_low" if strict else "tick_sell_low"
    for rule, col in (("path (hourly)", "path_low"), ("tick (tape)", tick_col)):
        if col not in q.columns:
            continue
        r = simulate(q, anchor, offset, strict, mode=mode, low_col=col)
        if r.get("n_fills", 0) < 2:
            print(f"{rule:>12}{r.get('n_fills', 0):>8}{'-':>8}")
            continue
        delta = "" if base is None else f"{r['adverse_pp'] - base:>+10.2f}"
        base = r["adverse_pp"] if base is None else base
        print(f"{rule:>12}{r['n_fills']:>8}{r['fill_rate']:>7.0f}%"
              f"{r['adverse_pp']:>+14.2f}{r['adverse_t']:>+8.2f}{delta:>10}")

    if "tick_low" not in q.columns:
        return
    p = simulate(q, anchor, offset, strict, mode=mode, low_col="path_low")
    if p.get("n_fills", 0) < 2:
        return
    filled = ok & ((q["path_low"] < quote) if strict else (q["path_low"] <= quote)).fillna(False)
    unfilled_mean = adv[ok & ~filled].mean()
    deficit = -p["adverse_pp"] * p["n_fills"]
    need = deficit / unfilled_mean if unfilled_mean > 0 else float("inf")
    print(f"\n  unfilled pool (n={int((ok & ~filled).sum())}) mean (close-quote) "
          f"{unfilled_mean:+.2f}pp")
    print(f"  BAR: tape must add >= {np.ceil(need):.0f} fills at that quality "
          f"to reach breakeven ({100*np.ceil(need)/p['n_fills']:.0f}% more than the "
          f"{p['n_fills']} curve fills)")

    if tape_idx is None:
        return
    # How much size was actually behind the tape fills. A fill is only worth what
    # crossed it: median size is the honest per-bet ticket, not the mean.
    cap = fill_capacity(q, quote, tape_idx, strict).dropna()
    if cap.empty:
        return
    notional = cap * quote[cap.index]
    print(f"  TICKET at the {len(cap)} tape fills, size of the crossing print (shares): "
          f"median {cap.median():.1f}, p25 {cap.quantile(.25):.1f}, p75 {cap.quantile(.75):.1f}")
    print(f"       as notional at our limit ($): median {notional.median():.2f}, "
          f"total {notional.sum():.0f} across {q['event_id'].nunique()} fixtures")
    print(f"       fills under 10 shares: {100*(cap < 10).mean():.0f}%   "
          f"under 50 shares: {100*(cap < 50).mean():.0f}%")


def report_book_sweep(q: pd.DataFrame, anchor: str, offset_c: float, strict: bool):
    """How often all three legs fill -- the maker's structural prize.

    Owning all three outcomes costs sum(quotes) and always pays exactly 1, so any
    sweep with sum below 1 is locked-in profit that needs no view at all.
    """
    quote = q[anchor] - offset_c / 100.0
    q = q.assign(_q=quote,
                 _f=(q["path_low"] < quote) if strict else (q["path_low"] <= quote))
    g = q.groupby("event_id").agg(n_fill=("_f", "sum"), sum_q=("_q", "sum"),
                                  sum_filled=("_q", lambda s: s[q.loc[s.index, "_f"]].sum()))
    swept = g[g["n_fill"] == 3]
    print(f"\nall-three-legs sweep at {offset_c:.0f}c: {len(swept)}/{len(g)} fixtures", end="")
    if len(swept):
        print(f" | mean cost of the full book {swept['sum_q'].mean():.4f} "
              f"(pays 1.0000 => {100*(1-swept['sum_q'].mean()):+.2f}pp risk-free)")
    else:
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-start", default="2025-10-01")
    ap.add_argument("--lead", type=float, default=72.0, help="hours before kickoff to post quotes")
    ap.add_argument("--touch", action="store_true",
                    help="count a fill when the market merely touches the quote (optimistic)")
    ap.add_argument("--offset-mode", choices=("cents", "ev"), default="cents",
                    help="quote below the anchor by fixed cents, or by a proportional EV cushion")
    ap.add_argument("--ev-margin", type=float, default=0.20,
                    help="EV cushion used for the fill-rule comparison (--offset-mode ev)")
    ap.add_argument("--fill-rule", choices=("path", "tick", "both"), default="both",
                    help="detect fills on the hourly curve, the trade tape, or compare them")
    args = ap.parse_args()

    df = _load()
    probs = _model_probs(df, args.test_start)
    snaps = price_quality.load_polymarket()

    idx = None
    if args.fill_rule in ("tick", "both"):
        tape = polymarket_trades.load()
        if tape.empty:
            print("no trade tape cached -- run `python -m ligamx.odds.polymarket_trades` first; "
                  "falling back to the hourly curve only\n")
            args.fill_rule = "path"
        else:
            idx = tape_index(tape)
            print(f"trade tape: {len(tape)} prints over {tape['event_id'].nunique()} fixtures")

    q = build_quotes(df, probs, snaps, args.lead, row_index(df), idx)
    strict = not args.touch
    mode = args.offset_mode

    print(f"quotes posted at T-{args.lead:g}h | {q['event_id'].nunique()} fixtures, {len(q)} legs "
          f"| mean actual lead {q['post_lead_h'].mean():.1f}h "
          f"| fill: {'trades through' if strict else 'touches'} the quote")
    if "tick_low" in q.columns:
        cov = q["n_prints"].gt(0).mean()
        print(f"legs with any bid-hitting print after posting: {100*cov:.0f}% "
              f"(median {q['n_prints'].median():.0f} prints)")
    print("\n(close-fill) is the primary gate: positive => the market moved TO our price,")
    print("negative => we were picked off.  placebo = the same number when every quote is")
    print("awarded regardless of the path, so it carries drift but no selection; the")
    print("difference between them is the cost of only filling when someone wants to trade.")

    low_col = "path_low"
    if args.fill_rule == "tick":
        low_col = "tick_low" if strict else "tick_sell_low"
    report(q, "mid", "MARKET MID  (pure market making, no model -- the control)", strict, mode, low_col)
    report(q, "model", "MODEL FAIR  (quote around the model's probability)", strict, mode, low_col)

    if args.fill_rule == "both" and "tick_low" in q.columns:
        off = args.ev_margin if mode == "ev" else 2.0
        report_fill_rules(q, "model", off, mode, strict, idx)
        report_fill_rules(q, "mid", off, mode, strict, idx)

    if mode == "cents":
        for off in (2.0, 3.0):
            report_book_sweep(q, "mid", off, strict)


if __name__ == "__main__":
    main()
