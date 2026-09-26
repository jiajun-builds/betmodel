#!/usr/bin/env python3
"""Walk-forward replay of the production fit, for trialling a model setting.

Before every match date the production recipe is refitted on strictly earlier
matches, and that date's fixtures are priced and calibrated exactly as the engine
calibrates them. Each season is then scored three ways:

* **against Pinnacle's no-vig close**, on the home share ``p_home / (p_home +
  p_away)``. That split is all the model contributes once the anchor has set the
  draw, and the close is a far less noisy yardstick than the result.
* **against the result**, by log loss.
* **by the signal rule replayed on the bet books' historical openers**, scored by
  closing-line value: the Pinnacle no-vig close times the price taken, minus one.
  That is how thresholds here are validated (D31); profit is too noisy.

It imports the recipe rather than restating it -- the fit, the blend, the devig
and the draw anchor -- because a copy is free to drift, and five copies of the
fit once agreed on a 27% error. The replay has two known gaps against
production. Historical openers carry no capture proof, so `require_price_proof`
is not applied. Fixtures with no Pinnacle opener are priced raw rather than
refused, which is how the uncalibrated rule was measured in D31.

    python scripts/walkforward.py csl --xi 0.004
    python scripts/walkforward.py ligamx --blend 0.5 --start 2024-01-01
    python scripts/walkforward.py ligamx --book duel

Research only: it reads the committed tables and writes nothing.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from betmodel import paths  # noqa: E402
from betmodel.config import load_league  # noqa: E402
from betmodel.config.schema import LeagueConfig  # noqa: E402
from betmodel.dates import parse_date_only_series  # noqa: E402
from betmodel.models.dc import (  # noqa: E402
    FitError, fit_production_model, prepare_training_frame,
)
from betmodel.signals import debias  # noqa: E402
from betmodel.signals.ev import devig  # noqa: E402
from betmodel.xg.blend import recompute  # noqa: E402

RESULT = {"H": 0, "D": 1, "A": 2}
SIDES = ("home", "draw", "away")
LEGS = ("h", "d", "a")
START = {"csl": "2024-03-01", "ligamx": "2024-01-01"}


def configure(config: LeagueConfig, *, xi=None, blend=None, shrink=None) -> LeagueConfig:
    """The league's config with any trialled model setting replaced.

    ``shrink`` is a k to shrink with, or 0 to turn shrinkage off.
    """
    model = config.model
    changes = {}
    if xi is not None:
        changes["xi"] = xi
    if blend is not None:
        changes["xg_blend"] = dataclasses.replace(
            model.xg_blend, xg=blend, goals=round(1.0 - blend, 6))
    if shrink is not None:
        changes["shrinkage"] = dataclasses.replace(
            model.shrinkage, enabled=shrink > 0, k=shrink or model.shrinkage.k)
    return dataclasses.replace(config, model=dataclasses.replace(model, **changes))


def predict(league: str, config: LeagueConfig, start: str,
            matches_path: str | None = None) -> pd.DataFrame:
    """Played matches from ``start`` on, each priced by a fit on earlier ones.

    Adds the raw 1X2 as ``raw_h/raw_d/raw_a`` and the weaker side's weighted
    evidence as ``evidence``. A club absent from the training window is skipped,
    as the engine skips it.
    """
    frame = pd.read_csv(matches_path or paths.for_league(league).matches_csv)
    frame, _ = recompute(frame, config)  # the target under this config's blend
    frame["_day"] = parse_date_only_series(frame["Date"])
    frame = frame[frame["Res"].isin(RESULT)].sort_values("_day", kind="stable")

    priced = []
    for day in sorted(frame.loc[frame["_day"] >= pd.Timestamp(start), "_day"].unique()):
        try:
            train = prepare_training_frame(frame[frame["_day"] < day].drop(columns="_day"),
                                           config)
            fit = fit_production_model(train, config)
        except FitError:
            continue
        evidence = dict(zip(fit.teams, fit.model.appearances()[1]))
        for index, row in frame[frame["_day"] == day].iterrows():
            if row["Home"] not in evidence or row["Away"] not in evidence:
                continue
            raw = fit.model.outcome_probabilities(row["Home"], row["Away"])
            priced.append({
                "index": index, "raw_h": raw.home, "raw_d": raw.draw, "raw_a": raw.away,
                "evidence": min(evidence[row["Home"]], evidence[row["Away"]]),
            })
    return frame.join(pd.DataFrame(priced).set_index("index"), how="inner")


def _prices(row, prefix: str, phase: str):
    odds = tuple(row.get(f"{prefix}_{phase}_{leg}") for leg in LEGS)
    return odds if all(pd.notna(o) and o > 1.0 for o in odds) else None


def score(league: str, config: LeagueConfig, frame: pd.DataFrame,
          book: str | None = None) -> pd.DataFrame:
    """One row per fixture: calibrated probabilities, the close, and any bet.

    ``book`` replays the rule on that one bet book's prices, rather than on the
    best of all of them -- how a league earns back a book it was paused on (D35).
    """
    anchor = config.odds.book(config.signals.debias.anchor_book).schema_prefix \
        if config.signals.debias.anchor_book else None
    books = [b.schema_prefix for b in config.odds.bet_books
             if b.schema_prefix and (book is None or b.key == book)]
    signals = config.signals
    rows = []
    for _, row in frame.iterrows():
        raw = (row["raw_h"], row["raw_d"], row["raw_a"])
        opener = _prices(row, anchor, "open") if anchor else None
        probs, _ = debias.apply(raw, signals.debias, anchor_odds=opener)
        close_odds = _prices(row, "pinnacle", "close")
        close = devig(close_odds) if close_odds else None
        out = {"season": row["Season"], "result": RESULT[row["Res"]],
               "p_h": probs[0], "p_d": probs[1], "p_a": probs[2],
               "c_h": np.nan, "c_d": np.nan, "c_a": np.nan, "clv": np.nan, "pnl": np.nan,
               "side": ""}
        if close:
            out.update(c_h=close[0], c_d=close[1], c_a=close[2])
            # The rule, as `engine._decide` applies it: best price per side,
            # highest-EV side, then the bar, then the long-shot cap.
            best = {}
            for i, side in enumerate(SIDES):
                if side not in signals.sides:
                    continue
                quoted = [row.get(f"{p}_open_{LEGS[i]}") for p in books]
                quoted = [o for o in quoted if pd.notna(o) and o > 1.0]
                if quoted:
                    best[i] = max(quoted)
            if best:
                pick = max(best, key=lambda i: probs[i] * best[i])
                odds = best[pick]
                fires = (
                    probs[pick] * odds - 1.0 > signals.ev_min
                    and (signals.odds_cap is None or odds <= signals.odds_cap)
                    and row["evidence"] >= signals.min_team_evidence
                )
                if fires:
                    out.update(side=SIDES[pick], clv=close[pick] * odds - 1.0,
                               pnl=odds - 1.0 if out["result"] == pick else -1.0)
        rows.append(out)
    return pd.DataFrame(rows)


def _interval(values: np.ndarray, draws: int = 2000) -> tuple[float, float]:
    if len(values) < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(0)
    means = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(draws)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def report(scored: pd.DataFrame) -> pd.DataFrame:
    """Per season and overall: the three scores."""
    def summarise(group: pd.DataFrame) -> dict:
        closed = group.dropna(subset=["c_h"])
        share = closed["p_h"] / (closed["p_h"] + closed["p_a"])
        target = closed["c_h"] / (closed["c_h"] + closed["c_a"])
        probs = group[["p_h", "p_d", "p_a"]].to_numpy()
        taken = probs[np.arange(len(group)), group["result"].to_numpy()]
        bets = group.dropna(subset=["clv"])
        low, high = _interval(bets["clv"].to_numpy())
        return {
            "fixtures": len(group), "with_close": len(closed),
            "xent_close": float(-(target * np.log(share)
                                  + (1 - target) * np.log(1 - share)).mean()),
            "log_loss": float(-np.log(taken).mean()),
            "bets": len(bets), "draws": int((bets["side"] == "draw").sum()),
            "clv": float(bets["clv"].mean()) if len(bets) else float("nan"),
            "clv_low": low, "clv_high": high, "clv_sum": float(bets["clv"].sum()),
            "roi": float(bets["pnl"].mean()) if len(bets) else float("nan"),
        }
    rows = {season: summarise(g) for season, g in scored.groupby("season", sort=False)}
    rows["all"] = summarise(scored)
    return pd.DataFrame.from_dict(rows, orient="index")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("league")
    parser.add_argument("--xi", type=float)
    parser.add_argument("--blend", type=float, help="xG weight of the target, 0..1")
    parser.add_argument("--shrink", type=float, help="shrinkage k, or 0 for none")
    parser.add_argument("--start", help="first match date priced")
    parser.add_argument("--book", help="replay on this bet book's prices only")
    args = parser.parse_args(argv)

    logging.disable(logging.WARNING)
    config = configure(load_league(args.league), xi=args.xi, blend=args.blend,
                       shrink=args.shrink)
    frame = predict(args.league, config, args.start or START[args.league])
    table = report(score(args.league, config, frame, book=args.book))
    model = config.model
    print(f"{args.league}: xi={model.xi} blend={model.xg_blend.xg} "
          f"shrink={model.shrinkage.k if model.shrinkage.enabled else 'off'} "
          f"ev_min={config.signals.ev_min} sides={','.join(config.signals.sides)}"
          f"{f' book={args.book}' if args.book else ''}")
    print(table.to_string(float_format="{:.4f}".format))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
