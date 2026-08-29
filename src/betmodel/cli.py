"""One entry point for every league and every stage.

    betmodel leagues
    betmodel csl model
    betmodel ligamx capture-opens --dry-run
    betmodel all publish

Replaces a shell script per repository, each with its own verbs. The league list
is discovered from ``leagues/``, so a new league gets every command without this
file changing, which is the whole point of the merge.

``all`` as the league runs a stage for every league. A stage that fails for one
league does not stop the others: the leagues are independent, and a Liga MX
outage must not cost a Chinese Super League capture.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone

from betmodel.config import ConfigError, available_leagues, load_league

log = logging.getLogger("betmodel")

ALL = "all"


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #

def _fixtures(league: str, args) -> int:
    from betmodel.fixtures.sync import sync

    stats = sync(
        league, load_league(league), dry_run=args.dry_run,
        allow_empty_upcoming=args.allow_empty_upcoming,
    )
    log.info("%s: fixtures %s", league, stats)
    return 0


def _xg(league: str, args) -> int:
    from betmodel.xg.sync import sync

    stats = sync(league, load_league(league), dry_run=args.dry_run, limit=args.limit)
    log.info("%s: xg %s", league, stats)
    return 0


def _freshness(league: str, args) -> int:
    from betmodel.xg.freshness import check

    reading = check(league, load_league(league), dry_run=args.dry_run)
    return 0


def _model(league: str, args) -> int:
    from betmodel.models.run import run_model

    fit = run_model(league, write=not args.dry_run)
    log.info("%s: fitted %d matches over %d teams", league, fit.n_matches, len(fit.teams))
    return 0


def _capture_opens(league: str, args) -> int:
    from betmodel.odds.capture_open import capture_opens

    stats = capture_opens(
        league, load_league(league), dry_run=args.dry_run,
        providers=tuple(args.providers) if args.providers else ("oddsapiio", "theoddsapi"),
    )
    log.info("%s: opens %s", league, stats)
    return 0


def _capture_closes(league: str, args) -> int:
    from betmodel.odds.capture_close import capture_closes

    stats = capture_closes(league, load_league(league), dry_run=args.dry_run)
    log.info("%s: closes %s", league, stats)
    return 0


def _reduce(league: str, args) -> int:
    from betmodel.odds import reduce as reduce_module

    config = load_league(league)
    records = reduce_module.build_records(league, config)
    stats = reduce_module.merge(
        league, records, dry_run=args.dry_run, create_columns=args.create_columns
    )
    log.info("%s: reduce %s", league, stats)
    missing = reduce_module.missing_columns(league, records)
    if missing:
        log.warning("%s: captured data with nowhere to go: %s", league, missing)
    return 0


def _signals(league: str, args) -> int:
    from betmodel.publish import public
    from betmodel.signals.engine import build_signals

    config = load_league(league)
    signals = build_signals(league, config)
    firing = [s for s in signals if s.fires]
    log.info("%s: %d priced fixtures, %d firing", league, len(signals), len(firing))
    if args.dry_run:
        for signal in firing:
            log.info("  %s %s @ %.2f (ev %+.3f)", signal.fixture_id, signal.pick,
                     signal.best[signal.pick].odds, signal.ev or 0.0)
        return 0
    public.publish_league(league, config, signals, generated_at=_now())
    return 0


def _legacy(league: str, args) -> int:
    """Write the compatibility payload the downstream board still reads."""
    import json
    import os

    from betmodel import paths
    from betmodel.publish import legacy
    from betmodel.signals.engine import build_signals

    config = load_league(league)
    if not config.publish.published:
        log.info("%s: not published, writing no compatibility payload", league)
        return 0
    signals = build_signals(league, config)
    at = _now()
    # All three the board fetches. One it cannot do without, and two it asks for
    # on every tick: a file the consumer requests and does not get is a failed
    # fetch it has to tolerate on every poll.
    payloads = {
        "upcoming_market_comparison": legacy.market_comparison(config, signals, generated_at=at),
        "upcoming_fixtures": legacy.upcoming_fixtures(config, signals, generated_at=at),
        "match_predictions": legacy.match_predictions(config, signals, generated_at=at),
    }
    if args.dry_run:
        for name, payload in payloads.items():
            log.info("%s: %s %d rows", league, name, len(payload["rows"]))
        return 0
    lp = paths.for_league(league)
    lp.ensure_dirs()
    for name, payload in payloads.items():
        path = lp.public_legacy_json(name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=1)
            handle.write("\n")
    log.info("%s: wrote %d legacy payload(s)", league, len(payloads))
    return 0


def _notify(league: str, args) -> int:
    from betmodel.notify.telegram import notify

    sent = notify(league, load_league(league), dry_run=args.dry_run)
    log.info("%s: %d alert(s)", league, sent)
    return 0


def _publish(league: str, args) -> int:
    for stage in (_signals, _legacy):
        stage(league, args)
    return 0


def _all(league: str, args) -> int:
    """The full local chain, in dependency order."""
    for stage in (_fixtures, _xg, _freshness, _reduce, _model,
                  _signals, _legacy, _notify):
        stage(league, args)
    return 0


#: League-scoped stages.
STAGES = {
    "fixtures": _fixtures,
    "xg": _xg,
    "freshness": _freshness,
    "model": _model,
    "capture-opens": _capture_opens,
    "capture-closes": _capture_closes,
    "reduce": _reduce,
    "signals": _signals,
    "legacy": _legacy,
    "notify": _notify,
    "publish": _publish,
    "all": _all,
}

NOT_YET: dict[str, str] = {}


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="betmodel", description=__doc__)
    parser.add_argument("league", help=f"a league id, or {ALL!r}")
    parser.add_argument("stage", nargs="?", help=", ".join(sorted(STAGES)))
    parser.add_argument("--dry-run", action="store_true",
                        help="decide and report, write nothing and spend no quota")
    parser.add_argument("--providers", nargs="*",
                        help="restrict a capture to these providers")
    parser.add_argument("--create-columns", action="store_true",
                        help="let reduce add a missing column to the match table")
    parser.add_argument("--allow-empty-upcoming", action="store_true",
                        help="accept a provider returning no upcoming fixture "
                             "(season end); refused by default")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap how many per-match fetches one xG run makes")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(message)s",
    )

    if args.league == "leagues":
        for league in available_leagues():
            config = load_league(league)
            print(f"{config.id:10s} {config.name:24s} {config.season:15s} "
                  f"ev_min={config.signals.ev_min}")
        return 0

    if args.league == "index":
        from betmodel.publish import public

        print(public.publish_index(generated_at=_now()))
        return 0

    if not args.stage:
        print("a stage is required; one of: " + ", ".join(sorted(STAGES)), file=sys.stderr)
        return 2
    if args.stage in NOT_YET:
        print(f"{args.stage}: {NOT_YET[args.stage]}", file=sys.stderr)
        return 2
    if args.stage not in STAGES:
        print(f"unknown stage {args.stage!r}; one of: " + ", ".join(sorted(STAGES)),
              file=sys.stderr)
        return 2

    leagues = list(available_leagues()) if args.league == ALL else [args.league]
    for league in leagues:
        if league not in available_leagues():
            print(f"unknown league {league!r}; have: {', '.join(available_leagues())}",
                  file=sys.stderr)
            return 2

    failures = 0
    for league in leagues:
        try:
            STAGES[args.stage](league, args)
        except (ConfigError, Exception) as exc:  # noqa: BLE001
            # One league's outage must not cost another league's capture, and a
            # capture missed is not recoverable.
            failures += 1
            log.error("%s %s failed: %s", league, args.stage, exc)
            if len(leagues) == 1:
                raise
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
