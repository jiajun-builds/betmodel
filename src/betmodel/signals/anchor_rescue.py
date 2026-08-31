"""Fetch the de-bias anchor for an edge that cleared the bar without one.

A fixture's prices arrive over a window. The bet books are polled every few
minutes because they are cheap; the anchor is polled slowly because it is
metered. So a newly listed match can be priced, and can clear the signal
threshold, minutes before the anchor is due -- and the engine now refuses to fire
on the raw grid, which means that edge is stranded until the next anchor tick,
up to three hours later.

This closes that window by asking the question the other way round: instead of
polling the anchor often enough to be lucky, poll it exactly when there is
something at stake. If any fixture is sitting in `unanchored`, the anchor is
worth a request right now.

**Spend is bounded by the trigger, not by a timer.** The caller runs this only on
a tick where the capture actually appended something -- the same gate the
republish already uses -- so the number of attempts for a stranded fixture is
bounded by the number of price appends, not by the clock. A fixture the anchor
book has genuinely not listed yet is retried when the next book opens on it and
then stops, rather than looping every five minutes until kickoff.

One request covers the whole league: the provider returns every event, so the
cost of rescuing one stranded fixture and rescuing five is identical.
"""

from __future__ import annotations

import logging

from betmodel.config.schema import LeagueConfig
from betmodel.odds.capture_open import capture_opens
from betmodel.signals import debias
from betmodel.signals.engine import STATE_UNANCHORED, build_signals

log = logging.getLogger(__name__)


def stranded_fixtures(league: str, config: LeagueConfig, **kwargs) -> list:
    """Signals that cleared the bar but could not be calibrated."""
    if config.signals.debias.method != debias.MARKET_ANCHOR:
        return []
    return [
        s for s in build_signals(league, config, **kwargs)
        if s.state == STATE_UNANCHORED
    ]


def rescue(
    league: str,
    config: LeagueConfig,
    *,
    dry_run: bool = False,
    capture=capture_opens,
    **kwargs,
) -> dict[str, int]:
    """Fetch the anchor if, and only if, an edge is stranded without one.

    Returns what it did. Never raises for the ordinary reasons -- a league with
    no anchor configured and a league with nothing stranded both mean the same
    thing here, which is "spend nothing".
    """
    stranded = stranded_fixtures(league, config, **kwargs)
    if not stranded:
        log.info("%s: no edge is waiting on an anchor; spending nothing", league)
        return {"stranded": 0, "fetched": 0}

    anchor_key = config.signals.debias.anchor_book
    anchor = next((b for b in config.odds.books if b.key == anchor_key), None)
    if anchor is None:
        # The config validator refuses market_anchor without an anchor_book, so
        # reaching here means the book was renamed out from under it.
        log.warning("%s: anchor book %r is not in the book list", league, anchor_key)
        return {"stranded": len(stranded), "fetched": 0}

    for signal in stranded:
        log.info("%s: stranded without an anchor: %s v %s (EV %.4f)",
                 league, signal.home_team, signal.away_team, signal.ev or 0.0)

    # `ignore_schedule` because the whole point is that the anchor's own interval
    # has not come round. `pending_fixtures` still gates it: a league whose
    # fixtures all have an anchor open spends nothing even when called.
    stats = capture(
        league, config,
        providers=(anchor.provider,),
        ignore_schedule=True,
        dry_run=dry_run,
        **kwargs,
    )
    return {"stranded": len(stranded), "fetched": 1, **stats}
