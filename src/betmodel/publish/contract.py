"""The published contract, version 1.

This tree is the repository's only public surface. Everything else here is an
implementation detail; these files are an interface, and the point of writing it
down is that a consumer should never have to read the producer to use it.

Three properties the pre-merge payloads did not have.

**One shape for every league.** Adding a league adds a directory, not a consumer
adapter. The two pre-merge shapes needed two adapters downstream and a heuristic
to tell them apart.

**One unit and one clock.** Expected value is a fraction, always. Timestamps are
UTC with a ``Z``, always, in fields whose names say so. The old shapes expressed
value in different units and used ``kickoff_at`` and ``match_time`` with opposite
meanings, so a consumer reading them as one contract was reading two.

**A manifest.** ``index.json`` lists the leagues and where their files are, so a
consumer hardcodes one address instead of a league list. That is what makes a new
league cost nothing downstream.

``schema`` is on every file. A consumer that reads it can accept two versions at
once, which is what lets the producer and the consumer be deployed separately
rather than in the same second.
"""

from __future__ import annotations

from datetime import datetime, timezone

SCHEMA_VERSION = 1

#: Files published per league. The manifest names them, so a consumer never
#: builds these paths itself, and never has to guess whether one exists.
#:
#: Three, not six. The agreed scope is signals and results, the pair that closes
#: the monitoring loop, plus the staleness metadata without which a consumer can
#: see a signal and not tell whether the model behind it was fitted last night or
#: three weeks ago. Fixtures, predictions and strength still ship in the legacy
#: tree. A manifest that named files nobody writes would break the consumer it
#: exists to serve.
LEAGUE_FILES = ("signals", "results", "meta")

SIDES = ("home", "draw", "away")

#: Signal states. Empty means no edge cleared the bar.
STATE_BET = "bet"
STATE_ODDS_CAP = "odds_cap"
#: Would have fired, but the de-bias anchor was missing when it ran. Published
#: so the board can show it and say why, never with a `bet` attached.
STATE_UNANCHORED = "unanchored"
STATE_NONE = ""
STATES = (STATE_BET, STATE_ODDS_CAP, STATE_UNANCHORED, STATE_NONE)


class ContractError(ValueError):
    """A payload does not satisfy the contract. Never published."""


def utc(moment: datetime | None) -> str | None:
    """A timestamp in the one format this contract uses."""
    if moment is None:
        return None
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #

def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def validate_signals(payload: dict) -> None:
    """Check a signals payload before it is written.

    Validation runs before writing, not after, so a payload that would mislead a
    consumer never reaches the tree at all. The checks are the ones that would be
    silent otherwise: a probability triple that does not sum to one still renders,
    an expected value in percentage points still renders, and a duplicate fixture
    id still renders while quietly making the file self-contradictory.
    """
    _require(payload.get("schema") == SCHEMA_VERSION,
             f"signals: schema must be {SCHEMA_VERSION}, got {payload.get('schema')!r}")
    _require(isinstance(payload.get("signals"), list), "signals: 'signals' must be a list")

    seen: set[str] = set()
    for signal in payload["signals"]:
        fixture_id = signal.get("fixture_id")
        _require(bool(fixture_id), "signals: a row has no fixture_id")
        _require(fixture_id not in seen, f"signals: duplicate fixture_id {fixture_id!r}")
        seen.add(fixture_id)

        _require(str(signal.get("kickoff_utc", "")).endswith("Z"),
                 f"signals: {fixture_id} kickoff_utc must be UTC with a Z suffix")

        model = signal.get("model") or {}
        probabilities = [model.get(side) for side in SIDES]
        _require(all(isinstance(p, (int, float)) for p in probabilities),
                 f"signals: {fixture_id} has a non-numeric probability")
        total = sum(probabilities)
        _require(abs(total - 1.0) < 1e-6,
                 f"signals: {fixture_id} probabilities sum to {total:.6f}, not 1")

        for quote in signal.get("quotes", []):
            _require(quote.get("side") in SIDES,
                     f"signals: {fixture_id} quote has side {quote.get('side')!r}")
            _require(isinstance(quote.get("odds"), (int, float)) and quote["odds"] > 1.0,
                     f"signals: {fixture_id} quote has unusable odds")
            value = quote.get("ev")
            _require(isinstance(value, (int, float)),
                     f"signals: {fixture_id} quote has a non-numeric ev")
            # A fraction, never percentage points. An EV above 5 is a unit error,
            # not a 500% edge, and it is the mistake the old contract invited.
            _require(-1.0 <= value <= 5.0,
                     f"signals: {fixture_id} ev {value} is outside a plausible "
                     "fractional range; percentage points are not accepted")

        _require(signal.get("state") in STATES,
                 f"signals: {fixture_id} has state {signal.get('state')!r}")
        bet = signal.get("bet")
        _require((signal["state"] == STATE_BET) == (bet is not None),
                 f"signals: {fixture_id} state and bet disagree")
        if bet is not None:
            _require(bool(bet.get("books")),
                     f"signals: {fixture_id} fires but names no book to bet with")


def validate_results(payload: dict) -> None:
    _require(payload.get("schema") == SCHEMA_VERSION, "results: wrong schema")
    _require(isinstance(payload.get("results"), list), "results: 'results' must be a list")
    seen: set[str] = set()
    for result in payload["results"]:
        fixture_id = result.get("fixture_id")
        _require(bool(fixture_id), "results: a row has no fixture_id")
        _require(fixture_id not in seen, f"results: duplicate fixture_id {fixture_id!r}")
        seen.add(fixture_id)
        _require(result.get("status") in ("played", "scheduled"),
                 f"results: {fixture_id} has status {result.get('status')!r}")
        if result["status"] == "played":
            _require(result.get("home_goals") is not None
                     and result.get("away_goals") is not None,
                     f"results: {fixture_id} is played but has no score")
            _require(result.get("result") in ("H", "D", "A"),
                     f"results: {fixture_id} has result {result.get('result')!r}")


def validate_index(payload: dict) -> None:
    _require(payload.get("schema") == SCHEMA_VERSION, "index: wrong schema")
    leagues = payload.get("leagues")
    _require(isinstance(leagues, list) and leagues, "index: 'leagues' must be a non-empty list")
    seen: set[str] = set()
    for league in leagues:
        league_id = league.get("id")
        _require(bool(league_id), "index: a league has no id")
        _require(league_id not in seen, f"index: duplicate league {league_id!r}")
        seen.add(league_id)
        files = league.get("files") or {}
        missing = [name for name in LEAGUE_FILES if name not in files]
        _require(not missing, f"index: {league_id} does not name {missing}")
        # A consumer that hardcodes a threshold cannot follow a league that
        # changes one. Publishing it is what stops the board and the producer
        # drifting apart, which they already had.
        _require(isinstance(league.get("ev_min"), (int, float)),
                 f"index: {league_id} does not publish its signal threshold")
