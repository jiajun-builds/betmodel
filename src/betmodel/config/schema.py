"""Typed league configuration.

Every value here was a module-level constant in one of the two pre-merge repos.
The point of moving them is that the engine stops knowing which league it is
running: it reads a :class:`LeagueConfig` and behaves accordingly.

Validation is deliberately strict and happens at load time. A silently wrong
parameter is far more expensive than a refused start, because the pipeline runs
unattended in CI and publishes to a live board.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

# A book's role decides what its price is allowed to do, not merely where it
# came from. Kept as an enum-ish tuple so an unknown role is a load error.
BOOK_ROLES = ("bet", "anchor", "reference")

#: Stage network requirement. ``residential`` means the provider blocks
#: datacenter IPs, so the stage cannot run on a stock CI runner without a
#: residential egress path. This is a property of the provider, not the league.
NETWORKS = ("cloud", "residential")

DEBIAS_METHODS = ("none", "market_anchor")

LEGACY_CONTRACTS = ("per_book", "composite")

EV_UNITS = ("fraction", "percent")


class ConfigError(ValueError):
    """A league config is malformed. Always fatal; never degrade to a default."""


def _req(mapping: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"{where}: missing required key {key!r}")
    return mapping[key]


def _one_of(value: Any, allowed: Sequence[str], where: str) -> str:
    if value not in allowed:
        raise ConfigError(f"{where}: {value!r} is not one of {list(allowed)}")
    return str(value)


# --------------------------------------------------------------------------- #
# providers and sources
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ProviderConfig:
    """An upstream API and its league-specific parameters.

    Parameters stay an open mapping so a new provider can be introduced by
    adding a module plus YAML, without editing this schema. Access them through
    :meth:`require` so a missing one fails with the league's name attached.
    """

    name: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def require(self, key: str) -> Any:
        if key not in self.params:
            raise ConfigError(
                f"provider {self.name!r}: missing parameter {key!r}"
            )
        return self.params[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)


@dataclass(frozen=True)
class SourceConfig:
    """Where one kind of input data comes from, and whether CI can reach it."""

    provider: str
    network: str
    params: Mapping[str, Any] = field(default_factory=dict)

    @property
    def needs_residential_ip(self) -> bool:
        return self.network == "residential"

    def require(self, key: str) -> Any:
        if key not in self.params:
            raise ConfigError(
                f"source via {self.provider!r}: missing parameter {key!r}"
            )
        return self.params[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)

    @classmethod
    def parse(cls, raw: Mapping[str, Any], where: str) -> "SourceConfig":
        provider = str(_req(raw, "provider", where))
        network = _one_of(raw.get("network", "cloud"), NETWORKS, f"{where}.network")
        params = {k: v for k, v in raw.items() if k not in ("provider", "network")}
        return cls(provider=provider, network=network, params=params)


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class XgBlend:
    """Weights of the continuous scoring target ``ExpG+ = xg*xG + goals*G``.

    Both leagues fit on a blend rather than on goals, and both arrived at very
    different weights by walk-forward search (CSL 0.7/0.3, Liga MX 0.25/0.75).
    That divergence is real signal about the two leagues, not drift, so the
    weights stay per-league.
    """

    xg: float
    goals: float

    def __post_init__(self) -> None:
        total = self.xg + self.goals
        if abs(total - 1.0) > 1e-9:
            raise ConfigError(
                f"model.xg_blend must sum to 1.0, got {self.xg} + {self.goals} = {total}"
            )
        if self.xg < 0 or self.goals < 0:
            raise ConfigError("model.xg_blend weights must be non-negative")


@dataclass(frozen=True)
class Shrinkage:
    """Empirical-Bayes shrink of team coefficients, ``w = n / (n + k)``."""

    enabled: bool = False
    k: float = 6.0

    def __post_init__(self) -> None:
        if self.enabled and self.k <= 0:
            raise ConfigError("model.shrinkage.k must be > 0 when enabled")


@dataclass(frozen=True)
class ModelConfig:
    xi: float                       # Dixon-Coles time decay
    lookback_months: int            # training window
    min_train: int                  # refuse to fit below this many matches
    max_goals: int                  # scoreline grid truncation
    xg_blend: XgBlend
    shrinkage: Shrinkage
    draw_calibration_alpha: float   # 1.0 = identity; see note below
    ah_lines: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not 0 < self.xi < 1:
            raise ConfigError(f"model.xi out of range: {self.xi}")
        if self.lookback_months <= 0:
            raise ConfigError("model.lookback_months must be > 0")
        if self.min_train <= 0:
            raise ConfigError("model.min_train must be > 0")
        # The grid is truncated at max_goals; too small silently loses tail mass.
        if not 5 <= self.max_goals <= 40:
            raise ConfigError(f"model.max_goals out of range: {self.max_goals}")
        if self.draw_calibration_alpha <= 0:
            raise ConfigError("model.draw_calibration_alpha must be > 0")

    @property
    def draw_calibration_is_identity(self) -> bool:
        """At alpha 1.0 the calibration step is a no-op.

        Both leagues now sit at 1.0: the draw bias it used to correct turned out
        to be an artifact of a fitter that truncated non-integer goals. It is
        kept as a knob rather than deleted so the correction can be re-derived
        if a future league genuinely needs one.
        """
        return abs(self.draw_calibration_alpha - 1.0) < 1e-12

    @classmethod
    def parse(cls, raw: Mapping[str, Any], where: str) -> "ModelConfig":
        blend_raw = _req(raw, "xg_blend", where)
        shrink_raw = raw.get("shrinkage", {}) or {}
        return cls(
            xi=float(_req(raw, "xi", where)),
            lookback_months=int(_req(raw, "lookback_months", where)),
            min_train=int(raw.get("min_train", 100)),
            max_goals=int(_req(raw, "max_goals", where)),
            xg_blend=XgBlend(
                xg=float(_req(blend_raw, "xg", f"{where}.xg_blend")),
                goals=float(_req(blend_raw, "goals", f"{where}.xg_blend")),
            ),
            shrinkage=Shrinkage(
                enabled=bool(shrink_raw.get("enabled", False)),
                k=float(shrink_raw.get("k", 6.0)),
            ),
            draw_calibration_alpha=float(raw.get("draw_calibration_alpha", 1.0)),
            ah_lines=tuple(float(x) for x in raw.get("ah_lines", ())),
        )


# --------------------------------------------------------------------------- #
# odds
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class BookConfig:
    """One bookmaker, and what the pipeline is allowed to do with its price.

    ``role`` is the load-bearing field:

    * ``bet``       — a price we could actually take. EV is computed against it
                      and a signal can fire on it.
    * ``anchor``    — never bet and never displayed; feeds the de-bias step only.
    * ``reference`` — captured for research and CLV, shown at most as context.

    ``poll_interval_minutes`` replaces the two pre-merge capture mechanisms with
    one knob. The predicted-open-window machinery existed to ration a 500/month
    provider; measured usage was ~10% of that allowance while the window itself
    missed 17% of opens and backfilled them with a Now line, which is not an
    opening price. Uniform polling at a provider-appropriate cadence is both
    simpler and more accurate.
    """

    key: str                       # canonical stored key, e.g. "onexbet"
    provider: str                  # which provider module fetches it
    role: str
    provider_name: str = ""        # the provider's own spelling, e.g. "1xbet"
    label: str = ""                # human display name
    poll_interval_minutes: int | None = None   # None = not polled for opens
    #: How far ahead of kickoff this book is worth asking about. ``None`` falls
    #: back to ``odds.open.lookahead_days``.
    #:
    #: Per book because the books do not publish at the same time and the
    #: accounts do not bill at the same rate. Pinnacle does not price a fixture
    #: until roughly a week out -- measured median 6.1 days, max 7.0 -- so under a
    #: 21-day league lookahead every anchor slot spent a request asking about
    #: fixtures that provably could not be priced yet. On a monthly allowance that
    #: is where the allowance went, and it ran out at the end of the period
    #: exactly when the anchor was needed.
    lookahead_days: int | None = None
    credential: str = "default"    # which API account to spend
    schema_prefix: str | None = None   # column prefix in matches.csv; None = store only
    legacy_prefix: str | None = None   # published legacy column prefix
    legacy_prefix_note: str = ""       # required when legacy_prefix is non-default
    url: str = ""                      # where a human would go to take this price

    def __post_init__(self) -> None:
        where = f"odds.books[{self.key}]"
        _one_of(self.role, BOOK_ROLES, f"{where}.role")
        if not self.key or self.key != self.key.lower():
            raise ConfigError(f"{where}: key must be lowercase")
        if self.lookahead_days is not None and self.lookahead_days <= 0:
            raise ConfigError(f"{where}.lookahead_days must be > 0")
        if self.poll_interval_minutes is not None:
            if self.poll_interval_minutes <= 0:
                raise ConfigError(f"{where}.poll_interval_minutes must be > 0")
            # The interval is applied as "minutes since midnight is a multiple of
            # this", so one that does not divide the day evenly would fire at an
            # inconsistent time each day and skip the slot straddling midnight.
            if 1440 % self.poll_interval_minutes:
                raise ConfigError(
                    f"{where}.poll_interval_minutes must divide 1440 evenly, "
                    f"got {self.poll_interval_minutes}")
            # The timer fires every five minutes, so a book asking for a finer
            # cadence than that would simply never come due on some slots.
            if self.poll_interval_minutes % 5:
                raise ConfigError(
                    f"{where}.poll_interval_minutes must be a multiple of the "
                    f"5-minute timer tick, got {self.poll_interval_minutes}")
        # The published legacy column names are derived from the key downstream.
        # A prefix that does not follow "<key>_open" is a frozen legacy name and
        # silently blanks every price on the board if it drifts, so it has to be
        # declared with a reason rather than merely written down.
        if self.legacy_prefix is not None:
            if self.legacy_prefix != f"{self.key}_open" and not self.legacy_prefix_note:
                raise ConfigError(
                    f"{where}: legacy_prefix {self.legacy_prefix!r} differs from the "
                    f"derived {self.key}_open!r; set legacy_prefix_note to record why "
                    "it is frozen"
                )

    @property
    def effective_legacy_prefix(self) -> str:
        return self.legacy_prefix or f"{self.key}_open"

    @property
    def is_bet_book(self) -> bool:
        return self.role == "bet"

    @classmethod
    def parse(cls, raw: Mapping[str, Any], where: str) -> "BookConfig":
        key = str(_req(raw, "key", where))
        return cls(
            key=key,
            provider=str(_req(raw, "provider", where)),
            role=str(_req(raw, "role", where)),
            provider_name=str(raw.get("provider_name", "") or ""),
            label=str(raw.get("label", "") or key),
            poll_interval_minutes=(
                int(raw["poll_interval_minutes"])
                if raw.get("poll_interval_minutes") is not None else None
            ),
            lookahead_days=(
                int(raw["lookahead_days"])
                if raw.get("lookahead_days") is not None else None
            ),
            credential=str(raw.get("credential", "default")),
            schema_prefix=raw.get("schema_prefix", key),
            legacy_prefix=raw.get("legacy_prefix"),
            legacy_prefix_note=str(raw.get("legacy_prefix_note", "") or ""),
            url=str(raw.get("url", "") or ""),
        )


@dataclass(frozen=True)
class CloseConfig:
    """Closing-line capture: a burst inside a short pre-kickoff window.

    ``window_minutes`` is how early we start trying; ``target_minutes`` is how
    close to kickoff a capture has to land before the fixture is considered
    finalised and stops costing quota. Anything caught outside the window is
    discarded rather than stored: a price taken hours out is not a close, and
    storing one as if it were poisons every CLV number computed afterwards.
    """

    books: tuple[str, ...]
    window_minutes: float = 60.0
    target_minutes: float = 5.0
    credential: str = "default"
    min_remaining: int = 50

    def __post_init__(self) -> None:
        if not self.books:
            raise ConfigError("odds.close.books must not be empty")
        if self.window_minutes <= 0:
            raise ConfigError("odds.close.window_minutes must be > 0")
        if not 0 < self.target_minutes <= self.window_minutes:
            raise ConfigError(
                "odds.close.target_minutes must be > 0 and <= window_minutes"
            )

    @classmethod
    def parse(cls, raw: Mapping[str, Any], where: str) -> "CloseConfig":
        return cls(
            books=tuple(str(b) for b in _req(raw, "books", where)),
            window_minutes=float(raw.get("window_minutes", 60.0)),
            target_minutes=float(raw.get("target_minutes", 5.0)),
            credential=str(raw.get("credential", "default")),
            min_remaining=int(raw.get("min_remaining", 50)),
        )


@dataclass(frozen=True)
class OpenConfig:
    """Opening-line capture: continuous polling of a pending set.

    A ``(fixture, book)`` pair is pending from the moment the fixture enters the
    lookahead until the book prices it or the match kicks off. There is no
    predicted window: the first price we ever see for a pending pair is its
    opener, and ``capture_watch`` records that we were watching beforehand.
    """

    lookahead_days: int = 21
    max_requests: int = 4

    def __post_init__(self) -> None:
        if self.lookahead_days <= 0:
            raise ConfigError("odds.open.lookahead_days must be > 0")
        if self.max_requests <= 0:
            raise ConfigError("odds.open.max_requests must be > 0")

    @classmethod
    def parse(cls, raw: Mapping[str, Any], where: str) -> "OpenConfig":
        return cls(
            lookahead_days=int(raw.get("lookahead_days", 21)),
            max_requests=int(raw.get("max_requests", 4)),
        )


@dataclass(frozen=True)
class OddsConfig:
    books: tuple[BookConfig, ...]
    close: CloseConfig
    open: OpenConfig
    providers: Mapping[str, ProviderConfig] = field(default_factory=dict)
    quota_floor: int = 50

    def __post_init__(self) -> None:
        keys = [b.key for b in self.books]
        dupes = {k for k in keys if keys.count(k) > 1}
        if dupes:
            raise ConfigError(f"odds.books: duplicate keys {sorted(dupes)}")
        for b in self.books:
            if b.provider not in self.providers:
                raise ConfigError(
                    f"odds.books[{b.key}]: provider {b.provider!r} is not declared "
                    f"under odds.providers ({sorted(self.providers)})"
                )
        unknown = set(self.close.books) - set(keys)
        if unknown:
            raise ConfigError(
                f"odds.close.books references undeclared books: {sorted(unknown)}"
            )
        if not self.bet_books:
            raise ConfigError("odds.books: at least one book must have role 'bet'")

    @property
    def bet_books(self) -> tuple[BookConfig, ...]:
        """Books a signal may fire on, in declared order.

        Order is load-bearing: it sets published column order and breaks ties
        when two books quote the same best price.
        """
        return tuple(b for b in self.books if b.role == "bet")

    @property
    def polled_books(self) -> tuple[BookConfig, ...]:
        return tuple(b for b in self.books if b.poll_interval_minutes)

    def book(self, key: str) -> BookConfig:
        for b in self.books:
            if b.key == key:
                return b
        raise ConfigError(f"odds.books: no book with key {key!r}")

    @classmethod
    def parse(cls, raw: Mapping[str, Any], where: str) -> "OddsConfig":
        providers = {
            name: ProviderConfig(name=name, params=dict(params or {}))
            for name, params in (raw.get("providers") or {}).items()
        }
        return cls(
            books=tuple(
                BookConfig.parse(b, f"{where}.books") for b in _req(raw, "books", where)
            ),
            close=CloseConfig.parse(_req(raw, "close", where), f"{where}.close"),
            open=OpenConfig.parse(raw.get("open", {}) or {}, f"{where}.open"),
            providers=providers,
            quota_floor=int(raw.get("quota_floor", 50)),
        )


# --------------------------------------------------------------------------- #
# signals
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class DebiasConfig:
    """Market-anchored correction of the model's draw probability.

    ``market_anchor`` replaces the model draw with the anchor book's no-vig draw,
    shrunk by ``lam``; at ``lam = 1.0`` the model contributes nothing to the draw
    and only splits the remaining mass between home and away. ``none`` leaves the
    model untouched.
    """

    method: str = "none"
    lam: float = 1.0
    anchor_book: str | None = None

    def __post_init__(self) -> None:
        _one_of(self.method, DEBIAS_METHODS, "signals.debias.method")
        if self.method == "market_anchor":
            if not self.anchor_book:
                raise ConfigError(
                    "signals.debias: market_anchor requires anchor_book"
                )
            if not 0.0 <= self.lam <= 1.0:
                raise ConfigError(f"signals.debias.lam out of [0,1]: {self.lam}")

    @property
    def enabled(self) -> bool:
        return self.method != "none"

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "DebiasConfig":
        return cls(
            method=str(raw.get("method", "none")),
            lam=float(raw.get("lam", raw.get("lambda", 1.0))),
            anchor_book=raw.get("anchor_book"),
        )


@dataclass(frozen=True)
class SignalConfig:
    """When a modelled edge becomes a bet.

    ``ev_min`` is a fraction, never percentage points, everywhere inside the
    engine. The two pre-merge repos disagreed on this and the disagreement
    reached the board, which now guesses the unit from the data.
    """

    ev_min: float
    odds_cap: float | None = None
    allow_draw: bool = False
    require_price_proof: bool = False
    #: Weighted matches a team needs before a fixture of theirs may fire.
    #:
    #: Shrinkage regularises a thin rating; it does not make it trustworthy. The
    #: empirical-Bayes target is the *league* mean, and for a promoted side that is
    #: the wrong prior -- it pulls a team we have barely seen toward average and
    #: therefore overrates them. Measured on Atlante, six matches in: the model put
    #: them 6.0 points above Pinnacle's no-vig line on their own fixture, which
    #: turned a -1.5% price into a +21.3% signal.
    #:
    #: Weighted rather than raw, because that is what the coefficients actually
    #: rest on. For the case this exists for -- a side promoted mid-window, whose
    #: matches are all recent and barely decayed -- the two agree within 5%
    #: (Atlante: 6 raw, 5.8 weighted), while weighting also covers a team that has
    #: stopped playing.
    #:
    #: Zero disables the check.
    min_team_evidence: float = 0.0
    debias: DebiasConfig = field(default_factory=DebiasConfig)

    def __post_init__(self) -> None:
        # A threshold above 1.0 almost certainly means someone wrote percent.
        if not 0.0 < self.ev_min < 1.0:
            raise ConfigError(
                f"signals.ev_min must be a fraction in (0,1), got {self.ev_min}; "
                "percentage points are not accepted"
            )
        if self.odds_cap is not None and self.odds_cap <= 1.0:
            raise ConfigError(f"signals.odds_cap must be > 1.0, got {self.odds_cap}")
        if self.min_team_evidence < 0:
            raise ConfigError("signals.min_team_evidence must be >= 0")

    @property
    def sides(self) -> tuple[str, ...]:
        return ("home", "draw", "away") if self.allow_draw else ("home", "away")

    @classmethod
    def parse(cls, raw: Mapping[str, Any], where: str) -> "SignalConfig":
        return cls(
            ev_min=float(_req(raw, "ev_min", where)),
            odds_cap=(float(raw["odds_cap"]) if raw.get("odds_cap") is not None else None),
            allow_draw=bool(raw.get("allow_draw", False)),
            require_price_proof=bool(raw.get("require_price_proof", False)),
            min_team_evidence=float(raw.get("min_team_evidence", 0.0)),
            debias=DebiasConfig.parse(raw.get("debias", {}) or {}),
        )


# --------------------------------------------------------------------------- #
# publish / notify
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PublishConfig:
    """How this league appears in the published contract.

    ``legacy_contract`` and ``legacy_ev_unit`` describe only the compatibility
    files under ``public/legacy/``, which exist so the downstream board keeps
    working across the cutover. The canonical contract under ``public/`` is
    identical for every league and always expresses EV as a fraction.

    ``published`` is the difference between a league that exists in this repo and
    a league that is in production. The capture timer discovers its work from the
    published manifest, so a league added for development is otherwise dispatched
    for real captures the moment it is published, against credentials nobody set
    up for it. It is a separate question from ``validated``: that one warns a
    reader about the numbers, this one decides whether there is anything for a
    reader to see.
    """

    legacy_contract: str
    legacy_ev_unit: str = "fraction"
    validated: bool = False
    caveat: str = ""
    published: bool = True

    def __post_init__(self) -> None:
        _one_of(self.legacy_contract, LEGACY_CONTRACTS, "publish.legacy_contract")
        _one_of(self.legacy_ev_unit, EV_UNITS, "publish.legacy_ev_unit")

    @classmethod
    def parse(cls, raw: Mapping[str, Any], where: str) -> "PublishConfig":
        return cls(
            legacy_contract=str(_req(raw, "legacy_contract", where)),
            legacy_ev_unit=str(raw.get("legacy_ev_unit", "fraction")),
            validated=bool(raw.get("validated", False)),
            caveat=str(raw.get("caveat", "") or ""),
            published=bool(raw.get("published", True)),
        )


@dataclass(frozen=True)
class NotifyConfig:
    telegram: bool = False

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "NotifyConfig":
        return cls(telegram=bool(raw.get("telegram", False)))


# --------------------------------------------------------------------------- #
# the league
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class LeagueConfig:
    """Everything that makes one league different from another."""

    id: str
    name: str
    code: str
    season: str
    timezone: str
    total_rounds: int
    sources: Mapping[str, SourceConfig]
    model: ModelConfig
    odds: OddsConfig
    signals: SignalConfig
    publish: PublishConfig
    notify: NotifyConfig = field(default_factory=NotifyConfig)
    model_name: str = ""
    model_version: str = ""

    def __post_init__(self) -> None:
        for required in ("fixtures", "xg"):
            if required not in self.sources:
                raise ConfigError(f"sources: missing required source {required!r}")
        # A de-bias anchor has to name a book that exists and is declared an
        # anchor. Pointing it at a bet book would let the price we intend to
        # take also define the probability we price it against.
        anchor = self.signals.debias.anchor_book
        if anchor:
            book = self.odds.book(anchor)
            if book.role != "anchor":
                raise ConfigError(
                    f"signals.debias.anchor_book {anchor!r} has role {book.role!r}; "
                    "the de-bias anchor must be declared role 'anchor'"
                )
        if self.total_rounds <= 0:
            raise ConfigError("total_rounds must be > 0")

    @property
    def residential_stages(self) -> tuple[str, ...]:
        """Stages that cannot run on a stock CI runner.

        Drives which workflow job needs a residential egress path, so the answer
        is data rather than a hardcoded league name in a YAML workflow.
        """
        return tuple(
            sorted(k for k, s in self.sources.items() if s.needs_residential_ip)
        )

    @classmethod
    def parse(cls, raw: Mapping[str, Any], where: str) -> "LeagueConfig":
        sources = {
            kind: SourceConfig.parse(cfg or {}, f"{where}.sources.{kind}")
            for kind, cfg in (_req(raw, "sources", where)).items()
        }
        return cls(
            id=str(_req(raw, "id", where)),
            name=str(_req(raw, "name", where)),
            code=str(_req(raw, "code", where)),
            season=str(_req(raw, "season", where)),
            timezone=str(_req(raw, "timezone", where)),
            total_rounds=int(_req(raw, "total_rounds", where)),
            sources=sources,
            model=ModelConfig.parse(_req(raw, "model", where), f"{where}.model"),
            odds=OddsConfig.parse(_req(raw, "odds", where), f"{where}.odds"),
            signals=SignalConfig.parse(_req(raw, "signals", where), f"{where}.signals"),
            publish=PublishConfig.parse(_req(raw, "publish", where), f"{where}.publish"),
            notify=NotifyConfig.parse(raw.get("notify", {}) or {}),
            model_name=str(raw.get("model_name", "") or ""),
            model_version=str(raw.get("model_version", "") or ""),
        )
