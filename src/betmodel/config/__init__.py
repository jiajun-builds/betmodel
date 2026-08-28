"""League configuration: the schema, and the loader that validates it.

Adding a league should mean adding one file under ``leagues/`` and nothing else.
Everything a league differs by therefore has to be expressible here.
"""

from betmodel.config.schema import (
    BookConfig,
    LeagueConfig,
    ModelConfig,
    OddsConfig,
    ProviderConfig,
    SignalConfig,
    SourceConfig,
)
from betmodel.config.loader import (
    ConfigError,
    available_leagues,
    load_all,
    load_league,
)

__all__ = [
    "BookConfig", "LeagueConfig", "ModelConfig", "OddsConfig", "ProviderConfig",
    "SignalConfig", "SourceConfig",
    "ConfigError", "available_leagues", "load_all", "load_league",
]
