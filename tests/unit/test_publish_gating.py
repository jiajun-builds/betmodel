"""A league in this repository is not automatically a league in production.

Adding a third league proved the abstraction, and then published it, and the
capture timer — which discovers its work from the manifest, exactly as designed —
started dispatching real captures for a league that had no odds-api.io
credential. Every open tick for it failed. The manifest is the enlistment, so
the flag that keeps a league out of it has to be honoured in all three places
that write the public tree.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

from betmodel.config import load_all, load_league
from betmodel.publish import public

AT = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)


def test_published_defaults_to_true():
    # A league that says nothing about it is a production league, so an existing
    # config cannot be silently withdrawn by an upgrade.
    from betmodel.config.schema import PublishConfig

    parsed = PublishConfig.parse({"legacy_contract": "composite"}, "publish")
    assert parsed.published is True


def _withheld_copy(league: str):
    """A real config with only its published flag flipped.

    Built rather than borrowed from disk on purpose. These tests used to require
    that some league on disk was unpublished, which was true only while the
    acceptance-test league existed; deleting it would have made one test vacuous
    and crashed the other. What is under test is the flag, not the roster.
    """
    config = load_league(league)
    return replace(config, publish=replace(config.publish, published=False))


def test_an_unpublished_league_is_absent_from_the_manifest():
    configs = dict(load_all())
    victim = sorted(configs)[0]
    configs[victim] = _withheld_copy(victim)
    named = {entry["id"] for entry in public.index_payload(configs, AT)["leagues"]}
    assert victim not in named
    # The invariant is "exactly the published ones", not "everything but the
    # victim" -- otherwise the test depends on no other league being withheld.
    assert named == {k for k, c in configs.items() if c.publish.published}


def test_an_unpublished_league_writes_no_canonical_files(tmp_path):
    league = sorted(load_all())[0]
    written = public.publish_league(
        league, _withheld_copy(league), [], generated_at=AT)
    assert written == {}, "a withheld league must not write, even over its own files"


def test_every_league_the_manifest_names_declares_its_files():
    # The consumer builds no paths of its own, so a named league must carry the
    # full file map or the board is left guessing.
    for entry in public.index_payload(load_all(), AT)["leagues"]:
        assert set(entry["files"]) == set(
            __import__("betmodel.publish.contract", fromlist=["x"]).LEAGUE_FILES
        )
