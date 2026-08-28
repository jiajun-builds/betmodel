"""Tests for the gate that decides whether a captured price may fire a signal.

The strategy behind SIGNAL_EV_THRESHOLD was fitted on prices proven to be
openers. A price captured from a market that was already quoting is out of
sample, and out of sample in the direction that loses money: an already-open
price is a sharp one, and EV measured against a sharp price is model-vs-market
disagreement, which this project has separately measured at negative CLV.

So the gate has two failure modes and both are expensive:

1. Too loose -- a first sighting fires as a bet. That is the bug myevbettracker
   calls "the most costly available", an untested threshold presented as
   actionable.

2. Too tight -- a genuine opener never fires. Silent forever, and silence is
   indistinguishable from "no edge this week" without a test saying otherwise.

The cold start of 2026-08-12 makes (2) the live risk: the whole round-4 slate was
captured from already-quoting markets and is correctly suppressed, so the only
evidence that the gate ever opens is the round-5 case exercised here.
"""

import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

import pandas as pd

from ligamx.odds.capture_watch import opener_proof, record_unpriced, watched_before

KICKOFF = datetime(2026, 8, 22, 23, 0, tzinfo=timezone.utc)
HORIZON = pd.Timedelta(days=14)


def _proof(*, captured_at, kickoff=KICKOFF, watched=None, watching_since):
    return opener_proof(home="Atlante", away="Toluca", bookmaker="duel",
                        captured_at=pd.Timestamp(captured_at),
                        kickoff=pd.Timestamp(kickoff),
                        watched=watched or {},
                        watching_since=pd.Timestamp(watching_since),
                        horizon=HORIZON)


class OpenerProof(unittest.TestCase):
    """The two proofs, and the gap between them that must stay shut."""

    def test_observed_when_seen_unpriced_first(self):
        """The round-5 case: watched unpriced, then priced. This must open."""
        seen = KICKOFF - timedelta(days=10)
        proof = _proof(captured_at=seen + timedelta(hours=2),
                       watched={("Atlante", "Toluca", "duel"): pd.Timestamp(seen)},
                       # Late enough that the window proof cannot rescue it, so a
                       # pass here is attributable to the observation alone.
                       watching_since=KICKOFF - timedelta(days=11))
        self.assertEqual(proof, "observed")

    def test_cold_start_first_sighting_is_not_an_opener(self):
        """The round-4 case: no prior unpriced look, inside the lookahead."""
        proof = _proof(captured_at=KICKOFF - timedelta(days=10),
                       watching_since=KICKOFF - timedelta(days=10))
        self.assertEqual(proof, "")

    def test_window_when_fixture_became_visible_after_capture_began(self):
        """Never caught unpriced, but it cannot have opened before we looked."""
        proof = _proof(captured_at=KICKOFF - timedelta(days=13),
                       watching_since=KICKOFF - timedelta(days=20))
        self.assertEqual(proof, "window")

    def test_unpriced_sighting_after_the_capture_proves_nothing(self):
        """Ordering is the whole proof: seeing it unpriced later is not evidence.

        This is exactly the cold-start shape -- 2026-08-12 captured at 09:03 and
        first recorded an unpriced look at 11:28 -- so an implementation that
        forgot the comparison would light up the entire round-4 slate.
        """
        captured = KICKOFF - timedelta(days=10)
        proof = _proof(captured_at=captured,
                       watched={("Atlante", "Toluca", "duel"):
                                pd.Timestamp(captured + timedelta(hours=2))},
                       watching_since=captured)
        self.assertEqual(proof, "")

    def test_proof_is_per_book(self):
        """One book's opener says nothing about the other's price."""
        seen = KICKOFF - timedelta(days=10)
        proof = opener_proof(home="Atlante", away="Toluca", bookmaker="betano",
                             captured_at=pd.Timestamp(seen + timedelta(hours=2)),
                             kickoff=pd.Timestamp(KICKOFF),
                             watched={("Atlante", "Toluca", "duel"): pd.Timestamp(seen)},
                             watching_since=pd.Timestamp(KICKOFF - timedelta(days=11)),
                             horizon=HORIZON)
        self.assertEqual(proof, "")


class ProofReachesTheExport(unittest.TestCase):
    """The verdict has to survive the trip into the comparison rows."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.history = os.path.join(self.dir.name, "history.csv")
        self.watch = os.path.join(self.dir.name, "watch.csv")

    def _write(self, rows):
        pd.DataFrame(rows).to_csv(self.history, index=False)

    def _row(self, book, captured_at, home_odds):
        return {
            "event_id": f"e-{book}", "commence_time": KICKOFF.isoformat(),
            "api_home_team": "Atlante", "api_away_team": "Toluca",
            "home_team": "Atlante", "away_team": "Toluca",
            "home_odds": home_odds, "draw_odds": 4.0, "away_odds": 1.5,
            "bookmaker": book, "market": "h2h", "regions": "eu",
            "last_update": captured_at.isoformat(),
            "fetched_at": captured_at.isoformat(),
            "snapshot_type": "open", "target_round": 5, "capture_reason": "test",
        }

    def _load(self):
        from ligamx.odds import export_upcoming_market_comparison as mod
        with mock.patch("ligamx.paths.odds_capture_history_csv", return_value=self.history), \
             mock.patch("ligamx.paths.capture_watch_csv", return_value=self.watch):
            return mod.load_captured_opens()

    def test_watched_then_priced_carries_observed(self):
        seen = KICKOFF - timedelta(days=10)
        record_unpriced([("Atlante", "Toluca", "duel")],
                        observed_at=seen.isoformat().replace("+00:00", "Z"),
                        path=self.watch)
        self._write([self._row("duel", seen + timedelta(hours=2), 6.6)])
        books = self._load()[("Atlante", "Toluca")]
        self.assertEqual(books["duel"]["proof"], "observed")

    def test_first_sighting_carries_no_proof_but_keeps_the_price(self):
        """Suppressed as a bet, still shown: the dashboard displays it either way."""
        self._write([self._row("duel", KICKOFF - timedelta(days=2), 6.6)])
        books = self._load()[("Atlante", "Toluca")]
        self.assertEqual(books["duel"]["proof"], "")
        self.assertEqual(books["duel"]["home"], 6.6)

    def test_watching_since_spans_the_whole_history_not_just_opens(self):
        """A close tick is also a moment we looked.

        Taking the minimum over opens alone would date the start of watching
        later than it was and hand out an unearned "window".
        """
        early_close = {**self._row("pinnacle", KICKOFF - timedelta(days=20), 6.6),
                       "snapshot_type": "close"}
        self._write([early_close,
                     self._row("duel", KICKOFF - timedelta(days=2), 6.6)])
        books = self._load()[("Atlante", "Toluca")]
        # Watching began 20 days out, so KICKOFF - 14d is after it: window proof.
        self.assertEqual(books["duel"]["proof"], "window")


class WatchRoundTrip(unittest.TestCase):
    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.path = os.path.join(d.name, "watch.csv")

    def test_a_repeat_observation_does_not_move_the_timestamp(self):
        """Ticks arrive in order, so the first recorded is the earliest.

        Were one ever recorded out of order the stored value would be later than
        the truth, which makes the gate stricter, not looser -- a capture between
        the true earliest and the stored one would fail to prove itself. That is
        the safe direction for this to be wrong in.
        """
        first = "2026-08-11T09:00:00Z"
        record_unpriced([("Atlante", "Toluca", "duel")], observed_at=first, path=self.path)
        added = record_unpriced([("Atlante", "Toluca", "duel")],
                                observed_at="2026-08-12T11:28:15Z", path=self.path)
        self.assertEqual(added, 0)
        self.assertEqual(watched_before(self.path)[("Atlante", "Toluca", "duel")],
                         pd.Timestamp(first))

    def test_duplicate_rows_in_the_file_resolve_to_the_earliest(self):
        """Defensive: record_unpriced cannot make duplicates, an edit or a
        concatenation can, and the earliest is the one that bounds the proof."""
        pd.DataFrame([
            {"home_team": "Atlante", "away_team": "Toluca", "bookmaker": "duel",
             "first_seen_unpriced_at": "2026-08-12T11:28:15Z"},
            {"home_team": "Atlante", "away_team": "Toluca", "bookmaker": "duel",
             "first_seen_unpriced_at": "2026-08-11T09:00:00Z"},
        ]).to_csv(self.path, index=False)
        self.assertEqual(watched_before(self.path)[("Atlante", "Toluca", "duel")],
                         pd.Timestamp("2026-08-11T09:00:00Z"))


if __name__ == "__main__":
    unittest.main()
