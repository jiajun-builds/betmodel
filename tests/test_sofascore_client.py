"""Unit tests for the SofaScore client's xG aggregation and round labelling.

The network layer is stubbed: these pin the two decisions that the rest of the
pipeline depends on -- that xG is the sum of periods (not SofaScore's ALL) and that
liguilla ties are labelled by name rather than by their non-sequential round number.
"""

import unittest

from ligamx.sofascore_client import SofascoreClient, event_goals, round_label


def _stats_payload(periods: dict) -> dict:
    """Build a /event/{id}/statistics payload from {period: (home, away)}."""
    return {
        "statistics": [
            {
                "period": key,
                "groups": [
                    {"groupName": "Attacking", "statisticsItems": [
                        {"name": "Total shots", "home": "10", "away": "8"},
                        {"name": "Expected goals", "home": str(h), "away": str(a)},
                    ]},
                ],
            }
            for key, (h, a) in periods.items()
        ]
    }


class StubClient(SofascoreClient):
    def __init__(self, payload):
        super().__init__(pause=0.0)
        self._payload = payload

    def _get(self, path: str):
        return self._payload


class TestEventXG(unittest.TestCase):
    def test_sums_halves_and_ignores_all(self):
        # ALL is deliberately inconsistent with the halves, as SofaScore's often is.
        client = StubClient(_stats_payload({
            "ALL": (0.16, 2.40),
            "1ST": (0.55, 0.40),
            "2ND": (1.03, 0.70),
        }))
        self.assertEqual(client.event_xg(1), (1.58, 1.10))

    def test_excludes_extra_time_periods(self):
        # Extra time must not leak in: event_goals reports regulation, so xG has to
        # stay on the same 90-minute clock.
        client = StubClient(_stats_payload({
            "ALL": (1.15, 1.00),
            "1ST": (0.50, 0.30),
            "2ND": (0.40, 0.60),
            "ET1": (0.20, 0.05),
            "ET2": (0.05, 0.05),
        }))
        self.assertEqual(client.event_xg(1), (0.90, 0.90))

    def test_falls_back_to_all_when_no_periods_published(self):
        client = StubClient(_stats_payload({"ALL": (1.42, 0.83)}))
        self.assertEqual(client.event_xg(1), (1.42, 0.83))

    def test_returns_none_when_xg_absent(self):
        payload = {"statistics": [{"period": "ALL", "groups": [
            {"statisticsItems": [{"name": "Total shots", "home": "9", "away": "4"}]}]}]}
        self.assertEqual(StubClient(payload).event_xg(1), (None, None))

    def test_returns_none_on_missing_statistics(self):
        self.assertEqual(StubClient(None).event_xg(1), (None, None))

    def test_by_period_exposes_every_period(self):
        client = StubClient(_stats_payload({
            "ALL": (1.0, 1.0), "1ST": (0.4, 0.5), "2ND": (0.7, 0.6),
        }))
        self.assertEqual(
            client.event_xg_by_period(1),
            {"ALL": (1.0, 1.0), "1ST": (0.4, 0.5), "2ND": (0.7, 0.6)},
        )


class TestEventGoals(unittest.TestCase):
    def test_penalty_shootout_reports_regulation_not_aggregate(self):
        # Real payload shape: a 2-1 result SofaScore serves as current 11-9.
        event = {
            "homeScore": {"current": 11, "display": 2, "normaltime": 2, "penalties": 9},
            "awayScore": {"current": 9, "display": 1, "normaltime": 1, "penalties": 8},
        }
        self.assertEqual(event_goals(event), (2, 1))

    def test_extra_time_win_reports_the_90_minute_score(self):
        # 0-0 at 90', three goals in extra time; the goals model works on 90 minutes.
        event = {
            "homeScore": {"current": 3, "normaltime": 0, "extra1": 2, "extra2": 1},
            "awayScore": {"current": 0, "normaltime": 0, "extra1": 0, "extra2": 0},
        }
        self.assertEqual(event_goals(event), (0, 0))

    def test_prefers_normaltime_when_current_is_self_inconsistent(self):
        # Observed: current=4 while normaltime=5 and the halves sum to 5.
        event = {
            "homeScore": {"current": 4, "normaltime": 5, "period1": 2, "period2": 3},
            "awayScore": {"current": 0, "normaltime": 0, "period1": 0, "period2": 0},
        }
        self.assertEqual(event_goals(event), (5, 0))

    def test_falls_back_to_current_without_normaltime(self):
        event = {"homeScore": {"current": 2}, "awayScore": {"current": 1}}
        self.assertEqual(event_goals(event), (2, 1))

    def test_none_when_no_score(self):
        self.assertEqual(event_goals({}), (None, None))


class TestRoundLabel(unittest.TestCase):
    def test_league_round_uses_number(self):
        self.assertEqual(round_label({"roundInfo": {"round": 7}}), "7")

    def test_playoff_prefers_name(self):
        self.assertEqual(
            round_label({"roundInfo": {"round": 28, "name": "Semifinals"}}),
            "Semifinals",
        )

    def test_play_in_recovered_from_sub_tournament(self):
        # roundInfo is null on these; the stage only shows up in the tournament slug.
        event = {"roundInfo": None,
                 "tournament": {"name": "Liga MX, Apertura, Play In",
                                "slug": "liga-mx-apertura-play-in"}}
        self.assertEqual(round_label(event), "Play-in")

    def test_round_info_wins_over_tournament_slug(self):
        event = {"roundInfo": {"round": 3},
                 "tournament": {"slug": "liga-mx-apertura-play-in"}}
        self.assertEqual(round_label(event), "3")

    def test_blank_when_no_round_info(self):
        self.assertEqual(round_label({}), "")
        self.assertEqual(round_label({"roundInfo": None}), "")
        self.assertEqual(round_label({"tournament": {"slug": "liga-mx-apertura"}}), "")


if __name__ == "__main__":
    unittest.main()
