"""
EV Calculator — expected value for 1X2 (match-odds) betting lines.

EV per outcome as a fraction of stake: EV = model_prob * decimal_odds - 1.
"""

from typing import Dict


class EVCalculator:
    def calculate_1x2_evs(
        self,
        home_win_prob: float,
        draw_prob: float,
        away_win_prob: float,
        home_odds: float,
        draw_odds: float,
        away_odds: float,
    ) -> Dict[str, float]:
        """
        EV per 1X2 outcome as a fraction of stake: EV = p * decimal_odds - 1.
        Missing/invalid odds yield 0.0 for that outcome.
        """
        return {
            "home_ev": self._one_x_two_ev(home_win_prob, home_odds),
            "draw_ev": self._one_x_two_ev(draw_prob, draw_odds),
            "away_ev": self._one_x_two_ev(away_win_prob, away_odds),
        }

    @staticmethod
    def _one_x_two_ev(prob: float, odds: float) -> float:
        if prob is None or odds is None:
            return 0.0
        try:
            return EVCalculator._safe_prob(prob) * float(odds) - 1.0
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _safe_prob(value: float) -> float:
        if value is None:
            return 0.0
        return max(0.0, float(value))
