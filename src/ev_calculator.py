"""
EV Calculator - calculates expected value for betting lines using spread-based formulas.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class EVContext:
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    home_minus_1_prob: float
    away_minus_1_prob: float
    home_odds: float
    away_odds: float
    spread: float


@dataclass
class PayoutStates:
    win: float = 0.0
    half_win: float = 0.0
    push: float = 0.0
    half_loss: float = 0.0
    loss: float = 0.0


class EVCalculator:
    SUPPORTED_SPREADS = {
        -1.5,
        -1.25,
        -1.0,
        -0.75,
        -0.5,
        -0.25,
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
    }

    def calculate_ev(
        self,
        model_prob: float,
        decimal_odds: float,
        spread: float = 0,
        draw_prob: float = 0,
        home_minus_1_prob: float = 0,
        away_minus_1_prob: float = 0,
        is_home: bool = True,
    ) -> float:
        """
        Compatibility API: calculate EV for one side from legacy arguments.
        """
        if model_prob is None or decimal_odds is None:
            return 0.0

        spread = self._normalize_spread(spread)
        draw_prob = self._safe_prob(draw_prob)
        home_minus_1_prob = self._safe_prob(home_minus_1_prob)
        away_minus_1_prob = self._safe_prob(away_minus_1_prob)

        if is_home:
            home_win_prob = self._safe_prob(model_prob)
            away_win_prob = self._safe_prob(1 - home_win_prob - draw_prob)
            context = EVContext(
                home_win_prob=home_win_prob,
                draw_prob=draw_prob,
                away_win_prob=away_win_prob,
                home_minus_1_prob=home_minus_1_prob,
                away_minus_1_prob=away_minus_1_prob,
                home_odds=decimal_odds,
                away_odds=0.0,
                spread=spread,
            )
            return self.calculate_side_ev(context, side="home")

        away_win_prob = self._safe_prob(model_prob)
        home_win_prob = self._safe_prob(1 - away_win_prob - draw_prob)
        context = EVContext(
            home_win_prob=home_win_prob,
            draw_prob=draw_prob,
            away_win_prob=away_win_prob,
            home_minus_1_prob=home_minus_1_prob,
            away_minus_1_prob=away_minus_1_prob,
            home_odds=0.0,
            away_odds=decimal_odds,
            spread=spread,
        )
        return self.calculate_side_ev(context, side="away")

    def calculate_side_ev(self, context: EVContext, side: str) -> float:
        """
        Calculate EV for one side using canonical probability buckets and payout states.
        """
        if side not in {"home", "away"}:
            return 0.0

        spread = self._normalize_spread(context.spread)
        odds = context.home_odds if side == "home" else context.away_odds
        if odds is None:
            return 0.0

        if side == "home":
            states = self._states_for_home_bet(context, spread)
        else:
            mirrored = self._mirror_context(context)
            states = self._states_for_home_bet(mirrored, self._normalize_spread(-spread))

        return self._ev_from_states(states, odds)

    def calculate_match_evs(
        self,
        home_win_prob: float,
        draw_prob: float,
        away_win_prob: float,
        home_minus_1_prob: float,
        away_minus_1_prob: float,
        home_odds: float,
        away_odds: float,
        spread: float,
    ) -> Dict[str, float]:
        """
        Calculate both home and away EV values for one match.
        """
        context = EVContext(
            home_win_prob=self._safe_prob(home_win_prob),
            draw_prob=self._safe_prob(draw_prob),
            away_win_prob=self._safe_prob(away_win_prob),
            home_minus_1_prob=self._safe_prob(home_minus_1_prob),
            away_minus_1_prob=self._safe_prob(away_minus_1_prob),
            home_odds=home_odds,
            away_odds=away_odds,
            spread=self._normalize_spread(spread),
        )
        return {
            "home_ev": self.calculate_side_ev(context, side="home"),
            "away_ev": self.calculate_side_ev(context, side="away"),
        }

    def calculate_home_ev(
        self,
        home_win_prob: float,
        draw_prob: float,
        away_win_prob: float,
        home_minus_1_prob: float,
        away_minus_1_prob: float,
        home_odds: float,
        spread: float,
    ) -> float:
        """
        Compatibility API: calculate EV for betting on home team.
        """
        context = EVContext(
            home_win_prob=self._safe_prob(home_win_prob),
            draw_prob=self._safe_prob(draw_prob),
            away_win_prob=self._safe_prob(away_win_prob),
            home_minus_1_prob=self._safe_prob(home_minus_1_prob),
            away_minus_1_prob=self._safe_prob(away_minus_1_prob),
            home_odds=home_odds,
            away_odds=0.0,
            spread=self._normalize_spread(spread),
        )
        return self.calculate_side_ev(context, side="home")

    def calculate_away_ev(
        self,
        home_win_prob: float,
        draw_prob: float,
        away_win_prob: float,
        home_minus_1_prob: float,
        away_minus_1_prob: float,
        away_odds: float,
        spread: float,
    ) -> float:
        """
        Compatibility API: calculate EV for betting on away team.
        """
        context = EVContext(
            home_win_prob=self._safe_prob(home_win_prob),
            draw_prob=self._safe_prob(draw_prob),
            away_win_prob=self._safe_prob(away_win_prob),
            home_minus_1_prob=self._safe_prob(home_minus_1_prob),
            away_minus_1_prob=self._safe_prob(away_minus_1_prob),
            home_odds=0.0,
            away_odds=away_odds,
            spread=self._normalize_spread(spread),
        )
        return self.calculate_side_ev(context, side="away")

    @staticmethod
    def _safe_prob(value: float) -> float:
        if value is None:
            return 0.0
        return max(0.0, float(value))

    @staticmethod
    def _normalize_spread(spread: float) -> float:
        try:
            return round(float(spread), 2)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _ev_from_states(states: PayoutStates, odds: float) -> float:
        net_profit = odds - 1
        return (
            states.win * net_profit
            + states.half_win * 0.5 * net_profit
            + states.half_loss * (-0.5)
            + states.loss * (-1)
        )

    @staticmethod
    def _mirror_context(context: EVContext) -> EVContext:
        return EVContext(
            home_win_prob=context.away_win_prob,
            draw_prob=context.draw_prob,
            away_win_prob=context.home_win_prob,
            home_minus_1_prob=context.away_minus_1_prob,
            away_minus_1_prob=context.home_minus_1_prob,
            home_odds=context.away_odds,
            away_odds=context.home_odds,
            spread=-context.spread,
        )

    def _states_for_home_bet(self, context: EVContext, spread: float) -> PayoutStates:
        """
        Settlement states for betting the home side at a given home handicap spread.
        """
        spread = self._normalize_spread(spread)
        h2 = min(context.home_minus_1_prob, context.home_win_prob)
        a2 = min(context.away_minus_1_prob, context.away_win_prob)
        h1 = max(0.0, context.home_win_prob - h2)
        a1 = max(0.0, context.away_win_prob - a2)
        h_or_d = context.home_win_prob + context.draw_prob
        d_or_a = context.draw_prob + context.away_win_prob

        if spread not in self.SUPPORTED_SPREADS:
            return PayoutStates(win=context.home_win_prob, loss=context.away_win_prob)

        if spread == 0.0:
            return PayoutStates(win=context.home_win_prob, push=context.draw_prob, loss=context.away_win_prob)
        if spread == 0.25:
            return PayoutStates(win=context.home_win_prob, half_win=context.draw_prob, loss=context.away_win_prob)
        if spread == 0.5:
            return PayoutStates(win=h_or_d, loss=context.away_win_prob)
        if spread == 0.75:
            return PayoutStates(win=h_or_d, half_loss=a1, loss=a2)
        if spread == 1.0:
            return PayoutStates(win=h_or_d, push=a1, loss=a2)
        if spread == 1.25:
            return PayoutStates(win=h_or_d, half_win=a1, loss=a2)
        if spread == 1.5:
            return PayoutStates(win=1.0 - a2, loss=a2)

        if spread == -0.25:
            return PayoutStates(win=context.home_win_prob, half_loss=context.draw_prob, loss=context.away_win_prob)
        if spread == -0.5:
            return PayoutStates(win=context.home_win_prob, loss=d_or_a)
        if spread == -0.75:
            return PayoutStates(win=h2, half_win=h1, loss=d_or_a)
        if spread == -1.0:
            return PayoutStates(win=h2, push=h1, loss=d_or_a)
        if spread == -1.25:
            return PayoutStates(win=h2, half_loss=h1, loss=d_or_a)
        if spread == -1.5:
            return PayoutStates(win=h2, loss=1.0 - h2)

        return PayoutStates(win=context.home_win_prob, loss=context.away_win_prob)

    def is_positive_ev(self, ev: float, threshold: float = 0.0) -> bool:
        """Check if the bet has positive expected value."""
        return ev > threshold

    def calculate_line_ev(self, match: Dict, line_type: str = "home") -> Dict:
        """Calculate EV for a specific betting line."""
        return {"ev": 0.0, "is_ev": False}

    def calculate_batch_ev(self, matches: List[Dict]) -> List[Dict]:
        """Calculate EV for a batch of matches."""
        return matches

    def filter_positive_ev(self, matches: List[Dict]) -> List[Dict]:
        """Filter to only positive EV lines."""
        return [m for m in matches if m.get("home_is_ev") or m.get("away_is_ev")]
