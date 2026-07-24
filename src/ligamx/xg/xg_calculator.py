"""
xG Calculator - calculates HExpG+ and AExpG+ using formula:
HExpG+ = 0.25 × HxG + 0.75 × HG
AExpG+ = 0.25 × AxG + 0.75 × AG

The 0.25/0.75 blend (down from 0.55/0.45) was chosen by the RPS walk-forward
sweep (ligamx.eval.hyperparam_sweep): a lower xG weight scored marginally better
than the old blend and than pure xG on Liga MX 1X2 forecasts.
"""

from typing import Dict, List


class XGCalculator:
    WEIGHT_XG = 0.25
    WEIGHT_GOALS = 0.75

    def calculate_hext(self, hg: int, hxg: float) -> float:
        """Calculate home team's expected goals."""
        if hg is None or hxg is None:
            return 0.0
        return self.WEIGHT_XG * hxg + self.WEIGHT_GOALS * hg

    def calculate_aext(self, ag: int, axg: float) -> float:
        """Calculate away team's expected goals."""
        if ag is None or axg is None:
            return 0.0
        return self.WEIGHT_XG * axg + self.WEIGHT_GOALS * ag

    def calculate_match_xg(self, hg: int, ag: int, hxg: float, axg: float) -> Dict:
        """Calculate xG for a complete match."""
        return {
            "HExpG+": self.calculate_hext(hg, hxg),
            "AExpG+": self.calculate_aext(ag, axg),
        }

    def calculate_batch(self, matches: List[Dict]) -> List[Dict]:
        """Calculate xG for a batch of matches."""
        results = []
        for match in matches:
            hg = match.get("home_goals")
            ag = match.get("away_goals")
            hxg = match.get("HxG", 0)
            axg = match.get("AxG", 0)

            xg = self.calculate_match_xg(hg, ag, hxg, axg)

            result = match.copy()
            result.update(xg)
            results.append(result)

        return results
