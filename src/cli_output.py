"""
CLI Output - displays results in terminal and saves to CSV.
"""

import os
from datetime import datetime
from typing import List, Dict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import config


class CLIOutput:
    def __init__(self):
        self.console = Console()
        self.output_dir = config.OUTPUT_DIR

    def display_ev_lines(self, matches: List[Dict], run_date: str = None):
        """Display +EV lines in a rich table."""
        if not run_date:
            run_date = datetime.now().strftime("%Y-%m-%d")
        
        table = Table(title=f"LIGAMX +EV BETTING LINES  Run: {run_date}")
        
        table.add_column("Match", style="cyan", no_wrap=True, width=24)
        table.add_column("Spread", justify="center", width=8)
        table.add_column("Home Odds", justify="right", width=9)
        table.add_column("Home EV", justify="right", width=8)
        table.add_column("Away Odds", justify="right", width=9)
        table.add_column("Away EV", justify="right", width=8)
        table.add_column("Bet Option", style="magenta", width=44)
        
        total_matches = len(matches)
        ev_count = 0
        
        for match in matches:
            home = match.get("home_team", "")
            away = match.get("away_team", "")
            match_name = f"{home} vs {away}"
            
            spread = match.get("spread", "-")
            home_ev = match.get("home_ev", 0)
            away_ev = match.get("away_ev", 0)
            home_odds = match.get("home_odds", 0)
            away_odds = match.get("away_odds", 0)
            spread_val = match.get("spread", 0)

            best_ev = max(home_ev, away_ev)
            if best_ev > 0:
                if home_ev > away_ev:
                    sign = f"-{abs(spread_val):.2g}" if spread_val < 0 else f"+{abs(spread_val):.2g}" if spread_val > 0 else "0"
                    ev_str = f"+{home_ev:.1f}%"
                    bet_option = f"{home} {sign} @{home_odds:.2f} EV {ev_str}"
                else:
                    sign = f"-{abs(spread_val):.2g}" if spread_val > 0 else f"+{abs(spread_val):.2g}" if spread_val < 0 else "0"
                    ev_str = f"+{away_ev:.1f}%"
                    bet_option = f"{away} {sign} @{away_odds:.2f} EV {ev_str}"
                ev_count += 1
            else:
                bet_option = ""
            
            table.add_row(
                match_name,
                str(spread),
                f"{home_odds:.2f}",
                f"{home_ev:.1f}%",
                f"{away_odds:.2f}",
                f"{away_ev:.1f}%",
                bet_option,
            )
        
        panel = Panel(
            table,
            subtitle=f"Total: {total_matches} matches  |  +EV: {ev_count}",
        )
        
        self.console.print(panel)
        
        return {
            "total": total_matches,
            "positive_ev": ev_count,
        }
    
    def save_to_csv(self, matches: List[Dict], run_date: str = None) -> str:
        """Save results to CSV file."""
        if not run_date:
            run_date = datetime.now().strftime("%Y-%m-%d")
        
        base_dir = os.path.dirname(os.path.dirname(__file__))
        output_path = os.path.join(base_dir, self.output_dir)
        os.makedirs(output_path, exist_ok=True)
        
        filename = f"ligamx_ev_{run_date}.csv"
        filepath = os.path.join(output_path, filename)
        
        if not matches:
            return filepath
        
        import pandas as pd
        
        df = pd.DataFrame(matches)
        df.to_csv(filepath, index=False)
        
        self.console.print(f"\nSaved: {filepath}")
        
        return filepath
