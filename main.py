#!/usr/bin/env python3
"""
LigaMX Betting Tool - Main Entry Point

Usage:
    python main.py --date 2026-04-18
    python main.py              # Run for today
"""

import argparse
import subprocess
import sys
import os
from datetime import datetime

import pandas as pd

from src.odds_client import OddsClient
from src.sofascore_scraper import SofascoreScraper
from src.prediction_model import PredictionModel
from src.ev_calculator import EVCalculator
from src.cli_output import CLIOutput
from src.data_updater import DataUpdater
import config


def _parse_dates_safe(date_series: pd.Series) -> pd.Series:
    """Parse dates safely across slash and dash formats; invalid values become NaT."""
    cleaned = date_series.astype(str).str.strip()
    parsed = pd.to_datetime(cleaned, format="%Y/%m/%d", errors="coerce")

    missing = parsed.isna()
    if missing.any():
        parsed_dash = pd.to_datetime(cleaned[missing], format="%Y-%m-%d", errors="coerce")
        parsed.loc[missing] = parsed_dash

    return parsed


def get_latest_data_date() -> datetime:
    """Get the latest match date from the data CSV."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, config.DATA_FILE)
    df = pd.read_csv(data_path)
    parsed_dates = _parse_dates_safe(df["Date"]).dropna()
    if parsed_dates.empty:
        return datetime.min
    return parsed_dates.max().to_pydatetime()


def get_latest_model_date() -> datetime:
    """Get the latest date from the model CSV files."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    
    dates = []
    for filename in ["MEX_team_stats.csv", "MEX_team_stats_match_simulations.csv"]:
        filepath = os.path.join(models_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            if "Date" in df.columns:
                parsed_dates = _parse_dates_safe(df["Date"]).dropna()
                if not parsed_dates.empty:
                    dates.append(parsed_dates.max().to_pydatetime())
    
    return max(dates) if dates else datetime.min


def run_pipeline(date: str = None, update_model: bool = True):
    """Run the complete betting tool pipeline."""
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("=" * 60)
    print("LIGAMX BETTING TOOL")
    print("=" * 60)
    
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"\nRun date: {date}")
    
    odds_client = OddsClient()
    sofascore = SofascoreScraper()
    model = PredictionModel()
    ev_calc = EVCalculator()
    cli = CLIOutput()
    updater = DataUpdater()
    model_regenerated_in_phase1 = False
    
    # Skip Phase 1 if --no-update (just use existing model)
    if not update_model:
        print("\n[--no-update] Skipping Phase 1...")
    else:
        print("\n" + "=" * 60)
        print("PHASE 1: INGEST NEW DATA")
        print("=" * 60)
        
        print("\n[1a] Fetching completed matches from ScraperFC...")
        completed = odds_client.get_completed_matches(days_from=config.XG_REFRESH_DAYS)
        completed = [m for m in completed if m.get("home_goals") is not None and m.get("away_goals") is not None]
        print(f"      Found: {len(completed)} finished matches")
        
        reconciled_matches = []
        skipped_matches = []
        if completed:
            print("\n[1b] Fetching xG for completed matches...")
            for match in completed:
                home = match.get("home_team")
                away = match.get("away_team")
                
                # Get xG by matching team names + match date (not run date)
                match_date = match.get("commence_time", date)[:10]
                xg_data = sofascore.get_match_xg(home, away, match_date)
                match["HxG"] = xg_data.get("HxG", 0)
                match["AxG"] = xg_data.get("AxG", 0)
                match["event_id"] = xg_data.get("event_id")

                if xg_data.get("HxG", 0) <= 0 and xg_data.get("AxG", 0) <= 0:
                    skipped_matches.append(f"{home} vs {away}")
                reconciled_matches.append(match)

            print(f"      Retrieved xG payloads for {len(reconciled_matches)} matches")
            if skipped_matches:
                print(f"      Skipped (no xG found): {len(skipped_matches)} matches")

        if reconciled_matches:
            print("\n[1c] Reconciling data/MEX_ligamx.csv...")
            stats = updater.upsert_matches(reconciled_matches)
            print(
                f"      Added: {stats['added']} | Updated: {stats['updated']} | "
                f"Unchanged: {stats['unchanged']} | Skipped: {stats['skipped']}"
            )

            if (stats["added"] + stats["updated"]) > 0 and update_model:
                print("\n[1e] Running DC model...")
                result = subprocess.run(
                    [sys.executable, "DC_MEX.py"],
                    capture_output=True,
                    text=True,
                    cwd=base_dir
                )
                if result.returncode == 0:
                    print("      Model regenerated successfully")
                    model_regenerated_in_phase1 = True
                else:
                    print(f"      Error: {result.stderr}")
    
    if update_model:
        if model_regenerated_in_phase1:
            print("\n[*] Model freshness already ensured in Phase 1; skipping duplicate check.")
        else:
            print("\n[*] Checking model freshness...")
            latest_data = get_latest_data_date()
            latest_model = get_latest_model_date()
            
            if latest_data > latest_model:
                print(f"      Model is stale (data: {latest_data.date()}, model: {latest_model.date()})")
                print("      Regenerating model...")
                result = subprocess.run(
                    [sys.executable, "DC_MEX.py"],
                    capture_output=True,
                    text=True,
                    cwd=base_dir
                )
                if result.returncode == 0:
                    print("      Model regenerated successfully")
                else:
                    print(f"      Error: {result.stderr}")
            else:
                print(f"      Model is up-to-date ({latest_model.date()})")
    
    print("\n" + "=" * 60)
    print("PHASE 2: CALCULATE +EV")
    print("=" * 60)
    
    print("\n[2a] Fetching Pinnacle odds...")
    odds = odds_client.get_odds()
    print(f"      Found: {len(odds)} upcoming matches with odds")
    
    print("\n[2b] Loading model probabilities...")
    model.load()
    print(f"      Loaded pre-computed model")
    
    print("\n[2d] Calculating +EV...")
    
    ev_results = []
    for odds_data in odds:
        home = odds_data.get("home_team")
        away = odds_data.get("away_team")
        
        spread = odds_data.get("spread", 0)
        home_odds = odds_data.get("home_odds")
        away_odds = odds_data.get("away_odds")
        
        # Get all probabilities from model
        probs = model.get_probabilities(home, away)
        if not probs:
            continue
        
        home_win = probs.get("home_win", 0)
        draw = probs.get("draw", 0)
        away_win = probs.get("away_win", 0)
        home_minus_1 = probs.get("Home -1", 0)
        away_minus_1 = probs.get("Away -1", 0)
        
        ev_values = ev_calc.calculate_match_evs(
            home_win_prob=home_win,
            draw_prob=draw,
            away_win_prob=away_win,
            home_minus_1_prob=home_minus_1,
            away_minus_1_prob=away_minus_1,
            home_odds=home_odds,
            away_odds=away_odds,
            spread=spread,
        )
        home_ev = ev_values["home_ev"]
        away_ev = ev_values["away_ev"]
        
        ev_results.append({
            "home_team": home,
            "away_team": away,
            "spread": spread,
            "home_win": home_win,
            "draw": draw,
            "away_win": away_win,
            "home_minus_1": home_minus_1,
            "away_minus_1": away_minus_1,
            "home_odds": home_odds,
            "away_odds": away_odds,
            "home_ev": home_ev * 100,
            "away_ev": away_ev * 100,
            "home_is_ev": home_ev > 0,
            "away_is_ev": away_ev > 0,
        })
    
    ev_count = len([r for r in ev_results if r.get("home_is_ev") or r.get("away_is_ev")])
    print(f"      Analyzed: {len(ev_results)} matches")
    print(f"      +EV found: {ev_count}")
    
    print("\n" + "=" * 60)
    print("OUTPUT")
    print("=" * 60)
    
    cli.display_ev_lines(ev_results, date)
    cli.save_to_csv(ev_results, date)
    
    print("\nDone!")
    
    return ev_results


def main():
    parser = argparse.ArgumentParser(description="LigaMX Betting Tool")
    parser.add_argument("--date", type=str, help="Date in YYYY-MM-DD format")
    parser.add_argument("--no-update", action="store_true", help="Skip model update")
    
    args = parser.parse_args()
    
    run_pipeline(
        date=args.date, 
        update_model=not args.no_update
    )


if __name__ == "__main__":
    main()
