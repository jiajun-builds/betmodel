"""
The Odds API client for Pinnacle Asian Handicap odds.
Uses ScraperFC for match results and The Odds API for odds.
"""

import requests
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import config


class OddsClient:
    SEASON_YEARS = ["25/26", "24/25"]
    SEASON_LEAGUES = [
        "Mexico Liga MX Apertura",
        "Mexico Liga MX Clausura",
    ]
    
    def __init__(self):
        self.base_url = config.THE_ODDS_API_BASE_URL
        self.api_key = config.THE_ODDS_API_KEY
        self._scraper = None
        self._season_cache = {}

    @property
    def scraper(self):
        """Lazy load ScraperFC."""
        if self._scraper is None:
            from ScraperFC import Sofascore
            self._scraper = Sofascore()
        return self._scraper

    def _get_season_matches(self, year: str, league: str) -> List[Dict]:
        """Get all matches for a season (cached)."""
        key = (year, league)
        if key in self._season_cache:
            return self._season_cache[key]
        
        try:
            matches = self.scraper.get_match_dicts(year=year, league=league)
            self._season_cache[key] = matches
            return matches
        except Exception as e:
            print(f"Error fetching {year} {league}: {e}")
            return []

    def get_completed_matches(self, days_from: int = 7) -> List[Dict]:
        """Get completed matches from ScraperFC.
        
        Args:
            days_from: Number of days to look back for completed matches
        
        Returns:
            List of completed matches with scores and team names (sofascore_team format)
        """
        results = []
        cutoff = datetime.now() - timedelta(days=days_from)
        
        for year in self.SEASON_YEARS:
            for league in self.SEASON_LEAGUES:
                matches = self._get_season_matches(year, league)
                
                for match in matches:
                    try:
                        # Filter: status == finished (code 100)
                        status = match.get("status", {})
                        if status.get("code") != 100:
                            continue
                        
                        # Filter: date within range
                        timestamp = match.get("startTimestamp", 0)
                        if timestamp == 0:
                            continue
                        
                        match_time = datetime.fromtimestamp(timestamp)
                        if match_time < cutoff:
                            continue
                        
                        # Extract team names and scores
                        home_team = match.get("homeTeam", {}).get("name", "")
                        away_team = match.get("awayTeam", {}).get("name", "")
                        home_goals = match.get("homeScore", {}).get("current", 0)
                        away_goals = match.get("awayScore", {}).get("current", 0)
                        
                        if home_team and away_team is not None:
                            results.append({
                                "home_team": home_team,
                                "away_team": away_team,
                                "home_goals": home_goals,
                                "away_goals": away_goals,
                                "commence_time": match_time.isoformat(),
                                "timestamp": timestamp,
                            })
                    except Exception as e:
                        print(f"Error processing match: {e}")
                        continue
        
        print(f"Found {len(results)} completed matches in last {days_from} days")
        return results

    def get_upcoming_fixtures(self, days_ahead: int = 7) -> List[Dict]:
        """Get upcoming fixtures (not implemented - use get_odds instead)."""
        return []

    def _get(self, endpoint: str, params: dict = None) -> dict:
        url = f"{self.base_url}{endpoint}"
        params = params or {}
        params["apiKey"] = self.api_key
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def get_odds(self, regions: str = "eu", markets: str = "spreads") -> List[Dict]:
        """Get odds for LigaMX from The Odds API."""
        params = {
            "sport": config.ODDS_SPORT_KEY,
            "regions": regions,
            "markets": markets,
            "oddsFormat": "decimal",
            "bookmakers": config.BOOKMAKER,
        }
        
        endpoint = f"/sports/{config.ODDS_SPORT_KEY}/odds"
        data = self._get(endpoint, params)
        
        results = []
        for event in data:
            bookmakers = event.get("bookmakers", [])
            if not bookmakers:
                continue
            
            bm = bookmakers[0]
            markets_data = bm.get("markets", [])
            if not markets_data:
                continue
            
            spreads_market = markets_data[0]
            outcomes = spreads_market.get("outcomes", [])
            
            home_team = event.get("home_team")
            away_team = event.get("away_team")
            
            home_odds = None
            away_odds = None
            spread = None
            
            for outcome in outcomes:
                if outcome.get("name") == home_team:
                    home_odds = outcome.get("price")
                    spread = outcome.get("point")
                elif outcome.get("name") == away_team:
                    away_odds = outcome.get("price")
            
            if home_odds and away_odds:
                results.append({
                    "event_id": event.get("id"),
                    "commence_time": event.get("commence_time"),
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_odds": home_odds,
                    "away_odds": away_odds,
                    "spread": spread,
                })
        
        return results

    def get_odds_for_date(self, target_date: str) -> List[Dict]:
        """Get odds for a specific date."""
        all_odds = self.get_odds()
        
        filtered = []
        for o in all_odds:
            commence = o.get("commence_time", "")
            if commence.startswith(target_date):
                filtered.append(o)
        
        return filtered