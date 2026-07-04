"""
Sofascore scraper for xG data using ScraperFC.
"""

from typing import Dict, Optional, List
import config
import os
from datetime import datetime


class SofascoreScraper:
    SEASON_YEARS = ["25/26", "24/25"]  # Current and previous seasons
    SEASON_LEAGUES = [
        "Mexico Liga MX Apertura",
        "Mexico Liga MX Clausura",
    ]
    
    def __init__(self):
        self._scraper = None
        self._season_cache = {}  # Cache by (year, league) tuple
    
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
            print(f"Loaded {len(matches)} matches for {year} {league}")
            return matches
        except Exception as e:
            print(f"Error fetching {year} {league}: {e}")
            return []
    
    def _extract_xg(self, match_id) -> Dict:
        """Extract xG from match statistics."""
        try:
            stats = self.scraper.scrape_team_match_stats(match_id)
            
            hxg = 0
            axg = 0
            
            # Look for 'Expected goals' in the 'name' column
            for _, row in stats.iterrows():
                stat_name = str(row.get("name", "")).lower()
                if "expected" in stat_name and "goal" in stat_name:
                    # Use homeValue and awayValue columns
                    hxg = float(row.get("homeValue", 0) or 0)
                    axg = float(row.get("awayValue", 0) or 0)
                    print(f"    Found xG: home={hxg}, away={axg}")
                    break
            
            return {"HxG": hxg, "AxG": axg}
        except Exception as e:
            print(f"Error extracting xG for match {match_id}: {e}")
            return {"HxG": 0, "AxG": 0}
    
    def _get_date_from_timestamp(self, timestamp: int) -> str:
        """Convert Unix timestamp to YYYY-MM-DD format."""
        try:
            dt = datetime.fromtimestamp(timestamp)
            return dt.strftime("%Y-%m-%d")
        except:
            return ""
    
    def get_match_xg(self, home_team: str, away_team: str, date: str) -> Dict:
        """Get xG for a match by matching team names and date.
        
        Args:
            home_team: Home team name (Odds API format - odds_team)
            away_team: Away team name (Odds API format - odds_team)
            date: Date string YYYY-MM-DD (or full ISO format)
        
        Returns:
            Dict with HxG, AxG, event_id (Sofascore match ID)
        """
        # Extract just the date part (YYYY-MM-DD)
        date_only = date[:10] if date else ""
        
        # Map to ScraperFC team names
        sf_home = config.ODDS_TO_SOFASCORE.get(home_team, home_team)
        sf_away = config.ODDS_TO_SOFASCORE.get(away_team, away_team)
        
        print(f"  Looking for: {sf_home} vs {sf_away} on {date_only}")
        
        # Try both current and previous season
        for year in self.SEASON_YEARS:
            for league in self.SEASON_LEAGUES:
                matches = self._get_season_matches(year, league)
                
                for match in matches:
                    # Extract match info
                    try:
                        sf_match_home = match.get("homeTeam", {}).get("name", "")
                        sf_match_away = match.get("awayTeam", {}).get("name", "")
                        
                        # Use startTimestamp instead of startTime
                        timestamp = match.get("startTimestamp", 0)
                        sf_date = self._get_date_from_timestamp(timestamp)
                        
                        # Check if teams and dates match
                        if (sf_match_home == sf_home and 
                            sf_match_away == sf_away and 
                            sf_date == date_only):
                            
                            match_id = match.get("id")
                            print(f"    Found match ID: {match_id} (timestamp: {timestamp}, date: {sf_date})")
                            
                            # Get xG
                            xg = self._extract_xg(match_id)
                            xg["event_id"] = str(match_id)
                            return xg
                            
                    except Exception as e:
                        print(f"    Error processing match: {e}")
                        continue
        
        # No match found - return zeros
        print(f"    No match found - returning zeros")
        return {"HxG": 0, "AxG": 0, "event_id": None}

    def get_team_xg(self, team_name: str, last_n: int = 5) -> Dict:
        """Get average xG for a team in last N matches."""
        pass

    def map_team_to_event(self, home_team: str, away_team: str, date: str) -> Optional[str]:
        """Map team names to Sofascore event ID."""
        result = self.get_match_xg(home_team, away_team, date)
        return result.get("event_id")