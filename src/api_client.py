"""
API-Football client for fixtures and historical results.
"""

import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import config


class APIClient:
    def __init__(self):
        self.base_url = config.API_FOOTBALL_BASE_URL
        self.api_key = config.API_FOOTBALL_API_KEY
        self.headers = {
            "x-apisports-key": self.api_key,
        }

    def _get(self, endpoint: str, params: dict = None) -> dict:
        url = f"{self.base_url}{endpoint}"
        response = requests.get(url, headers=self.headers, params=params)
        
        if response.status_code == 404:
            raise Exception(f"Endpoint not found: {endpoint}")
        
        response.raise_for_status()
        data = response.json()
        
        if "results" in data and data["results"] == 0:
            return {"response": [], "errors": []}
        
        if data.get("errors"):
            raise Exception(f"API error: {data.get('errors')}")
        
        return data

    def get_fixtures(self, date: str, status: Optional[str] = None, season: int = 2025) -> List[Dict]:
        """Get fixtures for a specific date."""
        params = {
            "league": config.LIGA_MX_LEAGUE_ID,
            "season": season,
            "date": date,
            "timezone": "America/Mexico_City",
        }
        if status:
            params["status"] = status

        data = self._get("/fixtures", params)
        fixtures = data.get("response", [])
        
        results = []
        for f in fixtures:
            fixture = f.get("fixture", {})
            league = f.get("league", {})
            teams = f.get("teams", {})
            goals = f.get("goals", {})
            
            results.append({
                "fixture_id": fixture.get("id"),
                "date": fixture.get("date", "")[:10],
                "time": fixture.get("date", "")[11:16],
                "league_id": league.get("id"),
                "league_name": league.get("name"),
                "round": league.get("round"),
                "home_team": teams.get("home", {}).get("name"),
                "away_team": teams.get("away", {}).get("name"),
                "home_goals": goals.get("home"),
                "away_goals": goals.get("away"),
                "status": fixture.get("status", {}).get("short"),
            })
        
        return results

    def get_completed_matches(self, days_back: int = 7) -> List[Dict]:
        """Get completed matches from the last N days."""
        results = []
        today = datetime.now()
        
        for i in range(days_back):
            date = (today - timedelta(days=i)).strftime("%Y-%m-%d")
            matches = self.get_fixtures(date, status="FT")
            results.extend(matches)
        
        return results

    def get_upcoming_fixtures(self, days_ahead: int = 7) -> List[Dict]:
        """Get upcoming fixtures for the next N days."""
        results = []
        today = datetime.now()
        
        for i in range(days_ahead):
            date = (today + timedelta(days=i)).strftime("%Y-%m-%d")
            matches = self.get_fixtures(date)
            for m in matches:
                if m.get("status") in [None, "NS", "TBD"]:
                    results.append(m)
        
        return results