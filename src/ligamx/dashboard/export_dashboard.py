"""
Export the dashboard's 5-file JSON contract from the pipeline outputs.

Reads:  MEX_ligamx.csv (history), MEX_upcoming_fixtures.csv (schedule),
        MEX_team_stats.csv + _match_simulations.csv (model),
        MEX_upcoming_market_comparison.csv (1X2 EV; optional).
Writes: data/dashboard/json/{dashboard_meta, upcoming_fixtures, match_predictions,
        team_strength_rankings, upcoming_market_comparison}.json
"""

from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime, timezone

import pandas as pd

from ligamx import paths
from ligamx.date_utils import parse_date_only_series

COMPETITION_CODE = "MEX"
COMPETITION_NAME = "Liga MX"
MODEL_NAME = "Negative Binomial with Dixon-Coles Time Decay"
MODEL_VERSION = "v1.0"
DISPLAY_TZ = "America/Mexico_City"

MARKET_FIELDS = [
    "home_team", "away_team", "round", "match_time", "kickoff_at",
    "home_win_prob", "draw_prob", "away_win_prob",
    # The bettable price: best of Betano UK / Duel per outcome, with the book
    # that supplied it. EV is computed against this, not against Pinnacle.
    "home_odds", "draw_odds", "away_odds",
    "home_book", "draw_book", "away_book",
    "home_ev", "draw_ev", "away_ev",
    "signal_pick", "signal_state", "bookmaker",
    # Pinnacle as the low-vig reference, so a reader can tell a model edge from a
    # soft book simply being soft.
    "pinnacle_home_odds", "pinnacle_draw_odds", "pinnacle_away_odds",
    # These are opening prices, captured once and never refreshed, so how old the
    # quote is decides whether it is still gettable.
    "price_captured_at", "price_age_h",
    "last_update", "fetched_at",
]


def _slug(value) -> str:
    slug = str(value).strip().lower().replace(" ", "-")
    slug = re.sub(r"[^a-z0-9-]", "", slug)
    return re.sub(r"-{2,}", "-", slug).strip("-")


def _fixture_id(season, rnd, date, home, away) -> str:
    try:
        rnd_s = str(int(float(rnd)))
    except (TypeError, ValueError):
        rnd_s = str(rnd)
    return f"{COMPETITION_CODE}-{_slug(season)}-{rnd_s}-{date}-{_slug(home)}-{_slug(away)}"


def _fair(p):
    try:
        p = float(p)
    except (TypeError, ValueError):
        return None
    if p <= 0 or math.isnan(p):
        return None
    return round(1.0 / p, 4)


def _clean(v):
    if isinstance(v, float):
        if math.isnan(v):
            return None
        if v.is_integer():
            return int(v)
    if pd.isna(v) if not isinstance(v, (list, dict)) else False:
        return None
    return v


def _records(df: pd.DataFrame) -> list:
    return [{k: _clean(v) for k, v in row.items()} for row in df.to_dict("records")]


def _read_model_updated_at(default: str) -> str:
    try:
        with open(paths.model_meta_json(), encoding="utf-8") as fh:
            val = json.load(fh).get("model_updated_at")
        return str(val) if val else default
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def _form_map(matches: pd.DataFrame) -> dict:
    played = matches.copy()
    played["_d"] = parse_date_only_series(played["Date"])
    played["Res"] = played["Res"].astype(str).str.strip()
    played = played[played["_d"].notna() & played["Res"].isin(["H", "D", "A"])]
    played = played.sort_values("_d")
    token = {("H", True): "W", ("D", True): "D", ("A", True): "L",
             ("H", False): "L", ("D", False): "D", ("A", False): "W"}
    acc: dict[str, list] = {}
    for _, r in played.iterrows():
        h, a, res = str(r["Home"]).strip(), str(r["Away"]).strip(), r["Res"]
        acc.setdefault(h, []).append(token[(res, True)])
        acc.setdefault(a, []).append(token[(res, False)])
    return {t: ",".join(v[-5:][::-1]) for t, v in acc.items()}


def run() -> None:
    out_dir = paths.data_dashboard_json_dir()
    os.makedirs(out_dir, exist_ok=True)

    matches = pd.read_csv(paths.ligamx_data_csv())
    upcoming = pd.read_csv(paths.upcoming_fixtures_csv())
    stats = pd.read_csv(paths.team_stats_csv())
    sims = pd.read_csv(paths.match_simulations_csv())

    # Current season = the season carrying upcoming fixtures (fallback: latest played).
    if not upcoming.empty:
        season = str(upcoming["Season"].mode().iloc[0])
    else:
        m = matches.copy()
        m["_d"] = parse_date_only_series(m["Date"])
        season = str(m.sort_values("_d").iloc[-1]["Season"])

    now_utc = datetime.now(timezone.utc)
    updated_at = now_utc.isoformat(timespec="seconds")
    model_updated_at = _read_model_updated_at(updated_at)

    # ---- upcoming fixtures -------------------------------------------------
    up = upcoming.copy()
    up = up.sort_values("kickoff_utc") if "kickoff_utc" in up.columns else up
    up_rows = []
    for _, r in up.iterrows():
        up_rows.append({
            "fixture_id": _fixture_id(r.get("Season", season), r.get("round", ""), r["Date"], r["Home"], r["Away"]),
            "round": _clean(r.get("round", "")),
            "match_date": str(r["Date"]),
            "match_time": str(r.get("Time", "")),
            "kickoff_at": str(r.get("kickoff_utc", "")),
            "home_team": str(r["Home"]),
            "away_team": str(r["Away"]),
        })

    # ---- match predictions (upcoming joined to model 1X2) ------------------
    prob_lut = {}
    for _, r in sims.iterrows():
        prob_lut[(str(r["Home Team"]), str(r["Away Team"]))] = (
            r["Home Win Probability"], r["Draw Probability"], r["Away Win Probability"]
        )
    pred_rows = []
    for row in up_rows:
        probs = prob_lut.get((row["home_team"], row["away_team"]))
        if not probs:
            continue
        hw, dr, aw = float(probs[0]), float(probs[1]), float(probs[2])
        total = hw + dr + aw
        if total > 0:
            hw, dr, aw = hw / total, dr / total, aw / total
        pred_rows.append({
            "fixture_id": row["fixture_id"], "round": row["round"],
            "match_date": row["match_date"], "kickoff_at": row["kickoff_at"],
            "home_team": row["home_team"], "away_team": row["away_team"],
            "home_win_prob": round(hw, 6), "draw_prob": round(dr, 6), "away_win_prob": round(aw, 6),
            "home_win_fair_odds": _fair(hw), "draw_fair_odds": _fair(dr), "away_win_fair_odds": _fair(aw),
        })

    # ---- team strength (current-season teams only) -------------------------
    current_teams = set(up["Home"]).union(set(up["Away"]))
    current_teams |= set(matches[matches["Season"].astype(str) == season]["Home"])
    current_teams |= set(matches[matches["Season"].astype(str) == season]["Away"])

    form = _form_map(matches)
    st = stats.copy()
    st["attack_rating"] = pd.to_numeric(st["Attack"], errors="coerce")
    st["defense_rating"] = pd.to_numeric(st["Defense"], errors="coerce")
    st["team"] = st["Team"].astype(str)
    if current_teams:
        st = st[st["team"].isin(current_teams)]
    st["overall_rating"] = st["attack_rating"] - st["defense_rating"]
    st["attack_rank"] = st["attack_rating"].rank(method="min", ascending=False).astype(int)
    st["defense_rank"] = st["defense_rating"].rank(method="min", ascending=True).astype(int)
    st = st.sort_values(["overall_rating", "team"], ascending=[False, True]).reset_index(drop=True)
    st["rank_overall"] = st.index + 1
    st["form"] = st["team"].map(form).fillna("")
    strength_rows = _records(st[[
        "rank_overall", "team", "attack_rating", "defense_rating",
        "overall_rating", "attack_rank", "defense_rank", "form",
    ]])

    # ---- market comparison (1X2 EV; optional) ------------------------------
    market_rows = []
    if os.path.exists(paths.market_comparison_csv()):
        mc = pd.read_csv(paths.market_comparison_csv())
        mc = mc.rename(columns={"kickoff_utc": "kickoff_at"})
        for col in MARKET_FIELDS:
            if col not in mc.columns:
                mc[col] = None
        market_rows = _records(mc[MARKET_FIELDS])

    # ---- round progress + meta --------------------------------------------
    season_played = matches[(matches["Season"].astype(str) == season) &
                            (matches["Res"].astype(str).str.strip().isin(["H", "D", "A"]))]
    matches_played = int(len(season_played))
    up_rounds = pd.to_numeric(up["round"], errors="coerce").dropna()
    total_rounds = 17
    current_round = int(up_rounds.min()) if not up_rounds.empty else total_rounds

    m2 = matches.copy()
    m2["_d"] = parse_date_only_series(m2["Date"])
    last_completed = m2["_d"].max()
    next_fixture_date = up_rows[0]["match_date"] if up_rows else None

    meta = {
        "competition_code": COMPETITION_CODE,
        "competition_name": COMPETITION_NAME,
        "season": season,
        "updated_at": updated_at,
        "model_updated_at": model_updated_at,
        "timezone": DISPLAY_TZ,
        "last_completed_match_date": last_completed.strftime("%Y-%m-%d") if pd.notna(last_completed) else None,
        "next_fixture_date": next_fixture_date,
        "matches_played": matches_played,
        "current_round": current_round,
        "total_rounds": total_rounds,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
    }
    common = {"competition_code": COMPETITION_CODE, "season": season, "updated_at": updated_at}

    def _write(name, payload):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        n = len(payload["rows"]) if isinstance(payload, dict) and "rows" in payload else 1
        print(f"  wrote {name} ({n} rows)")

    _write("dashboard_meta.json", meta)
    _write("upcoming_fixtures.json", {"meta": common, "rows": up_rows})
    _write("match_predictions.json", {"meta": {**common, "model_name": MODEL_NAME, "model_version": MODEL_VERSION}, "rows": pred_rows})
    _write("team_strength_rankings.json", {"meta": common, "rows": strength_rows})
    _write("upcoming_market_comparison.json", {"meta": common, "rows": market_rows})
    print(f"Dashboard JSON exported to {out_dir}")


if __name__ == "__main__":
    run()
