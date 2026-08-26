"""
Export the dashboard's 5-file JSON contract from the pipeline outputs.

Reads:  MEX_ligamx.csv (history), MEX_upcoming_fixtures.csv (schedule),
        MEX_team_stats.csv + _match_simulations.csv (model),
        MEX_upcoming_market_comparison.csv (1X2 EV; optional).
Writes: data/dashboard/json/{dashboard_meta, upcoming_fixtures, match_predictions,
        team_strength_rankings, upcoming_market_comparison, match_results}.json
"""

from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime, timezone

import pandas as pd

from ligamx import paths, stage_meta
from ligamx.date_utils import parse_date_only_series

COMPETITION_CODE = "MEX"
COMPETITION_NAME = "Liga MX"
MODEL_NAME = "Negative Binomial with Dixon-Coles Time Decay"
MODEL_VERSION = "v1.0"
DISPLAY_TZ = "America/Mexico_City"

# Finished matches, as facts only: the score and who won. Deliberately NOT settled
# outcomes -- whether a bet won depends on which side was backed, at what price and
# on what stake, none of which this pipeline knows. Settlement belongs to whoever
# holds the bet, the same way EV belongs here and not on the board.
#
# Field-for-field identical to cslmonitor's match_results, so a consumer reads one
# contract rather than a per-league adapter.
RESULTS_FIELDS = [
    "fixture_id",
    "season",
    "round",
    "match_date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "result",
    "status",
]

# How far back match_results carries. A trailing window rather than "the current
# season" on purpose, and the reason is sharper here than in CSL: Liga MX runs two
# short seasons a year, so a season-scoped export would empty itself twice annually
# -- each time right when something downstream is still settling the final.
RESULTS_WINDOW_DAYS = 180

MARKET_FIELDS = [
    # Stable join key, derived with the same _fixture_id() that stamps
    # upcoming_fixtures.json and match_predictions.json, so the three payloads
    # join on equal terms. The market CSV does not carry it -- it is rebuilt here
    # rather than added there so there is exactly one construction of a fixture's
    # identity in this repo. A downstream ledger keys a signal's lifecycle on this;
    # re-deriving it from the (home_team, away_team) pair is lossy, because team
    # names are display strings that a rename upstream silently changes.
    "fixture_id",
    "home_team", "away_team", "round", "match_time", "kickoff_at",
    "home_win_prob", "draw_prob", "away_win_prob",
    # The bettable price: best of Betano UK / Duel per outcome, with the book
    # that supplied it. EV is computed against this, not against Pinnacle.
    "home_odds", "draw_odds", "away_odds",
    "home_book", "draw_book", "away_book",
    "home_ev", "draw_ev", "away_ev",
    "signal_pick", "signal_state", "bookmaker",
    # Why the best outcome is or is not a bet. "observed"/"window" mean its price
    # is a proven opener; "" means a first sighting of a market that was already
    # quoting, which the signal gate suppresses. Carried so a silent board can say
    # which of the two it is -- no edge, or no proof.
    "price_proof",
    # Pinnacle as the low-vig reference, so a reader can tell a model edge from a
    # soft book simply being soft. Its two clocks ride along so the anchor can be dated
    # like any other quote. They roll with every fetch rather than being banked once, so
    # a consumer presenting them as history has to freeze them at the moment it means.
    "pinnacle_home_odds", "pinnacle_draw_odds", "pinnacle_away_odds",
    "pinnacle_last_update", "pinnacle_fetched_at",
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


def _int_or_none(value):
    """A plain Python int, or None. `_clean` only unwraps floats, and pandas hands
    back numpy int64 here, which json.dump refuses."""
    n = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(n) else int(n)


def _build_results(matches: pd.DataFrame, now_utc: datetime) -> list:
    """Finished matches inside the trailing window, keyed by the same fixture_id.

    The id is rebuilt from each row's *own* Season and Round rather than the export's
    current season, so a match played in Clausura keeps the id it had while it was an
    upcoming fixture. That is the whole point of the file: it lets a consumer join a
    signal it recorded weeks ago to the score.

    `result` is derived from the score, not read from the CSV's `Res` column. The two
    disagree on real rows today -- both legs of the Clausura 2026 final -- and picking
    a winner between the two spellings would mean settling a bet on a guess. Those
    rows go out as `disputed` instead, which a consumer must not settle.

    Measured against six months of this repo's own committed fixture snapshots, 92.1%
    of fixtures whose scheduled date has passed join to a result on fixture_id
    exactly. 3.0% are postponed and not yet played, so their absence here is correct.
    The remaining 5.0% are the real weakness: the id embeds the match date, so a match
    moved even by a day gets a different id once played, and the signal recorded
    against the old id never finds its result.

    Those two compound -- a postponed match, once played, is by definition on a new
    date and so lands in the 5%. A consumer should fall back to
    (season, home_team, away_team), the same two-key shape the kickoff join uses.
    Missing a join leaves a bet unsettled rather than settled wrongly, which is the
    survivable direction.
    """
    df = matches.copy()
    df["_d"] = parse_date_only_series(df["Date"])
    df["_res"] = df["Res"].astype(str).str.strip()
    df = df[df["_d"].notna() & df["_res"].isin(["H", "D", "A"])].copy()

    cutoff = pd.Timestamp(now_utc).tz_convert("UTC").tz_localize(None).normalize() - pd.Timedelta(
        days=RESULTS_WINDOW_DAYS
    )
    df = df[df["_d"] >= cutoff].copy()
    if df.empty:
        return []

    df["home_goals"] = pd.to_numeric(df["HG"], errors="coerce")
    df["away_goals"] = pd.to_numeric(df["AG"], errors="coerce")
    # A row whose Res parsed but whose score did not is dropped rather than exported
    # with a null score: a consumer settling on it would read the missing side as zero.
    df = df[df["home_goals"].notna() & df["away_goals"].notna()].copy()
    if df.empty:
        return []
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)

    rows = []
    disputed = []
    for _, r in df.iterrows():
        derived = "H" if r["home_goals"] > r["away_goals"] else ("A" if r["home_goals"] < r["away_goals"] else "D")
        status = "finished" if derived == r["_res"] else "disputed"
        if status == "disputed":
            disputed.append(
                f"{r['_d'].date()} {r['Home']} {r['home_goals']}-{r['away_goals']} {r['Away']} "
                f"(Res={r['_res']}, score says {derived})"
            )
        rows.append({
            "fixture_id": _fixture_id(r["Season"], r.get("Round", ""), str(r["_d"].date()), r["Home"], r["Away"]),
            # str because cslmonitor's numeric season would otherwise be an int in one
            # league's payload and a string in the other's, for the same contract field.
            "season": str(r["Season"]),
            "round": _int_or_none(r.get("Round")),
            "match_date": str(r["_d"].date()),
            "home_team": str(r["Home"]),
            "away_team": str(r["Away"]),
            "home_goals": int(r["home_goals"]),
            "away_goals": int(r["away_goals"]),
            "result": derived,
            "status": status,
        })

    if disputed:
        print(f"  [WARN] {len(disputed)} match(es) where Res disagrees with the score, "
              f"exported as disputed and not settleable:")
        for d in disputed:
            print(f"    {d}")

    ids = [r["fixture_id"] for r in rows]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"match_results contains duplicate fixture_id values: {dupes[:5]}")
    for r in rows:
        if list(r.keys()) != RESULTS_FIELDS:
            raise ValueError(f"match_results row fields do not match the contract: {list(r.keys())}")

    rows.sort(key=lambda r: (r["match_date"], r["home_team"], r["away_team"]))
    return rows


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
    # These three answer different questions and must not be collapsed:
    # updated_at is when this export ran (CI re-runs it after every odds
    # capture), while the other two are when their stage last really ran. Null
    # when never stamped -- an unknown age must not render as a fresh one.
    model_updated_at = stage_meta.read(paths.model_meta_json(), "model_updated_at")
    fixtures_updated_at = stage_meta.read(paths.fixtures_meta_json(), "fixtures_updated_at")

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
        # Before the fill-missing loop below, or every id would be None: the market
        # CSV has no fixture_id column of its own. `season` is the export's season
        # rather than a per-row one because the comparison only ever covers upcoming
        # fixtures, which are by definition in it.
        mc["fixture_id"] = [
            _fixture_id(season, r["round"], r["match_date"], r["home_team"], r["away_team"])
            for _, r in mc.iterrows()
        ]
        for col in MARKET_FIELDS:
            if col not in mc.columns:
                mc[col] = None
        # "" is this field's "no proof", and read_csv turns a blank cell into NaN,
        # which would reach the JSON as null. Two spellings of absent is one more
        # than a consumer gating a bet on it should have to handle.
        if "price_proof" in mc.columns:
            mc["price_proof"] = mc["price_proof"].fillna("")
        market_rows = _records(mc[MARKET_FIELDS])

        # Fail here rather than let a consumer join on a key that matches nothing.
        # The comparison is built from the same upcoming fixtures, so a miss means
        # the two derivations have drifted (a season string, a round that arrived as
        # a float, a renamed team) -- and a silent drop downstream is the failure
        # mode this key exists to remove. Nothing has been written to disk yet.
        up_ids = {row["fixture_id"] for row in up_rows}
        mkt_ids = [row["fixture_id"] for row in market_rows]
        if len(set(mkt_ids)) != len(mkt_ids):
            raise ValueError("market comparison contains duplicate fixture_id values")
        orphans = sorted(set(mkt_ids) - up_ids)
        if orphans:
            raise ValueError(
                "market comparison fixture_id values not found in upcoming fixtures: "
                f"{orphans[:5]}"
            )

    result_rows = _build_results(matches, now_utc)

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
        "fixtures_updated_at": fixtures_updated_at,
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
    # `season` in the shared meta is the dashboard's current season, while these rows
    # carry their own -- the window is a trailing 180 days and, in a league with two
    # seasons a year, crosses a boundary roughly every other export.
    _write("match_results.json", {"meta": common, "rows": result_rows})
    print(f"Dashboard JSON exported to {out_dir}")


if __name__ == "__main__":
    run()
