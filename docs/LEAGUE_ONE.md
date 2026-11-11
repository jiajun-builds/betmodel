# China League One: integration framework

League One is **auxiliary evidence for the CSL model**, not a league of its own.
It has no odds capture, no anchor, no signals and no board. Its matches enter the
CSL fit so that a promoted club arrives in the CSL with a rating that rests on
something, and nothing else about the tool changes.

## Why this shape

The September 2026 walk-forward (four promoted clubs, 2025-26) found:

- pooled at weight 0.5 with League One's own scoring level and home advantage,
  cross-entropy against Pinnacle's close fell 2.26 (x1000), 95% CI [-4.89, -0.36];
- weight 0.25 and 0.5 could not be told apart, 1.0 was slightly worse;
- CLV was unchanged (+14.0% against +13.7%);
- fixtures priced from League One data alone were very poor (excess about 195);
- all of the gain was in promoted-club fixtures; no other fixture moved.

So it is worth having before a season starts, it is not worth betting on by
itself, and it must not be able to make a fixture fire.

Rerun on 2026-11-11 against today's model, which since September also corrects
clubs by their market offset and pools with the opener (D38), the gain is much
smaller: -0.23 [-0.58, +0.07] over 2025-26, -0.97 [-2.39, +0.28] on
promoted-club fixtures, nothing on the rest, and the same 21 bets. It is in as a
change that harms nothing measurable and probably helps promoted clubs. D42 has
the table.

A separate `leagues/league_one.yml` was rejected: a league file means captures,
an anchor and a threshold, and none of that has evidence behind it here.

## Invariants

These are the rules the code and tests state. Each is there for a measured reason.

1. **Off means identical.** A league with no auxiliary configured fits to the
   same floating point as before. G1 stays green with no exemption, and Liga MX
   is untouched.
2. **The primary's weighting does not move.** The training window, the as-of
   date and the Dixon-Coles reference date come from the primary match table
   alone. Auxiliary rows dated after the primary's latest match are left out of
   the fit, and `min_train` counts primary rows only.
3. **Auxiliary matches are down-weighted and on their own level.** Their weight
   is the Dixon-Coles weight times `weight` (0.5). They get their own intercept
   shift and their own home-advantage shift, so League One's lower home
   advantage (0.24 against 0.28) cannot drag the CSL's down. The division gap is
   then identified only by clubs that have played in both, which is the honest
   source of it.
4. **Only CSL clubs are priced.** `team_stats.csv` and `simulations.csv` list
   exactly the clubs with a primary match in the window, as today. A club with
   only League One matches is not in them, so it cannot be priced until it has
   played a CSL match.
5. **Auxiliary matches are not evidence.** `Matches` and `WeightedMatches` count
   primary matches at their undiluted weight. `min_team_evidence` and the
   walk-forward's `evidence` therefore read the same numbers with or without
   League One. The auxiliary counts are reported in separate columns.
6. **No leak in the replay.** `team_offset.walk_forward` and
   `scripts/walkforward.py` give each day's fit the auxiliary rows strictly
   before that day, through the same function production uses.
7. **A blank target stays blank.** A League One match with no xG has no target
   and leaves the fit, exactly as a CSL match does. An all-zero xG pair is the
   provider's placeholder and is stored blank.
8. **A failed auxiliary sync is loud and not fatal.** It logs what it could not
   do and leaves the primary xG stage's result standing.

## Data

`data/csl/league_one_matches.csv`, committed. One row per finished match, keyed
by SofaScore event id, sorted by kickoff.

| Column | Meaning |
|---|---|
| `event_id` | SofaScore event id; the upsert key |
| `Season` | season year, e.g. `2026` |
| `Round` | `providers.sofascore.round_label` |
| `Date`, `Time` | kickoff in UTC, `YYYY-MM-DD` and `HH:MM` |
| `kickoff_utc` | `betmodel.dates.stamp` of the kickoff |
| `home_sofascore`, `away_sofascore` | SofaScore's team names, as served |
| `home_team_id`, `away_team_id` | SofaScore team ids |
| `HG`, `AG` | regulation goals, `providers.sofascore.event_goals` |
| `HxG`, `AxG` | xG per the CSL source's `period_strategy`; blank if unpublished |

The file stores SofaScore's spelling. The loader resolves each name through the
CSL mapping's `sofascore_team` column to the canonical CSL name, and a club the
mapping does not know keeps its SofaScore name. That is why relegated and
promoted clubs need their `sofascore_team` cell filled in
`data/csl/team_name_mapping.csv`: an unlinked club is two clubs to the fitter and
the division gap loses one of the few bridges it has.

The blended target is computed at load time from the league's `xg_blend`, so a
blend change cannot leave the two tables on different targets.

SofaScore: unique tournament 782. Season ids 2024 = 58844, 2025 = 72904,
2026 = 90905. 2023 (50458) has no xG at all and is not fetched.

## Configuration

In `leagues/csl.yml`:

```yaml
model:
  auxiliary:
    - id: league_one
      name: China League One
      file: league_one_matches.csv
      weight: 0.5
      own_level: true
      enabled: true
      source:
        provider: sofascore
        unique_tournament_id: 782
```

## Code

| Where | What |
|---|---|
| `config/schema.py` | `AuxiliarySource`, parsed into `ModelConfig.auxiliary` |
| `models/auxiliary.py` | load a file, resolve names, compute the target, select the rows a fit may use |
| `models/poisson.py` | optional per-match group with its own intercept and home-advantage shift |
| `models/dc.py` | `fit_production_model(train, config, auxiliary=...)`; `ProductionFit` reports primary clubs and primary evidence |
| `models/team_offset.py`, `scripts/walkforward.py` | thread the auxiliary rows through the replay; `--aux-weight` to trial |
| `models/run.py` | read the auxiliary, write primary clubs only, add the auxiliary count columns |
| `xg/auxiliary_sync.py`, `cli.py` | the `xg` stage also upserts new League One matches and refetches the review window |

## Plan

1. **Backfill** seasons 2024-2026 from SofaScore into the file above, and report
   xG coverage per season, the team list with ids, and whether per-half xG is
   real or zeros.
2. **Build** the code in the table, with unit tests for each invariant.
3. **Validate** with `scripts/walkforward.py csl` against `--aux-weight 0`:
   cross-entropy against the close must be lower on 2025-26 and CLV not worse,
   or the framework ships with `enabled: false`. Both held, narrowly; see above.
4. **Record** the decision as D42 in `docs/DECISIONS.md`.
5. **Launch**: full suite green, then a pull request to `main`. Model artefacts
   under `data/csl/model/` are not committed by hand; the next refresh refits.

## Not covered

- Whether Pinnacle, 1xBet or Duel price League One was never checked. It does
  not matter for this shape.
- The evidence is four promoted clubs. The next test is the 2027 promoted
  clubs, and it only arrives once this is already running.
