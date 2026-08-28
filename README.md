# betmodel

One betting-model engine. Leagues are configuration, not forks.

Merged from `cslmonitor` (Chinese Super League) and `ligamxterminal` (Liga MX), which
had diverged into two implementations sharing a set of filenames. Both full histories
are preserved in this repo.

## Status: mid-migration

The engine is being assembled under `src/betmodel/`. Until a stage lands there, the
original code still lives under `legacy/csl/` and `legacy/ligamx/`. `legacy/` is
deleted once every stage has moved.

| | |
|---|---|
| Target layout | `src/betmodel/` engine, `leagues/*.yml` config, `data/<league>/`, `public/` contract |
| Adding a league should cost | one `leagues/<id>.yml` plus one team-name mapping CSV |
| Downstream consumer | `myevbettracker`, which reads `public/` over raw.githubusercontent |

## Layout

```
leagues/<id>.yml        every league-specific parameter; the only file a new league needs
src/betmodel/           the engine
  config/               league config schema + loader
  providers/            sofascore thesportsdb theoddsapi oddsapiio polymarket
  fixtures/ xg/ models/ odds/ signals/ publish/ notify/ eval/
data/<league>/          committed CSV/JSON; git history IS the database
public/                 the published contract (see docs/CONTRACT.md)
  index.json            league manifest; consumers hardcode only this address
  <league>/             signals.json results.json fixtures.json predictions.json ...
  legacy/<league>/      pre-merge JSON shapes, kept only until the tracker migrates
tests/golden/           frozen pre-merge outputs; the equivalence baseline
legacy/                 pre-merge source trees, deleted as stages land
```

## Why the data is committed

An opening line exists only while a bookmaker shows it. Neither odds provider sells
opener history at any tier, so `data/<league>/odds_capture_history.csv` is
irreproducible. It is append-only and committed. Two mechanisms depend on that:
the capture workflow gates its republish job on whether a commit actually appended,
and the signal alert derives its dedup baseline from `git show HEAD:`.

## Network split

SofaScore blocks datacenter IPs (HTTP 403), not just non-browser TLS fingerprints.
`curl_cffi` handles the handshake; the IP still has to be residential. Stages are
therefore tagged `network: cloud | residential` in the league config, and the
residential lane is a swappable implementation (proxy, self-hosted runner, or laptop).

## Verification

`tests/golden/` holds every published JSON, master CSV and model output captured from
both repos immediately before the merge, with `SHA256SUMS` and the two source commit
SHAs in `PROVENANCE.txt`. The merge is correct when the new engine reproduces those
outputs from those inputs. See the gates G1–G7 in the migration plan.
