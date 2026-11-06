#!/usr/bin/env python3
"""Alert when a league's configured de-bias is not reaching its published signals.

Falling back to raw probabilities for one fixture is normal and deliberate: no
anchor book has opened that match yet, and `signals.debias.apply` says so in its
own docstring. The label on each signal records which path ran, so nothing is
hidden.

A whole league falling back is a different condition wearing the same clothes. It
means the anchor is not being captured at all, and the league has been publishing
uncalibrated probabilities under a config that says otherwise. That happened here:
the Chinese Super League ran raw for three days because its anchor poll was being
refused by its own quota floor, and the only thing that surfaced it was reading
the config and the output side by side by hand.

The correction is not cosmetic. Measured on six fixtures, de-biasing moved EV by
2.3 to 6.4 percentage points, always downward, against a signal threshold of 20 --
enough to turn a fixture that should not fire into one that does.

**Not captured and not proven look the same in the output, and they are not.**
Both publish `raw`. The first means the poll never got the price; the second means
it did, and the anchor gate refused it because nothing showed it was the opener.
Liga MX round 11 was entirely the second, after an international break, under an
alert that said "not being captured" and sent the reader to the provider instead
of the proof. So the alert now names the fixtures and says which case each is.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
import sys

#: Below this share of anchored signals, a league configured for market_anchor is
#: not merely waiting on a few unopened fixtures. Deliberately forgiving: early in
#: a round most fixtures legitimately have no anchor yet, and an alarm that fires
#: on a normal Monday gets muted.
MIN_ANCHORED_SHARE = 0.5

ANCHORED = "market_anchor"
RAW = "raw"


#: What ``anchors`` says about a raw signal with no anchor open on file at all.
MISSING = None


def _classify(
    raw: list[dict], anchors: dict[str, str | None],
) -> tuple[list[dict], dict[str, list[dict]]]:
    """Raw signals with no anchor opener on file, and the rest by its proof."""
    missing = [s for s in raw if anchors.get(s.get("fixture_id"), MISSING) is MISSING]
    refused: dict[str, list[dict]] = {}
    for s in raw:
        proof = anchors.get(s.get("fixture_id"), MISSING)
        if proof is not MISSING:
            refused.setdefault(proof or "none", []).append(s)
    return missing, refused


def _diagnosis(raw: list[dict], anchors: dict[str, str | None]) -> str:
    """Which raw signals lack the price and which had it refused."""
    missing, refused = _classify(raw, anchors)

    def names(rows):
        return ", ".join(f"{s.get('home_team')} v {s.get('away_team')}" for s in rows)

    parts = []
    if missing:
        parts.append(
            f"{len(missing)} have no anchor opening price on file -- the anchor "
            f"book's openers are not being captured: {names(missing)}"
        )
    for proof, rows in sorted(refused.items()):
        if proof == "observed":
            parts.append(
                f"{len(rows)} have an observed anchor opener on file and were "
                f"still published raw -- the engine did not apply it: {names(rows)}"
            )
            continue
        parts.append(
            f"{len(rows)} have an anchor opener on file that the anchor gate "
            f"refused (proof {proof!r}, not 'observed') -- nothing showed the book "
            f"unpriced shortly before it, so read capture_watch.csv: {names(rows)}"
        )
    return "; ".join(parts)


def assess(
    league: str, method: str, signals: list[dict],
    anchors: dict[str, str | None] | None = None,
) -> tuple[bool, str]:
    """Decide for one league. Pure, so the threshold can be tested.

    ``anchors`` maps a fixture id to the proof of the anchor opener on file for
    it, or to ``MISSING`` when there is none. Optional: without it the alert still
    fires, it just cannot say which of the two causes it is.
    """
    if method != ANCHORED:
        return True, f"{league}: de-bias is {method!r}; nothing to check"
    if not signals:
        return True, f"{league}: no priced fixture to judge"

    anchored = sum(1 for s in signals if s.get("model", {}).get("method") == ANCHORED)
    share = anchored / len(signals)
    if share >= MIN_ANCHORED_SHARE:
        return True, (f"{league}: {anchored}/{len(signals)} signals anchored "
                      f"({share:.0%})")
    head = (
        f"{league} is configured for market_anchor but only {anchored} of "
        f"{len(signals)} published signals are anchored ({share:.0%}), so the "
        f"league is publishing uncalibrated probabilities."
    )
    if anchors is None:
        return False, (f"{head} The anchor book's opening prices are either not "
                       f"being captured or not proven as openers.")
    raw = [s for s in signals if s.get("model", {}).get("method") != ANCHORED]
    return False, f"{head} Of those that are not: {_diagnosis(raw, anchors)}."


def anchor_proofs(league: str, config, signals: list[dict]) -> dict[str, str | None]:
    """The anchor opener on file for each signal's fixture, by its proof.

    Read through `reduce.collapse_opens`, the same reduction the engine uses, so
    this cannot disagree with the gate about what is on file.
    """
    import pandas as pd

    from betmodel.dates import local_matchday
    from betmodel.odds import reduce

    book = config.signals.debias.anchor_book
    opens = reduce.collapse_opens(league, config)
    out: dict[str, str | None] = {}
    for s in signals:
        kickoff = pd.Timestamp(s["kickoff_utc"]).to_pydatetime()
        day = local_matchday(kickoff, config.timezone)
        held = opens.get((s["home_team"], s["away_team"], day, book))
        out[s["fixture_id"]] = MISSING if held is None else held["proof"]
    return out


def _escape(text) -> str:
    return html.escape(str(text), quote=False)


def format_alert(
    name: str, signals: list[dict], anchors: dict[str, str | None] | None,
) -> str:
    """The Telegram message for one league, in the shape of the other alerts.

    :func:`assess` stays the console's account; this is the reader's. Same facts,
    one message per league, naming each fixture under the case it is.
    """
    anchored = sum(1 for s in signals if s.get("model", {}).get("method") == ANCHORED)
    raw = [s for s in signals if s.get("model", {}).get("method") != ANCHORED]

    def names(rows):
        return "、".join(f"{_escape(s.get('home_team'))} vs {_escape(s.get('away_team'))}"
                        for s in rows)

    lines = [
        "⚠️ <b>锚定盘未校准</b>",
        f"<b>{_escape(name)}</b>",
        f"已校准: <b>{anchored}/{len(signals)} 场</b>"
        f"（{anchored / len(signals):.0%}，低于 {MIN_ANCHORED_SHARE:.0%}）",
        "其余场次为未校准概率，不会发出下注信号。",
    ]
    if anchors is None:
        lines.append("原因未知: 锚定盘开盘价未抓取，或未被证明为开盘价。")
        return "\n".join(lines)
    missing, refused = _classify(raw, anchors)
    if missing:
        lines.append(f"❌ 无锚定开盘价（未抓取）{len(missing)} 场: {names(missing)}")
    for proof, rows in sorted(refused.items()):
        if proof == "observed":
            lines.append(f"❌ 有 observed 开盘价但引擎未应用 {len(rows)} 场: {names(rows)}")
            continue
        lines.append(f"⚠️ 开盘价无法确认是首个报价 {len(rows)} 场: {names(rows)}")
    if any(proof != "observed" for proof in refused):
        lines.append("无法确认的场次请查看 capture_watch.csv。")
    return "\n".join(lines)


def alert(text: str) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not (token and chat):
        print("no telegram credentials; alerting skipped", file=sys.stderr)
        return
    try:
        subprocess.run(
            ["curl", "-sS", "--max-time", "20", "-o", "/dev/null",
             f"https://api.telegram.org/bot{token}/sendMessage",
             "--data-urlencode", f"chat_id={chat}",
             "--data-urlencode", "parse_mode=HTML",
             "--data-urlencode", "disable_web_page_preview=true",
             "--data-urlencode", f"text={text}"],
            check=True, capture_output=True)
    except Exception as exc:  # noqa: BLE001
        print(f"alert could not be sent: {exc}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from betmodel import paths
    from betmodel.config import available_leagues, load_league

    problems = []
    for league in available_leagues():
        config = load_league(league)
        if not config.publish.published:
            continue
        path = paths.for_league(league).public_json("signals")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            signals = json.load(fh).get("signals", [])
        method = config.signals.debias.method
        anchors = None
        if method == ANCHORED and signals:
            try:
                anchors = anchor_proofs(league, config, signals)
            except Exception as exc:  # noqa: BLE001
                # The diagnosis is a courtesy; the alarm must not depend on it.
                print(f"{league}: could not read the anchor openers: {exc}",
                      file=sys.stderr)
        ok, message = assess(league, method, signals, anchors)
        print(message)
        if not ok:
            problems.append(format_alert(config.name, signals, anchors))

    for text in problems:
        if args.dry_run:
            print(f"would send:\n{text}")
        else:
            alert(text)
    # Reported, not fatal: the refresh has its own work and its own failure path.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
