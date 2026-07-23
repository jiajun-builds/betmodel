"use strict";

/* ------------------------------------------------------------------ *
 * Liga MX Terminal — static dashboard.
 * Reads the 5-file JSON contract:
 *   dashboard_meta.json, upcoming_fixtures.json, match_predictions.json,
 *   team_strength_rankings.json, upcoming_market_comparison.json
 * from ./data (built site) or ../data/dashboard/json (repo run).
 *
 * EV surface is Asian-Handicap (Pinnacle). EV values are percentages.
 * ------------------------------------------------------------------ */

const DATA_BASES = ["./data", "../data/dashboard/json"];
const DISPLAY_TZ = "America/Mexico_City";

const el = {
  overviewHero: document.getElementById("overview-hero"),
  overviewBody: document.getElementById("overview-body"),
  signalBody: document.getElementById("signal-body"),
  marketBody: document.getElementById("market-body"),
  strengthBody: document.getElementById("strength-body"),
  contextBody: document.getElementById("context-body"),
  roundFill: document.getElementById("round-fill"),
};

/* ---------- helpers ---------- */
function setText(bind, value) {
  document.querySelectorAll(`[data-bind="${bind}"]`).forEach((n) => { n.textContent = value; });
}
function pct(v) { return v == null ? "--" : `${(Number(v) * 100).toFixed(1)}%`; }
function rating(v) { return Number(v).toFixed(3); }
function odds(v) { return v == null ? "--" : Number(v).toFixed(2); }
function ev(v) { if (v == null || v === "") return "--"; const n = Number(v); return (n >= 0 ? "+" : "") + n.toFixed(1) + "%"; }
function evClass(v) { if (v == null || v === "") return "zero"; const n = Number(v); return n > 0.1 ? "pos" : n < -0.1 ? "neg" : "zero"; }
function sideLetter(k) { return k === "home" ? "H" : k === "away" ? "A" : "D"; }
function sideWord(k) { return k === "home" ? "HOME" : k === "away" ? "AWAY" : "DRAW"; }
function fmtSpread(s) { if (s == null || s === "") return ""; const n = Number(s); return (n > 0 ? "+" : "") + n; }

function fmtStamp(v) {
  if (!v) return "--";
  const d = new Date(v);
  if (Number.isNaN(d.getTime())) return String(v);
  return new Intl.DateTimeFormat("en-GB", {
    day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit",
    hour12: false, timeZone: DISPLAY_TZ,
  }).format(d).replace(",", "");
}
function fmtDay(v, fb) {
  if (!v) return fb || "--";
  const d = new Date(v);
  if (Number.isNaN(d.getTime())) return fb || "--";
  return new Intl.DateTimeFormat("en-CA", { year: "numeric", month: "2-digit", day: "2-digit", timeZone: DISPLAY_TZ }).format(d);
}
function fmtTime(v, fb) {
  if (!v) return fb || "--";
  const d = new Date(v);
  if (Number.isNaN(d.getTime())) return fb || "--";
  return new Intl.DateTimeFormat("en-GB", { hour: "2-digit", minute: "2-digit", hour12: false, timeZone: DISPLAY_TZ }).format(d);
}
function clock() {
  return new Intl.DateTimeFormat("en-GB", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false, timeZone: DISPLAY_TZ }).format(new Date());
}
function esc(s) { return String(s == null ? "" : s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }

/* ---------- OVERVIEW (best-bet hero + firing signals) ---------- */
function renderOverview(market) {
  const bets = market
    .filter((r) => r.signal_state === "bet" && r.signal_pick && r.signal_pick !== "none")
    .map((r) => {
      const home = r.signal_pick === "home";
      return {
        time: r.match_time, kickoff_at: r.kickoff_at,
        match: `${r.home_team} vs ${r.away_team}`,
        team: home ? r.home_team : r.away_team,
        key: r.signal_pick,
        spread: home ? r.spread : (r.spread == null ? null : -r.spread),
        odds: home ? r.home_odds : r.away_odds,
        ev: home ? r.home_ev : r.away_ev,
      };
    })
    .sort((a, b) => b.ev - a.ev);

  setText("ov-signal-count", String(bets.length));
  setText("ov-signal-total", String(market.length));

  renderHero(bets, market);

  if (!el.overviewBody) return;
  if (!bets.length) {
    el.overviewBody.innerHTML = `<tr class="ov-empty"><td colspan="6">— No +EV signals on the current slate.</td></tr>`;
    return;
  }
  el.overviewBody.innerHTML = bets.map((b) => `<tr>
    <td class="ov-time">${fmtTime(b.kickoff_at, b.time)}</td>
    <td class="ov-match">${esc(b.match)}</td>
    <td class="ov-pick">${esc(b.team)} <span class="ov-side">(${sideLetter(b.key)} ${fmtSpread(b.spread)})</span></td>
    <td class="num ov-odds">${odds(b.odds)}</td>
    <td class="num ov-ev${b.ev >= 5 ? " strong" : ""}">${ev(b.ev)}</td>
    <td class="ov-sig"><span class="badge">● BET</span></td>
  </tr>`).join("");
}

function renderHero(bets, market) {
  if (!el.overviewHero) return;

  let pick = bets[0] || null;
  const firing = Boolean(pick);
  if (!pick) {
    market.forEach((r) => {
      [["home", r.home_ev, r.home_odds, r.home_team, r.spread],
       ["away", r.away_ev, r.away_odds, r.away_team, (r.spread == null ? null : -r.spread)]].forEach(([k, e, o, t, sp]) => {
        if (e == null || e === "") return;
        if (!pick || Number(e) > pick.ev) {
          pick = { ev: Number(e), odds: o, key: k, team: t, spread: sp, match: `${r.home_team} vs ${r.away_team}`, time: r.match_time, kickoff_at: r.kickoff_at };
        }
      });
    });
  }

  if (!pick) {
    el.overviewHero.className = "hero hero--empty";
    el.overviewHero.innerHTML = `<div class="hero__main"><span class="hero__label">★ BEST BET</span><span class="hero__team">No market data</span></div>`;
    return;
  }

  const when = fmtTime(pick.kickoff_at, pick.time);
  const cta = firing ? `<span class="badge">● BET</span>` : `<span class="badge badge--cap">BELOW THRESHOLD</span>`;

  el.overviewHero.className = "hero" + (firing ? " hero--bet" : " hero--flat");
  el.overviewHero.innerHTML = `
    <div class="hero__main">
      <span class="hero__label">${firing ? "★ BEST BET" : "★ TOP EDGE"}</span>
      <div class="hero__headline">
        <span class="hero__team">${esc(pick.team)}</span>
        <span class="hero__side">${sideWord(pick.key)} ${fmtSpread(pick.spread)}</span>
      </div>
      <span class="hero__ctx">${esc(pick.match)}${when ? ` · ${when}` : ""}</span>
    </div>
    <div class="hero__metrics">
      <div class="hero__metric"><span class="hero__metric-k">PINNACLE</span><span class="hero__metric-v">${odds(pick.odds)}</span></div>
      <div class="hero__metric"><span class="hero__metric-k">EDGE (EV)</span><span class="hero__metric-v ${evClass(pick.ev)}">${ev(pick.ev)}</span></div>
      <div class="hero__cta">${cta}</div>
    </div>`;
}

/* ---------- EV BET (Asian-Handicap sides) ---------- */
function renderEvBet(rows) {
  el.signalBody.innerHTML = rows.map((row) => {
    const outs = [
      { key: "home", label: row.home_team, spread: row.spread, prob: row.home_win_prob, odds: row.home_odds, ev: row.home_ev },
      { key: "away", label: row.away_team, spread: row.spread == null ? null : -row.spread, prob: row.away_win_prob, odds: row.away_odds, ev: row.away_ev },
    ];
    const isBet = row.signal_state === "bet";

    const tr = outs.map((o, i) => {
      const timeCell = i === 0 ? `<td class="sig-time" rowspan="2">${fmtTime(row.kickoff_at, row.match_time)}</td>` : "";
      const nameCls = "sig-name" + (isBet && row.signal_pick === o.key ? " is-pick" : "");
      const evStrong = o.ev != null && Math.abs(Number(o.ev)) >= 5 ? " strong" : "";
      const action = (row.signal_pick === o.key && isBet)
        ? `<span class="sig-action"><span class="badge">● BET</span></span>` : "";
      return `<tr>
        ${timeCell}
        <td class="${nameCls}">${esc(o.label)} <span class="ov-side">${fmtSpread(o.spread)}</span></td>
        <td class="num sig-prob">${pct(o.prob)}</td>
        <td class="num c-grp sig-odds">${odds(o.odds)}</td>
        <td class="num sig-ev ${evClass(o.ev)}${evStrong}">${ev(o.ev)}</td>
        <td class="c-grp">${action}</td>
      </tr>`;
    }).join("");

    return `<tbody class="fixture${isBet ? " fixture--bet" : ""}">${tr}</tbody>`;
  }).join("");
}

/* ---------- SCHEDULE (1X2 model probabilities) ---------- */
function renderMarket(rows) {
  el.marketBody.innerHTML = rows.map((r) => {
    const trio = [
      { k: "home", v: r.home_win_prob, f: r.home_win_fair_odds },
      { k: "draw", v: r.draw_prob, f: r.draw_fair_odds },
      { k: "away", v: r.away_win_prob, f: r.away_win_fair_odds },
    ];
    const maxKey = trio.reduce((a, b) => (b.v > a.v ? b : a)).k;
    const cell = (c) => `<td class="num prob-cell${c.k === maxKey ? " is-max" : ""}"><span class="pv">${pct(c.v)}</span><span class="pf">F ${odds(c.f)}</span></td>`;
    return `<tr>
      <td class="mk-rnd">${esc(r.round)}</td>
      <td class="mk-date">${fmtDay(r.kickoff_at, r.match_date)}</td>
      <td class="mk-time">${fmtTime(r.kickoff_at, r.match_time)}</td>
      <td class="mk-match">${esc(r.home_team)} vs ${esc(r.away_team)}</td>
      ${cell({ ...trio[0] }).replace('class="num prob-cell', 'class="num c-grp prob-cell')}
      ${cell(trio[1])}
      ${cell(trio[2])}
    </tr>`;
  }).join("");
}

/* ---------- TEAM STRENGTH ---------- */
function renderStrength(rows) {
  el.strengthBody.innerHTML = rows.map((t) => {
    const form = (t.form || "").split(",").filter(Boolean).map((x) => {
      const c = x.trim();
      const cls = c === "W" ? "w" : c === "D" ? "d" : "l";
      return `<span class="${cls}">${c}</span>`;
    }).join("");
    return `<tr>
      <td class="num st-rk${t.rank_overall <= 3 ? " top" : ""}">${t.rank_overall}</td>
      <td class="st-team">${esc(t.team)}</td>
      <td class="num c-grp st-ovr">${rating(t.overall_rating)}</td>
      <td class="num st-r">${rating(t.attack_rating)}</td>
      <td class="num st-r">${rating(t.defense_rating)}</td>
      <td class="c-grp"><div class="form">${form}</div></td>
    </tr>`;
  }).join("");
}

/* ---------- MODEL CONTEXT ---------- */
function renderContext(m) {
  const rows = [
    ["Competition", m.competition_name], ["Season", m.season],
    ["Last Update", m.updated_at], ["Model Update", m.model_updated_at],
    ["Timezone", DISPLAY_TZ], ["Last Completed", m.last_completed_match_date],
    ["Next Fixture", m.next_fixture_date], ["Model Name", m.model_name],
    ["Version", m.model_version], ["Matches Played", m.matches_played],
  ];
  el.contextBody.innerHTML = rows.map(([k, v]) => `<div class="meta__row"><dt>${k}</dt><dd>${esc(v)}</dd></div>`).join("");
}

/* ---------- HEADER + KPI ---------- */
function renderHeader(meta, fixtures, predictions, strength, market, marketFetched) {
  setText("masthead-trail", `${meta.competition_name} · Season ${meta.season} · ${meta.model_name} · ${meta.model_version}`);
  setText("masthead-next-date", meta.next_fixture_date);
  setText("masthead-updated", fmtStamp(meta.updated_at));
  setText("played", String(meta.matches_played));
  setText("round-label", `${meta.current_round}/${meta.total_rounds}`);
  if (el.roundFill) el.roundFill.style.width = `${Math.round((meta.current_round / meta.total_rounds) * 100)}%`;

  setText("panel-signal-meta", `Model ${fmtStamp(meta.model_updated_at)} · Pinnacle ${fmtStamp(marketFetched)}`);
  setText("panel-market-meta", `${predictions.length} matches · ${meta.model_name}`);
  setText("panel-strength-meta", `${strength.length} clubs · recent 5 form`);

  const nf = predictions[0] || fixtures[0];
  if (nf) {
    setText("next-fixture", `${nf.home_team} vs ${nf.away_team}`);
    setText("next-note", `${fmtDay(nf.kickoff_at, nf.match_date)} · ${fmtTime(nf.kickoff_at, nf.match_time)}`);
  }
  const sc = strength[0];
  if (sc) {
    setText("strong-team", sc.team);
    setText("strong-note", `OVR ${rating(sc.overall_rating)} · ATT ${rating(sc.attack_rating)} · DEF ${rating(sc.defense_rating)}`);
  }
}

/* ---------- nav + clock + boot ---------- */
function initNav() {
  const links = Array.from(document.querySelectorAll("[data-view-target]"));
  const views = Array.from(document.querySelectorAll("[data-view]"));
  links.forEach((link) => {
    link.addEventListener("click", () => {
      const target = link.dataset.viewTarget;
      views.forEach((v) => v.classList.toggle("view--active", v.dataset.view === target));
      links.forEach((l) => {
        const active = l.dataset.viewTarget === target;
        l.classList.toggle("tab--active", active);
        l.setAttribute("aria-current", active ? "page" : "false");
      });
    });
  });
}
function startClock() { const tick = () => setText("masthead-clock", clock()); tick(); setInterval(tick, 1000); }

async function loadJson(name) {
  const errs = [];
  for (const base of DATA_BASES) {
    try {
      const res = await fetch(`${base}/${name}`);
      if (res.ok) return res.json();
      errs.push(`${base}/${name} -> ${res.status}`);
    } catch (e) { errs.push(`${base}/${name} -> ${e}`); }
  }
  throw new Error(`Failed to load ${name}. Tried: ${errs.join(", ")}`);
}

async function bootstrap() {
  try {
    const [meta, fixturesP, predictionsP, strengthP, marketP] = await Promise.all([
      loadJson("dashboard_meta.json"),
      loadJson("upcoming_fixtures.json"),
      loadJson("match_predictions.json"),
      loadJson("team_strength_rankings.json"),
      loadJson("upcoming_market_comparison.json"),
    ]);
    const fixtures = fixturesP.rows;
    const predictions = predictionsP.rows;
    const strength = strengthP.rows;
    const kickoffByTeams = new Map(fixtures.map((f) => [`${f.home_team}|${f.away_team}`, f.kickoff_at]));
    const market = marketP.rows.map((row) => ({
      ...row,
      kickoff_at: row.kickoff_at ?? kickoffByTeams.get(`${row.home_team}|${row.away_team}`),
    }));
    const marketFetched = market.reduce((m, r) => (r.fetched_at && r.fetched_at > m ? r.fetched_at : m), "");

    renderEvBet(market);
    renderOverview(market);
    renderMarket(predictions);
    renderStrength(strength);
    renderContext(meta);
    renderHeader(meta, fixtures, predictions, strength, market, marketFetched);
    startClock();
  } catch (error) {
    console.error(error);
    const banner = document.createElement("div");
    banner.className = "state-banner";
    banner.textContent = "Terminal data load failed. Ensure dashboard JSON files exist in ./data or ../data/dashboard/json.";
    document.querySelector(".term").prepend(banner);
  }
}

initNav();
bootstrap();
