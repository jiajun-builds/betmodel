"use strict";

/* ------------------------------------------------------------------ *
 * CSL Terminal — native drop-in dashboard.
 * Reads the same JSON contract as the existing build:
 *   dashboard_meta.json, upcoming_fixtures.json, match_predictions.json,
 *   team_strength_rankings.json, upcoming_market_comparison.json
 * from ./data (built site) or ../data/dashboard/json (repo run).
 * ------------------------------------------------------------------ */

const DATA_BASES = ["./data", "../data/dashboard/json"];
const DISPLAY_TZ = "Europe/London";
const SIGNAL_ODDS_CAP = 7;

/* The books we bet into. Mirrors src/csl/odds/books.py — same keys, same order.
 * Order drives the left-to-right odds columns; the Python side additionally uses it
 * as the best-price tie-break, so keep the two in sync.
 *
 * `key` is the single identity: it is the column-name root (`${key}_open_...`), the
 * token inside signal_book / signal_books, and the logo filename stem. Logo files
 * MUST stay lowercase — macOS resolves any case locally but GitHub Pages serves from
 * Linux and 404s, so a capitalised stem breaks in production only. */
const BOOKS = [
  {
    key: "onexbet",
    label: "1xBet",
    logo: "./assets/onexbet.png",
    url: "https://1xbetjap.com/en/line/football/58043-china-super-league",
  },
  {
    key: "duel",
    label: "Duel",
    logo: "./assets/duel.png",
    url: "https://duel.com/sports?bt-path=/soccer/china/chinese-super-league-1669818818899349504",
  },
];
const BOOK_BY_KEY = new Map(BOOKS.map((b) => [b.key, b]));
const oddsCol = (key, side) => `${key}_open_${side}_odds`;

const el = {
  overviewHero: document.getElementById("overview-hero"),
  overviewBody: document.getElementById("overview-body"),
  signalBody: document.getElementById("signal-body"),
  marketBody: document.getElementById("market-body"),
  marketFilter: document.getElementById("market-filter"),
  strengthBody: document.getElementById("strength-body"),
  contextBody: document.getElementById("context-body"),
  roundFill: document.getElementById("round-fill"),
};

/* ---------- helpers ---------- */
function setText(bind, value) {
  document.querySelectorAll(`[data-bind="${bind}"]`).forEach((n) => { n.textContent = value; });
}
function pct(v) { return `${(v * 100).toFixed(1)}%`; }
function rating(v) { return Number(v).toFixed(3); }
function goals(v) { return Number(v).toFixed(2); }
function odds(v) { return v == null ? "--" : Number(v).toFixed(2); }
function ev(v) { if (v == null) return "--"; const n = Number(v); return (n >= 0 ? "+" : "") + n.toFixed(3); }
function evClass(v) { if (v == null) return "zero"; return v > 0.0005 ? "pos" : v < -0.0005 ? "neg" : "zero"; }
function sideLetter(k) { return k === "home" ? "H" : k === "away" ? "A" : "D"; }
function sideWord(k) { return k === "home" ? "HOME WIN" : k === "away" ? "AWAY WIN" : "DRAW"; }

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

/* ---------- books ---------- */
/* signal_books is a "|"-joined string, and null on every non-firing row — hence the
 * String(v || "") coercion. Unknown keys are dropped rather than rendered broken, so a
 * third book shipped in the JSON ahead of this file degrades to a missing logo instead
 * of a broken page. */
function bookList(v) {
  return String(v || "").split("|").filter(Boolean)
    .map((k) => BOOK_BY_KEY.get(k)).filter(Boolean);
}
function bookLogo(b) {
  if (!b) return "";
  return `<span class="booklogo"><img src="${esc(b.logo)}" alt="${esc(b.label)}" title="${esc(b.label)}" loading="lazy" /></span>`;
}
/* Each logo is itself the link — clicking a book's mark opens that book's league page. */
function bookLogoLink(b) {
  if (!b) return "";
  return `<a class="booklink" href="${esc(b.url)}" target="_blank" rel="noopener noreferrer" title="${esc(b.label)} ↗">${bookLogo(b)}</a>`;
}
function bookLogoLinks(v) {
  const bs = bookList(v);
  return bs.length ? `<span class="booklogos">${bs.map(bookLogoLink).join("")}</span>` : "";
}

/* ---------- OVERVIEW (best-bet hero + firing signals) ---------- */
function renderOverview(market) {
  const bets = market
    .filter((r) => r.signal_state === "bet")
    .map((r) => ({
      time: r.match_time,
      kickoff_at: r.kickoff_at,
      match: `${r.home_team} vs ${r.away_team}`,
      team: r.signal_pick === "home" ? r.home_team : r.signal_pick === "away" ? r.away_team : "Draw",
      key: r.signal_pick,
      // Best price across books — the line we would actually take.
      odds: r[`best_open_${r.signal_pick}_odds`],
      ev: r[`best_open_${r.signal_pick}_ev`],
      book: r.signal_book,
      books: r.signal_books,
    }))
    .sort((a, b) => b.ev - a.ev);

  setText("ov-signal-count", String(bets.length));
  setText("ov-signal-total", String(market.length));

  renderHero(bets, market);

  if (!el.overviewBody) return;
  if (!bets.length) {
    el.overviewBody.innerHTML = `<tr class="ov-empty"><td colspan="6">— No signals firing. No model pick clears EV &gt; 0.20 &amp; odds ≤ ${SIGNAL_ODDS_CAP} on the current slate.</td></tr>`;
    return;
  }
  el.overviewBody.innerHTML = bets.map((b) => `<tr>
    <td class="ov-time">${fmtTime(b.kickoff_at, b.time)}</td>
    <td class="ov-match">${esc(b.match)}</td>
    <td class="ov-pick">${esc(b.team)} <span class="ov-side">(${sideLetter(b.key)})</span></td>
    <td class="num ov-odds"><span class="ov-odds-v">${odds(b.odds)}</span>${bookLogo(BOOK_BY_KEY.get(b.book))}</td>
    <td class="num ov-ev${b.ev >= 0.2 ? " strong" : ""}">${ev(b.ev)}</td>
    <td class="ov-sig"><span class="badge">● BET</span>${bookLogoLinks(b.books)}</td>
  </tr>`).join("");
}

function renderHero(bets, market) {
  if (!el.overviewHero) return;

  // Prefer a firing signal; otherwise surface the single best available edge.
  let pick = bets[0] || null;
  let firing = Boolean(pick);
  if (!pick) {
    market.forEach((r) => {
      ["home", "draw", "away"].forEach((k) => {
        const e = r[`best_open_${k}_ev`];
        if (e == null) return;
        if (!pick || e > pick.ev) {
          pick = {
            ev: e, odds: r[`best_open_${k}_odds`], key: k,
            // No signal fired here, so signal_book is empty — name the book holding
            // the best price instead, so the tile still says where the price is.
            book: r[`best_open_${k}_book`], books: "",
            team: k === "home" ? r.home_team : k === "away" ? r.away_team : "Draw",
            match: `${r.home_team} vs ${r.away_team}`, time: r.match_time, kickoff_at: r.kickoff_at,
          };
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
  const cta = firing
    ? `<span class="badge">● BET</span>${bookLogoLinks(pick.books)}`
    : `<span class="badge badge--cap">BELOW THRESHOLD</span>`;

  el.overviewHero.className = "hero" + (firing ? " hero--bet" : " hero--flat");
  el.overviewHero.innerHTML = `
    <div class="hero__main">
      <span class="hero__label">${firing ? "★ BEST BET" : "★ TOP EDGE"}</span>
      <div class="hero__headline">
        <span class="hero__team">${esc(pick.team)}</span>
        <span class="hero__side">${sideWord(pick.key)}</span>
      </div>
      <span class="hero__ctx">${esc(pick.match)}${when ? ` · ${when}` : ""}</span>
    </div>
    <div class="hero__metrics">
      <div class="hero__metric"><span class="hero__metric-k">BEST OPEN</span><span class="hero__metric-v">${odds(pick.odds)}${bookLogo(BOOK_BY_KEY.get(pick.book))}</span></div>
      <div class="hero__metric"><span class="hero__metric-k">EDGE (EV)</span><span class="hero__metric-v ${evClass(pick.ev)}">${ev(pick.ev)}</span></div>
      <div class="hero__cta">${cta}</div>
    </div>`;
}

/* ---------- EV BET (bet signals) ---------- */
function renderSignals(rows) {
  let openMax = "";
  el.signalBody.innerHTML = rows.map((row) => {
    // One entry per outcome; `byBook` carries every book's raw price so the table can
    // show them side by side, while EV is driven by the best of them.
    const outs = [
      { key: "home", label: row.home_team, prob: row.home_win_prob },
      { key: "draw", label: "Draw", prob: row.draw_prob },
      { key: "away", label: row.away_team, prob: row.away_win_prob },
    ].map((o) => ({
      ...o,
      byBook: BOOKS.map((b) => ({ book: b, odds: row[oddsCol(b.key, o.key)] })),
      bestBook: row[`best_open_${o.key}_book`],
      ev: row[`best_open_${o.key}_ev`],
    }));
    const isBet = row.signal_state === "bet";
    // The board's "open as of" stamp is now a max across books — they do not open
    // together, so either one can be the most recent.
    BOOKS.forEach((b) => {
      const u = row[`${b.key}_open_last_update`];
      if (u && u > openMax) openMax = u;
    });

    const tr = outs.map((o, i) => {
      // match_time on the market rows is UTC; kickoff_at (joined from fixtures in
      // bootstrap) is what fmtTime renders in DISPLAY_TZ, as every other view does.
      const timeCell = i === 0 ? `<td class="sig-time" rowspan="3">${esc(fmtTime(row.kickoff_at, row.match_time))}</td>` : "";
      const isPick = isBet && row.signal_pick === o.key;
      const nameCls = "sig-name" + (isPick ? " is-pick" : "");
      const probCls = "sig-prob" + (isPick ? " is-pick" : "");
      const evStrong = o.ev != null && o.ev >= 0.2 ? " strong" : "";
      let action = "";
      if (row.signal_pick === o.key && row.signal_state === "bet") {
        // Logos are the links: one per book that independently clears both bars.
        action = `<span class="sig-action"><span class="badge">● BET</span>${bookLogoLinks(row.signal_books)}</span>`;
      } else if (row.signal_pick === o.key && row.signal_state === "odds_cap") {
        action = `<span class="badge badge--cap">ODDS &gt; ${SIGNAL_ODDS_CAP}</span>`;
      }
      // One cell per book. Only the FIRST carries c-grp — that class draws the group's
      // left border, and on the second it would read as a section break between the
      // two books rather than around them. A book with no line renders a dim placeholder
      // rather than an empty cell, so "hasn't opened yet" is distinguishable from a
      // rendering fault (relevant: the books do not open together, and Duel's coverage
      // is permanently sparser — it has no backfill_open safety net).
      const oddsCells = o.byBook.map(({ book, odds: v }, j) => {
        const cls = ["num", j === 0 ? "c-grp" : "", "sig-odds",
                     book.key === o.bestBook ? "is-best" : "",
                     v == null ? "sig-odds--none" : ""].filter(Boolean).join(" ");
        return `<td class="${cls}">${v == null ? "" : Number(v).toFixed(2)}</td>`;
      }).join("");
      return `<tr>
        ${timeCell}
        <td class="${nameCls}">${esc(o.label)}</td>
        <td class="num ${probCls}">${pct(o.prob)}</td>
        ${oddsCells}
        <td class="num sig-ev ${evClass(o.ev)}${evStrong}">${ev(o.ev)}</td>
        <td class="c-grp">${action}</td>
      </tr>`;
    }).join("");

    return `<tbody class="fixture${isBet ? " fixture--bet" : ""}">${tr}</tbody>`;
  }).join("");

  return openMax;
}

/* ---------- SCHEDULE ---------- */
let marketRows = [];
let marketFilterKey = "all";
let marketMetaSuffix = "";

function dayKey(r) { return fmtDay(r.kickoff_at, r.match_date); }
function todayKey() { return fmtDay(new Date()); }
function offsetKey(n) { const d = new Date(); d.setDate(d.getDate() + n); return fmtDay(d); }
function dayChipLabel(key) {
  const d = new Date(`${key}T12:00:00Z`);
  if (Number.isNaN(d.getTime())) return key;
  return new Intl.DateTimeFormat("en-GB", { weekday: "short", day: "2-digit", month: "short", timeZone: "UTC" })
    .format(d).replace(",", "").toUpperCase();
}

function renderMarketFilter() {
  if (!el.marketFilter) return;
  const counts = new Map();
  marketRows.forEach((r) => { const k = dayKey(r); counts.set(k, (counts.get(k) || 0) + 1); });
  const keys = [...counts.keys()].sort();
  const today = todayKey();
  const tomorrow = offsetKey(1);

  const chips = [{ key: "all", label: "ALL", n: marketRows.length }].concat(
    keys.map((k) => ({
      key: k,
      label: k === today ? "TODAY" : k === tomorrow ? "TOMORROW" : dayChipLabel(k),
      n: counts.get(k),
    }))
  );
  if (!chips.some((c) => c.key === marketFilterKey)) marketFilterKey = "all";

  el.marketFilter.innerHTML = chips.map((c) => `<button type="button" class="dchip${c.key === marketFilterKey ? " dchip--active" : ""}" data-day="${esc(c.key)}" aria-pressed="${c.key === marketFilterKey}">${esc(c.label)}<span class="dchip__n">${c.n}</span></button>`).join("");
}

function renderMarket(rows) {
  if (rows) marketRows = rows;
  renderMarketFilter();
  const view = marketFilterKey === "all" ? marketRows : marketRows.filter((r) => dayKey(r) === marketFilterKey);
  const label = marketFilterKey === "all" ? `${marketRows.length} matches` : `${view.length} of ${marketRows.length} matches`;
  setText("panel-market-meta", marketMetaSuffix ? `${label} · ${marketMetaSuffix}` : label);
  if (!view.length) {
    el.marketBody.innerHTML = `<tr><td class="mk-empty" colspan="7">No matches on this date.</td></tr>`;
    return;
  }
  el.marketBody.innerHTML = view.map((r) => {
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
      ${cell({ ...trio[0], grp: true }).replace('class="num prob-cell', 'class="num c-grp prob-cell')}
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
    // Relegated clubs still carry a rating — how they would fare against this
    // season's field — but they are not in the league, so they are greyed rather
    // than dropped, and the top-3 accent skips them.
    const rel = t.in_current_season === false;
    const flags = [
      rel ? `<span class="st-flag" title="Relegated — not in this season’s league">REL</span>` : "",
      t.low_sample ? `<span class="st-flag st-flag--warn" title="Only ${goals(t.weighted_matches)} weighted matches in the model’s 18-month window — this rating is less certain">!</span>` : "",
    ].join("");
    const coefs = `Model coefficients — attack ${rating(t.attack_coef)}, defence ${rating(t.defense_coef)} (log scale, mean-centred); ${goals(t.weighted_matches)} weighted matches`;
    return `<tr class="${rel ? "st-row--rel" : ""}" title="${esc(coefs)}">
      <td class="num st-rk${!rel && t.rank_overall <= 3 ? " top" : ""}">${t.rank_overall}</td>
      <td class="st-team">${esc(t.team)}${flags}</td>
      <td class="num c-grp st-ovr">${rating(t.overall_rating)}</td>
      <td class="num st-r">${goals(t.attack_rating)}</td>
      <td class="num st-r">${goals(t.defense_rating)}</td>
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
function renderHeader(meta, fixtures, predictions, strength, market, openMax) {
  setText("masthead-trail", `${meta.competition_name} · Season ${meta.season} · ${meta.model_name} · ${meta.model_version}`);
  setText("masthead-next-date", meta.next_fixture_date);
  setText("masthead-updated", fmtStamp(meta.updated_at));
  setText("played", String(meta.matches_played));
  setText("round-label", `${meta.current_round}/${meta.total_rounds}`);
  if (el.roundFill) el.roundFill.style.width = `${Math.round((meta.current_round / meta.total_rounds) * 100)}%`;

  setText("panel-signal-meta", `Model ${fmtStamp(meta.model_updated_at)} · OPENS ${fmtStamp(openMax)}`);
  // renderMarket owns panel-market-meta now (it appends the filtered-count prefix).
  marketMetaSuffix = meta.model_name || "";
  renderMarket();
  // Count the league, not the table: the model's 18-month window straddles two
  // seasons, so relegated clubs are listed (greyed) but must not inflate this.
  const inLeague = strength.filter((t) => t.in_current_season !== false);
  setText("panel-strength-meta", `${inLeague.length} clubs · recent 5 form`);
  setText(
    "strength-legend-avg",
    `A league-average team is ${goals(meta.league_avg_goals)} in both ATT and DEF.`,
  );

  const nf = predictions[0] || fixtures[0];
  if (nf) {
    setText("next-fixture", `${nf.home_team} vs ${nf.away_team}`);
    setText("next-note", `${fmtDay(nf.kickoff_at, nf.match_date)} · ${fmtTime(nf.kickoff_at, nf.match_time)}`);
  }
  const sc = inLeague[0];
  if (sc) {
    setText("strong-team", sc.team);
    setText("strong-note", `OVR ${rating(sc.overall_rating)} · ATT ${goals(sc.attack_rating)} · DEF ${goals(sc.defense_rating)}`);
  }
  // Best bet + firing signals now render on the Overview view (renderOverview / renderHero).
}

/* ---------- nav + clock + boot ---------- */
function initMarketFilter() {
  if (!el.marketFilter) return;
  el.marketFilter.addEventListener("click", (e) => {
    const btn = e.target.closest("[data-day]");
    if (!btn) return;
    marketFilterKey = btn.dataset.day;
    renderMarket();
  });
}

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

    const openMax = renderSignals(market);
    renderOverview(market);
    renderMarket(predictions);
    renderStrength(strength);
    renderContext(meta);
    renderHeader(meta, fixtures, predictions, strength, market, openMax);
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
initMarketFilter();
bootstrap();
