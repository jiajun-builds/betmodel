/**
 * Fires the capture ticks for every league.
 *
 * GitHub's own cron is not usable as the trigger here. Measured on this project
 * over 299 runs, a requested every-10-minutes schedule landed at a median of 10
 * minutes but a p90 of 137 and a worst case of 232. A closing line has a hard
 * deadline -- the fixture leaves the pre-match feed at kickoff and no backfill
 * exists at any price -- so the trigger has to be something that actually fires
 * on time. This does; the workflow keeps a slow cron purely as a heartbeat.
 *
 * Two ticks, because the two captures have different economics.
 *
 *   every 5 minutes   close tick, and only when a fixture is near kickoff.
 *                     Gated because an ungated 5-minute tick would spend the
 *                     monthly allowance in days.
 *   every 15 minutes  open tick. Ungated: the pending set is what decides
 *                     whether anything is spent, and an idle tick costs nothing.
 *
 * Leagues are discovered from the published manifest rather than listed here, so
 * adding one needs no change to this file.
 */

const OWNER = "jiajun-builds";
const REPO = "betmodel";

/**
 * How close to kickoff the close tick starts firing.
 *
 * Wider than the window the capture itself uses, deliberately. The workflow
 * still has to be queued, a runner started and dependencies installed, which is
 * roughly a minute, and cron itself has jitter. Firing early costs nothing --
 * the capture re-reads the window from config and declines -- while firing late
 * costs a close that cannot be recovered.
 */
const CLOSE_LEAD_MINUTES = 20;

const API = "https://api.github.com";

async function github(env, path, init = {}) {
  const response = await fetch(`${API}${path}`, {
    ...init,
    headers: {
      Authorization: `Bearer ${env.GITHUB_PAT}`,
      Accept: "application/vnd.github+json",
      "User-Agent": "betmodel-capture-timer",
      ...(init.headers || {}),
    },
  });
  return response;
}

/** A file from the default branch. Works while the repository is private. */
async function readFile(env, path) {
  const response = await github(env, `/repos/${OWNER}/${REPO}/contents/${path}`, {
    headers: { Accept: "application/vnd.github.raw" },
  });
  if (!response.ok) {
    console.log(`could not read ${path}: HTTP ${response.status}`);
    return null;
  }
  return await response.text();
}

async function leagues(env) {
  const raw = await readFile(env, "public/index.json");
  if (!raw) return [];
  try {
    return JSON.parse(raw).leagues.map((entry) => entry.id);
  } catch (error) {
    console.log(`manifest unreadable: ${error}`);
    return [];
  }
}

/** True when some fixture kicks off inside the lead window. */
async function closeIsDue(env, league, now) {
  const csv = await readFile(env, `data/${league}/upcoming_fixtures.csv`);
  if (!csv) {
    // Fail open. A missed close is unrecoverable, a wasted request is not, so
    // an unreadable schedule dispatches rather than skips.
    console.log(`${league}: no schedule readable, dispatching anyway`);
    return true;
  }
  const lines = csv.trim().split("\n");
  const header = lines[0].split(",").map((h) => h.trim());
  const column = header.indexOf("kickoff_utc");
  if (column === -1) {
    console.log(`${league}: schedule has no kickoff_utc, dispatching anyway`);
    return true;
  }
  const horizon = now + CLOSE_LEAD_MINUTES * 60_000;
  for (const line of lines.slice(1)) {
    const kickoff = Date.parse(line.split(",")[column]);
    if (!Number.isNaN(kickoff) && kickoff > now && kickoff <= horizon) return true;
  }
  return false;
}

async function dispatch(env, league, kind) {
  const response = await github(env, `/repos/${OWNER}/${REPO}/dispatches`, {
    method: "POST",
    body: JSON.stringify({ event_type: kind, client_payload: { league } }),
  });
  const ok = response.status === 204;
  console.log(`${league} ${kind}: HTTP ${response.status}`);
  if (!ok && env.TELEGRAM_BOT_TOKEN && env.TELEGRAM_CHAT_ID) {
    // A dispatch that stops working is invisible otherwise: no run appears, and
    // an absent run looks exactly like an idle tick. One outage went unnoticed
    // for roughly 41 hours that way.
    await fetch(`https://api.telegram.org/bot${env.TELEGRAM_BOT_TOKEN}/sendMessage`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        chat_id: env.TELEGRAM_CHAT_ID,
        text: `capture-timer: ${league} ${kind} dispatch returned ${response.status}`,
      }),
    });
  }
  return ok;
}

export default {
  async scheduled(event, env, ctx) {
    const now = Date.now();
    const minute = new Date(now).getUTCMinutes();
    const ids = await leagues(env);
    if (ids.length === 0) {
      console.log("no leagues discovered; nothing dispatched");
      return;
    }

    for (const league of ids) {
      if (minute % 15 === 0) {
        ctx.waitUntil(dispatch(env, league, "open-tick"));
      }
      if (await closeIsDue(env, league, now)) {
        ctx.waitUntil(dispatch(env, league, "close-tick"));
      } else {
        console.log(`${league}: no fixture within ${CLOSE_LEAD_MINUTES}m, skipping close`);
      }
    }
  },

  // No public endpoint. An open trigger would let anyone drain a metered
  // allowance, and there is nothing here worth exposing.
  async fetch() {
    return new Response("Not found", { status: 404 });
  },
};
