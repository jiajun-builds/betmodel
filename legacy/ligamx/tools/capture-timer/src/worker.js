/**
 * Capture timer for ligamxterminal.
 *
 * GitHub's own `schedule:` cron cannot drive this pipeline. Measured on the sister
 * repo over 299 runs, a cron asking for every 10 minutes landed at median 10min but
 * p90 137min and worst 232min -- wider than the entire 20-minute closing window,
 * and a missed close is unrecoverable because the fixture leaves the pre-match feed
 * at kickoff. An API-fired repository_dispatch starts within seconds instead.
 *
 * (Cron expressions are spelled out in prose above on purpose: an asterisk-slash
 * inside a block comment closes it, which is exactly the build error this file
 * shipped with the first time.)
 *
 * Two crons, two event types, because the two prices have opposite deadlines:
 *
 *   every 15 min  -> open-tick   Betano UK + Duel openers. No window; a late tick
 *                                costs latency only, and Betano opens a median
 *                                T-293h out so 15 min is 0.09% of the lead.
 *   every  5 min  -> close-tick  Pinnacle close, but ONLY when a fixture is
 *                                actually approaching kickoff (see below).
 *
 * Why close-tick is gated here and not just in the workflow: firing every 5 minutes
 * unconditionally is 288 workflow runs a day, of which ~250 do nothing but check a
 * CSV and exit. Filtering on the fixture list first drops that to ~4 per fixture --
 * roughly 36 on a match day and zero otherwise -- which keeps the Actions log
 * readable enough that a real failure is visible in it.
 *
 * The workflow still re-checks the window in Python. This is an optimisation, not
 * the safety gate: if this Worker dies the repo's fallback cron keeps capturing,
 * and if it fires spuriously the Python side spends nothing.
 */

const OWNER = "jiajun-builds";
const REPO = "ligamxterminal";

const FIXTURES_URL =
  `https://raw.githubusercontent.com/${OWNER}/${REPO}/main/data/MEX_upcoming_fixtures.csv`;

const OPEN_CRON = "*/15 * * * *";

// Fire a close-tick when kickoff is this close. Wider than the workflow's own
// 20-minute window so cron jitter plus ~45s of runner startup still lands inside it.
const CLOSE_LEAD_MINUTES = 25;

/** Kickoff times (ms) from the fixtures CSV's kickoff_utc column. */
async function kickoffs() {
  const resp = await fetch(FIXTURES_URL, {
    headers: { "User-Agent": `${REPO}-capture-timer` },
    cf: { cacheTtl: 300, cacheEverything: true },
  });
  if (!resp.ok) throw new Error(`fixtures fetch failed: ${resp.status}`);

  const [header, ...lines] = (await resp.text()).trim().split("\n");
  const col = header.split(",").indexOf("kickoff_utc");
  if (col < 0) throw new Error("fixtures CSV has no kickoff_utc column");

  return lines
    .map((line) => Date.parse(line.split(",")[col]))
    .filter((t) => !Number.isNaN(t));
}

/** True when some fixture is inside [now, now + CLOSE_LEAD_MINUTES). */
function closeDue(times, now) {
  const until = now + CLOSE_LEAD_MINUTES * 60_000;
  return times.some((t) => t > now && t <= until);
}

async function dispatch(eventType, env) {
  const resp = await fetch(
    `https://api.github.com/repos/${OWNER}/${REPO}/dispatches`,
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${env.GITHUB_PAT}`,
        Accept: "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": `${REPO}-capture-timer`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ event_type: eventType }),
    },
  );
  // 204 No Content is success. Anything else is worth seeing in `wrangler tail`.
  if (resp.status !== 204) {
    throw new Error(`dispatch ${eventType} failed: ${resp.status} ${await resp.text()}`);
  }
  console.log(`fired ${eventType}`);
}

export default {
  async scheduled(event, env, ctx) {
    // Logged every tick so a cron/constant mismatch is visible in `wrangler tail`
    // rather than silently routing every tick to close-tick and never opening.
    console.log(`cron fired: ${event.cron}`);

    if (event.cron === OPEN_CRON) {
      // Openers have no window, so there is nothing to check -- always fire.
      ctx.waitUntil(dispatch("open-tick", env));
      return;
    }

    ctx.waitUntil(
      (async () => {
        let times;
        try {
          times = await kickoffs();
        } catch (err) {
          // Fail OPEN, not closed. A CSV we could not read must not be the reason
          // a close is missed -- the Python side spends nothing if it was a false
          // alarm, whereas a skipped tick can lose the line for good.
          console.log(`fixture check failed (${err.message}); firing anyway`);
          return dispatch("close-tick", env);
        }
        if (closeDue(times, event.scheduledTime)) {
          return dispatch("close-tick", env);
        }
        console.log("no fixture within the close window; idle");
      })(),
    );
  },

  /**
   * Manual trigger, disabled by default.
   *
   * wrangler.toml sets workers_dev = false, so nothing routes here at all. This
   * handler stays behind a shared secret anyway: firing a dispatch spends real
   * quota against a 500-per-MONTH allowance, so an open endpoint is a way to
   * drain the budget, not a convenience. To use it, set TRIGGER_SECRET
   * (`wrangler secret put TRIGGER_SECRET`), add a route, then:
   *
   *   curl -H "X-Trigger-Secret: <secret>" "https://<host>/?type=open-tick"
   *
   * For ordinary manual runs prefer the GitHub API directly, or
   * `gh workflow run capture-odds.yml` -- neither needs this to exist.
   */
  async fetch(request, env) {
    const expected = env.TRIGGER_SECRET;
    // 404, not 403: an unconfigured endpoint should not confirm it is here.
    if (!expected || request.headers.get("X-Trigger-Secret") !== expected) {
      return new Response("not found\n", { status: 404 });
    }
    const type = new URL(request.url).searchParams.get("type");
    if (!["open-tick", "close-tick", "capture-tick"].includes(type)) {
      return new Response("pass ?type=open-tick|close-tick|capture-tick\n", { status: 400 });
    }
    await dispatch(type, env);
    return new Response(`fired ${type}\n`);
  },
};
