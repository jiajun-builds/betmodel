/**
 * capture-tick timer for cslmonitor.
 *
 * Fires a `repository_dispatch` of type `capture-tick` at `.github/workflows/capture-odds.yml`
 * every ten minutes. An API-fired dispatch starts within seconds; GitHub's own cron does not.
 * Measured over 299 real runs, a schedule asking for every ten minutes landed at a median of
 * 10min but a p90 of 137min and a max of 232min -- wider than capture_close's 60min window,
 * and a missed close is unrecoverable because the fixture leaves the pre-match feed at kickoff.
 *
 * (Cron expressions are spelled out in prose above on purpose: an asterisk-slash inside a
 * block comment closes it, which is the build error the sister repo's worker shipped with.)
 *
 * Why this runs on Cloudflare and not on the 2015 MacBook Air it used to: the MBA sleeps.
 * With the launchd job confirmed healthy on an exact 600s interval there was still an 11h15m
 * hole from 2026-08-21T23:53Z to 2026-08-22T11:09Z with the lid closed, and a 7h39m hole the
 * week before. A healthy `launchctl list` proves nothing about coverage; only the dispatch
 * timestamps do. See roadmap #12 in AGENTS.md.
 *
 * Deliberately a DUMB timer. Unlike ligamxterminal's, it does no fixture-proximity gating and
 * has no open/close split: capture-odds.yml re-checks every window in Python and an idle tick
 * spends zero API quota, so there is nothing worth gating on and no fail-open branch to get
 * wrong. The workflow's own `schedule:` cron stays as the fallback heartbeat.
 */

const OWNER = "jiajun-builds";
const REPO = "cslmonitor";
const EVENT_TYPE = "capture-tick";
const UA = "cslmonitor-capture-timer";

const RUNS_URL =
  `https://github.com/${OWNER}/${REPO}/actions/workflows/capture-odds.yml`;

/** POST the dispatch. Returns the HTTP status; 204 No Content is success. */
async function dispatch(env) {
  const resp = await fetch(`https://api.github.com/repos/${OWNER}/${REPO}/dispatches`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${env.GITHUB_PAT}`,
      Accept: "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
      "User-Agent": UA,
      "Content-Type": "application/json",
    },
    // No client_payload: every job's `if` in capture-odds.yml falls back to
    // `(github.event.inputs.only || 'all')`, so a bare event_type runs all three legs.
    body: JSON.stringify({ event_type: EVENT_TYPE }),
  });
  if (resp.status !== 204) {
    console.log(`dispatch failed: ${resp.status} ${await resp.text()}`);
  }
  return resp.status;
}

/**
 * Push a failure to Telegram, using the same bot as the repo's signal alerts.
 *
 * Never throws and never fails a tick -- same contract as src/csl/notify/signal_alert.py.
 * Absent secrets are a no-op, not an error.
 */
async function notify(env, text) {
  if (!env.TELEGRAM_BOT_TOKEN || !env.TELEGRAM_CHAT_ID) {
    console.log("telegram secrets absent; alert skipped");
    return;
  }
  try {
    const resp = await fetch(
      `https://api.telegram.org/bot${env.TELEGRAM_BOT_TOKEN}/sendMessage`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json", "User-Agent": UA },
        body: JSON.stringify({
          chat_id: env.TELEGRAM_CHAT_ID,
          text,
          disable_web_page_preview: true,
        }),
      },
    );
    console.log(`telegram alert: ${resp.status}`);
  } catch (err) {
    console.log(`telegram alert threw: ${err.message}`);
  }
}

/**
 * At most one alert per hour, without any stored state.
 *
 * The failure this exists for -- an expired PAT -- fails EVERY tick, so an ungated alert is
 * 144 messages a day and would be muted within one. Firing only on the top-of-hour tick caps
 * it at 24, which is loud enough: the incident this replaces went unnoticed for 41h, so a
 * sustained failure reported up to ~50min late is still three orders of magnitude better.
 * Transient single-tick failures are absorbed by the retry below and never alert at all.
 *
 * If the hourly band ever proves too coarse, the upgrade is a KV namespace holding a
 * `last_alert_at` key written with `expirationTtl: 3600`; it is not worth a binding today.
 */
function alertDue(scheduledTime) {
  return new Date(scheduledTime).getUTCMinutes() === 0;
}

export default {
  async scheduled(event, env, ctx) {
    console.log(`cron fired: ${event.cron}`);

    ctx.waitUntil(
      (async () => {
        let status = await dispatch(env);

        // One retry, because a single GitHub 5xx must not page anyone. A dispatch is
        // idempotent from the pipeline's point of view: capture-odds.yml serialises on
        // `concurrency: capture-odds` and an idle tick spends nothing.
        if (status !== 204) {
          console.log("retrying once");
          status = await dispatch(env);
        }

        if (status === 204) {
          console.log(`fired ${EVENT_TYPE}`);
          return;
        }

        const msg =
          `⚠️ capture-timer: dispatch failed ${status}\n` +
          `The capture tick is no longer firing; capture has fallen back to GitHub's ` +
          `throttled cron (~29min median). Check the PAT first -- that is what broke last time.\n` +
          RUNS_URL;
        if (alertDue(event.scheduledTime)) await notify(env, msg);

        // Throw so the tick is recorded as an error in observability, not just a log line.
        throw new Error(`dispatch ${EVENT_TYPE} failed: ${status}`);
      })(),
    );
  },

  /**
   * Manual trigger, disabled by default.
   *
   * wrangler.toml sets workers_dev = false, so nothing routes here at all. The handler stays
   * behind a shared secret anyway: firing a dispatch spends real quota against a 500-per-MONTH
   * allowance, so an open endpoint is a way to drain the budget, not a convenience. To use it,
   * `wrangler secret put TRIGGER_SECRET`, add a route, then:
   *
   *   curl -H "X-Trigger-Secret: <secret>" https://<host>/
   *
   * For ordinary manual runs prefer the GitHub API directly (SETUP_ALERTS.md section 2.2) or
   * `gh workflow run capture-odds.yml` -- neither needs this to exist.
   */
  async fetch(request, env) {
    const expected = env.TRIGGER_SECRET;
    // 404, not 403: an unconfigured endpoint should not confirm it is here.
    if (!expected || request.headers.get("X-Trigger-Secret") !== expected) {
      return new Response("not found\n", { status: 404 });
    }
    const status = await dispatch(env);
    return new Response(`dispatch ${EVENT_TYPE}: ${status}\n`, {
      status: status === 204 ? 200 : 502,
    });
  },
};
