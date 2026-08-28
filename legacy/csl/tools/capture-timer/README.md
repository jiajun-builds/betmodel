# `cslmonitor-capture-timer` — the capture tick

A Cloudflare Worker cron that POSTs a `repository_dispatch` of type `capture-tick` to
`jiajun-builds/cslmonitor` every 10 minutes, driving
[`.github/workflows/capture-odds.yml`](../../.github/workflows/capture-odds.yml).

That is the whole job. It fetches nothing, reads no fixture list and makes no decisions —
`capture-odds.yml` re-checks every capture window in Python and an idle tick spends zero API
quota, so there is nothing for the timer to gate on.

## Why it is not on the MacBook Air any more

It used to be a `launchd` job with `StartInterval 600` on the 2015 MBA. Two failure modes,
both silent from this repo (roadmap #12 in `AGENTS.md`):

- **The host sleeps.** The job was confirmed healthy on an exact 600 s interval and there was
  *still* an 11h15m hole from `2026-08-21T23:53Z` to `2026-08-22T11:09Z` with the lid closed,
  and a 7h39m hole the week before. A healthy `launchctl list` proves nothing about coverage.
- **The PAT expired** and nobody noticed for ~41 h. Nothing went red: the workflow's fallback
  `schedule:` cron kept firing and every run stayed green. Only capture *resolution* degraded,
  to a ~29 min median against a 10 min target.

The second one is why this Worker alerts to Telegram (see below) rather than staying a purely
dumb timer.

The **xG fetcher stays on the MBA** and must not be moved here: SofaScore Cloudflare-blocks
datacenter IPs and the scrape needs `curl_cffi` TLS impersonation, which the Workers runtime
has no equivalent for. See `scripts/LOCAL_XG_SETUP.md`.

## Secrets

Set with `npx wrangler secret put <NAME>` from this directory.

| Secret | Required | What |
| --- | --- | --- |
| `GITHUB_PAT` | yes | Fine-grained token. Resource owner + repository access: **`jiajun-builds/cslmonitor` only**. Repository permissions → **Contents: Read and write** (that is what authorises `repository_dispatch`; Metadata:read is added automatically). |
| `TELEGRAM_BOT_TOKEN` | no | Same bot as the repo's Actions secrets (`SETUP_ALERTS.md` §1). |
| `TELEGRAM_CHAT_ID` | no | Same chat id. |
| `TRIGGER_SECRET` | no | Leave **unset**. Without it the `fetch` handler 404s unconditionally, which is what we want. |

The alert no-ops if either Telegram secret is missing — it never fails a tick.

### PAT expiry — record it here

> **`GITHUB_PAT` expires: 2027-08-22.** Minted 2026-08-22, one-year expiry.

An unrecorded expiry date is what caused the 41 h outage. GitHub emails a warning before a
fine-grained token expires; the Telegram alert below is the backstop if that mail is missed.

## Deploy

```bash
cd tools/capture-timer
npx wrangler deploy
npx wrangler tail cslmonitor-capture-timer   # watch a tick or two
```

A healthy tick logs `cron fired: ...` then `fired capture-tick`.

## Alerting

On a dispatch that does not return `204 No Content`, the Worker retries once, then — on the
top-of-hour tick only — pushes a Telegram message and throws so the failure is recorded in
observability. The hourly gate is deliberate: an expired PAT fails *every* tick, so an ungated
alert would be 144 messages a day and muted within one.

## Checking it is actually working

The only test that matters is dispatch spacing across a span when the MBA was closed:

```bash
gh run list --workflow=capture-odds.yml --event repository_dispatch -L 100 \
  --json createdAt -q '.[].createdAt'
```

Clean ~10 min spacing overnight = the Worker is the source. A hole shaped like the ones above
means it is not.

## Retiring the old timer

Do **not** stop the MBA `launchd` job until the check above passes — running both for a while
is harmless (`capture-odds.yml` serialises on `concurrency: capture-odds`). The runbook for
decommissioning it, to be run *on the MBA*, is
[`scripts/DECOMMISSION_CAPTURE_TIMER.md`](../../scripts/DECOMMISSION_CAPTURE_TIMER.md).
