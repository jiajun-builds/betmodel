# Retire the `capture-tick` launchd timer (run this ON the 2015 MacBook Air)

**Audience:** an agent or operator working on `Jordans-MacBook-Air-2015`, with no context from
the session that wrote this file. Everything you need is below.

**What this does:** removes the `launchd` job on this Mac that POSTs a `repository_dispatch`
of type `capture-tick` to `jiajun-builds/cslmonitor` every 10 minutes. That timer has been
replaced by a Cloudflare Worker (`tools/capture-timer/` in this repo), which does the same POST
from the cloud.

---

## 🤝 Handover — read this first

**Handed over 2026-08-22** from the main dev Mac (`jordan@Developer/python/cslmonitor`,
branch `ops/capture-timer-worker`, PR #52) **to you, on the 2015 MacBook Air.**

**You are the only machine that can finish this job.** The Worker half is done and live; the
launchd job it replaces is on *your* disk and nowhere else. Nobody can stop it remotely, and
nobody has a copy of its plist.

### What is already done (do not redo any of it)

| | State | Evidence |
| --- | --- | --- |
| Worker deployed | ✅ live | `cslmonitor-capture-timer`, version `3ff4ad06`, trigger `*/10 * * * *` |
| Secrets set | ✅ | `GITHUB_PAT` (expires **2027-08-22**), `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` |
| Reachable over HTTP | ✅ no | `workers_dev = false`, `preview_urls = false` |
| End-to-end proven | ✅ once | Worker logged `fired capture-tick` at `2026-08-22T13:20:16Z`; the `repository_dispatch` run was created `13:20:18Z` and finished green — 2 s latency |
| Overnight coverage | ❓ **not yet proven** | that is Step 0 below, and it is your gate |

### What you must do

1. Run **Step 0**. It is a gate, not a formality — if it fails, stop and report.
2. Remove the launchd timer (Steps 1–3), deal with the token (Steps 4–5).
3. Run the post-checks (Step 6) and report back.

### The one thing that makes this readable

**Both timers are running in parallel right now, and their timestamps separate cleanly:**

| Source | Lands on | Observed |
| --- | --- | --- |
| This Mac (launchd, `StartInterval 600`) | `:x2:5x` | `13:12:58Z`, `13:02:57Z` |
| The Worker (Cloudflare cron) | `:x0:1x` | `13:20:18Z` |

So in Step 0 do **not** just count runs — a healthy-looking list may be entirely *your* ticks.
Look at where the seconds and minutes land, and specifically at what happens **after this Mac
goes to sleep**.

### Known-untested

The Worker's Telegram-on-failure alert is deployed but has **never fired**. Testing it required
deliberately breaking `GITHUB_PAT` on a live timer, which was judged not worth it. Do not assume
you will be paged if the Worker dies — the Step 6 post-checks are the real verification.

---

> ## ⚠️ Do NOT touch the xG fetcher
>
> `~/Library/LaunchAgents/com.cslmonitor.fetch-xg.plist` and `scripts/fetch_xg_local.sh` **must
> keep running on this Mac**. The xG scrape is pinned here by residential IP — SofaScore
> Cloudflare-blocks datacenter IPs and the scrape needs `curl_cffi` TLS impersonation, which no
> cloud runtime here can do. Only the *dispatch timer* moves. If you are unsure which job you
> are looking at: the xG job uses `StartCalendarInterval` (once a day) and runs a shell script;
> the timer being retired uses `StartInterval` (600 s) and runs a `curl` to `api.github.com`.
>
> A dead xG fetcher is **silent** — the merge is no-erase, so every downstream step rebuilds
> green on frozen data. In July 2026 that went unnoticed for 10 days. Do not risk it.

## Why the timer moved

Two failure modes, neither visible from the repo (roadmap #12 in `AGENTS.md`):

1. **This Mac sleeps.** With the job confirmed healthy on an exact 600 s interval
   (`07:05:12Z` / `07:15:12Z` / `07:25:13Z`), there was still a **7h39m** hole overnight on
   2026-08-17/18 and an **11h15m** hole from `2026-08-21T23:53Z` to `2026-08-22T11:09Z`, lid
   closed. A healthy `launchctl list` proves nothing about coverage; only dispatch timestamps do.
2. **The PAT expired** on 2026-08-16 and it was not noticed for **~41 h**. Nothing went red —
   the workflow's fallback `schedule:` cron kept firing and every run stayed green. Only capture
   *resolution* degraded, to a ~29 min median against a 10 min target.

---

## Step 0 — Precondition: prove the replacement is live

**Do not stop anything until this passes.** Running both timers for a few days is harmless —
`capture-odds.yml` has `concurrency: {group: capture-odds, cancel-in-progress: false}`, so a
double-fire queues instead of racing, and an idle tick spends zero API quota. Stopping the old
one too early is *not* harmless.

```bash
cd <this repo on this Mac> && git pull
gh run list --workflow=capture-odds.yml --event repository_dispatch -L 100 \
  --json createdAt -q '.[].createdAt'
```

Read the timestamps. You need **clean ~10 minute spacing across a multi-hour span during which
this Mac was asleep or closed** — typically overnight. That span is the entire point of the
move: it is the one thing the Worker can do that this machine cannot.

The shape you are ruling out, from the night before the Worker existed:

    2026-08-21T23:53:56Z     <- last tick before the lid closed
    2026-08-22T11:09:44Z     <- first tick the next morning.  11h15m of nothing.

- ✅ Ticks land on `:x0:1x` right through the night, no multi-hour gap → continue to Step 1.
- ❌ Still a multi-hour hole overnight → **stop and report.** The Worker is not doing its job,
  and this Mac is still the only thing keeping capture at 10 min. Leaving the old job installed
  but tokenless is the worst possible outcome: it looks configured and does nothing.
- ⚠️ If the only ticks you see land on `:x2:5x`, those are **this Mac's**, not the Worker's.
  That is a fail, however healthy the list looks.

## Step 1 — Find the job

The plist was never checked into the repo, so **discover it, do not assume a name**:

```bash
launchctl list | grep -iE 'cslmonitor|capture|csl'
ls -la ~/Library/LaunchAgents/
grep -rls 'dispatches\|capture-tick' ~/Library/LaunchAgents/ 2>/dev/null
```

You are looking for a job with `StartInterval` of `600` whose program runs `curl` against
`https://api.github.com/repos/jiajun-builds/cslmonitor/dispatches`. The label is probably
something like `com.cslmonitor.capture-tick`, but confirm from the file, not the name.

If nothing matches, the job may already be gone — check `crontab -l` too, then report and stop.

## Step 2 — Record it before deleting it

This plist is the only copy that has ever existed and it is needed for rollback.

```bash
cat ~/Library/LaunchAgents/<LABEL>.plist                       # paste this into your report
cp ~/Library/LaunchAgents/<LABEL>.plist ~/Desktop/capture-timer-plist.bak
```

Also note where it wrote its log (`StandardOutPath` / `StandardErrorPath`) and keep the last
~50 lines of it — that log is the only local record of the dispatch history.

## Step 3 — Stop and remove the job

```bash
launchctl bootout gui/$(id -u)/<LABEL>          # on older macOS: launchctl unload -w <plist>
rm ~/Library/LaunchAgents/<LABEL>.plist
launchctl list | grep -i <LABEL>                # expect: no output
```

## Step 4 — The PAT file: check before deleting

The token lives at `~/.config/cslbet/pat` as a standalone file (it is not inlined in the
plist). The xG job does **not** use it — that was proved on 2026-08-17, when the xG push kept
working normally while this very token was expired. So deleting it should be safe. Confirm
anyway, because the xG job pushes with a plain `git push origin`
(`scripts/fetch_xg_local.sh:102`) and a broken push is silent:

```bash
grep -rl 'cslbet/pat' ~/Library/LaunchAgents ~/bin ~/.zshrc ~/.zprofile ~/.config 2>/dev/null
cd <this repo on this Mac>
git config --get remote.origin.url        # a token embedded in the URL?
git config --get credential.helper        # osxkeychain means the push does NOT need the file
```

- Nothing else references it, and the remote uses the keychain → `rm ~/.config/cslbet/pat`
  (and `rmdir ~/.config/cslbet` if now empty).
- **Anything else references it → leave the file in place and say so in your report.** A broken
  xG push is a silent multi-day data freeze; a stray unused token file is not.

## Step 5 — Revoke the old token on GitHub

`GitHub → Settings → Developer settings → Fine-grained tokens →` revoke the token scoped to
`jiajun-builds/cslmonitor` that this Mac was using.

The Worker uses a **different, newer** token, so this is safe. It matters because it makes an
accidental re-install of the plist fail loudly rather than quietly double-firing forever.

If you cannot tell the two tokens apart, do not guess — report the token names and expiry dates
and let a human pick.

## Step 6 — Post-checks

1. **The tick survives without this Mac.** Wait ~30 min after Step 3, then re-run the Step 0
   command. Dispatches should still be landing every ~10 min. That proves the Worker, not
   `launchd`, is now the source.
2. **The xG push still works.** After the next daily xG run, confirm a fresh `chore(xg)` commit
   landed on `main`:
   ```bash
   git -C <this repo on this Mac> fetch origin main
   git -C <this repo on this Mac> log origin/main --oneline -5 --grep='chore(xg)'
   ```
   Also check `~/Library/Logs/cslmonitor-fetch-xg.log` for a clean run. If the push now fails
   with an auth error, Step 4 removed a credential it needed — restore it from
   `~/Desktop/capture-timer-plist.bak`'s token or mint a new one.

## Rollback

If the Worker turns out to be unreliable and the launchd timer needs to come back:

```bash
cp ~/Desktop/capture-timer-plist.bak ~/Library/LaunchAgents/<LABEL>.plist
# restore the token the plist expects
mkdir -p ~/.config/cslbet && pbpaste > ~/.config/cslbet/pat && chmod 600 ~/.config/cslbet/pat
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/<LABEL>.plist
launchctl list | grep -i <LABEL>
```

Then disable the Worker's cron instead, so the two do not double-fire indefinitely.

## Report back

Include:

- the full plist contents from Step 2, and the tail of its log;
- the Step 0 timestamps you judged the precondition on;
- what was removed, and whether `~/.config/cslbet/pat` was deleted or kept (and why);
- both Step 6 post-check results.

The plist contents are the important one — that configuration exists nowhere else, and pasting
it back is what makes it recoverable.
