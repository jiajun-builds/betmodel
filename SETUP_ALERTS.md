# Signal alerts + fast capture — setup

Two independent pieces ship together (each works without the other):

- **P0-2 Telegram alerts** — a new bet signal is pushed to your phone the moment it
  fires, with the full bet instruction (no dashboard visit needed).
- **P0-3 fast capture** — an external timer fires the capture workflow every ~10 min
  via `repository_dispatch`, instead of relying on GitHub's throttled `schedule` cron
  (measured median ~78 min, worst ~197 min between landed runs — wider than the window).

Both read GitHub **Actions secrets**: `Repo → Settings → Secrets and variables →
Actions → New repository secret`.

---

## 1. Telegram alerts

### 1.1 Create the bot and get the token
1. In Telegram, open a chat with **@BotFather**.
2. Send `/newbot`, follow the prompts (name + username).
3. BotFather replies with a token like `8123456789:AAH...xyz`. That is
   **`TELEGRAM_BOT_TOKEN`**.

### 1.2 Get your chat id
1. Send any message (e.g. `hi`) to your new bot so it has an update to return.
2. Open in a browser (replace `<TOKEN>`):
   `https://api.telegram.org/bot<TOKEN>/getUpdates`
3. Find `"chat":{"id":123456789,...}`. That number is **`TELEGRAM_CHAT_ID`**.
   - For a **group**, add the bot to the group, post a message there, and use the
     group's (negative) `chat.id` instead.

### 1.3 Add the two secrets
| Secret name | Value |
| --- | --- |
| `TELEGRAM_BOT_TOKEN` | the BotFather token |
| `TELEGRAM_CHAT_ID` | the numeric chat id |

That's it. The capture-publish job and the CSL Refresh job already pass these to the
notifier. If the secrets are absent, the notifier logs and no-ops — it never fails a
build.

### 1.4 What fires, and dedup
- A message is sent only for a **newly** firing signal — `signal_state == "bet"`
  (EV > 0.20 and 1xBet odds ≤ 7) for a `(fixture, pick)` that was **not** already a
  "bet" in the previously committed comparison CSV. A price that merely moved on an
  already-alerted pick is **not** re-sent (that's what the bottom-line-odds line in the
  message is for at execution time).
- Test it without waiting for a real signal:
  ```bash
  PYTHONPATH=src TELEGRAM_BOT_TOKEN=... TELEGRAM_CHAT_ID=... \
    python -m csl.notify.signal_alert --dry-run   # prints, sends nothing
  ```
  Drop `--dry-run` to actually send (it will send only genuinely-new signals).

### 1.5 The other alert: a stale xG feed
The same two secrets also power a health alert. xG is fetched outside CI on a home Mac
(see `scripts/LOCAL_XG_SETUP.md`), and because the xG merge is no-erase, a dead fetcher
is otherwise **silent** — every downstream step rebuilds green on frozen data. In July
2026 that went unnoticed for 10 days.

- Runs once a day, inside `./scripts/csl.sh all` (STEP 1b) — not on the every-3h odds
  refresh, which never touches xG.
- Fires when xG trails the results feed by more than **`XG_STALE_DAYS`** (default 3)
  **and** at least 4 played matches sit past the xG frontier. Both conditions are
  needed: SofaScore does not cover every CSL fixture, so an isolated result with no xG
  is normal and must not alert forever.
- Test it:
  ```bash
  PYTHONPATH=src python -m csl.xg.check_freshness --dry-run
  PYTHONPATH=src python -m csl.xg.check_freshness --stale-days 0 --dry-run  # force it
  ```

---

## 2. Fast capture via `repository_dispatch`

An external timer POSTs to the GitHub API every ~10 min; the API-fired run starts
within seconds. The `schedule` cron stays in the workflow as a fallback heartbeat.

### 2.1 Create a fine-grained PAT
`GitHub → Settings → Developer settings → Fine-grained tokens → Generate new token`:
- **Resource owner / Repository access**: only this repo (`jiajun-builds/cslmonitor`).
- **Repository permissions → Contents: Read and write** (this is what authorises
  `repository_dispatch`; Metadata:read is added automatically).
- Copy the token (`github_pat_...`).

### 2.2 The trigger call
```bash
curl -sS -X POST \
  -H "Authorization: Bearer <PAT>" \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  https://api.github.com/repos/jiajun-builds/cslmonitor/dispatches \
  -d '{"event_type":"capture-tick"}'
```
A `204 No Content` means it fired. Check `Actions → Capture Odds` for the run.

### 2.3 Where to run the timer (pick one)
- **cron-job.org** (no server): new cronjob, method POST, the URL + headers + body
  above, schedule every 10 minutes. Store the PAT in its header field.
- **Home Mac `launchd`** (already the xG-refresh host): a `.plist` running the curl on
  a `StartInterval` of 600s. Survives as long as the Mac is awake/online.
- **Cloudflare Worker Cron Trigger** (free): a Worker with `crons = ["*/10 * * * *"]`
  doing the same `fetch()`; keep the PAT in a Worker secret.

### 2.4 Nothing to change if you skip this
Without the timer the workflow still runs on its `schedule` cron — just at GitHub's
throttled cadence. The `repository_dispatch` trigger is purely additive.

---

## Notes
- The capture tick now chases **both** Pinnacle and 1xBet opens (it used to stop at
  Pinnacle), and the 3-hourly refresh backfills a missed 1xBet open at zero quota. More
  1xBet opens captured ⇒ more fixtures with a live EV/signal. Extra API cost is bounded
  by the 12 h capture window, the kickoff cap, and the `min-remaining ≥ 50` guard.
- A backfilled 1xBet "open" is the line as it stood at the 3 h refresh, not the true
  open — it is labelled as a fallback in the history. Always re-check the live 1xBet
  price against the message's **bottom-line odds** before staking.
