# Local xG refresh (home Mac)

SofaScore's Cloudflare blocks GitHub Actions' datacenter IPs (HTTP 403), so xG
can't be fetched from CI. Instead, a home Mac on a **residential IP** fetches xG
daily and pushes `data/raw_data/xg_data.csv`; GitHub CI builds everything else on
top of it. Because the merge is **no-erase**, a CI run that 403s just retains the
file — no conflict, no data loss.

```
Home Mac (daily, residential IP)      GitHub CI (unchanged)
──────────────────────────────      ─────────────────────────
xg_pipeline --full-season      ──►   scheduled full run reads the fresh
commit + push xg_data.csv             xg_data.csv → merge → model → odds
                                      → dashboard → deploy Pages
```

It does **not** need to be on 24/7 — only awake at the scheduled minute. Keep it
**plugged in** (battery health is irrelevant on AC) and, optionally, let it sleep
and auto-wake for the job (see `pmset` below).

## One-time setup on the home Mac

1. **Install conda** (Miniconda) if not present.
2. **Clone the repo** and create the env:
   ```bash
   git clone https://github.com/jiajun-builds/cslmonitor.git
   cd cslmonitor
   conda env create -f environment.yml     # includes curl_cffi
   ```
3. **Configure git identity + non-interactive push.** `git push` must work with
   no prompt. Easiest: an SSH deploy key with no passphrase, or a stored PAT via
   the macOS keychain helper:
   ```bash
   git config user.name  "CSL Home Bot"
   git config user.email "you@example.com"
   git config credential.helper osxkeychain   # then do one manual push to store creds
   ```
4. **(Optional) `.env.local`** — only needed if you also run odds locally; xG
   needs no key.
5. **Sanity check** the fetch by hand:
   ```bash
   ./scripts/fetch_xg_local.sh
   ```
   You should see it fetch, then either commit+push or "No xG changes".

## Schedule it

```bash
./scripts/install_local_xg.sh 8      # runs daily at 08:00 local
```
This writes and loads `~/Library/LaunchAgents/com.cslmonitor.fetch-xg.plist`,
baking in your conda path so launchd can find the env. Pick an hour **before**
GitHub's 09:17 London full run.

Verify:
```bash
launchctl start com.cslmonitor.fetch-xg          # run it now
tail -f ~/Library/Logs/cslmonitor-fetch-xg.log   # watch output
```

## Let it sleep between runs (optional)

So the Mac isn't on all day, have it auto-wake ~5 min before the job, run, then
sleep on its own:
```bash
sudo pmset repeat wakeorpoweron MTWRFSU 07:55:00
```
launchd also **catches up on the next wake** if the machine was asleep at the
scheduled minute, so a missed run self-heals.

## Notes / troubleshooting
- **Mode:** the job runs `--full-season` (robust, ~a few minutes). For the faster
  two-round refresh, edit the plist / set `XG_MODE=""` in the environment.
- **Nothing pushed?** That's normal when xG is unchanged — check the log.
- **"cannot switch branch while rebasing"** — a wedged clone. This killed the feed for
  10 days in July 2026: a hand-edit to `xg_data.csv` on GitHub collided with the same
  rows the fetcher regenerated (identical numbers, different bytes: `2` vs `2.0`), the
  rebase conflicted, and the old retry loop just re-ran `git pull --rebase`, which can
  only fatal while files are unmerged. Every later run then died before reaching the
  API. The script now unwinds a leftover rebase on startup and, on a rejected push,
  resets to the remote tip and re-runs the pipeline instead of rebasing — the pipeline's
  no-erase merge is the right resolver, and it cannot leave a conflict behind. It also
  refuses to reset if anything **other than** `xg_data.csv` is uncommitted, so don't
  keep work-in-progress in this clone.
- **Did anyone notice?** Now, yes: the daily CI refresh sends a Telegram alert when xG
  falls behind the results feed (`csl.xg.check_freshness`, see `SETUP_ALERTS.md` §1.5).
- **`conda: not found` in the log:** re-run `install_local_xg.sh` from a shell
  where `conda` works (it bakes the path in).
- **Uninstall:**
  ```bash
  launchctl unload ~/Library/LaunchAgents/com.cslmonitor.fetch-xg.plist
  rm ~/Library/LaunchAgents/com.cslmonitor.fetch-xg.plist
  sudo pmset repeat cancel   # if you set an auto-wake
  ```
