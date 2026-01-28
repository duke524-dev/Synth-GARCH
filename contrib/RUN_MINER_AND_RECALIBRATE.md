# Running the miner and recalibration daemon in real time

Run both the Synth miner and the GARCH recalibration daemon so the miner serves requests while models are refreshed on a schedule.

## Option 1: Two terminals

**Terminal 1 — miner (PM2 or foreground)**

```bash
cd /path/to/synth-subnet-garch
source bt_venv/bin/activate   # if you use a venv
pm2 start miner.config.js
# Or run in foreground: python neurons/miner.py --netuid 50 --wallet.name miner --wallet.hotkey default --axon.port 8091 ...
```

**Terminal 2 — recalibration daemon (foreground)**

```bash
cd /path/to/synth-subnet-garch
./scripts/cron_recalibrate.sh --daemon
```

The daemon wakes every few minutes (see `RECALIBRATE_CHECK_INTERVAL_MINUTES`), checks if `RECALIBRATE_PERIOD_HOURS` have passed since the last successful run, then runs `fetch_history.py` and `recalibrate_models.py` when due.

## Option 2: One PM2 config (miner + recalibrate daemon)

From the project root:

```bash
pm2 start miner-and-recalibrate.config.js
```

This starts both:

- **miner** — Bittensor miner (same as `miner.config.js`).
- **recalibrate-daemon** — Recalibration loop (`scripts/cron_recalibrate.sh --daemon`).

Useful commands:

```bash
pm2 status
pm2 logs miner
pm2 logs recalibrate-daemon
pm2 stop miner
pm2 stop recalibrate-daemon
pm2 restart all
```

## Option 3: Miner with PM2, daemon in background (no second terminal)

```bash
cd /path/to/synth-subnet-garch
pm2 start miner.config.js
nohup ./scripts/cron_recalibrate.sh --daemon >> logs/recalibrate.log 2>&1 &
```

To stop the daemon later:

```bash
pkill -f "cron_recalibrate.sh --daemon"
```

## Daemon settings (recalibration)

- **RECALIBRATE_PERIOD_HOURS** — Run fetch + recalibrate at most this often (default: 24).
- **RECALIBRATE_CHECK_INTERVAL_MINUTES** — How often to wake and check (default: 5).

Example: recalibrate at most every 6 hours, check every 5 minutes:

```bash
RECALIBRATE_PERIOD_HOURS=6 RECALIBRATE_CHECK_INTERVAL_MINUTES=5 ./scripts/cron_recalibrate.sh --daemon
```

In the PM2 config, set these via `env` in the `recalibrate-daemon` app if needed.

## Makefile shortcuts (from project root)

```bash
make miner                     # pm2 start miner.config.js
make recalibrate-daemon        # ./scripts/cron_recalibrate.sh --daemon (foreground)
make miner-and-recalibrate    # pm2 start miner-and-recalibrate.config.js (both)
```

## Summary

| Process             | Command / config                               |
|---------------------|------------------------------------------------|
| Miner only          | `make miner` or `pm2 start miner.config.js`   |
| Recalibrate only    | `make recalibrate-daemon` or `./scripts/cron_recalibrate.sh --daemon` |
| Miner + recalibrate | `make miner-and-recalibrate` or `pm2 start miner-and-recalibrate.config.js` |
