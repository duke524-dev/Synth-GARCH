// Run both miner and recalibration daemon: pm2 start miner-and-recalibrate.config.js
// From project root so scripts resolve correctly.
module.exports = {
  apps: [
    {
      name: "garmin-miner",
      interpreter: "./venv/bin/python3",
      script: "./neurons/miner.py",
      args: "--netuid 247 --logging.debug --logging.trace --wallet.name duke524 --wallet.hotkey test --axon.port 9000 --blacklist.force_validator_permit true --blacklist.validator_min_stake 1000 --subtensor.network test",
      env: {
        PYTHONPATH: ".",
      },
    },
    {
      name: "garmin-recalibrate-daemon",
      interpreter: "bash",
      script: "./scripts/cron_recalibrate.sh",
      args: "--daemon",
      env: {
        RECALIBRATE_PERIOD_HOURS: "24",
        RECALIBRATE_CHECK_INTERVAL_MINUTES: "5",
      },
    },
  ],
};
