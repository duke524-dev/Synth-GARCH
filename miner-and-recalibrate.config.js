// Run both miner and recalibration daemon: pm2 start miner-and-recalibrate.config.js
// From project root so scripts resolve correctly.
module.exports = {
  apps: [
    {
      name: "miner",
      interpreter: "python3",
      script: "./neurons/miner.py",
      args: "--netuid 50 --logging.debug --logging.trace --wallet.name miner --wallet.hotkey default --axon.port 8091 --blacklist.force_validator_permit true --blacklist.validator_min_stake 1000",
      env: {
        PYTHONPATH: ".",
      },
    },
    {
      name: "recalibrate-daemon",
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
