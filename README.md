# f1-predictor
An experimental F1 predictor using Python, OpenF1 and XGBoost for qualifying, sprint and race predictions. Adapted from existing work and being developed further.

# F1 Predictor

A personal project to learn about machine learning through Formula 1 predictions.

The current version uses Python, OpenF1 and XGBoost to estimate qualifying, sprint qualifying, sprint and race outcomes using available session data and recent performance.

## Credits

I started with the prediction scripts by [farhansh911](https://github.com/farhansh911):

- [baku-2026](https://github.com/farhansh911/baku-2026)
- [bahrain-malaysia-2026](https://github.com/farhansh911/bahrain-malaysia-2026)

Those projects credit [Mariana Antaya’s prediction work](https://github.com/mar-antaya/2026_f1_predictions).

I used these scripts as the starting point for a reusable predictor and have been adapting the code with assistance from ChatGPT. This repository will track further changes, testing and results.

## Current version

The predictor now uses OpenF1 for schedules, drivers, historical classifications,
practice/Sprint lap evidence and weather. FastF1 is no longer a dependency and no
F1TV account is required. Published-grid source preference and the model's scoring
policy are retained. Completed-session data is cached outside Git.

Run `python -m pip install -r requirements.txt`, then
`python predict_anytime.py --year 2026 --race Austin --session qualifying`.
The first download can take several minutes; later runs reuse the cache.

See [START_HERE.txt](START_HERE.txt) for all commands, source cutoffs, derived lap
quality, output columns, validation limits and remaining work. Simulation shares
are not calibrated real-world probabilities. Section telemetry stays disabled.

Lap audits distinguish missing evidence from excluded laps. Pit entry/exit
mapping and cache renewal are checked explicitly. FP1 reserve drivers no longer
automatically replace the assumed competitive entrants; confirmed substitutions
can be supplied with `--roster`. Historical comparisons use matching roster rules
and actual winner identities. See START_HERE for results and remaining limits.
