# f1-predictor
An experimental F1 predictor using Python, FastF1 and XGBoost for qualifying, sprint and race predictions. Adapted from existing work and being developed further.

# F1 Predictor

A personal project to learn about machine learning through Formula 1 predictions.

The current version uses Python, FastF1 and XGBoost to estimate qualifying, sprint qualifying, sprint and race outcomes using available session data and recent performance.

## Credits

I started with the prediction scripts by [farhansh911](https://github.com/farhansh911):

- [baku-2026](https://github.com/farhansh911/baku-2026)
- [bahrain-malaysia-2026](https://github.com/farhansh911/bahrain-malaysia-2026)

Those projects credit [Mariana Antaya’s prediction work](https://github.com/mar-antaya/2026_f1_predictions).

I used these scripts as the starting point for a reusable predictor and have been adapting the code with assistance from ChatGPT. This repository will track further changes, testing and results.

## Planned improvements

- Better balance between current-weekend evidence and recent form
- Tyre-aware practice and sprint pace comparisons
- Circuit similarity and weather inputs
- More historical testing and clearer prediction outputs

These improvements are planned and are not all implemented yet.

## Status

This is an experimental learning project. Its simulation percentages are not validated real-world probabilities, and prediction accuracy is still being evaluated.

See [START_HERE.txt](START_HERE.txt) for setup, commands, cache handling,
session weights, grid sources, validation results and remaining limits. Manual circuit
categories and tyre-aware pace comparisons are implemented; section telemetry
remains disabled.

New timing downloads require FastF1 3.9+ and F1TV Access or higher after F1's
October 2026 authentication change. Install requirements, then run
`python -m fastf1 auth f1tv --authenticate` locally before predicting.

Published grids are now selected from dated FIA documents, with F1/OpenF1 and
qualifying-order fallbacks. Pit-lane starts are identified separately.
Race/Sprint scoring keeps qualifying performance separate from the starting
position and splits their existing evidence budget equally. This reduced average
ranking error in the 13-race comparison; recovery after penalties remains limited.

Prediction tables now show an explicit position ordered by mean simulated rank.
Race/Sprint tables show mean position and close matchups on the lower driver's
row, for example: "Hamilton beats Antonelli in 49.8% of simulations".
Q/SQ tables show predicted position and pole share. These shares are uncalibrated.

The existing backtest section also compares the model with starting-grid and
qualifying order on matching eligible weekends; Q/SQ use a history-only reference.
Scoring is unchanged. Across 13 eligible GP replays, the mean-position display
averaged 3.259 places out versus 3.510 for grid order, with both picking 8/13 winners.
See START_HERE.txt for coverage, limitations and the effect of changing display order.

## FastF1 v2 checkpoint

Fresh timing downloads require FastF1 3.9+ and an F1TV Access-or-higher account; see
START_HERE.txt for local authentication. Existing cached data can run offline:

```powershell
python predict_anytime.py --year 2026 --race Singapore --session race --offline
```

The schedule and required sessions must already be cached. No automatic OpenF1
timing fallback is implemented. V2 retains its scoring weights and excludes
section telemetry. FP1 reserve-driver selection and ambiguous FIA grid dates
are handled explicitly; use `--roster` for confirmed entry changes.
