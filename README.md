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

Published grids are now selected from dated FIA documents, with F1/OpenF1 and
qualifying-order fallbacks. Pit-lane starts are identified separately.
Race/Sprint scoring keeps qualifying performance separate from the starting
position and splits their existing evidence budget equally. This reduced average
ranking error in the 13-race comparison; recovery after penalties remains limited.

Weather context now includes Open-Meteo forecasts, earlier-session FastF1 weather
with OpenF1 filling missing fields, and dated FIA car-presentation submissions.
Manual weather scenarios stay separate. These additions do not change scores:
reducing the weight of weather-mismatched sessions did not improve the 13-race
comparison. Upgrade declarations are not measured performance gains.
Use `--weather-source off` to skip these context downloads. Source details,
forecast hours and limitations are saved outside Git in `weather_context.json`
alongside the prediction. See START_HERE.txt for historical forecast assumptions.
