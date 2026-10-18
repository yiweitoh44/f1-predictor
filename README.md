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

See [START_HERE.txt](START_HERE.txt) for commands and cache handling, and
[SESSION_INFLUENCE.md](SESSION_INFLUENCE.md) for the implemented session weights,
historical windows, validation results and remaining limits. Manual circuit
categories and tyre-aware pace comparisons are implemented; section telemetry
remains disabled.

Published grids are now selected from dated FIA documents, with F1/OpenF1 and
qualifying-order fallbacks. Pit-lane starts are identified separately. See
[GRID_SOURCES.md](GRID_SOURCES.md) for source timing, commands and validation.
Correcting grid inputs did not improve overall ranking accuracy in the tested
sample; recovery after penalties remains limited.
