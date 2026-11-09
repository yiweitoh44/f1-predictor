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

Lap audits distinguish missing evidence from excluded laps. If a session's
driver list omits a number present in its lap feed, only that driver's laps
stay unverified; other drivers' valid pace is retained and a warning is printed.
Missing feeds are reported. When pit records are missing, observed lap flags
and same-stint continuation can retain checked interior laps; uncertain stint
ends and missing boundary evidence stay unverified. Other drivers' usable laps
and Sprint classifications are retained. Empty timing, tyre or flag feeds still
cannot supply validated pace.
Pit entry/exit
mapping and cache renewal are checked explicitly. FP1 reserve drivers no longer
automatically replace the assumed competitive entrants; confirmed substitutions
can be supplied with `--roster`. Historical comparisons use matching roster rules
and actual winner identities. See START_HERE for results and remaining limits.

## v3 development and the v2 checkpoint

The [v2 release](https://github.com/yiweitoh44/f1-predictor/releases/tag/v2)
preserves the FastF1-based predictor. This branch develops the OpenF1 version;
it does not read FastF1 caches or require an F1TV account.

Each run now exports `data_coverage.csv` and `source_manifest.json` beside the
prediction files. The manifest identifies the downloaded responses used, and
local snapshots preserve them when the cache refreshes. These are observations
from download time, not proof of what was available at an earlier race cutoff.
See START_HERE.txt for the data limitations and qualifying experiments.

The data-resilience update validates whole-lap tyre ages and stint identifiers,
keeps malformed tyre evidence out of pace scoring, and reports the reason in the
lap audit. A missing or damaged active OpenF1 cache file can be restored from a
verified local snapshot. Keep the complete `cache/openf1` folder for offline use;
this recovers saved data and does not supply sessions that were never downloaded.


## Qualifying practice evidence

GP qualifying after practice, without completed SQ/Sprint evidence, now gives
native XGBoost separate FP1/FP2/FP3 pace scores, matched gaps and support/context.
Missing session scores remain unknown; they are not replaced with historical
form. The explicit history/practice budgets and 85% evidence / 15% learned blend
are retained. Race, Sprint and competitively anchored GP qualifying keep
their existing models. SQ has the separate treatment below. Section telemetry
stays disabled.

Q/SQ rows export the practice fields in `prediction_inputs.csv` and `training.csv`.
`run_info.json` identifies the actual model columns and which extra fields are
active. Lap counts and comparison-field sizes are context, not confidence.

The expanded review covered 139 settings across 5,580 forecast attempts,
including controls and failed cases. The selected input treatment retained
9/19 GP pole picks, improved top-three matches from 28/57 to 29/57 and reduced
ranking error from 2.000 to 1.977 over 16 complete fields. These are inspected
historical cases, including the original Mexico forecast stage, not unseen
future validation. That review left Race/Sprint/SQ results unchanged. Alternative lap-matching
methods were tested and the current matching was retained. See START_HERE for
method coverage, stage-specific results and limits.

## Sprint qualifying

After FP1, SQ now uses a dedicated native model with separate compound pace,
comparison support and earlier SQ form. It compares all validated tyre ages
within the existing compound, freshness-proxy, age and time bands; it still
requires two matched laps per compound and four usable comparison drivers.
No fuel or engine correction is assumed. Other requested formats are unchanged.

With sufficient earlier-season history, SQ uses the learned score directly,
rather than blending a small adjustment into the history-heavy anchor. This is
not 100% practice: history remains an input. Early-season SQ retains the labelled
evidence-only fallback. `run_info.json` records the actual policy and columns;
`prediction_inputs.csv` contains the `sq_*` evidence. Commands are unchanged.

The review covered 1,132 trial settings including controls, or 6,792 forecast
attempts across the same six SQ weekends. Pole picks improved from 1/6 to 3/6,
top-three matches from 9/18 to 10/18; ranking error stayed at 1.773 over only two
complete fields. The pole gain held across five fixed simulation seeds. These
are extensively inspected historical cases, not an estimate of future accuracy.
See START_HERE for method coverage, individual picks and limits.

## Early season and saved forecasts

Rounds 1–4 can use a labelled evidence-only fallback when training history is
insufficient. It uses available current-season evidence and refuses an arbitrary
ranking when no evidence distinguishes drivers. Later model scoring is retained.

Each run also saves a forecast copy outside `latest`, with inputs, source/code
hashes and runtime versions. After the session, score it against a published-results
CSV using `python predict_anytime.py --score-forecast "<saved folder>" --actual-results "results.csv"`.
See START_HERE.txt for CSV columns, coverage checks and validation limits.

## Qualifying pole correction

After FP3, normal-weekend qualifying uses a separate pole model to reorder the
main model's predicted top five. Other score values and the simulation spread
are retained. Before FP3 and after SQ/Sprint, the existing policy stays active.

The same 19-event comparison improves poles **9/19 → 11/19**, top-three matches
**29/57 → 30/57**, and average ranking error **1.977 → 1.943** over 16 complete
fields. The two-pole gain held across five fixed simulation seeds. Monaco's
podium prediction worsened; these historical results do not establish future
accuracy. The separate after-FP2 comparison is unchanged.

The auxiliary model can consider one accepted matched lap per compound with four
usable comparison drivers. The main model keeps its two-lap rule and all raw
quality checks remain. Both components use native XGBoost. Saved forecasts include
`pole_model.json` and `qualifying_front.csv` when this correction is active.
See START_HERE for the policy and saved-model formula.

## Sprint winner correction

After FP1 and SQ have loaded, a separate Sprint model reorders the main model's
predicted top eight, keeping the same score values and simulation spread. It uses
earlier Sprint/GP form, SQ performance, the starting grid and existing FP1 pace
comparisons. Early-season runs use an evidence-only version. Before FP1/SQ, the
existing policy remains active.

The same six-event historical comparison improves winner picks **3/6 → 5/6**,
retains **14/18** podium-driver matches and reduces ranking error **1.273 → 1.091**
on the only complete Sprint field available. Miami and Silverstone winner picks
improve; Zandvoort remains wrong. The winner gain holds across five fixed
simulation seeds. Six inspected events and one complete ranking are limited
retrospective evidence, not a future success-rate estimate.

Active saved forecasts include `sprint_model.json` and `sprint_front.csv`.
GP race, qualifying and SQ predictions retain their existing models. Commands,
dependencies, lap-quality checks and the two-lap comparison rule are unchanged.
See START_HERE for the auxiliary model and validation details.

## GP race correction

With FP1, a completed Sprint and qualifying available, GP forecasts use two small
ordering models: one retains the existing session-score emphasis, and one uses
separate GP/Sprint history and FP1 long runs. Their equal mean-rank vote reorders
only the existing top three, preserving base score values and simulation spread.
Normal weekends and earlier stages keep the existing model.

The same saved benchmark improves winners **15/19 → 16/19**, retains **37/57**
podium-driver matches, and improves average places out **3.344 → 3.330** over
19 complete fields. Silverstone changes from Antonelli to the correct Leclerc
pick; no previously correct winner becomes wrong. Montreal, Monza and Mexico
remain winner misses. Original Mexico forecasts are retained unchanged.

The winner gain holds across five simulation seeds. A broader top-five treatment
reached one extra podium match but lowered the latest normal-weekend rolling
backtest from 7/8 to 6/8, so it was rejected. This conservative version keeps that
7/8 and improves the latest Sprint-weekend GP checks from 6/8 to 7/8. These are
inspected historical cases, not an estimate of future accuracy.

Active forecasts include `race_score_model.json`, `race_long_model.json` and
`race_front.csv`. Qualifying, SQ and Sprint models, commands, dependencies and
lap-quality checks are unchanged. See START_HERE for formulas and limits.

## Recovery from a poor starting position

Race and Sprint forecasts now allow a bounded recovery adjustment after the
existing ordering models. A driver starting outside the front quarter can move
closer to the performance suggested by qualifying, recent driver/team form and
accepted practice long runs. This requires an independently known starting grid;
confirmed pit-lane starters retain their existing treatment. Missing evidence
stays missing. No extra data feed or dependency is required.

The retained 50-event benchmark keeps GP winners **16/19**, Sprint winners
**5/6**, qualifying poles **11/19** and SQ poles **3/6**. GP average places out
improves **3.330 → 3.316**; Sprint improves **1.091 → 1.000**, with only one
complete Sprint field available. Across 23 non-pit GP driver-event cases starting
behind their qualifying position, error falls **2.83 → 2.39 places**. Individual
weekends can worsen. These are inspected historical comparisons.

The review covered 292 settings across 16 practical method families, including
separate learned recovery models. The bounded rule retains all winner/podium
picks across five fixed simulation seeds and all saved rolling backtest counts.
Original Mexico forecasts remain preserved. See START_HERE for the formula,
comparison coverage and the existing CSV columns used to inspect adjustments.
