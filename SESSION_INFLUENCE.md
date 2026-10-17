# Session and historical influence

The circuit version could reward a worse Sprint result and let an extreme
practice gap dominate a driver's prediction. This version uses explicit evidence
weights and a small, constrained learned adjustment. Section telemetry remains
disabled. Circuit category weights are unchanged.

## Scoring policy

Lower scores mean better expected finishing positions. The final score is
**85% weighted evidence + 15% native XGBoost prediction**, with the native
prediction bounded to 0..1. Every native input is constrained so that worsening
only that input cannot improve the score. This does not promise an unchanged
rank when other drivers, data availability or the training set change.

These are weights **within the 85% evidence component**, with all listed sources
available. Missing sources have their weight redistributed for each driver.
They are not percentages of winning probability.

| Target | History | Starting grid | Earlier Sprint | Earlier SQ | Practice |
|---|---:|---:|---:|---:|---:|
| Grand Prix | 20% | 65% | 10% | 2.5% | 2.5% |
| Sprint | 20% | 75% | 0% | 0% | 5% |
| Qualifying | 30% | 0% | 10% | 50% | 10% |
| Sprint qualifying | 85% | 0% | 0% | 0% | 15% |

Before current sessions exist, history supplies the evidence component. On a
normal weekend, qualifying with only history and practice uses 75%/25% within
that component. Grid means Q order for a GP or SQ order for a Sprint, unless a
complete verified `--grid` is supplied. Automatic penalties remain future work.

Historical evidence combines driver form (65%), current-team form (25%) and
driver form at related circuits (10%), redistributing absent components.
Driver/team form uses the previous **four rounds**, with weight halved every
three rounds. GP/Q gets three times the vote of Sprint/SQ within a weekend;
Sprint weekends do not get two weekend budgets. Team form averages teammates
and uses the driver's current team. If all history is missing, use an explicit
neutral score of 0.5; this is a fallback, not measured performance.

Related-circuit form also uses the four-round window and requires two matching
weekends. Its existing within-weekend averaging and 0.85-per-round decay remain.
Team-related circuit form and numeric geometry comparisons are exported but
are not separate active inputs. `circuit_matches.csv` shows every earlier track;
weights outside the window are zero. Positive track weight alone does not prove
that a particular driver has two usable matching results.

Training still uses **all usable earlier rounds in the selected season**. Its
example weights halve every six rounds and are divided by the number of rows
in each weekend, so Sprint weekends do not receive twice the total weight.
Q/SQ share a training family, as do GP/Sprint; their explicit policies differ.
Every historical feature row uses only earlier weekends and eligible earlier
sessions that weekend. At least four earlier usable training weekends and an
eligible chronological test of the requested session type are still required.

## Practice and Sprint evidence

- Practice uses ranks of acceptable tyre-matched gaps, not raw FP finishing
  order. Q/SQ uses short runs; GP/Sprint uses short and longer runs. FP1/FP2/FP3
  have relative weights 1/2/3 among available practice observations.
- Sprint evidence combines classified position (80%) and comparable longer-run
  pace (20%), redistributing missing parts. A known non-finisher or driver below
  75% distance supplies no classified-position evidence. Unknown completion
  status does not automatically exclude a recorded classification.
- A compound needs four eligible drivers and at least two matched samples per
  driver. Legacy summaries have only aggregate sample counts; the fallback
  cannot establish per-compound support. Rebuilt summaries record those counts.
- Gaps outside 0..5% are omitted, not clipped into a plausible value. Current
  omissions appear in `pace_rejections.csv`. This file is an extreme-gap audit,
  not a complete account of every missing or insufficiently supported observation.
- Newly built summaries remove laps over 7% slower than the driver's fastest
  valid lap on that compound and reject comparison groups with over 7% spread.
  Removed laps break consecutive runs. Existing pit, accuracy, track-status,
  compound, tyre-age and session-window checks remain.

These thresholds are conservative assumptions, not proof of equal fuel loads,
traffic or programmes. Singapore's legacy 16.3898% medium-tyre practice gap for
Antonelli is excluded. His other acceptable evidence can still contribute.
The cloud lacks his raw Singapore lap trace, so the exact cause of that gap has
**not** been confirmed. Synthetic lap tests verify the new filters; the recorded
real-data comparisons below use legacy summaries with the new scoring guard.

`--rebuild-summaries` recalculates compact summaries using normal raw-cache
reuse. Add `--offline` to forbid downloads. Failed rebuilds retain previously
valid completed summaries and report that pace was not rebuilt. No cache is
deleted. `pace_summary_version: 2` identifies summaries built with the filters.
`--refresh` remains a separate explicit request to renew downloaded data.

## Validation recorded on 17 October 2026

The comparison used the user's cached 2026 summaries, the merged circuit
baseline (`0b5de91`), and chronological fits: each tested round trained only on
earlier rounds. The target-stage masks were FP1/SQ/S/Q for GP, FP1/SQ/S for Q,
FP1/SQ for Sprint, and FP1 for SQ, subject to actual schedule and availability.
These are late-weekend comparisons, not validation of every possible input stage.

Twelve candidates compared three source-weight policies, four/six-round history
windows, and 0%/15% learned adjustment. Selection used mean ranking error across
event-sessions in rounds 4–8. The selected competition-focused policy tied across
the two windows on that development score; the four-round version was retained.
Rounds 9–16 were then evaluated with the selected settings fixed. Singapore
round 17's race result was not loaded for selection or testing.

**Held-out rounds 9–16:** lower ranking error is better. Ranking error averages
the absolute difference between predicted and actual classified position across
drivers with valid classifications, then averages the event results. Missing
classifications do not rescale the remaining actual ranks. Winner/pole picks use the lowest
score; ties use driver abbreviation, not actual finishing order.

| Session | Events | Old ranking error | New ranking error | Old winner/pole picks | New winner/pole picks |
|---|---:|---:|---:|---:|---:|
| Grand Prix | 8 | 3.77 | 3.31 | 2/8 | 5/8 |
| Qualifying | 8 | 2.22 | 2.01 | 2/8 | 4/8 |
| Sprint | 2 | 2.00 | 1.18 | 0/2 | 0/2 |
| Sprint qualifying | 2 | 1.55 | 1.68 | 0/2 | 0/2 |

GP top-three overlap improved from 11/24 to 16/24 driver slots. Simply following
the known grid achieved GP error 3.42 and 5/8 winners; history alone achieved
3.88 and 1/8. For Sprint, grid alone scored 1.23 and 1/2 winners. For SQ, history
alone scored 1.45 and 0/2 poles, better ranking error than this selected policy.
The policy was retained without retuning against the held-out outcomes. These
small samples do not establish general accuracy, particularly for Sprint/SQ.

The normal CLI still reports only its **latest four** eligible tests. The final
Singapore GP replay reported **2/4 winners**, mean ranking error 3.50 and 8/12
top-three slots. This is a different test set from the eight-event table.

All four target modes completed offline 22-driver Singapore replays with
25,000 sampled rankings. Saved models contain only six evidence columns and
no section inputs. The saved native component plus the recorded blend
reproduces the score. A real-data check changed only Antonelli's Sprint P4 to
P20: his score worsened from 0.15155 to 0.20566 (lower is better). Changing only
his grid to P20 also worsened it. No driver-specific correction was added.

The authoritative external `test_circuit_profiles_final.py` passed **91 tests**,
covering existing cutoff/grid/section rules plus history windows, weekend
weighting, missing sources, practice guards, monotonic fitting, saved-model
reproduction, grid overrides, missing-classification metrics, failed-rebuild
cache retention and explicit refresh of corrupt summaries. Raw ZIPs,
caches, replay outputs, experiment records and the test file remain outside Git.
The test file is therefore not supplied by a fresh clone of this repository.

## Outputs and remaining scope

`session_influence.csv` contains each driver's evidence scores and effective
weights. `influence_policy.json` records the settings. `model.json` is **only
the native 15% component**; full inference also requires the evidence anchor
and the blend recorded in `run_info.json`. Its six input columns are
`influence_history`, `influence_grid`, `influence_sprint`,
`influence_qualifying`, `influence_practice`, and `influence_anchor`.
Missing source inputs use history in the learned component. Raw diagnostics in
`prediction_inputs.csv` are not all model inputs.

Weather fields and `--air-temp`/`--wet` currently record diagnostics only and
do not change predictions. Weather transfer and forecasts belong to the
weather-context branch. Grid penalties, pit-lane handling, richer output and
probability calibration also remain separate work. Simulation percentages are
uncalibrated sampled-ranking shares, not measured winning probabilities.

A clean installation needs historical data downloaded from FastF1; a cache
stores those observations, not a pretrained model or prior predictions. Each
run retrains. Missing history or blocked timing downloads still limit what can
run; these cached replays do not demonstrate successful fresh downloads.
