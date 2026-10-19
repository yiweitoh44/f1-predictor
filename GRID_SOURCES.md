# Published starting grids

Qualifying order can differ from the starting grid after penalties or pit-lane
changes. After Q/SQ has loaded, Race/Sprint predictions now try published grids
automatically. Qualifying order remains the explicit fallback.

```powershell
python -m pip install -r requirements.txt
python predict_anytime.py --year 2026 --race Singapore --session race
```

For a completed race, answer `y` to the existing pre-session replay prompt.
The new PDF reader, `pypdf`, is pure Python. Native XGBoost is unchanged.

## Selection and conflicts

1. A user-supplied `--grid` CSV overrides automatic sources.
2. Dated FIA starting-grid documents take priority. The latest usable revision
   before the cutoff is selected; recalled documents are skipped.
3. Formula1.com starting-grid tables are the next preference.
4. OpenF1's `starting_grid` endpoint is the final published-grid fallback.
5. If no eligible complete grid is available, use the loaded Q/SQ order.

Every candidate must match the event/session and contain each entered driver
exactly once, without duplicate or out-of-range slots. GP and Sprint grids are
handled separately. Missing drivers and unsupported document layouts reject
the candidate. Sources are never combined into a guessed partial grid.
Conflicting eligible sources are recorded; the higher-priority source wins.
A complete older provisional FIA grid may be used if a later document cannot
be validated; the rejection remains in the audit. Check the selected title.

OpenF1 grid records are attached to the corresponding **qualifying session key**.
The adapter matches year, location, session name and scheduled time, then maps
driver numbers through that session's entry list. It does not query the target
race's result or running order. OpenF1 grid rows do not identify pit-lane status;
that remains unknown if this fallback is selected.

Source references:

- [FIA documents](https://www.fia.com/documents)
- [F1 results and starting grids](https://www.formula1.com/en/results/2026/races)
- [OpenF1 starting-grid documentation](https://openf1.org/docs/#starting-grid)

## Publication time and historical replay

FIA website publication times are interpreted as Europe/Paris wall time. The
site labels these CET throughout the year; the seasonal UTC offset is applied.
When a document cover gives an issue date/time, its local event offset comes
from the FastF1 schedule. The later of publication and issue time is used.
Disagreements larger than two hours reject the document. Session identity,
year and date proximity are also checked.

F1/OpenF1 responses lack a publication timestamp. Their retrieval time is the
earliest availability this application can establish. A record first fetched
today is not silently allowed into a replay of last week. Normal predictions
before the target starts may use a grid fetched during the request; explicit
historical cutoffs cannot. Dated FIA documents make many older replays possible.

The external `F1Data/cache/grids` directory holds raw responses and immutable
parsed snapshots. A new revision does not overwrite an earlier grid snapshot.
Normal online requests refresh responses after five minutes; `--refresh`
renews them immediately. `--offline` uses cached responses/snapshots only.
If a download fails, cached data is checked against the same time/identity rules.

These are current copies of dated historical publications. They are not proof
that the bytes were archived at the original issue time; later corrections
and document availability remain limitations of retrospective evaluation.

## Pit-lane starters and manual overrides

FIA's explicit pit-lane section and recognised F1 grid notes identify pit-lane
starters. They have no invented numbered grid slot. Each receives the same
back-of-field score for the existing model. Their pit-exit order, lost time,
strategy and detailed recovery are not modelled. The scoring split below
allows qualifying performance to offset some of the starting disadvantage.

The manual CSV still accepts `driver,grid`. Use `PIT` for a pit-lane start:

```csv
driver,grid
VER,1
LEC,2
RUS,19
ALO,PIT
```

This is an excerpt: the real file must contain the entire entered field.
Vacant grid slots are permitted. An optional `pit_lane` true/false column is
accepted; a true row must have `grid=PIT`, zero or empty. Optional `known_utc`
timestamps must include an offset and be no later than the cutoff. Without
timestamps, a manual file is the user's assertion of the pre-session grid.

```powershell
python predict_anytime.py --year 2026 --race Singapore --session race --grid grid.csv
python predict_anytime.py --year 2026 --race Singapore --session race --grid-source qualifying
```

The second command explicitly chooses the qualifying-order grid assumption.
It changes the grid source, not the scoring policy; it does not reproduce the
older six-input model after adopting the split below.

## Separate qualifying performance and starting position

Qualifying performance is preserved separately in `qualifying_grid_norm` and
`starting_grid.csv`; a grid penalty does not overwrite the observed Q/SQ result.
Race/Sprint split their **existing grid evidence budget equally** between
actual starting position and Q/SQ position when both are available. For a GP
with all sources, the 65% grid budget becomes 32.5% each within the evidence
component. Other budgets and the 85% evidence / 15% learned blend are unchanged.
Missing/invalid Q/SQ position retains the existing anchor. Equal Q/SQ and grid
positions leave the anchor unchanged. Pit-lane starters still share the same
back-of-field starting score, with qualifying performance retained separately.

Race/Sprint also feed `influence_grid_qualifying` to the native adjustment as a
seventh input. Q/SQ keep six model inputs and their original behavior.
Qualifying position is a performance proxy, not a measured recovery probability.
Cached lap-time gaps mix qualifying stages and are not used here.

`session_influence.csv` records the combined `influence_weight_grid` budget and
its two applied components, `influence_weight_starting_grid` and
`influence_weight_grid_qualifying`. The policy and exact model inputs are saved
in the existing JSON exports. These components sum to the combined budget;
they are not additional full source weights.

A circuit-dependent split was tested using existing grid-stickiness assumptions,
but performed worse than the fixed split and was not adopted. Overtaking,
traffic, strategy, pit-exit delay and sophisticated recovery remain deferred.

## Outputs

- `starting_grid.csv`: driver, qualifying position, grid slot, pit-lane status,
  applied grid score and source. Blank pit-lane status means unknown.
- `grid_sources.json`: selected source URL, dated/observed time, source
  conflicts, rejected/unavailable sources and fallback details.
- `run_info.json`: includes the same grid audit alongside the existing metadata.

The terminal identifies the selected source, pit-lane starters, conflicts or
qualifying fallback. Grids are only fetched after the corresponding qualifying
has loaded. Q/SQ forecasts do not use target-race grid information.

## Validation on 19 October 2026

The authoritative external test file passes 124 checks, including grid timing,
identity, source conflicts, pit-lane flags, offline fallback, cached revisions,
split/missing-source rules, and separate Race/Sprint model inputs.
All four session modes completed real cached replays. Q/SQ results are unchanged.
The adopted Race/Sprint implementation reproduces the tested fixed-split candidate.

All **17 GP and 6 Sprint** grids through Singapore were obtained from dated FIA
documents, with all 22 roster drivers identified. F1 and OpenF1 grid endpoints
were also exercised; undated responses first observed after historical cutoffs
were excluded. Singapore's final dated grid identifies Russell in slot 19 and
Alonso, Stroll and Hadjar as pit-lane starters.

Thirteen complete GP replays (Canada–Singapore) use the same cached sessions,
25,000 simulations, seed 42, and unchanged native training settings. Each trains only on earlier
2026 rounds. Training labels are identical, and target/future session-result
loads are prohibited. Source grids are dated before each prediction cutoff.

| Version | Average places out | Winner picks | Podium-driver hits |
|---|---:|---:|---:|
| Previous qualifying-order assumption | 3.273 | 8/13 | 24/39 |
| Published grid before scoring split | 3.371 | 8/13 | 25/39 |
| Published grid plus equal qualifying/grid split, adopted | 3.238 | 8/13 | 25/39 |
| Published grid plus circuit-dependent split, not adopted | 3.266 | 8/13 | 25/39 |

The adopted split improves seven races, ties five and worsens one against the
published-grid baseline. Direct model-score rank error, before sampled win-share
ordering, also improves: 3.406 to 3.252. The earlier seven-race average improves
3.597 to 3.494; the later six-race average improves 3.106 to 2.939. These races
have been examined previously and are not an untouched holdout. Candidate
settings were fixed before this comparison; no driver-specific rules were added.

The routine eight-test report remains 5/8 for Singapore, with ranking error
3.307 to 3.125. Singapore Russell improves from P16 to P9 (actual P9), Hungary
Antonelli P7 to P5 (actual P3), Italy Antonelli P15 to P11 (actual P1), and Belgium
Hadjar P17 to P13 (actual P6). Large recoveries remain underestimated. Netherlands
worsens 2.455 to 2.636; the change does not improve every race.

Historical training uses recorded historical grids where available. Historical
pit-lane records with grid=0 retain the existing qualifying fallback, a shared
limitation of this comparison. Simulated win-share ordering can also be unstable
near the back when shares are tiny or zero. No full finishing-order or recovery
calibration is established by these results.
FP2/FP3 are absent from the supplied cache. Early-season prediction still needs
four earlier weekends. Section telemetry stays disabled, the DNF experiment
remains discarded, and weather/live-update modelling is outside this branch.
