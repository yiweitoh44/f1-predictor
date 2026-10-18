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
strategy and recovery are not modelled.

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

The second command explicitly chooses the previous qualifying-order assumption.
Keep the six existing model evidence columns and all session/history weights.
Qualifying performance is preserved separately in `qualifying_grid_norm` and
`starting_grid.csv`; a grid penalty does not overwrite the observed Q/SQ result.
This is useful context, not an active new recovery model. A trial with a seventh
learned qualifying input failed to improve the comparison and was not adopted.
Sophisticated recovery remains deferred.

## Outputs

- `starting_grid.csv`: driver, qualifying position, grid slot, pit-lane status,
  applied grid score and source. Blank pit-lane status means unknown.
- `grid_sources.json`: selected source URL, dated/observed time, source
  conflicts, rejected/unavailable sources and fallback details.
- `run_info.json`: includes the same grid audit alongside the existing metadata.

The terminal identifies the selected source, pit-lane starters, conflicts or
qualifying fallback. Grids are only fetched after the corresponding qualifying
has loaded. Q/SQ forecasts do not use target-race grid information.

## Validation on 18 October 2026

The authoritative external test file passes 118 checks, including grid timing,
identity, source conflicts, pit-lane flags, offline fallback and cached revisions.
All four session modes completed real cached replays. Q/SQ results are unchanged;
explicit qualifying-grid fallback reproduces the previous race prediction.

All **17 GP and 6 Sprint** grids through Singapore were obtained from dated FIA
documents, with all 22 roster drivers identified. F1 and OpenF1 grid endpoints
were also exercised; undated responses first observed after historical cutoffs
were excluded. Singapore's final dated grid identifies Russell in slot 19 and
Alonso, Stroll and Hadjar as pit-lane starters.

Thirteen complete GP replays (Canada–Singapore) use the same cached sessions,
25,000 simulations and unchanged model settings. Each trains only on earlier
2026 rounds. Training labels are identical, and target/future session-result
loads are prohibited. Source grids are dated before each prediction cutoff.

| Version | Average places out | Winner picks | Podium-driver hits |
|---|---:|---:|---:|
| Previous qualifying-order assumption | 3.273 | 8/13 | 24/39 |
| Published grid, retained model | 3.371 | 8/13 | 25/39 |
| Published grid plus trial qualifying input, not adopted | 3.406 | 8/13 | 24/39 |

The retained change improves two race ranking errors, ties seven and worsens
four. This is a source-correctness improvement, **not demonstrated overall
prediction improvement**. The model's strong grid influence and limited
recovery handling remain important. These races have been examined previously;
they are not an untouched holdout, and no source/driver weights were tuned here.

The routine eight-test report remains 5/8 for Singapore: this branch changes
the target grid; historical training/backtests already use recorded historical
grid positions where available. The existing six model columns are unchanged.
FP2/FP3 are absent from the supplied cache. Early-season prediction still needs
four earlier weekends. Section telemetry stays disabled, the DNF experiment
remains discarded, and weather/live-update modelling is outside this branch.
