"""Predict GP/Sprint qualifying or races using only data available beforehand.
Run: python predict_anytime.py --race Singapore --session race
Requires the existing requirements.txt. No OpenAI API or LLM is used.
Uses a new training pipeline; it does not reproduce the original GitHub model.

SETUP: Keep session_influence.py, circuit_sections.py, grid_sources.py, openf1_data.py, tracks.csv and
requirements.txt beside this file. See START_HERE.txt for setup and scoring.
Select --session race, qualifying, sprint or sprint-qualifying. Rerun the same
command after another session's data becomes available. This predicts an
unstarted session, not an in-progress race. At least four usable earlier
weekends in the selected season are required. Missing sessions are reported
and omitted; missing driver values are not filled with invented positions.

TRAINING: Qualifying and Sprint Qualifying share one family; Race and Sprint
share another. Explicit source weights distinguish formats. Each rerun trains
with the current availability mask: preweekend predictions use earlier form
alone, while later runs can use practice/qualifying already completed. Old
weekends are tested chronologically, with no future labels used for features.
Pooling Sprint and normal sessions is a practical small-data assumption, not
evidence they are identical. Recent form covers four previous weekends. The
score combines 85% explicit session/history evidence with a 15% monotonic
native XGBoost adjustment. Backtests are small and probabilities uncalibrated.

TRACKS: Optional numeric geometry descriptors remain supported. tracks.csv also
supports manual overlapping straight/corner/braking/traction/tyre groups
and profile_source. Keep circuit_sections.py beside this script. Completed
sessions supply results for manual category matching. Straights, corners,
braking, traction and tyre demands can overlap. Category weights are manual
starting assumptions. Section telemetry is not downloaded or calculated in
normal runs and all section-pace columns remain excluded from model fitting.

WEATHER/UPGRADES: Observed weather comes from completed sessions, not forecasts.
--air-temp and --wet record observations/scenarios only; weather is not an
active model input in this version. There is no upgrade-news feed or quantified
upgrade effect. Recent team results can reflect changes without identifying them.

ROSTER/GRID: Before running this weekend, the last known competitive roster is
assumed unchanged. FP reserve-driver substitutions also need checking. Supply
--roster with driver,team,name columns to correct it. After qualifying, automatic
grid selection prefers dated FIA documents, then F1/OpenF1 records observed
before the cutoff. Qualifying order is the fallback. --grid overrides sources;
--grid-source qualifying explicitly uses qualifying order. Pit-lane starters
are identified separately and share a back-of-field score; delay is not modelled.

OUTPUTS: ~/F1Data/outputs/year_round_session/latest holds predictions, training,
prediction_inputs, backtest CSVs, model.json, run_info.json and section diagnostic
CSVs. Tables use mean simulated position. Race/Sprint show close adjacent
matchups on the lower driver's row; Q/SQ show pole share. Full simulation
diagnostics stay in simulation_details.csv. Backtest comparisons against grid
and qualifying order (history only for Q/SQ) use matching eligible weekends.
model.json stores the learned component; influence_policy.json records
the blend. Rerunning the same event/session replaces its latest files. Raw caches
are retained. --refresh explicitly renews data; --offline uses cached data only.
--rebuild-summaries recalculates summaries using ordinary raw cache reuse.
--as-of allows historical replay, but uses today's published historical
revisions, not original archived data snapshots.

VALIDATION: See START_HERE.txt for chronological comparisons, regression
checks and the Sprint Qualifying limitation. Cached replays do not validate
fresh timing downloads or the raw-lap cause of extreme legacy practice gaps.
Section pace remains disabled and excluded from training and predictions.
"""
from __future__ import annotations
import argparse
import json
import logging
import re
from pathlib import Path
import numpy as np
import pandas as pd
import session_influence as influence
import grid_sources
import openf1_data
from circuit_sections import (GROUP_COLUMNS, GROUP_WEIGHTS, circuit_group_match,
                              historical_section_features)

ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path.home() / "F1Data"
CACHE_ROOT = DATA_ROOT / "cache"
OUTPUT_ROOT = DATA_ROOT / "outputs"
SUMMARY_SCHEMA = 6
BACKTEST_WEEKENDS = 8
UNCERTAINTY_WEEKENDS = 4
CLOSE_CALL_MIN_PCT = 45.0
CLOSE_CALL_MAX_PCT = 55.0
SESSION_NAMES = {"Practice 1": "FP1", "Practice 2": "FP2", "Practice 3": "FP3",
                 "Sprint Qualifying": "SQ", "Sprint Shootout": "SQ",
                 "Sprint": "S", "Qualifying": "Q", "Race": "R"}
MODES = {"race": "R", "qualifying": "Q", "sprint": "S", "sprint-qualifying": "SQ"}
RACE_TYPES = {"R", "S"}
INPUTS = ("FP1", "FP2", "FP3", "SQ", "S", "Q")
TRACK_SCALES = {"grid_stickiness": 1.0, "length_km": 7.0,
                "longest_straight_km": 2.5, "slow_corner_fraction": 1.0,
                "high_speed_fraction": 1.0, "street": 1.0}


def number(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else np.nan
    except (TypeError, ValueError):
        return np.nan


def mean_recent(values):
    values = [number(v) for v in values[-8:] if np.isfinite(number(v))]
    if not values:
        return np.nan
    result = values[0]
    for value in values[1:]:
        result = 0.45 * value + 0.55 * result
    return float(result)


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def session_schedule(event):
    result = []
    for index in range(1, 6):
        code = SESSION_NAMES.get(str(event.get(f"Session{index}", "")))
        if code:
            start = pd.to_datetime(event.get(f"Session{index}DateUtc"), utc=True, errors="coerce")
            result.append((code, start))
    return result


def allowed_before(event, target):
    codes = [code for code, _ in session_schedule(event)]
    if target not in codes:
        return []
    return [code for code in codes[:codes.index(target)] if code in INPUTS]


def select_event(schedule, name):
    if name.isdigit():
        matches = schedule[schedule.RoundNumber == int(name)]
    else:
        exact = pd.Series(False, index=schedule.index)
        partial = exact.copy()
        for col in ("EventName", "Location", "Country"):
            if col in schedule:
                text = schedule[col].astype(str).str.casefold()
                exact |= text == name.casefold()
                partial |= text.str.contains(name.casefold(), regex=False)
        matches = schedule[exact if exact.any() else partial]
    if len(matches) != 1:
        raise ValueError("Race name missing or ambiguous. Use --list-races and choose its round number.")
    return matches.iloc[0].to_dict()


def read_tracks(path):
    if not path.exists():
        print("No tracks.csv: circuit descriptors will be missing. Predictions use performance data.")
        return {}
    frame = pd.read_csv(
        path, encoding="utf-8-sig", sep=None, engine="python"
    )
    if "location" not in frame or frame.location.isna().any() or frame.location.duplicated().any():
        raise ValueError("tracks.csv needs one unique, nonblank location per row.")
    result = {}
    for row in frame.to_dict("records"):
        values = {key: number(row.get(key)) for key in TRACK_SCALES}
        for key in ("grid_stickiness", "slow_corner_fraction", "high_speed_fraction", "street"):
            if np.isfinite(values[key]) and not 0 <= values[key] <= 1:
                raise ValueError(f"{row['location']}: {key} must be between 0 and 1.")
        for key in ("length_km", "longest_straight_km"):
            if np.isfinite(values[key]) and values[key] <= 0:
                raise ValueError(f"{row['location']}: {key} must be positive.")
        for key in GROUP_COLUMNS + ("profile_source", "profile_basis"):
            value = row.get(key)
            values[key] = "" if pd.isna(value) else str(value).strip()
        result[str(row["location"])] = values
    return result

PACE_COMPOUNDS = ("SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET")


def tyre_pace_summary(laps, code):
    """Matched-tyre gaps and net stint trends, not fuel correction."""
    required = {"Driver", "Compound", "TyreLife", "FreshTyre",
                "LapNumber", "Stint", "LapStartTime", "lap_seconds"}
    if laps.empty or not required.issubset(laps.columns):
        return {}

    data = laps.copy()
    data["Compound"] = data.Compound.astype("string").str.upper()

    for column in ("TyreLife", "LapNumber", "Stint", "lap_seconds"):
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data["start_s"] = pd.to_timedelta(
        data.LapStartTime, errors="coerce"
    ).dt.total_seconds()

    data = data[
        data.Compound.isin(PACE_COMPOUNDS)
        & data.FreshTyre.isin([True, False])
        & np.isfinite(data.TyreLife) & (data.TyreLife >= 1)
        & np.isfinite(data.LapNumber) & np.isfinite(data.Stint)
        & np.isfinite(data.start_s)
        & np.isfinite(data.lap_seconds) & (data.lap_seconds > 0)
    ].copy()

    if data.empty:
        return {}

    personal_best = data.groupby(["Driver", "Compound"], observed=True).lap_seconds.transform("min")
    data = data[data.lap_seconds <= personal_best * influence.PACE_MAX_LAP_SPREAD].copy()
    if data.empty:
        return {}

    data["age_band"] = ((data.TyreLife - 1) // 3).astype(int)

    # Practice: ten-minute windows. Race/Sprint: five-lap windows.
    data["phase"] = (
        (data.LapNumber - 1) // 5 if code in {"S", "R"}
        else data.start_s // 600
    ).astype(int)

    output = {}
    keys = ["Compound", "FreshTyre", "age_band", "phase"]

    def compare(frame, label, statistic, minimum):
        if frame.empty:
            return

        best = frame.groupby(keys + ["Driver"], observed=True).agg(
            pace=("lap_seconds", statistic),
            samples=("lap_seconds", "size")
        ).reset_index()

        best = best[best.samples >= minimum].copy()
        groups = best.groupby(keys, observed=True).pace
        best["peers"] = groups.transform("size")
        best["reference"] = groups.transform("min")
        best["spread"] = groups.transform("max") / best.reference
        best = best[(best.peers >= 2) & (best.spread <= influence.PACE_MAX_LAP_SPREAD)].copy()
        best["gap"] = 100 * (best.pace / best.reference - 1)

        for (driver, compound), group in best.groupby(
            ["Driver", "Compound"], observed=True
        ):
            row = output.setdefault(str(driver), {})
            row[f"{label}_{compound.lower()}_gap_pct"] = float(
                group.gap.median()
            )
            row[f"{label}_{compound.lower()}_matched_laps"] = int(group.samples.sum())
            row[label + "_matched_laps"] = (
                row.get(label + "_matched_laps", 0)
                + int(group.samples.sum())
            )

    if code.startswith("FP"):
        compare(data[data.TyreLife <= 5], "short", "min", 1)

    if not (code.startswith("FP") or code in {"S", "R"}):
        return output

    running = (
        data[data.LapNumber > 1].copy()
        if code in {"S", "R"} else data
    )
    sustained = []
    trends = {}

    for _, stint in running.groupby(
        ["Driver", "Stint", "Compound", "FreshTyre"], observed=True
    ):
        stint = stint.sort_values("LapNumber")
        breaks = (
            stint.LapNumber.diff().ne(1)
            | stint.TyreLife.diff().ne(1)
            | (stint.start_s.diff() <= 0)
        ).cumsum()

        for _, run in stint.groupby(breaks):
            if len(run) < 4:
                continue

            sustained.append(run)
            slope = np.polyfit(run.TyreLife, run.lap_seconds, 1)[0]
            trends.setdefault(str(run.Driver.iloc[0]), []).append(
                float(100 * slope / run.lap_seconds.median())
            )

    if sustained:
        compare(pd.concat(sustained), "long", "median", 2)

    for driver, slopes in trends.items():
        output.setdefault(driver, {})[
            "long_trend_pct_per_lap"
        ] = float(np.median(slopes))

    return output


def race_session_context(results, raw_laps):
    """Finishing gap, completion and neutralisation context."""
    output = {}
    winners = results[results.Position.eq(1)]
    if winners.empty:
        return output

    winner = winners.iloc[0]
    winner_time = pd.to_timedelta(winner.get("Time"), errors="coerce")
    winner_seconds = (
        winner_time.total_seconds()
        if pd.notna(winner_time) else np.nan
    )
    winner_laps = pd.to_numeric(winner.get("Laps"), errors="coerce")
    n = len(results)

    for record in results.to_dict("records"):
        driver = record["Abbreviation"]
        status_value = record.get("Status")
        status = (
            "" if pd.isna(status_value)
            else str(status_value).strip().casefold()
        )
        laps = pd.to_numeric(record.get("Laps"), errors="coerce")
        grid = pd.to_numeric(record.get("GridPosition"), errors="coerce")
        position = pd.to_numeric(record.get("Position"), errors="coerce")

        finished = (
            float(
                status == "finished"
                or (status.startswith("+") and "lap" in status)
            )
            if status else np.nan
        )

        duration = pd.to_timedelta(record.get("Time"), errors="coerce")
        seconds = (
            duration.total_seconds()
            if pd.notna(duration) else np.nan
        )

        valid_gap = (
            status == "finished"
            and np.isfinite(winner_seconds)
            and winner_seconds > 0
            and np.isfinite(seconds)
            and seconds >= winner_seconds
            and np.isfinite(laps)
            and np.isfinite(winner_laps)
            and laps == winner_laps
        )

        fraction = (
            laps / winner_laps
            if np.isfinite(laps) and np.isfinite(winner_laps)
            and winner_laps > 0 else np.nan
        )

        neutralized = np.nan
        if {"Driver", "TrackStatus"}.issubset(raw_laps.columns):
            statuses = raw_laps.loc[
                raw_laps.Driver.eq(driver), "TrackStatus"
            ].dropna().astype(str)
            if len(statuses):
                neutralized = float(
                    statuses.str.contains(r"[4567]", regex=True).mean()
                )

        output[driver] = {
            "finish_gap_pct": (
                100 * (seconds - winner_seconds) / winner_seconds
                if valid_gap else np.nan
            ),
            "laps_completed_fraction": float(fraction),
            "finished": finished,
            "grid_gain_norm": (
                float((grid - position) / max(1, n - 1))
                if np.isfinite(grid) and grid > 0
                and np.isfinite(position) else np.nan
            ),
            "neutralized_lap_fraction": neutralized,
        }

    return output

class Store(openf1_data.Store):
    def __init__(self, year, now, refresh=False, offline=False, rebuild_summaries=False):
        super().__init__(openf1_data.Client(CACHE_ROOT, offline, refresh), year, now, tyre_pace_summary)


def circuit_similarity(a, b):
    # Need at least two geometry descriptors; a grid assumption alone is not
    # evidence of similar straights/corners.
    keys = [k for k in TRACK_SCALES if k != "grid_stickiness"
            and np.isfinite(number(a.get(k))) and np.isfinite(number(b.get(k)))]
    if len(keys) < 2:
        return np.nan
    distance = np.mean([abs(number(a[k]) - number(b[k])) / TRACK_SCALES[k] for k in keys])
    return float(np.exp(-4 * distance))


def features(driver, target_code, event, history, inputs, tracks):
    code, team = driver["driver"], driver["team"]
    row = {"sprint_target": float(target_code in {"SQ", "S"}),
           "round": float(event["RoundNumber"])}
    circuit = tracks.get(event["Location"], {})
    row.update({"track_" + k: number(circuit.get(k)) for k in TRACK_SCALES})
    for label, family in (("qualifying", {"Q", "SQ"}), ("race", RACE_TYPES)):
        records = [r for r in history if r["session"] in family]
        individual = [r for r in records if r["driver"] == code]
        row[label + "_driver_form"], row[label + "_history_weekends"] = influence.historical_form(
            records, int(event["RoundNumber"]), driver=code)
        row[label + "_team_form"], row[label + "_team_history_weekends"] = influence.historical_form(
            records, int(event["RoundNumber"]), team=team)
        values = [number(r["position_norm"]) for r in individual[-8:]]
        row[label + "_consistency"] = float(np.std(values)) if len(values) >= 2 else np.nan
        weights, performances = [], []
        for prior in individual:
            similarity = circuit_similarity(circuit, tracks.get(prior["location"], {}))
            if np.isfinite(similarity):
                weights.append(similarity * 0.85 ** max(0, int(event["RoundNumber"]) - prior["round"]))
                performances.append(prior["position_norm"])
        row[label + "_similar_tracks"] = float(np.average(performances, weights=weights)) if weights else np.nan
    row["known_air_temp"] = row["known_track_temp"] = row["known_rain"] = np.nan
    grid_source = inputs.get("Q" if target_code == "R" else "SQ") if target_code in RACE_TYPES else None
    grid_driver = next((d for d in grid_source["drivers"] if d["driver"] == code), {}) if grid_source else {}
    row["start_grid_norm"] = number(grid_driver.get("position_norm"))
    row["qualifying_grid_norm"] = row["start_grid_norm"]
    for session_code in INPUTS:
        data = inputs.get(session_code)
        row[session_code + "_available"] = float(data is not None)
        participant = next(
            (d for d in data["drivers"] if d["driver"] == code), {}
        ) if data else {}

        for key in ("position_norm", "gap_pct", "laps"):
            row[session_code + "_" + key] = number(participant.get(key))

        if session_code.startswith("FP"):
            row[session_code + "_position_norm"] = np.nan

        if session_code.startswith("FP") or session_code == "S":
            # Raw fastest-lap gap stays in cache, but is not a pace input.
            row[session_code + "_gap_pct"] = np.nan

            labels = (
                ("short", "long")
                if session_code.startswith("FP") else ("long",)
            )

            for label in labels:
                for compound in PACE_COMPOUNDS:
                    key = f"{label}_{compound.lower()}_gap_pct"
                    row[session_code + "_" + key] = number(
                        participant.get(key)
                    )

                key = label + "_matched_laps"
                row[session_code + "_" + key] = number(
                    participant.get(key, 0)
                )

            row[session_code + "_long_trend_pct_per_lap"] = number(
                participant.get("long_trend_pct_per_lap")
            )

        if session_code == "S":
            for key in (
                "finish_gap_pct", "laps_completed_fraction", "finished",
                "grid_gain_norm", "neutralized_lap_fraction"
            ):
                row["S_" + key] = number(participant.get(key))
    # Iterate in actual scheduled order, not alphabetic order, for latest weather.
    for session_code, _ in session_schedule(event):
        if session_code in inputs:
            weather = inputs[session_code].get("weather", {})
            for field, key in (("AirTemp", "known_air_temp"), ("TrackTemp", "known_track_temp"), ("Rainfall", "known_rain")):
                if np.isfinite(number(weather.get(field))):
                    row[key] = number(weather[field])
    for label in ("qualifying", "race"):
        extra, _ = historical_section_features(
            code, label, int(event["RoundNumber"]),
            [r for r in history if int(r["round"]) >= int(event["RoundNumber"]) - influence.HISTORY_WEEKENDS], {},
            tracks, event["Location"], team=team
        )
        row.update(extra)
    practices = []
    for order, session_code in enumerate(("FP1", "FP2", "FP3"), 1):
        data = inputs.get(session_code)
        if data:
            score = influence.pace_score(code, data, ("long", "short") if target_code in RACE_TYPES else ("short",))
            practices.append((score, order))
    row["practice_evidence_score"] = influence.weighted_available(practices)
    sprint = inputs.get("S")
    row["sprint_evidence_score"] = np.nan
    if sprint:
        participant = next((d for d in sprint["drivers"] if d['driver'] == code), {})
        # An early retirement cannot establish race pace from finishing position.
        # Its observed longer-run pace can still be used when comparable.
        position = number(participant.get('position_norm'))
        if number(participant.get('finished')) == 0 or number(participant.get('laps_completed_fraction')) < .75:
            position = np.nan
        row["sprint_evidence_score"] = influence.weighted_available([
            (position, .8), (influence.pace_score(code, sprint, ('long',)), .2)])
    row.update(influence.add_influence(row, target_code))
    return row


def add_history(history, data, rnd, location):
    if data and data["code"] in {"Q", "SQ", "R", "S"}:
        for driver in data["drivers"]:
            if np.isfinite(number(driver.get("position_norm"))):
                history.append({**driver, "round": rnd, "location": location, "session": data["code"]})


def training_rows(events, all_data, family, mask, tracks):
    history, rows = [], []
    for event in events:
        rnd = int(event["RoundNumber"])
        data = all_data.get(rnd, {})
        # History is updated only after all examples for this weekend are built.
        # Same-weekend information enters only through explicit session inputs.
        for code in ("SQ", "Q", "S", "R"):
            label = data.get(code)
            if code not in family or label is None:
                continue
            inputs = {c: data[c] for c in allowed_before(event, code) if c in mask and c in data}
            for driver in label["drivers"]:
                y = number(driver.get("position_norm"))
                if not np.isfinite(y):
                    continue
                row = features(driver, code, event, history, inputs, tracks)
                # Actual historical grids are pre-race information, but use them
                # only at a stage where the corresponding qualifying is available.
                grid = number(driver.get("grid"))
                # Reporting only: never mistake a qualifying fallback for an
                # independently recorded historical starting grid.
                row['benchmark_grid_position'] = (
                    grid if code in RACE_TYPES and ("Q" if code == "R" else "SQ") in inputs
                    and np.isfinite(grid) and grid == int(grid) and 1 <= grid <= len(label['drivers'])
                    else np.nan)
                if code in RACE_TYPES and ("Q" if code == "R" else "SQ") in inputs and grid > 0:
                    row["start_grid_norm"] = (grid - 1) / max(1, len(label["drivers"]) - 1)
                    row.update(influence.add_influence(row, code))
                rows.append({**row, "label": y, "event_round": rnd,
                             "target_session": code, "driver": driver["driver"]})
        for code, _ in session_schedule(event):
            add_history(history, data.get(code), rnd, event["Location"])
    return pd.DataFrame(rows), history


def fit(frame, columns):
    import xgboost as xgb
    learned_share = influence.LEARNED_SHARE

    weights = 2. ** (-(frame.event_round.max() - frame.event_round.to_numpy()) / 6.)
    # A Sprint weekend must not get twice the total training weight.
    weekend_counts = frame.groupby("event_round").label.transform("size").to_numpy()
    weights = weights / weekend_counts * np.median(weekend_counts)

    training = xgb.DMatrix(
        frame[columns].to_numpy(dtype=np.float32),
        label=frame.label.to_numpy(dtype=np.float32),
        weight=weights,
        feature_names=columns,
    )

    booster = xgb.train(
        {
            "objective": "reg:squarederror",
            "max_depth": 2,
            "monotone_constraints": tuple(1 for _ in columns),
            "eta": 0.04,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "lambda": 3.0,
            "seed": 42,
            "nthread": 2,
            "verbosity": 0,
        },
        training,
        num_boost_round=200,
    )

    class NativeModel:
        def predict(self, data):
            inputs = xgb.DMatrix(
                data[columns].to_numpy(dtype=np.float32),
                feature_names=columns,
            )
            learned = np.clip(booster.predict(inputs), 0., 1.)
            return ((1 - learned_share) * data["influence_anchor"].to_numpy()
                    + learned_share * learned)

        def save_model(self, path):
            booster.save_model(str(path))

    return NativeModel()


def ranking_metrics(drivers, predicted, actual):
    """Same classified drivers for every method; tied scores share mean rank."""
    drivers = np.asarray(drivers)
    predicted, actual = np.asarray(predicted, dtype=float), np.asarray(actual, dtype=float)
    if len(drivers) < 3 or not (len(predicted) == len(actual) == len(drivers)):
        raise ValueError('Ranking comparison needs at least three matching drivers.')
    if not np.isfinite(predicted).all() or not np.isfinite(actual).all():
        raise ValueError('Ranking comparison requires complete scores and classifications.')
    order = np.lexsort((drivers, predicted))
    actual_order = np.lexsort((drivers, actual))
    return dict(pick=str(drivers[order[0]]), actual=str(drivers[actual_order[0]]),
                correct=bool(drivers[order[0]] == drivers[actual_order[0]]),
                mean_rank_error=float(np.mean(np.abs(pd.Series(predicted).rank().to_numpy()
                                                    - pd.Series(actual).rank().to_numpy()))),
                top3_hits=len(set(drivers[order[:3]]) & set(drivers[actual_order[:3]])))


def benchmark_records(group, predicted, rnd, target):
    """Reporting uses pre-target evidence; never fill a missing baseline with labels."""
    methods = {'Model': np.asarray(predicted, dtype=float)}
    if target in RACE_TYPES:
        methods.update({'Starting grid': group.get('benchmark_grid_position'),
                        'Qualifying order': group.get('qualifying_grid_norm')})
    else:
        # Q/SQ's own eventual order cannot be a pre-session benchmark.
        methods['History only'] = group.get('influence_history')
    records = []
    for method, values in methods.items():
        record = dict(round=int(rnd), session=target, method=method, eligible=False,
                      reason='', drivers=len(group), pick=None, actual=None, correct=None,
                      mean_rank_error=None, top3_hits=None)
        if len(group) < 3:
            record['reason'] = 'fewer than three classified drivers for a top-three comparison'
        elif values is None or not np.isfinite(np.asarray(values, dtype=float)).all():
            record['reason'] = 'missing pre-session evidence for one or more classified drivers'
        elif method == 'Starting grid' and (np.any(np.asarray(values) < 1)
                or len(set(values)) != len(values)):
            record['reason'] = 'missing/ambiguous grid slots; no qualifying fallback used'
        elif method == 'Qualifying order' and (np.any(np.asarray(values) < 0)
                or np.any(np.asarray(values) > 1)):
            record['reason'] = 'invalid qualifying positions'
        else:
            record.update(eligible=True, **ranking_metrics(group.driver, values, group.label))
        records.append(record)
    return records


def comparison_summary(records, target):
    methods = ['Model', 'Starting grid', 'Qualifying order'] if target in RACE_TYPES else ['Model', 'History only']
    eligible = [{r['round'] for r in records if r['method'] == method and r['eligible']} for method in methods]
    common = set.intersection(*eligible)
    rows = []
    for method in methods:
        selected = [r for r in records if r['round'] in common and r['method'] == method and r['eligible']]
        if selected:
            rows.append({'Method': method, 'First correct': f"{sum(r['correct'] for r in selected)}/{len(selected)}",
                         'Podium drivers' if target in RACE_TYPES else 'Top-three drivers':
                             f"{sum(r['top3_hits'] for r in selected)}/{3*len(selected)}",
                         'Avg places out': float(np.mean([r['mean_rank_error'] for r in selected]))})
    return sorted(common), rows


def train_and_check(frame, target_code, comparison_rows=None):
    # Only the interpretable evidence scores enter the learned adjustment.
    # Raw pace, consistency, counts, weather and section fields remain diagnostics.
    columns = list(influence.RACE_MODEL_COLUMNS if target_code in RACE_TYPES else influence.MODEL_COLUMNS)
    rounds = sorted(frame.event_round.unique())
    if len(rounds) < 4:
        raise ValueError("Need usable history from at least four earlier weekends in this season.")
    test_rounds = [
        rnd for rnd in rounds[3:]
        if (
            (frame.event_round == rnd)
            & (frame.target_session == target_code)
        ).any()
    ][-BACKTEST_WEEKENDS:]

    if not test_rounds:
        raise ValueError(
            f"No usable {target_code} backtests with at least "
            "three earlier training weekends."
        )

    errors, details, comparisons = [], [], []
    for rnd in test_rounds:
        train = frame[frame.event_round < rnd]
        test = frame[
            (frame.event_round == rnd)
            & (frame.target_session == target_code)
        ]
        model = fit(train, columns)
        predicted = model.predict(test[columns])
        for session_code in test.target_session.unique():
            loc = np.where(test.target_session.to_numpy() == session_code)[0]
            group = test.iloc[loc]
            guess = predicted[loc]
            errors.extend([{"session": session_code, "round": rnd, "error": float(e)}
                           for e in group.label.to_numpy() - guess])
            actual = group.loc[group.label.idxmin(), "driver"]
            ranks = pd.Series(guess).rank(method="average").to_numpy()
            # Compare ranks among drivers with valid classifications. Labels
            # retain the original roster denominator when some results are missing.
            actual_ranks = group.label.rank(method="average").to_numpy()
            order = np.lexsort((group.driver.to_numpy(), guess))
            actual_order = np.argsort(group.label.to_numpy())
            picked = group.driver.iloc[order[0]]
            details.append({"round": int(rnd), "session": session_code,
                            "pick": picked, "actual": actual, "correct": picked == actual,
                            "mean_rank_error": float(np.mean(np.abs(ranks-actual_ranks))),
                            "top3_hits": len(set(group.driver.iloc[order[:3]]) & set(group.driver.iloc[actual_order[:3]]))})
            comparisons.extend(benchmark_records(group, guess, rnd, session_code))
    uncertainty_rounds = set(test_rounds[-UNCERTAINTY_WEEKENDS:])
    recent_errors = [r for r in errors if r['round'] in uncertainty_rounds]
    exact = [r["error"] for r in recent_errors if r["session"] == target_code]
    residuals = exact if len(exact) >= 20 else [r["error"] for r in recent_errors]
    if len(residuals) < 2:
        raise ValueError("Insufficient earlier test errors to estimate uncertainty.")
    print(f"Backtest: {sum(d['correct'] for d in details)}/{len(details)} first-place picks; "
          f"{sum(d['session'] == target_code for d in details)} tests of the requested session type.")
    print(f"Backtest ranking: mean position error {np.mean([d['mean_rank_error'] for d in details]):.2f}; "
          f"top-three drivers found {sum(d['top3_hits'] for d in details)}/{3*len(details)}.")
    common, table = comparison_summary(comparisons, target_code)
    if table:
        print(f"Backtest comparison: same {len(common)} of these {len(test_rounds)} tests; lower places out is better.")
        print(pd.DataFrame(table).to_string(index=False, float_format=lambda x: f'{x:.2f}'))
    else:
        print('Backtest comparison unavailable: insufficient matching pre-session baseline evidence.')
    if len(common) < len(test_rounds):
        print('Incomplete comparisons are excluded from every method in the comparison table; reasons saved in backtest_comparisons.csv.')
    if comparison_rows is not None:
        comparison_rows.extend(comparisons)
    print(f"Simulation uncertainty uses the latest {len(uncertainty_rounds)} eligible tests; "
          f"the report shows up to {BACKTEST_WEEKENDS}.")
    if len(exact) < 20:
        print("Uncertainty uses pooled session-family errors; requested-session evidence is sparse.")
    sigma = max(float(np.std(residuals, ddof=1)), 0.03)
    return fit(frame, columns), columns, sigma, details


def simulate(scores, sigma, count, return_positions=False):
    rng = np.random.default_rng(42)
    ranks = np.argsort(scores[None, :] + rng.normal(0, sigma, (count, len(scores))), axis=1)
    win = np.bincount(ranks[:, 0], minlength=len(scores)) * 100 / count
    top3 = np.bincount(ranks[:, :min(3, len(scores))].ravel(), minlength=len(scores)) * 100 / count
    order_sum = np.zeros(len(scores))
    for place in range(len(scores)):
        order_sum += np.bincount(ranks[:, place], minlength=len(scores)) * (place + 1)
    result = (win, top3, order_sum / count)
    if return_positions:
        positions = np.empty(ranks.shape, dtype=np.int32)
        np.put_along_axis(positions, ranks, np.arange(1, len(scores)+1)[None, :], axis=1)
        return (*result, positions)
    return result


def prediction_tables(drivers, scores, win, top3, average, positions, target):
    """Display by mean position; pairwise percentages use the very same draws."""
    names = np.asarray([d['driver'] for d in drivers])
    order = np.lexsort((names, scores, average))
    short_names = [str(d.get('name', d['driver'])).split()[-1] for d in drivers]
    labels = [str(d.get('name', d['driver'])) if short_names.count(short_names[i]) > 1 else short_names[i]
              for i, d in enumerate(drivers)]
    rows, audit = [], []
    for rank, i in enumerate(order, 1):
        participant = drivers[i]
        row = dict(predicted_position=rank, driver=participant['driver'],
                   name=participant.get('name', participant['driver']), team=participant['team'])
        above, ahead, close_call = None, None, ''
        if rank > 1:
            j = order[rank-2]
            above = drivers[j]['driver']
            ahead = float(np.mean(positions[:, i] < positions[:, j]) * 100)
            if CLOSE_CALL_MIN_PCT <= ahead <= CLOSE_CALL_MAX_PCT:
                close_call = f'↑ {labels[i]} beats {labels[j]} in {ahead:.1f}% of simulations'
        if target in RACE_TYPES:
            row.update(mean_position=float(average[i]), close_call=close_call)
        else:
            row['pole_pct'] = float(win[i])
        rows.append(row)
        audit.append(dict(predicted_position=rank, driver=participant['driver'], score=float(scores[i]),
                          first_pct=float(win[i]), top3_pct=float(top3[i]), mean_position=float(average[i]),
                          driver_above=above, ahead_of_above_pct=ahead,
                          close_call=bool(close_call) if target in RACE_TYPES else False))
    return pd.DataFrame(rows), pd.DataFrame(audit)


def print_prediction_table(output):
    display = output.drop(columns='driver').rename(columns={
        'predicted_position': 'Position', 'name': 'Driver', 'team': 'Team',
        'mean_position': 'Mean position', 'pole_pct': 'Pole share', 'close_call': 'Close call'})
    if 'Pole share' in display:
        display['Pole share'] = display['Pole share'].map(lambda x: f'{x:.1f}%')
    rendered = display.to_string(index=False, float_format=lambda x: f'{x:.2f}')
    # Older Windows terminals may not support Unicode; preserve the meaning.
    import sys
    encoding = getattr(sys.stdout, 'encoding', None) or 'utf-8'
    try:
        '↑'.encode(encoding)
    except UnicodeEncodeError:
        rendered = rendered.replace('↑', '^')
    print(rendered)


def run(args):
    now = pd.Timestamp.now(tz="UTC")
    if args.as_of:
        now = pd.to_datetime(args.as_of, utc=True, errors="raise")
        if now > pd.Timestamp.now(tz="UTC"):
            raise ValueError("--as-of cannot be in the future.")
    schedule = openf1_data.Client(CACHE_ROOT, args.offline, args.refresh).schedule(args.year)
    schedule = schedule[schedule.RoundNumber > 0].sort_values("RoundNumber")
    if args.list_races:
        print(schedule[["RoundNumber", "EventName", "Location"]].to_string(index=False))
        return
    event = select_event(schedule, args.race)
    target = MODES[args.session]
    allowed = allowed_before(event, target)
    starts = dict(session_schedule(event))
    if target not in starts:
        raise ValueError("That session is not on this event's schedule (perhaps this is not a Sprint weekend).")
    target_start = starts[target]
    if pd.isna(target_start):
        raise ValueError("Target session start time is unavailable; cannot establish a safe cutoff.")
    if now >= target_start:
        print(f"The scheduled start for {args.session} has passed.")
        print("It may be ongoing or finished; live status is not checked.")
        print("Continue using only completed sessions from before the target?")

        try:
            answer = input("Continue? [y/N]: ").strip().casefold()
        except EOFError:
            raise ValueError(
                "No confirmation received. Run interactively or supply "
                "--as-of before the target start."
            ) from None

        if answer not in {"y", "yes"}:
            raise ValueError("Prediction cancelled.")

        now = target_start - pd.Timedelta(seconds=1)
        print(f"Pre-session replay cutoff: {now.isoformat()}")
    tracks = read_tracks(args.tracks)
    store = Store(args.year, now, args.refresh, args.offline,
                  getattr(args, "rebuild_summaries", False))
    print(f"Target: {args.year} {event['EventName']} / {args.session}. Cutoff: {now.isoformat()}")
    current = {}
    for code in allowed:
        data = store.get(event, code)
        if data:
            current[code] = data
    print("This weekend's inputs: " + (", ".join(current) if current else "none; using earlier performance only"))
    for code, data in current.items():
        audit = data.get('lap_quality_audit', [])
        if audit:
            print(f"  {code} OpenF1 derived lap quality: {sum(r['accepted'] for r in audit)}/{len(audit)} accepted ({data.get('quality_policy')}).")
        if code.startswith("FP") or code == "S":
            short = sum(
                number(d.get("short_matched_laps", 0)) > 0
                for d in data["drivers"]
            )
            long = sum(
                number(d.get("long_matched_laps", 0)) > 0
                for d in data["drivers"]
            )
            print(
                f"  {code} matched pace: "
                f"short {short}/{len(data['drivers'])}; "
                f"long {long}/{len(data['drivers'])} drivers"
            )
            labels = ('long',) if code == 'S' else (('long', 'short') if target in RACE_TYPES else ('short',))
            usable = sum(np.isfinite(influence.pace_score(d['driver'], data, labels)) for d in data['drivers'])
            print(f"  {code} usable pace evidence after quality/support checks: {usable}/{len(data['drivers'])} drivers")
    previous = schedule[schedule.RoundNumber < int(event["RoundNumber"])].to_dict("records")
    all_data, usable_events, requested_rounds = {}, [], []
    for past in previous:
        # Some earlier rounds may also still be in the future for a distant target.
        if not any(pd.notna(t) and t < now for _, t in session_schedule(past)):
            continue
        rnd = int(past["RoundNumber"])
        requested_rounds.append(rnd)
        #print(f"Loading earlier weekend R{rnd}: {past['EventName']}", flush=True)
        available = {}
        for code, _ in session_schedule(past):
            # Labels/history need competitive sessions. Practice is downloaded
            # only when that practice type is present in the current input stage.
            if code in {"Q", "SQ", "S", "R"} or code in current:
                data = store.get(past, code)
                if data:
                    available[code] = data
        if available:
            all_data[rnd] = available
            usable_events.append(past)
    if not usable_events:
        raise ValueError("No earlier season data could be loaded.")
    family = RACE_TYPES if target in RACE_TYPES else {"Q", "SQ"}
    frame, history = training_rows(usable_events, all_data, family, set(current), tracks)
    if frame.empty:
        raise ValueError("No valid historical labels available for this prediction type.")
    # Prefer the latest available weekend roster; otherwise use the last known
    # competitive entry list. Allow an explicit roster for replacements/new teams.
    if args.roster:
        roster = pd.read_csv(args.roster, encoding="utf-8-sig")
        if not {"driver", "team"}.issubset(roster.columns):
            raise ValueError("Roster CSV needs driver,team columns (name is optional).")
        roster["driver"] = roster.driver.astype(str).str.strip().str.upper()
        roster["name"] = roster["name"] if "name" in roster else roster.driver
        drivers = roster[["driver", "name", "team"]].to_dict("records")
        roster_source = str(args.roster)
    elif current:
        last_code = [c for c in allowed if c in current][-1]
        drivers = current[last_code]["drivers"]
        roster_source = "current " + last_code + "; check reserve-driver substitutions"
    else:
        candidates = [(e, c) for e in usable_events for c, _ in session_schedule(e)
                      if c in all_data[int(e["RoundNumber"])] and c in {"R", "Q", "S", "SQ"}]
        if not candidates:
            raise ValueError("No competitive driver roster available. Supply --roster.")
        last, last_code = candidates[-1]
        sessions = all_data[int(last["RoundNumber"])]
        drivers = sessions[last_code]["drivers"]
        roster_source = f"earlier R{int(last['RoundNumber'])} {last_code}; assumes unchanged drivers/teams"
    names = [d["driver"] for d in drivers]
    if len(drivers) < 3 or len(names) != len(set(names)) or any(not str(d.get("team", "")).strip() for d in drivers):
        raise ValueError("Roster is incomplete or contains duplicate drivers.")
    print("Roster: " + roster_source)
    inputs = pd.DataFrame([features(d, target, event, history, current, tracks) for d in drivers])
    grid_record = None
    grid_warnings = []
    grid_mode = getattr(args, 'grid_source', 'auto')
    qualifying_code = 'Q' if target == 'R' else 'SQ'
    if args.grid and (target not in RACE_TYPES or qualifying_code not in current):
        raise ValueError("Use --grid only for a race after its qualifying data is available.")
    if target in RACE_TYPES:
        if args.grid:
            grid_record = grid_sources.manual_grid(args.grid, drivers, now)
        elif qualifying_code in current and grid_mode != 'qualifying':
            grid_store = grid_sources.GridStore(CACHE_ROOT, args.offline, args.refresh)
            grid_record = grid_store.resolve(args.year, event, target, drivers, now,
                live=not args.as_of and pd.Timestamp.now(tz='UTC') < target_start)
            grid_warnings = grid_store.warnings
        if grid_record:
            grid_sources.apply_grid(inputs, drivers, grid_record)
            print(f"Starting grid: {grid_record['source']} ({grid_record['url']}).")
            pit_drivers = [r['driver'] for r in grid_record['rows'] if r.get('pit_lane') is True]
            if pit_drivers:
                print('Pit-lane starters: ' + ', '.join(pit_drivers) + '; back-of-field score, exit order/delay not modelled.')
            if any(r.get('pit_lane') is None for r in grid_record['rows']):
                print('This grid source does not identify pit-lane starters; status is unknown.')
            if grid_record.get('conflicts'):
                print('Grid sources disagree; using the higher-priority source. See grid_sources.json.')
        else:
            print('Starting grid: qualifying order fallback; penalties/pit-lane changes are not confirmed.'
                  if qualifying_code in current else 'Starting grid unavailable: qualifying has not loaded.')
            if grid_warnings:
                print('Published grid unavailable at cutoff; details saved in grid_sources.json.')
    for value, field in ((args.air_temp, "known_air_temp"), (args.wet, "known_rain")):
        if value is not None:
            inputs[field] = value
    if args.air_temp is not None or args.wet is not None:
        print("Weather scenario recorded for diagnostics; it does not change this version's prediction.")
    target_track = tracks.get(event["Location"], {})
    if not (any(np.isfinite(number(target_track.get(k))) for k in TRACK_SCALES)
            or any(target_track.get(k) for k in GROUP_COLUMNS)):
        print("No usable target circuit descriptors in tracks.csv.")
    circuit_columns = inputs.filter(regex=r"_(similar_tracks|related_circuit_form|section_gap_pct)$")
    if circuit_columns.isna().all().all():
        print("Circuit suitability unavailable: insufficient geometry or matching historical weekends.")
    # Legacy cached section evidence is intentionally ignored in normal runs.
    section_profile, section_evidence = [], []
    print(
        f"Training on {len(frame)} driver-session examples "
        f"from {frame.event_round.nunique()} earlier weekends."
    )
    coverage = {
        "training_rounds": sorted(int(r) for r in frame.event_round.unique()),
        "requested_earlier_rounds": requested_rounds,
        "current_sessions_with_lap_timing": [code for code, data in current.items() if data.get("lap_data_loaded", False)],
        "classification_only_current_sessions": [code for code, data in current.items() if not data.get("lap_data_loaded", False)],
    }
    coverage["incomplete"] = (len(coverage["training_rounds"]) < len(requested_rounds)
                              or bool(coverage["classification_only_current_sessions"]))
    timing_sessions = ", ".join(coverage["current_sessions_with_lap_timing"]) or "none"
    print(f"Training coverage: {len(coverage['training_rounds'])}/{len(requested_rounds)} earlier weekends; "
          f"current sessions with lap timing: {timing_sessions}.")
    if coverage["incomplete"]:
        print("Incomplete data coverage: inspect run_info.json alongside these experimental rankings.")
    print("Section pace: disabled; excluded from model training and predictions.")
    if target in RACE_TYPES:
        print("Grid evidence budget: equal starting-grid / qualifying-performance split when both are available.")
    inputs = inputs.assign(**pd.DataFrame([
        influence.add_influence(row, target) for row in inputs.to_dict('records')]))
    comparisons = []
    model, columns, sigma, tests = train_and_check(frame, target, comparison_rows=comparisons)
    scores = model.predict(inputs[columns])
    win, top3, average, positions = simulate(scores, sigma, args.simulations, return_positions=True)
    output, simulation_details = prediction_tables(drivers, scores, win, top3, average, positions, target)
    folder = OUTPUT_ROOT / f"{args.year}_{int(event['RoundNumber']):02d}_{args.session}" / "latest"
    folder.mkdir(parents=True, exist_ok=True)
    quality_rows = [dict(session=code, policy=data.get('quality_policy'), **row)
                    for code, data in current.items() for row in data.get('lap_quality_audit', [])]
    pd.DataFrame(quality_rows, columns=['session','policy','driver','lap_number','accepted','reason']).to_csv(
        folder / 'lap_quality.csv', index=False)
    output.to_csv(folder / "predictions.csv", index=False)
    simulation_details.to_csv(folder / 'simulation_details.csv', index=False)
    frame.to_csv(folder / "training.csv", index=False)
    inputs.assign(driver=names).to_csv(folder / "prediction_inputs.csv", index=False)
    grid_audit = dict(mode=grid_mode, selected=grid_record, warnings=grid_warnings,
                      fallback='qualifying order' if target in RACE_TYPES and grid_record is None else None)
    (folder / 'grid_sources.json').write_text(json.dumps(json_safe(grid_audit), indent=2), encoding='utf-8')
    grid_map = {r['driver']: r for r in grid_record['rows']} if grid_record else {}
    pd.DataFrame([dict(driver=d, qualifying_position=(inputs.iloc[i].qualifying_grid_norm*(len(names)-1)+1)
                      if np.isfinite(inputs.iloc[i].qualifying_grid_norm) else None,
                      grid_position=grid_map[d]['position'] if d in grid_map else
                          (inputs.iloc[i].start_grid_norm*(len(names)-1)+1 if np.isfinite(inputs.iloc[i].start_grid_norm) else None),
                      pit_lane=grid_map.get(d, {}).get('pit_lane'),
                      grid_score=inputs.iloc[i].start_grid_norm,
                      source=grid_record['source'] if grid_record else
                          ('qualifying fallback' if target in RACE_TYPES else 'not applicable'))
                  for i,d in enumerate(names)]).to_csv(folder/'starting_grid.csv',index=False)
    pd.DataFrame(tests).to_csv(folder / "backtest.csv", index=False)
    pd.DataFrame(comparisons).to_csv(folder / 'backtest_comparisons.csv', index=False)
    model.save_model(folder / "model.json")
    (folder / "influence_policy.json").write_text(json.dumps(influence.policy(), indent=2), encoding="utf-8")
    inputs[[c for c in inputs if c.startswith('influence_')]].assign(driver=names).to_csv(
        folder / "session_influence.csv", index=False)
    pace_rejections = []
    for session_code, data in current.items():
        for participant in data['drivers']:
            for key, value in participant.items():
                if key.startswith(('short_', 'long_')) and key.endswith('_gap_pct'):
                    reason = influence.pace_reason(value)
                    if reason and reason != 'missing':
                        pace_rejections.append(dict(session=session_code, driver=participant['driver'], feature=key,
                                                    value=value, reason=reason))
    pd.DataFrame(pace_rejections, columns=['session','driver','feature','value','reason']).to_csv(
        folder / "pace_rejections.csv", index=False)
    matches = []
    for past in usable_events:
        similarity, contributions = circuit_group_match(
            tracks.get(event["Location"], {}), tracks.get(past["Location"], {}))
        distance = int(event["RoundNumber"]) - int(past["RoundNumber"])
        in_window = 0 < distance <= influence.HISTORY_WEEKENDS
        matches.append({"round": int(past["RoundNumber"]), "location": past["Location"],
                        "similarity": similarity,
                        "recency_factor": 0.85 ** distance,
                        "within_history_window": in_window,
                        "historical_weight": similarity * 0.85 ** distance if in_window else 0.,
                        **contributions})
    pd.DataFrame(matches).to_csv(folder / "circuit_matches.csv", index=False)
    evidence_columns = ["driver", "family", "target_section", "kind", "target_share", "condition",
                        "historical_round", "historical_session", "historical_location",
                        "match_score", "source_factor", "session_weight", "weekend_weight",
                        "weight", "accepted", "gap_pct", "candidate_sections",
                        "candidate_gap_min_pct", "candidate_gap_max_pct"]
    pd.DataFrame(section_evidence, columns=evidence_columns).to_csv(
        folder / "section_evidence.csv", index=False)
    pd.DataFrame(section_profile, columns=["start", "end", "kind", "seconds", "length",
                                          "speed", "share", "condition"]).to_csv(
        folder / "target_sections.csv", index=False)
    # Preserve legacy CSV headers while section analysis stays disabled.
    # Session timing coverage remains useful independently of section telemetry.
    cohort_rows, rejection_rows, session_coverage = [], [], []
    diagnostic_sessions = [(int(past["RoundNumber"]), past["Location"], all_data.get(int(past["RoundNumber"]), {}))
                           for past in usable_events]
    diagnostic_sessions.append((int(event["RoundNumber"]), event["Location"], current))
    for rnd, location, sessions in diagnostic_sessions:
        for code, data in sessions.items():
            origin = {"year": args.year, "round": rnd, "location": location, "session": code}
            session_coverage.append({**origin, "lap_data_loaded": data.get("lap_data_loaded", False),
                                     "section_status": "disabled",
                                     "pace_summary_version": data.get("pace_summary_version", 1)})
    cohort_columns = ["year", "round", "location", "session", "reference_driver", "reference_lap",
                      "driver", "section", "kind", "start", "end", "compound", "fresh",
                      "age_band", "phase", "condition", "peer_drivers", "drivers", "laps",
                      "section_times", "driver_median_seconds", "cohort_median_seconds", "gap_pct"]
    pd.DataFrame(cohort_rows, columns=cohort_columns).to_csv(folder / "section_cohorts.csv", index=False)
    pd.DataFrame(rejection_rows, columns=["year", "round", "location", "session", "driver", "lap", "reason"]).to_csv(
        folder / "section_rejections.csv", index=False)
    metadata = {"target": args.session, "event": event["EventName"], "as_of": now.isoformat(),
                "used_sessions": list(current), "roster_source": roster_source,
                "weather": "diagnostic observations/scenarios only; not an active model input; no forecast API",
                "manual_weather": {"air_temp": args.air_temp, "wet": args.wet},
                "grid_source": grid_record['source'] if grid_record else "qualifying order when available",
                "grid_details": grid_audit,
                "sigma_normalized_positions": sigma, "simulations": args.simulations,
                "backtest_maximum_weekends": BACKTEST_WEEKENDS,
                "validation_comparison": {
                    "common_rounds": comparison_summary(comparisons, target)[0],
                    "model_order": "score order (expected mean-position order under shared simulation noise)",
                    "grid_basis": "historical OpenF1 grid when qualifying is available; current published revisions; no qualifying fallback for a missing benchmark grid",
                    "missing_baseline": "exclude the weekend from every method in the comparison table; retain full model backtest report",
                    "ties": "mean ranks for ranking error; driver code breaks tied first/podium picks"},
                "output_policy": {
                    "order": "mean simulated position, then model score, then driver code",
                    "close_call_ahead_pct_range": [CLOSE_CALL_MIN_PCT, CLOSE_CALL_MAX_PCT],
                    "close_call": "Race/Sprint lower row only; fraction of the same simulations where this driver finishes ahead of the adjacent driver above",
                    "simulation_details": "unchanged first/top-three shares, scores, means and adjacent comparisons saved separately"},
                "uncertainty_maximum_weekends": UNCERTAINTY_WEEKENDS,
                "model_features": columns,
                "section_pace_model_inputs": False,
                "session_influence": influence.policy(),
                "prediction_formula": "(1-learned_share)*influence_anchor + learned_share*clip(native_model_prediction,0,1)",
                "circuit_category_weights": GROUP_WEIGHTS,
                "circuit_category_method": "weighted tag overlap; one result average per weekend; minimum two matching weekends; recency 0.85 per round",
                "circuit_history_window_rounds": influence.HISTORY_WEEKENDS,
                "offline": args.offline,
                "summary_schema": SUMMARY_SCHEMA,
                "data_provider": "OpenF1; no FastF1 runtime or timing-cache dependency",
                "lap_quality": "derived OpenF1 checks, not an official IsAccurate flag; see lap_quality.csv",
                "unavailable_sessions": store.unavailable,
                "session_data_coverage": session_coverage,
                "data_coverage": coverage,
                "section_analysis": {"enabled": False,
                    "physical_alignment_validated": False,
                    "status": "disabled; legacy diagnostic evidence ignored"},
                "tracks": tracks, "upgrades": "not separately measured or modelled",
                "historical_data": "current published revisions, not archived as-of snapshots"}
    (folder / "run_info.json").write_text(json.dumps(json_safe(metadata), indent=2), encoding="utf-8")
    print("\nPrediction ordered by mean simulated position. Shares are experimental, not calibrated probabilities.")
    if target in RACE_TYPES:
        print('Close calls: the named row driver beats the driver above in 45–55% of simulations.')
    print_prediction_table(output)
    print(f"Saved: {folder}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--race")
    parser.add_argument("--session", choices=MODES, default="race")
    parser.add_argument("--list-races", action="store_true")
    parser.add_argument("--tracks", type=Path, default=ROOT / "tracks.csv")
    parser.add_argument("--roster", type=Path, help="Optional verified driver,team,name CSV")
    parser.add_argument("--grid", type=Path, help="Optional full driver,grid CSV after qualifying, for grid penalties")
    parser.add_argument("--grid-source", choices=['auto', 'qualifying'], default='auto',
                        help="Auto: dated FIA/F1/OpenF1 grids; qualifying: explicitly use Q/SQ order")
    parser.add_argument("--refresh", action="store_true", help="Refresh OpenF1 and local summary caches")
    parser.add_argument("--rebuild-summaries", action="store_true",
                        help="Recalculate compact summaries, reusing raw OpenF1 caches; combine with --offline to forbid downloads")
    parser.add_argument("--offline", action="store_true", help="Use cached OpenF1 data only; do not make network requests")
    parser.add_argument("--as-of", help="Historical replay cutoff, e.g. 2026-09-25T06:00:00Z")
    parser.add_argument("--simulations", type=int, default=25_000)
    parser.add_argument("--air-temp", type=float, help="Record temperature scenario in Celsius; diagnostic only, no score effect")
    parser.add_argument("--wet", type=int, choices=[0, 1], help="Record dry/wet scenario; diagnostic only, no score effect")
    args = parser.parse_args()
    if args.offline and args.refresh:
        parser.error("--offline cannot be combined with --refresh.")
    if not args.list_races and not args.race:
        parser.error("Supply --race Singapore or --list-races.")
    if not 100 <= args.simulations <= 1_000_000:
        parser.error("Use between 100 and 1000000 simulations.")
    try:
        run(args)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        raise SystemExit(f"Stopped: {exc}") from exc


if __name__ == "__main__":
    main()
