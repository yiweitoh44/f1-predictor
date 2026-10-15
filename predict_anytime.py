"""Predict GP/Sprint qualifying or races using only data available beforehand.
Run: python predict_anytime.py --race Singapore --session race
Requires the existing requirements.txt. No OpenAI API or LLM is used.
Uses a new training pipeline; it does not reproduce the original GitHub model.

SETUP: Put this file beside your existing tracks.csv and requirements.txt.
Select --session race, qualifying, sprint or sprint-qualifying. Rerun the same
command after another session's data becomes available. This predicts an
unstarted session, not an in-progress race. At least four usable earlier
weekends in the selected season are required. Missing sessions are reported
and omitted; missing driver values are not filled with invented positions.

TRAINING: Qualifying and Sprint Qualifying share one family; Race and Sprint
share another. A Sprint indicator distinguishes formats. Each rerun trains
with the current availability mask: preweekend predictions use earlier form
alone, while later runs can use practice/qualifying already completed. Old
weekends are tested chronologically, with no future labels used for features.
Pooling Sprint and normal sessions is a practical small-data assumption, not
evidence they are identical. Backtests are small and probabilities uncalibrated.

TRACKS: Optional numeric geometry descriptors remain supported. tracks.csv also
supports documented overlapping straight/corner/braking/traction/tyre groups
and profile_source. Keep circuit_sections.py beside this script. Completed
eligible sessions can supply approximate tyre-matched section telemetry.
Source-related result form is separate from measured section pace. Whole-lap
section suitability requires substantial coverage; missing data stays missing.
Section thresholds and source-group factors are experimental, not calibrated.

WEATHER/UPGRADES: Observed weather comes from completed sessions, not forecasts.
--air-temp and --wet are manual scenario overrides. There is no upgrade-news
feed or quantified upgrade effect. Later measured team pace is the proxy.

ROSTER/GRID: Before running this weekend, the last known competitive roster is
assumed unchanged. FP reserve-driver substitutions also need checking. Supply
--roster with driver,team,name columns to correct it. Grid defaults to qualifying
order; use --grid driver,grid CSV with every driver once after qualifying to
account for penalties. Pit-lane starters can only be approximated at the back.

OUTPUTS: outputs_anytime/year_round_session/stage holds predictions, training,
prediction_inputs, backtest CSVs, model.json and run_info.json. A rerun of the
same stage replaces that stage's files. Results from different stages persist.
Use --refresh for corrected data. --as-of allows historical replay, but uses
today's published historical revisions, not original archived data snapshots.

VALIDATION: Syntax, CLI and offline stage/cutoff/cache/simulation checks passed.
FastF1 network calls and real XGBoost fitting could not be exercised in the
authoring environment. Offline integration checks used stand-ins for those two
components. No live result or prediction-accuracy improvement is claimed.
"""
from __future__ import annotations
import argparse
import json
import logging
import re
from pathlib import Path
import numpy as np
import pandas as pd
from circuit_sections import (GROUP_COLUMNS, summarize_sections,
                              historical_section_features, target_sections)

ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path.home() / "F1Data"
CACHE_ROOT = DATA_ROOT / "cache"
OUTPUT_ROOT = DATA_ROOT / "outputs"
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
        for key in GROUP_COLUMNS + ("profile_source",):
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
        best = best[best.peers >= 2].copy()
        best["gap"] = 100 * (best.pace / best.reference - 1)

        for (driver, compound), group in best.groupby(
            ["Driver", "Compound"], observed=True
        ):
            row = output.setdefault(str(driver), {})
            row[f"{label}_{compound.lower()}_gap_pct"] = float(
                group.gap.median()
            )
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

def summarize(session, code):
    # FastF1 uses pandas subclasses. Convert to ordinary frames before validation.
    results = pd.DataFrame(session.results).copy()
    if results.empty or "Abbreviation" not in results:
        raise ValueError("no driver results/roster published")
    results = results[results.Abbreviation.notna()].copy()
    results["Abbreviation"] = results.Abbreviation.astype(str).str.strip()
    results = results[results.Abbreviation != ""]
    if results.Abbreviation.duplicated().any() or len(results) < 2:
        raise ValueError("driver roster is incomplete or ambiguous")
    n = len(results)
    pos = pd.to_numeric(results.get("Position", pd.Series(np.nan, index=results.index)), errors="coerce").astype(float)
    invalid = (~np.isfinite(pos)) | (pos < 1) | (pos > n) | (pos % 1 != 0)
    pos = pos.mask(invalid)
    pos = pos.mask(pos.duplicated(keep=False) & pos.notna())
    results["Position"] = pos
    raw_laps = pd.DataFrame(session.laps).copy()
    laps = raw_laps.copy()
    best, lap_counts = {}, {}
    if not laps.empty and {"Driver", "LapTime"}.issubset(laps.columns):
        laps = laps.dropna(subset=["Driver", "LapTime"])
        for col in ("PitInTime", "PitOutTime"):
            if col in laps:
                laps = laps[laps[col].isna()]

        required_quality = {"IsAccurate", "Deleted", "TrackStatus"}

        if required_quality.issubset(laps.columns):
            laps = laps[
                laps.IsAccurate.eq(True)
                & laps.Deleted.eq(False)
                & laps.TrackStatus.astype("string").isin(["1", "1.0"])
            ].copy()
        else:
            # Missing quality information is not proof of a valid pace lap.
            laps = laps.iloc[0:0].copy()
        seconds = pd.to_timedelta(laps.LapTime).dt.total_seconds()
        laps = laps.assign(lap_seconds=seconds)
        laps = laps[np.isfinite(laps.lap_seconds) & (laps.lap_seconds > 0)]
        best = laps.groupby("Driver").lap_seconds.min().to_dict()
        lap_counts = laps.groupby("Driver").size().to_dict()
    fastest = min(best.values()) if best else np.nan
    pace = tyre_pace_summary(laps, code)
    section_analysis = summarize_sections(session, laps, code)
    context = (
        race_session_context(results, raw_laps)
        if code in RACE_TYPES else {}
    )
    if code.startswith("FP") and not best:
        raise ValueError("no usable practice laps published")
    if code in {"Q", "SQ", "S", "R"} and pos.notna().sum() < max(2, n // 2):
        raise ValueError("too few valid classified positions published")
    drivers = []
    for row in results.to_dict("records"):
        driver = row["Abbreviation"]
        lap = number(best.get(driver))
        position = number(row.get("Position"))
        drivers.append({"driver": driver,
                        "name": str(row.get("FullName") or driver),
                        "team": str(row.get("TeamName") or "Unknown"),
                        "position": position,
                        "position_norm": (position - 1) / max(1, n - 1),
                        "grid": number(row.get("GridPosition")),
                        "gap_pct": 100 * (lap / fastest - 1) if np.isfinite(lap) and np.isfinite(fastest) else np.nan,
                        "laps": float(lap_counts.get(driver, 0)),
                        "sections": section_analysis.get("drivers", {}).get(driver, []),
                        **pace.get(driver, {}),
                        **context.get(driver, {})})
    try:
        weather = pd.DataFrame(session.weather_data)
    except Exception:
        weather = pd.DataFrame()
    observed = {}
    for column in ("AirTemp", "TrackTemp", "Rainfall"):
        values = pd.to_numeric(weather[column], errors="coerce").dropna() if column in weather else pd.Series(dtype=float)
        observed[column] = float(values.max() if column == "Rainfall" else values.median()) if len(values) else np.nan
    # Driver evidence is already attached above; avoid duplicating it in JSON.
    section_analysis = {k: v for k, v in section_analysis.items() if k != "drivers"}
    return {"code": code, "drivers": drivers, "weather": observed,
            "section_analysis": section_analysis}


class Store:
    def __init__(self, year, now, refresh=False):
        self.year, self.now, self.refresh = year, now, refresh
        self.folder = CACHE_ROOT / "anytime_v4_sections" / str(year)
        self.memory = {}

    def get(self, event, code):
        rnd = int(event["RoundNumber"])
        key = (rnd, code)
        if key in self.memory:
            return self.memory[key]
        start = dict(session_schedule(event)).get(code)
        # Never fetch a session that has not started as of this prediction cutoff.
        if start is None or pd.isna(start) or start >= self.now:
            return None
        path = self.folder / f"{rnd:02d}_{code}.json"
        if path.exists() and not self.refresh:
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.get("schema") == 4 and saved.get("event") == event["EventName"]:
                end = pd.to_datetime(saved.get("completed_utc"), utc=True, errors="coerce")
                if pd.notna(end) and end < self.now:
                    self.memory[key] = saved["data"]
                    return saved["data"]
        import fastf1
        try:
            # Retry recently published sessions against the server, rather than
            # repeatedly reading a partial FastF1 cache from an earlier attempt.
            fastf1.Cache.enable_cache(str(CACHE_ROOT / "fastf1"),
                                     force_renew=self.refresh)
            print(f"  Building section summary R{rnd} {code}", flush=True)
            session = fastf1.get_session(self.year, rnd, code)
            session.load(laps=True, telemetry=True, weather=True, messages=True)
            statuses = pd.DataFrame(session.session_status)
            completed = pd.NaT
            if {"Status", "Time"}.issubset(statuses.columns):
                final = statuses[statuses.Status.astype(str).str.casefold().isin(["finalised", "finalized"]) ]
                if not final.empty and pd.notna(session.t0_date):
                    completed = pd.to_datetime(session.t0_date, utc=True) + pd.to_timedelta(final.Time.iloc[-1])
            # Fail closed for live/incomplete sessions. Old published classifications
            # may lack status timing; allow only with a conservative past-day cutoff.
            if pd.isna(completed):
                if self.now < start + pd.Timedelta(days=1):
                    raise ValueError("session not confirmed finalised yet")
                completed = start + pd.Timedelta(days=1)
            if completed >= self.now:
                return None
            data = summarize(session, code)
        except Exception as exc:
            print(f"  R{rnd} {code} unavailable: {type(exc).__name__}: {str(exc)[:130]}")
            self.memory[key] = None
            return None
        self.folder.mkdir(parents=True, exist_ok=True)
        payload = {"schema": 4, "event": event["EventName"], "completed_utc": completed.isoformat(), "data": data}
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(json_safe(payload)), encoding="utf-8")
        temp.replace(path)
        self.memory[key] = json_safe(data)
        return self.memory[key]


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
        team_records = [r for r in records if r["team"] == team]
        row[label + "_driver_form"] = mean_recent([r["position_norm"] for r in individual]) 
        team_sessions = {}
        for record in team_records:
            key = (record["round"], record["session"])
            team_sessions.setdefault(key, []).append(
                record["position_norm"]
            )

        row[label + "_team_form"] = mean_recent([
            float(np.mean(positions))
            for positions in team_sessions.values()
        ])
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
            code, label, int(event["RoundNumber"]), history, inputs,
            tracks, event["Location"]
        )
        row.update(extra)
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
                if code in RACE_TYPES and ("Q" if code == "R" else "SQ") in inputs and grid > 0:
                    row["start_grid_norm"] = (grid - 1) / max(1, len(label["drivers"]) - 1)
                rows.append({**row, "label": y, "event_round": rnd,
                             "target_session": code, "driver": driver["driver"]})
        for code, _ in session_schedule(event):
            add_history(history, data.get(code), rnd, event["Location"])
    return pd.DataFrame(rows), history


def fit(frame, columns):
    import xgboost as xgb

    weights = 0.95 ** (
        frame.event_round.max() - frame.event_round.to_numpy()
    )

    training = xgb.DMatrix(
        frame[columns].to_numpy(dtype=np.float32),
        label=frame.label.to_numpy(dtype=np.float32),
        weight=weights,
        feature_names=columns,
    )

    booster = xgb.train(
        {
            "objective": "reg:squarederror",
            "max_depth": 3,
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
            return booster.predict(inputs)

        def save_model(self, path):
            booster.save_model(str(path))

    return NativeModel()


def train_and_check(frame, target_code):
    excluded = {"label", "event_round", "target_session", "driver"}
    # Section timings remain inspectable, but are not validated model inputs.
    columns = [
        c for c in frame.columns
        if c not in excluded and "_section_" not in c
    ]
    rounds = sorted(frame.event_round.unique())
    if len(rounds) < 4:
        raise ValueError("Need usable history from at least four earlier weekends in this season.")
    test_rounds = [
        rnd for rnd in rounds[3:]
        if (
            (frame.event_round == rnd)
            & (frame.target_session == target_code)
        ).any()
    ][-4:]

    if not test_rounds:
        raise ValueError(
            f"No usable {target_code} backtests with at least "
            "three earlier training weekends."
        )

    errors, details = [], []
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
            errors.extend([{"session": session_code, "error": float(e)}
                           for e in group.label.to_numpy() - guess])
            picked = group.iloc[int(np.argmin(guess))].driver
            actual = group.loc[group.label.idxmin(), "driver"]
            details.append({"round": int(rnd), "session": session_code,
                            "pick": picked, "actual": actual, "correct": picked == actual})
    exact = [r["error"] for r in errors if r["session"] == target_code]
    residuals = exact if len(exact) >= 20 else [r["error"] for r in errors]
    if len(residuals) < 2:
        raise ValueError("Insufficient earlier test errors to estimate uncertainty.")
    print(f"Backtest: {sum(d['correct'] for d in details)}/{len(details)} first-place picks; "
          f"{sum(d['session'] == target_code for d in details)} tests of the requested session type.")
    if len(exact) < 20:
        print("Uncertainty uses pooled session-family errors; requested-session evidence is sparse.")
    sigma = max(float(np.std(residuals, ddof=1)), 0.03)
    return fit(frame, columns), columns, sigma, details


def simulate(scores, sigma, count):
    rng = np.random.default_rng(42)
    ranks = np.argsort(scores[None, :] + rng.normal(0, sigma, (count, len(scores))), axis=1)
    win = np.bincount(ranks[:, 0], minlength=len(scores)) * 100 / count
    top3 = np.bincount(ranks[:, :min(3, len(scores))].ravel(), minlength=len(scores)) * 100 / count
    order_sum = np.zeros(len(scores))
    for place in range(len(scores)):
        order_sum += np.bincount(ranks[:, place], minlength=len(scores)) * (place + 1)
    return win, top3, order_sum / count


def run(args):
    import fastf1
    now = pd.Timestamp.now(tz="UTC")
    if args.as_of:
        now = pd.to_datetime(args.as_of, utc=True, errors="raise")
        if now > pd.Timestamp.now(tz="UTC"):
            raise ValueError("--as-of cannot be in the future.")
    cache = CACHE_ROOT / "fastf1"
    cache.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache), force_renew=args.refresh)
    fastf1.set_log_level("ERROR")
    schedule = pd.DataFrame(fastf1.get_event_schedule(args.year, include_testing=False))
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
    store = Store(args.year, now, args.refresh)
    print(f"Target: {args.year} {event['EventName']} / {args.session}. Cutoff: {now.isoformat()}")
    current = {}
    for code in allowed:
        data = store.get(event, code)
        if data:
            current[code] = data
    print("This weekend's inputs: " + (", ".join(current) if current else "none; using earlier performance only"))
    for code, data in current.items():
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
    previous = schedule[schedule.RoundNumber < int(event["RoundNumber"])].to_dict("records")
    all_data, usable_events = {}, []
    for past in previous:
        # Some earlier rounds may also still be in the future for a distant target.
        if not any(pd.notna(t) and t < now for _, t in session_schedule(past)):
            continue
        rnd = int(past["RoundNumber"])
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
    if target in RACE_TYPES:
        print("Starting grid defaults to qualifying order when available; penalties require --grid.")
    if args.grid:
        if target not in RACE_TYPES or ("Q" if target == "R" else "SQ") not in current:
            raise ValueError("Use --grid only for a race after its qualifying data is available.")
        grid = pd.read_csv(args.grid, encoding="utf-8-sig")
        if not {"driver", "grid"}.issubset(grid.columns):
            raise ValueError("Grid CSV requires driver,grid columns with every entered driver.")
        grid["driver"] = grid.driver.astype(str).str.strip().str.upper()
        grid["grid"] = pd.to_numeric(grid.grid, errors="raise")
        if (grid.driver.duplicated().any() or set(grid.driver) != set(names)
                or set(grid.grid) != set(range(1, len(names) + 1))):
            raise ValueError("Grid CSV must contain every driver once and each grid position 1 through the field size once. Approximate pit-lane starters as last.")
        mapping = dict(zip(grid.driver, grid.grid))
        inputs["start_grid_norm"] = [(mapping[d] - 1) / (len(names) - 1) for d in names]
    for value, field in ((args.air_temp, "known_air_temp"), (args.wet, "known_rain")):
        if value is not None:
            inputs[field] = value
    if args.air_temp is not None or args.wet is not None:
        print("Weather override is your scenario, not an automatically fetched forecast.")
    if not any(np.isfinite(number(v)) for v in tracks.get(event["Location"], {}).values()):
        print("No usable target circuit descriptors in tracks.csv.")
    circuit_columns = inputs.filter(regex=r"_(similar_tracks|related_circuit_form|section_gap_pct)$")
    if circuit_columns.isna().all().all():
        print("Circuit suitability unavailable: insufficient geometry, sourced relationships or section evidence.")
    section_family = "race" if target in RACE_TYPES else "qualifying"
    section_source, section_profile = target_sections(current, section_family)
    for code, data in current.items():
        info = data.get("section_analysis", {})
        print(f"  {code} section analysis: {info.get('status', 'not available')}; "
              f"{info.get('sampled_laps', 0)} sampled laps, "
              f"{info.get('failed_laps', 0)} rejected telemetry traces.")
    section_evidence = []
    for name in names:
        _, evidence = historical_section_features(
            name, section_family, int(event["RoundNumber"]), history,
            current, tracks, event["Location"]
        )
        section_evidence.extend(evidence)
    if section_source:
        coverage = inputs[section_family + "_section_coverage"]
        print(f"Section reference: current {section_source}; historical coverage "
              f"{100 * coverage.min():.0f}..{100 * coverage.max():.0f}% across drivers.")
        print("Section matches are approximate; no fuel/traffic/banking correction.")
    else:
        print("Section suitability unavailable: no usable pre-target circuit telemetry.")
    print(
        f"Training on {len(frame)} driver-session examples "
        f"from {frame.event_round.nunique()} earlier weekends."
    )
    print("Section pace: diagnostic only; excluded from model training and predictions.")
    model, columns, sigma, tests = train_and_check(frame, target)
    scores = model.predict(inputs[columns])
    win, top3, average = simulate(scores, sigma, args.simulations)
    output = pd.DataFrame({"driver": names, "name": [d.get("name", d["driver"]) for d in drivers],
                           "team": [d["team"] for d in drivers], "first_pct": win,
                           "top3_pct": top3, "mean_simulated_position": average}).sort_values("first_pct", ascending=False)
    stage = "_".join(current) if current else "pre_weekend"
    folder = OUTPUT_ROOT / f"{args.year}_{int(event['RoundNumber']):02d}_{args.session}" / "latest"
    folder.mkdir(parents=True, exist_ok=True)
    if target in {"Q", "SQ"}:
        output = output.rename(columns={"first_pct": "pole_pct"})
        output = output.drop(columns=["top3_pct", "mean_simulated_position"])
    else:
        output = output.rename(columns={
        "first_pct": "win_pct",
        "top3_pct": "podium_pct",
        })
    output.to_csv(folder / "predictions.csv", index=False)
    frame.to_csv(folder / "training.csv", index=False)
    inputs.assign(driver=names).to_csv(folder / "prediction_inputs.csv", index=False)
    pd.DataFrame(tests).to_csv(folder / "backtest.csv", index=False)
    model.save_model(folder / "model.json")
    evidence_columns = ["driver", "family", "target_section", "kind", "target_share",
                        "historical_round", "historical_session", "historical_location",
                        "match_score", "source_factor", "weight", "gap_pct"]
    pd.DataFrame(section_evidence, columns=evidence_columns).to_csv(
        folder / "section_evidence.csv", index=False)
    pd.DataFrame(section_profile, columns=["start", "end", "kind", "seconds", "length",
                                          "speed", "share", "condition"]).to_csv(
        folder / "target_sections.csv", index=False)
    metadata = {"target": args.session, "event": event["EventName"], "as_of": now.isoformat(),
                "used_sessions": list(current), "roster_source": roster_source,
                "weather": "latest available session observations or manual scenario; no forecast API",
                "manual_weather": {"air_temp": args.air_temp, "wet": args.wet},
                "grid_source": str(args.grid) if args.grid else "qualifying order when available",
                "sigma_normalized_positions": sigma, "simulations": args.simulations,
                "model_features": columns,
                "section_pace_model_inputs": False,
                "section_analysis": {
                    "reference_session": section_source,
                    "minimum_overall_coverage": 0.70,
                    "minimum_historical_weekends": 2,
                    "maximum_speed_difference_kmh": 40,
                    "minimum_section_length_m": 30,
                    "minimum_section_duration_s": 0.3,
                    "section_length_ratio_range": [0.5, 2.0],
                    "documented_group_factor": 1.0,
                    "undocumented_group_factor": 0.75,
                    "recency_per_round": 0.85,
                    "limits": "Approximate marker windows and reference-lap speed bands; no fuel, traffic, gradient, banking or battery correction. Thresholds not calibrated."
                },
                "tracks": tracks, "upgrades": "not separately measured or modelled",
                "historical_data": "current published revisions, not archived as-of snapshots"}
    (folder / "run_info.json").write_text(json.dumps(json_safe(metadata), indent=2), encoding="utf-8")
    print("\nExperimental simulation shares; not calibrated real-world probabilities:")
    print(output.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    print("First = pole for qualifying; winner for a race. Top3 = qualifying top three or race podium.")
    print("These are sampled rankings, not lap-by-lap simulations. Race DNFs are learned through results; no extra retirement lottery.")
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
    parser.add_argument("--refresh", action="store_true", help="Refresh FastF1 and local summary caches")
    parser.add_argument("--as-of", help="Historical replay cutoff, e.g. 2026-09-25T06:00:00Z")
    parser.add_argument("--simulations", type=int, default=25_000)
    parser.add_argument("--air-temp", type=float, help="Optional manual weather scenario in Celsius")
    parser.add_argument("--wet", type=int, choices=[0, 1], help="Optional manual dry/wet scenario")
    args = parser.parse_args()
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
