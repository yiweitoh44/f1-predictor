"""Coarse, distance-aligned section evidence for predict_anytime.py.

Only call summarize_sections after the predictor confirms a session completed
before its cutoff. Corner markers locate approximate zones, not exact radii.
No fuel, traffic, banking or battery-state correction is claimed.
"""
from collections import defaultdict
import numpy as np
import pandas as pd

KINDS = ("braking", "traction", "slow_corner", "medium_corner",
         "fast_corner", "straight", "other")
MIN_COVERAGE = 0.70
MAX_LAPS_PER_DRIVER = 6
GROUP_COLUMNS = ("straight_groups", "corner_groups", "braking_groups", "traction_groups", "tyre_groups")


def source_match(kind, target_profile, past_profile):
    """Small, explicit expert prior; telemetry compatibility is still required.

    A shared documented group gets factor 1; unsupported matches get 0.75.
    No negative conclusion is drawn from blank group membership.
    """
    field = {"straight": "straight_groups", "braking": "braking_groups",
             "traction": "traction_groups"}.get(kind, "corner_groups")
    if kind in {"other", "slow_corner", "medium_corner"}:
        return 0.75
    a = set(str(target_profile.get(field, "")).split("|")) - {""}
    b = set(str(past_profile.get(field, "")).split("|")) - {""}
    return 1.0 if a & b else 0.75


def related_circuit_form(driver, family, target_round, history, tracks, location):
    """Pre-weekend fallback: related-circuit RESULTS, not inferred section speed."""
    target = tracks.get(location, {})
    codes = {"Q", "SQ"} if family == "qualifying" else {"R", "S"}
    rows = []
    for prior in history:
        if prior.get("driver") != driver or prior.get("session") not in codes or int(prior["round"]) >= target_round:
            continue
        past = tracks.get(prior["location"], {})
        matched = []
        for key in GROUP_COLUMNS:
            a = set(str(target.get(key, "")).split("|")) - {""}
            b = set(str(past.get(key, "")).split("|")) - {""}
            if a & b:
                matched.append(key)
        y = finite(prior.get("position_norm"))
        if matched and np.isfinite(y):
            weight = len(matched) / len(GROUP_COLUMNS) * 0.85 ** (target_round - int(prior["round"]))
            rows.append((weight, y, int(prior["round"])))
    result = {family + "_related_circuit_form": np.nan,
              family + "_related_circuit_weekends": float(len({r[2] for r in rows}))}
    if len({r[2] for r in rows}) >= 2:
        result[family + "_related_circuit_form"] = sum(w * y for w, y, _ in rows) / sum(w for w, _, _ in rows)
    return result


def finite(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else np.nan
    except (TypeError, ValueError):
        return np.nan


def normalize_trace(frame):
    """Use elapsed time at fixed distances, never compare raw cross-track seconds."""
    required = {"Distance", "Time", "Speed", "Throttle", "Brake"}
    if not required.issubset(frame.columns):
        raise ValueError("missing distance/time/speed/throttle/brake telemetry")
    data = frame.copy()
    data["seconds"] = pd.to_timedelta(data.Time).dt.total_seconds()
    for key in ("Distance", "Speed", "Throttle", "Brake"):
        data[key] = pd.to_numeric(data[key], errors="coerce")
    data = data.dropna(subset=["Distance", "seconds", "Speed", "Throttle", "Brake"])
    # Preserve sample order: sorting could conceal reversed or corrupt distances.
    if len(data) < 30 or (np.diff(data.Distance) <= 0).any():
        raise ValueError("too little telemetry or non-increasing distance")
    if (np.diff(data.seconds) <= 0).any() or (data.Speed <= 0).any():
        raise ValueError("non-increasing time or invalid speed")
    if np.diff(data.seconds).max() > 2:
        raise ValueError("telemetry gap exceeds two seconds")
    if data.Distance.iloc[0] > 50 or data.Distance.iloc[-1] < 1000:
        raise ValueError("incomplete lap telemetry")
    return data


def make_sections(reference, corners, start_distance=None, end_distance=None):
    """Partition every interval, including unclassified 'other', exactly once."""
    ref = normalize_trace(reference)
    end = float(ref.Distance.iloc[-1])
    start = max(0., float(ref.Distance.iloc[0]), start_distance or 0.)
    grid = np.arange(start, min(end, end_distance if end_distance is not None else end), 10.)
    if len(grid) < 100:
        raise ValueError("reference lap too short")
    seconds = np.interp(grid, ref.Distance, ref.seconds)
    speed = np.interp(grid, ref.Distance, ref.Speed)
    throttle = np.interp(grid, ref.Distance, ref.Throttle)
    brake = np.interp(grid, ref.Distance, ref.Brake)
    acceleration = np.gradient(speed / 3.6, seconds)
    corner_distances = pd.to_numeric(corners.get("Distance", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy()
    corner_distances = corner_distances[(corner_distances >= 0) & (corner_distances <= end)]
    if len(corner_distances) < 3:
        raise ValueError("usable corner marker distances unavailable")
    # Approximate windows, not automatically inferred geometric corner boundaries.
    delta = (grid[:, None] - corner_distances[None, :] + end / 2) % end - end / 2
    nearest = np.min(np.abs(delta), axis=1)
    after_corner = np.any((delta >= 0) & (delta <= 150), axis=1)
    kinds = np.full(len(grid), "other", dtype=object)
    corner = nearest <= 80
    kinds[corner & (speed < 150)] = "slow_corner"
    kinds[corner & (speed >= 150) & (speed < 220)] = "medium_corner"
    kinds[corner & (speed >= 220)] = "fast_corner"
    kinds[(~corner) & (throttle >= 95)] = "straight"
    kinds[after_corner & (acceleration > 0.5) & (speed < 200) & (throttle >= 50)] = "traction"
    kinds[brake >= 0.5] = "braking"
    sections = []
    start = 0
    # Each section has a contiguous distance range and one mutually exclusive type.
    for stop in range(1, len(grid)):
        if kinds[stop] != kinds[start] or stop == len(grid) - 1:
            terminal = stop
            duration = float(seconds[terminal] - seconds[start])
            length = float(grid[terminal] - grid[start])
            if length > 0 and duration > 0:
                sections.append({"start": float(grid[start]), "end": float(grid[terminal]),
                                 "kind": str(kinds[start]), "seconds": duration,
                                 "length": length, "speed": length / duration * 3.6})
            start = stop
    if not sections:
        raise ValueError("no sections")
    total = sum(s["seconds"] for s in sections)
    for section in sections:
        section["share"] = section["seconds"] / total
    return sections


def section_times(trace, sections):
    data = normalize_trace(trace)
    start, end = sections[0]["start"], sections[-1]["end"]
    if data.Distance.iloc[0] > start or data.Distance.iloc[-1] < end:
        raise ValueError("driver trace does not cover the common distance range")
    result = []
    for section in sections:
        a, b = np.interp([section["start"], section["end"]], data.Distance, data.seconds)
        if b <= a:
            raise ValueError("invalid section duration")
        result.append(float(b - a))
    return result


def compare_sections(samples, sections):
    """Within-session tyre/age/freshness/phase-matched gaps to cohort median.

    Positive gaps mean slower. At least three distinct drivers in each cohort.
    A driver receives one median contribution per section, regardless of lap count.
    """
    groups = defaultdict(list)
    for sample in samples:
        if len(sample["times"]) == len(sections):
            groups[tuple(sample["cohort"])].append(sample)
    by_driver = defaultdict(lambda: defaultdict(list))
    for group in groups.values():
        drivers = sorted({sample["driver"] for sample in group})
        if len(drivers) < 3:
            continue
        times = {d: np.median([s["times"] for s in group if s["driver"] == d], axis=0) for d in drivers}
        baseline = np.median(list(times.values()), axis=0)
        if not np.isfinite(baseline).all() or (baseline <= 0).any():
            continue
        condition = "wet" if str(group[0]["cohort"][0]) in {"INTERMEDIATE", "WET"} else "dry"
        for driver in drivers:
            gaps = 100 * (times[driver] / baseline - 1)
            for i, gap in enumerate(gaps):
                by_driver[driver][(i, condition)].append(float(gap))
    output = {}
    for driver, entries in by_driver.items():
        output[driver] = [{**sections[i], "condition": condition,
                           "gap_pct": float(np.median(gaps))}
                          for (i, condition), gaps in entries.items()]
    return output


def summarize_sections(session, clean_laps, code):
    """Bounded extraction. Errors disable section evidence, not existing pace."""
    required = {"Driver", "LapNumber", "Compound", "TyreLife", "FreshTyre", "LapStartTime", "lap_seconds"}
    if clean_laps.empty or not required.issubset(clean_laps.columns):
        return {"status": "missing usable laps/tyre metadata", "drivers": {}, "profile": []}
    try:
        corners = pd.DataFrame(session.get_circuit_info().corners)
    except Exception as exc:
        return {"status": "corner markers unavailable: " + type(exc).__name__, "drivers": {}, "profile": []}
    laps = clean_laps.copy()
    for key in ("LapNumber", "TyreLife", "lap_seconds"):
        laps[key] = pd.to_numeric(laps[key], errors="coerce")
    laps["start_s"] = pd.to_timedelta(laps.LapStartTime, errors="coerce").dt.total_seconds()
    laps["Compound"] = laps.Compound.astype("string").str.upper()
    laps = laps[laps.Compound.isin(["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"])
                & laps.FreshTyre.isin([True, False]) & (laps.TyreLife >= 1)
                & np.isfinite(laps.start_s) & np.isfinite(laps.lap_seconds)
                & np.isfinite(laps.LapNumber)].copy()
    if code in {"R", "S"}:
        laps = laps[laps.LapNumber > 1]
    laps["phase"] = ((laps.LapNumber - 1) // 5 if code in {"R", "S"} else laps.start_s // 600)
    laps["age_band"] = (laps.TyreLife - 1) // 3
    keys = ["Compound", "FreshTyre", "age_band", "phase"]
    # Do not spend telemetry work on cohorts which cannot produce a comparison.
    laps = laps.groupby(keys, dropna=False).filter(lambda x: x.Driver.nunique() >= 3)
    # Bound expensive per-lap extraction while retaining the same fixed policy.
    laps = laps.sort_values("lap_seconds").groupby("Driver", sort=False).head(MAX_LAPS_PER_DRIVER)
    if laps.empty:
        return {"status": "no three-driver matched tyre cohorts", "drivers": {}, "profile": []}
    raw = []
    failures = 0
    for record in laps.to_dict("records"):
        try:
            matches = session.laps[(session.laps.Driver == record["Driver"]) & (session.laps.LapNumber == record["LapNumber"])]
            if len(matches) != 1:
                raise ValueError("ambiguous lap")
            trace = matches.iloc[0].get_car_data().add_distance()
            trace = normalize_trace(trace)
            raw.append((record, trace))
        except Exception:
            failures += 1
    if not raw:
        return {"status": "no usable lap telemetry", "drivers": {}, "profile": [], "failed_laps": failures}
    try:
        typical_length = float(np.median([float(t.Distance.iloc[-1]) for _, t in raw]))
        raw = [(r, t) for r, t in raw if 0.97 <= float(t.Distance.iloc[-1]) / typical_length <= 1.03]
        common_start = max(float(t.Distance.iloc[0]) for _, t in raw)
        common_end = min(float(t.Distance.iloc[-1]) for _, t in raw)
        # Fastest usable matched lap is one common reference for all drivers.
        sections = make_sections(raw[0][1], corners, common_start, common_end)
        condition = "wet" if raw[0][0]["Compound"] in {"INTERMEDIATE", "WET"} else "dry"
        for section in sections:
            section["condition"] = condition
    except Exception as exc:
        return {"status": "reference unavailable: " + str(exc), "drivers": {}, "profile": []}
    samples = []
    for record, trace in raw:
        try:
            samples.append({"driver": str(record["Driver"]), "cohort": [record[k] for k in keys],
                            "times": section_times(trace, sections)})
        except ValueError:
            failures += 1
    drivers = compare_sections(samples, sections)
    return {"status": "ok" if drivers else "insufficient matching telemetry", "drivers": drivers,
            "profile": sections, "sampled_laps": len(samples), "failed_laps": failures,
            "reference_driver": str(raw[0][0]["Driver"]), "reference_lap": int(raw[0][0]["LapNumber"])}


def target_sections(inputs, family):
    """Inputs are supplied in actual session order."""
    for code in reversed(tuple(inputs)):
        data = inputs.get(code, {}).get("section_analysis", {})
        if data.get("status") == "ok" and data.get("profile"):
            return code, data["profile"]
    return None, []


def historical_section_features(driver, family, target_round, history, inputs, tracks=None, location=None):
    """Separate circuit suitability from driver form and overall session weighting.

    Match section type, representative speed and length. This is a coarse
    operating-condition match, not proof of equivalent radius/banking/gradient.
    """
    _, profile = target_sections(inputs, family)
    tracks = tracks or {}
    result = {family + "_section_coverage": 0., family + "_section_gap_pct": np.nan}
    result.update({family + "_" + k + "_section_gap_pct": np.nan for k in KINDS})
    result.update(related_circuit_form(driver, family, target_round, history, tracks, location))
    if not profile:
        return result, []
    codes = {"Q", "SQ"} if family == "qualifying" else {"R", "S"}
    prior = [r for r in history if r.get("driver") == driver and r.get("session") in codes
             and int(r["round"]) < target_round]
    evidence = []
    by_kind = defaultdict(list)
    covered = 0.
    weighted = 0.
    for index, target in enumerate(profile):
        # Very short fragments are especially sensitive to timing interpolation.
        if target["length"] < 30 or target["seconds"] < 0.3:
            continue
        observations = []
        for record in prior:
            candidates = []
            for past in record.get("sections", []):
                if past["length"] < 30 or past["seconds"] < 0.3:
                    continue
                if past["kind"] != target["kind"]:
                    continue
                if past.get("condition") != target.get("condition"):
                    continue
                speed_difference = abs(finite(past["speed"]) - finite(target["speed"]))
                ratio = finite(past["length"]) / target["length"]
                if speed_difference > 40 or not 0.5 <= ratio <= 2 or not np.isfinite(finite(past.get("gap_pct"))):
                    continue
                closeness = np.exp(-speed_difference / 40 - abs(np.log(ratio)))
                candidates.append((closeness, past["gap_pct"]))
            if candidates:
                # One contribution per historical session, not one per corner.
                total = sum(w for w, _ in candidates)
                gap = sum(w * g for w, g in candidates) / total
                match = max(w for w, _ in candidates)
                source_factor = source_match(target["kind"], tracks.get(location, {}), tracks.get(record["location"], {}))
                weight = match * source_factor * 0.85 ** (target_round - int(record["round"]))
                observations.append((weight, gap))
                evidence.append({"driver": driver, "family": family, "target_section": index,
                                 "kind": target["kind"], "target_share": target["share"],
                                 "historical_round": record["round"], "historical_session": record["session"],
                                 "historical_location": record["location"], "match_score": match,
                                 "source_factor": source_factor, "weight": weight, "gap_pct": gap})
        # Require at least two historical weekends; one event cannot establish suitability.
        rounds = {e["historical_round"] for e in evidence if e["target_section"] == index}
        if len(rounds) < 2:
            continue
        total = sum(w for w, _ in observations)
        if total <= 0:
            continue
        gap = sum(w * g for w, g in observations) / total
        share = target["share"]
        covered += share
        weighted += share * gap
        by_kind[target["kind"]].append((share, gap))
    result[family + "_section_coverage"] = covered
    # No overall score when evidence covers only a favourite piece of the lap.
    if covered >= MIN_COVERAGE:
        result[family + "_section_gap_pct"] = weighted / covered
    for kind, pairs in by_kind.items():
        result[family + "_" + kind + "_section_gap_pct"] = sum(s * g for s, g in pairs) / sum(s for s, _ in pairs)
    return result, evidence
