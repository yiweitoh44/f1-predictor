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
# Manual starting assumptions for comparing TRACKS, not shares of the final
# prediction. Review against the same historical cases before changing them.
GROUP_WEIGHTS = dict(zip(GROUP_COLUMNS, (0.25, 0.30, 0.15, 0.15, 0.15)))


def circuit_group_match(target, past):
    """Weighted overlap of manual tags; blank categories supply no evidence."""
    matches = {}
    for key, weight in GROUP_WEIGHTS.items():
        a = {v.strip() for v in str(target.get(key) or "").split("|") if v.strip()}
        b = {v.strip() for v in str(past.get(key) or "").split("|") if v.strip()}
        matches[key] = weight * len(a & b) / len(a | b) if a and b else 0.0
    return float(sum(matches.values())), matches


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


def related_circuit_form(driver, family, target_round, history, tracks, location, team=None):
    """Pre-weekend fallback: related-circuit RESULTS, not inferred section speed."""
    target = tracks.get(location, {})
    codes = {"Q", "SQ"} if family == "qualifying" else {"R", "S"}
    # Each weekend has one vote: Sprint + GP and two teammates must not
    # accidentally make a circuit count as several independent weekends.
    weekends = defaultdict(list)
    team_weekends = defaultdict(list)
    if team is None:
        team = next((r.get("team") for r in reversed(history)
                     if r.get("driver") == driver and int(r["round"]) < target_round), None)
    for prior in history:
        if prior.get("session") not in codes or int(prior["round"]) >= target_round:
            continue
        past = tracks.get(prior["location"], {})
        similarity, _ = circuit_group_match(target, past)
        y = finite(prior.get("position_norm"))
        if similarity > 0 and np.isfinite(y):
            weight = similarity * 0.85 ** (target_round - int(prior["round"]))
            if prior.get("driver") == driver:
                weekends[int(prior["round"])].append((weight, y))
            if team and prior.get("team") == team:
                team_weekends[int(prior["round"])].append((weight, y))
    def average(groups):
        if len(groups) < 2:
            return np.nan
        rows = [(values[0][0], float(np.mean([y for _, y in values])))
                for values in groups.values()]
        return sum(w * y for w, y in rows) / sum(w for w, _ in rows)
    result = {family + "_related_circuit_form": average(weekends),
              family + "_related_circuit_weekends": float(len(weekends)),
              family + "_team_related_circuit_form": average(team_weekends),
              family + "_team_related_circuit_weekends": float(len(team_weekends))}
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
    if not np.isfinite(data[["Distance", "seconds", "Speed", "Throttle", "Brake"]].to_numpy(dtype=float)).all():
        raise ValueError("non-finite car telemetry")
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


def normalize_positions(frame):
    """Require position updates, not just frequently repeated timestamps.

    A moving lap with stale XY cannot locate sub-second sections. Apply the
    same two-second maximum gap used for car telemetry to actual XY updates,
    including a stale tail. This is a quality screen, not alignment proof.
    """
    positions = pd.DataFrame(frame).copy()
    if not {"Time", "X", "Y"}.issubset(positions):
        raise ValueError("position channels unavailable")
    positions["seconds"] = pd.to_timedelta(positions.Time).dt.total_seconds()
    for key in ("X", "Y"):
        positions[key] = pd.to_numeric(positions[key], errors="coerce")
    positions = positions.dropna(subset=["seconds", "X", "Y"])
    if not np.isfinite(positions[["seconds", "X", "Y"]].to_numpy(dtype=float)).all():
        raise ValueError("non-finite positions")
    if (len(positions) < 30 or (np.diff(positions.seconds) <= 0).any()
            or np.diff(positions.seconds).max() > 2):
        raise ValueError("insufficient or discontinuous positions")
    xy = positions[["X", "Y"]].to_numpy()
    changed = np.r_[True, np.any(np.diff(xy, axis=0) != 0, axis=1)]
    update_times = positions.seconds.to_numpy()[changed]
    gaps = np.diff(np.r_[update_times, positions.seconds.iloc[-1]])
    maximum_gap = float(gaps.max())
    if maximum_gap > 2:
        raise ValueError(f"stale XY: {maximum_gap:.3f}s without a position update (maximum 2s)")
    if int(changed.sum()) < 30:
        raise ValueError("fewer than 30 distinct position updates")
    return positions


def reference_corners(corners, lap, reference, positions=None):
    """Locate marker XY coordinates on the exact selected reference lap.

    FastF1's supplied Distance values use its session-fastest lap, which need
    not be our clean reference. Map marker times onto our car-distance trace
    instead. This does not align different drivers' integrated distances.
    """
    if positions is None:
        positions = normalize_positions(lap.get_pos_data())
    markers = pd.DataFrame(corners).copy()
    if not {"X", "Y"}.issubset(markers):
        raise ValueError("corner marker coordinates unavailable")
    for key in ("X", "Y"):
        markers[key] = pd.to_numeric(markers[key], errors="coerce")
    markers = markers.dropna(subset=["X", "Y"])
    if not np.isfinite(markers[["X", "Y"]].to_numpy(dtype=float)).all():
        raise ValueError("non-finite corner marker coordinates")
    if len(markers) < 3:
        raise ValueError("usable corner marker coordinates unavailable")
    xy = positions[["X", "Y"]].to_numpy()
    times = []
    for marker in markers[["X", "Y"]].to_numpy():
        nearest = int(np.argmin(np.sum((xy - marker) ** 2, axis=1)))
        times.append(float(positions.seconds.iloc[nearest]))
    if min(times) < reference.seconds.iloc[0] or max(times) > reference.seconds.iloc[-1]:
        raise ValueError("corner position falls outside reference car telemetry")
    markers["Distance"] = np.interp(times, reference.seconds, reference.Distance)
    return markers


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


def compare_sections(samples, sections, diagnostics=None):
    """Within-session tyre/age/freshness/phase-matched gaps to cohort median.

    Positive gaps mean slower. At least three distinct drivers in each cohort.
    A driver receives one median contribution per section, regardless of lap count.
    """
    groups = defaultdict(list)
    for sample in samples:
        if len(sample["times"]) == len(sections):
            groups[tuple(sample["cohort"])].append(sample)
    by_driver = defaultdict(lambda: defaultdict(list))
    for cohort, group in groups.items():
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
                if diagnostics is not None:
                    selected = [s for s in group if s["driver"] == driver]
                    diagnostics.append({
                        "driver": driver, "section": i, "kind": sections[i]["kind"],
                        "start": sections[i]["start"], "end": sections[i]["end"],
                        "compound": cohort[0], "fresh": bool(cohort[1]),
                        "age_band": float(cohort[2]), "phase": float(cohort[3]),
                        "condition": condition, "peer_drivers": len(drivers),
                        "drivers": drivers,
                        "laps": [s.get("lap") for s in selected],
                        "section_times": [float(s["times"][i]) for s in selected],
                        "driver_median_seconds": float(times[driver][i]),
                        "cohort_median_seconds": float(baseline[i]),
                        "gap_pct": float(gap),
                    })
    output = {}
    for driver, entries in by_driver.items():
        output[driver] = [{**sections[i], "section": i, "condition": condition,
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
    return _summarize_selected_sections(session, laps, corners)


def _summarize_selected_sections(session, laps, corners):
    """Extract an already selected sample; also supports replaying exports."""
    keys = ["Compound", "FreshTyre", "age_band", "phase"]
    raw = []
    failures = 0
    rejections = []
    for record in laps.to_dict("records"):
        try:
            matches = session.laps[(session.laps.Driver == record["Driver"]) & (session.laps.LapNumber == record["LapNumber"])]
            if len(matches) != 1:
                raise ValueError("ambiguous lap")
            lap = matches.iloc[0]
            trace = lap.get_car_data().add_distance()
            trace = normalize_trace(trace)
            positions = normalize_positions(lap.get_pos_data())
            raw.append((record, trace, lap, positions))
        except Exception as exc:
            failures += 1
            rejections.append({"driver": str(record["Driver"]),
                               "lap": int(record["LapNumber"]),
                               "reason": type(exc).__name__ + ": " + str(exc)})
    if not raw:
        return {"status": "car/position telemetry unavailable", "drivers": {}, "profile": [],
                "failed_laps": failures, "rejections": rejections}
    try:
        typical_length = float(np.median([float(t.Distance.iloc[-1]) for _, t, _, _ in raw]))
        retained = []
        for record, trace, lap, positions in raw:
            if 0.97 <= float(trace.Distance.iloc[-1]) / typical_length <= 1.03:
                retained.append((record, trace, lap, positions))
            else:
                failures += 1
                rejections.append({"driver": str(record["Driver"]),
                                   "lap": int(record["LapNumber"]),
                                   "reason": "lap distance differs by more than 3% from sample median"})
        raw = retained
        if not raw:
            raise ValueError("no traces within the common lap-length range")
        cohorts = defaultdict(set)
        for record, _, _, _ in raw:
            cohorts[tuple(record[k] for k in keys)].add(str(record["Driver"]))
        if not any(len(drivers) >= 3 for drivers in cohorts.values()):
            return {"status": "no three-driver cohorts with usable car/position telemetry",
                    "drivers": {}, "profile": [], "sampled_laps": len(raw),
                    "failed_laps": failures, "rejections": rejections}
        common_start = max(float(t.Distance.iloc[0]) for _, t, _, _ in raw)
        common_end = min(float(t.Distance.iloc[-1]) for _, t, _, _ in raw)
        # Fastest usable matched lap is one common reference for all drivers.
        corners = reference_corners(corners, raw[0][2], raw[0][1], positions=raw[0][3])
        sections = make_sections(raw[0][1], corners, common_start, common_end)
        condition = "wet" if raw[0][0]["Compound"] in {"INTERMEDIATE", "WET"} else "dry"
        for section in sections:
            section["condition"] = condition
    except Exception as exc:
        return {"status": "reference unavailable: " + str(exc), "drivers": {}, "profile": [],
                "failed_laps": failures, "rejections": rejections}
    samples = []
    for record, trace, _, _ in raw:
        try:
            samples.append({"driver": str(record["Driver"]), "cohort": [record[k] for k in keys],
                            "lap": int(record["LapNumber"]),
                            "times": section_times(trace, sections)})
        except ValueError as exc:
            failures += 1
            rejections.append({"driver": str(record["Driver"]),
                               "lap": int(record["LapNumber"]), "reason": str(exc)})
    cohorts = []
    drivers = compare_sections(samples, sections, diagnostics=cohorts)
    return {"status": "ok" if drivers else "insufficient matching telemetry", "drivers": drivers,
            "profile": sections, "sampled_laps": len(samples), "failed_laps": failures,
            "cohorts": cohorts, "rejections": rejections,
            "marker_reference": "selected clean lap position samples",
            "physical_alignment_validated": False,
            "reference_driver": str(raw[0][0]["Driver"]), "reference_lap": int(raw[0][0]["LapNumber"])}


def target_sections(inputs, family):
    """Inputs are supplied in actual session order."""
    for code in reversed(tuple(inputs)):
        data = inputs.get(code, {}).get("section_analysis", {})
        if data.get("status") == "ok" and data.get("profile"):
            return code, data["profile"]
    return None, []


def historical_section_features(driver, family, target_round, history, inputs, tracks=None, location=None, team=None):
    """Separate circuit suitability from driver form and overall session weighting.

    Match section type, representative speed and length. This is a coarse
    operating-condition match, not proof of equivalent radius/banking/gradient.
    """
    _, profile = target_sections(inputs, family)
    tracks = tracks or {}
    result = {family + "_section_coverage": 0., family + "_section_gap_pct": np.nan}
    result.update({family + "_" + k + "_section_gap_pct": np.nan for k in KINDS})
    result.update(related_circuit_form(driver, family, target_round, history, tracks, location, team))
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
        if (not all(np.isfinite(finite(target.get(k))) for k in ("length", "seconds", "speed", "share"))
                or target["length"] < 30 or target["seconds"] < 0.3 or target["speed"] <= 0
                or not 0 < target["share"] <= 1):
            continue
        section_evidence = []
        for record in prior:
            candidates = []
            for past_index, past in enumerate(record.get("sections", [])):
                if (not all(np.isfinite(finite(past.get(k))) for k in ("length", "seconds", "speed", "gap_pct"))
                        or past["length"] < 30 or past["seconds"] < 0.3 or past["speed"] <= 0):
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
                candidates.append((closeness, past["gap_pct"], past.get("section", past_index)))
            if candidates:
                # One contribution per historical session, not one per corner.
                total = sum(w for w, _, _ in candidates)
                gap = sum(w * g for w, g, _ in candidates) / total
                match = max(w for w, _, _ in candidates)
                source_factor = source_match(target["kind"], tracks.get(location, {}), tracks.get(record["location"], {}))
                weight = match * source_factor * 0.85 ** (target_round - int(record["round"]))
                section_evidence.append({"driver": driver, "family": family, "target_section": index,
                                 "kind": target["kind"], "target_share": target["share"],
                                 "condition": target.get("condition"),
                                 "historical_round": record["round"], "historical_session": record["session"],
                                 "historical_location": record["location"], "match_score": match,
                                 "source_factor": source_factor, "weight": weight, "gap_pct": gap,
                                 "session_weight": weight,
                                 "candidate_sections": [i for _, _, i in candidates],
                                 "candidate_gap_min_pct": min(g for _, g, _ in candidates),
                                 "candidate_gap_max_pct": max(g for _, g, _ in candidates)})
        # A Sprint and GP in one weekend share one budget of evidence. Preserve
        # session rows for inspection, but distribute the weekend's best match
        # weight across them instead of treating each as another weekend.
        by_round = defaultdict(list)
        for entry in section_evidence:
            by_round[entry["historical_round"]].append(entry)
        for entries in by_round.values():
            weekend_weight = max(e["session_weight"] for e in entries)
            total = sum(e["session_weight"] for e in entries)
            for entry in entries:
                entry["weekend_weight"] = weekend_weight
                entry["weight"] = weekend_weight * entry["session_weight"] / total
                entry["accepted"] = len(by_round) >= 2
        evidence.extend(section_evidence)
        # Require at least two historical weekends; one event cannot establish suitability.
        if len(by_round) < 2:
            continue
        total = sum(e["weight"] for e in section_evidence)
        if total <= 0:
            continue
        gap = sum(e["weight"] * e["gap_pct"] for e in section_evidence) / total
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
