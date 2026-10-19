"""Explicit current-session and historical influence for v2.

Scores are normalised positions: smaller is better. These weights are model
assumptions, not calibrated probabilities or claims about physical car pace.
"""
import numpy as np

HISTORY_WEEKENDS = 4
HISTORY_HALF_LIFE = 3.0
LEARNED_SHARE = 0.15
PACE_MAX_GAP_PCT = 5.0
PACE_MAX_LAP_SPREAD = 1.07
SOURCE_NAMES = ('history', 'grid', 'sprint', 'qualifying', 'practice')
SOURCE_WEIGHTS = {
    'R': dict(history=.20, grid=.65, sprint=.10, qualifying=.025, practice=.025),
    'S': dict(history=.20, grid=.75, sprint=0., qualifying=0., practice=.05),
    'Q': dict(history=.30, grid=0., sprint=.10, qualifying=.50, practice=.10),
    'SQ': dict(history=.85, grid=0., sprint=0., qualifying=0., practice=.15),
}
MODEL_COLUMNS = ['influence_' + k for k in SOURCE_NAMES] + ['influence_anchor']
RACE_MODEL_COLUMNS = MODEL_COLUMNS + ['influence_grid_qualifying']
GRID_QUALIFYING_SHARE = 0.5


def numeric(value):
    try:
        x = float(value)
        return x if np.isfinite(x) else np.nan
    except (TypeError, ValueError):
        return np.nan


def weighted_available(items):
    pairs = [(numeric(v), w) for v, w in items if np.isfinite(numeric(v)) and w > 0]
    return float(np.average([v for v, _ in pairs], weights=[w for _, w in pairs])) if pairs else np.nan


def historical_form(records, target_round, driver=None, team=None):
    """One budget per weekend, bounded window, same current team for team form.

    GP/Q has three times the vote of Sprint/SQ inside a historical weekend.
    The across-weekend decay uses round distance, not the number of sessions.
    """
    sessions = {}
    for r in records:
        rnd = int(r['round'])
        if not target_round - HISTORY_WEEKENDS <= rnd < target_round:
            continue
        if driver is not None and r.get('driver') != driver:
            continue
        if team is not None and r.get('team') != team:
            continue
        y = numeric(r.get('position_norm'))
        if np.isfinite(y):
            sessions.setdefault((rnd, r['session']), []).append(y)
    weekends = {}
    for (rnd, code), values in sessions.items():
        weekends.setdefault(rnd, []).append((float(np.mean(values)), 1. if code in {'Q', 'R'} else 1/3))
    values = [(weighted_available(rows), 2 ** (-(target_round-rnd)/HISTORY_HALF_LIFE))
              for rnd, rows in weekends.items()]
    return weighted_available(values), len(weekends)


def pace_reason(value):
    value = numeric(value)
    if not np.isfinite(value):
        return 'missing'
    if value < 0 or value > PACE_MAX_GAP_PCT:
        return 'outside comparable-pace range (0..5%); omitted, not clipped'
    return None


def pace_score(driver, data, labels):
    """Rank usable matched comparisons; a compound needs four usable drivers.

    These remain programme-dependent observations. Sparse or extreme gaps
    cannot become a performance penalty just because a cache contains them.
    """
    scores = []
    for label in labels:
        for compound in ('soft', 'medium', 'hard', 'intermediate', 'wet'):
            key = f'{label}_{compound}_gap_pct'
            peers = [d for d in data.get('drivers', [])
                     if numeric(d.get(f'{label}_{compound}_matched_laps', d.get(label + '_matched_laps'))) >= 2
                     and pace_reason(d.get(key)) is None]
            me = next((d for d in peers if d['driver'] == driver), None)
            if me is None or len(peers) < 4:
                continue
            gaps = np.array([numeric(d[key]) for d in peers])
            gap = numeric(me[key])
            # Tied observations share the mean of their occupied ranks.
            rank = (np.sum(gaps < gap) + (np.sum(gaps == gap)-1)/2) / (len(peers)-1)
            scores.append((float(rank), 1.))
    return weighted_available(scores)


def add_influence(row, target):
    family = 'race' if target in {'R', 'S'} else 'qualifying'
    history = weighted_available([
        (row.get(family + '_driver_form'), .65),
        (row.get(family + '_team_form'), .25),
        (row.get(family + '_related_circuit_form'), .10),
    ])
    if not np.isfinite(history):
        history = .5  # Explicit neutral fallback for a new driver/team.
    values = dict(history=history, grid=numeric(row.get('start_grid_norm')),
                  sprint=numeric(row.get('sprint_evidence_score')),
                  qualifying=numeric(row.get('SQ_position_norm')),
                  practice=numeric(row.get('practice_evidence_score')))
    weights = {k: (SOURCE_WEIGHTS[target][k] if np.isfinite(values[k]) else 0.) for k in SOURCE_NAMES}
    total = sum(weights.values())
    weights = {k: w/total for k, w in weights.items()}
    anchor = sum(values[k]*w for k,w in weights.items() if w > 0)
    # Missing sources use the historical fallback in the learned adjustment;
    # they cannot provide an accidentally favourable missing-value split.
    out = {'influence_' + k: values[k] if np.isfinite(values[k]) else history for k in SOURCE_NAMES}
    out['influence_anchor'] = float(anchor)
    out.update({'influence_weight_' + k: weights[k] for k in SOURCE_NAMES})
    # Preserve qualifying performance when a penalty changes the starting grid.
    # Split the existing grid budget; do not add a second full evidence budget.
    qualifying = numeric(row.get('qualifying_grid_norm'))
    if np.isfinite(qualifying) and not 0 <= qualifying <= 1:
        qualifying = np.nan
    out['influence_grid_qualifying'] = qualifying if np.isfinite(qualifying) else history
    out['influence_weight_starting_grid'] = weights['grid']
    out['influence_weight_grid_qualifying'] = 0.
    if target in {'R', 'S'} and np.isfinite(qualifying) and np.isfinite(values['grid']):
        qualifying_weight = weights['grid'] * GRID_QUALIFYING_SHARE
        out['influence_anchor'] += qualifying_weight * (qualifying - values['grid'])
        out['influence_weight_starting_grid'] -= qualifying_weight
        out['influence_weight_grid_qualifying'] = qualifying_weight
    return out


def policy():
    return dict(history_weekends=HISTORY_WEEKENDS, history_half_life_rounds=HISTORY_HALF_LIFE,
                learned_adjustment_share=LEARNED_SHARE, source_weights=SOURCE_WEIGHTS,
                race_model_columns=RACE_MODEL_COLUMNS,
                grid_qualifying_share=GRID_QUALIFYING_SHARE,
                grid_budget='Race/Sprint split the existing grid budget equally between starting grid and Q/SQ position when both are available; otherwise retain the existing anchor',
                grid_weight_diagnostics='influence_weight_grid is the combined budget; influence_weight_starting_grid and influence_weight_grid_qualifying are its applied components',
                comparable_pace_gap_range_pct=[0, PACE_MAX_GAP_PCT],
                maximum_clean_lap_ratio=PACE_MAX_LAP_SPREAD,
                pace_minimum_driver_samples=2, pace_minimum_field_drivers=4,
                missing_source='redistribute anchor weight across available sources; learned input uses history',
                historical_session_weight='one budget per weekend; GP/Q 3, Sprint/SQ 1',
                limits='Manual starting policy. Practice remains affected by fuel/traffic/programmes. Simulation shares are uncalibrated.')
