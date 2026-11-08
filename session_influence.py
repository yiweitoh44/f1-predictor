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
QUALIFYING_PRACTICE_SCORE_COLUMNS = [
    f'qual_{code}_{field}' for code in ('FP1', 'FP2', 'FP3')
    for field in ('score', 'gap_norm')]
QUALIFYING_PRACTICE_CONTEXT_COLUMNS = [
    f'qual_{code}_{field}' for code in ('FP1', 'FP2', 'FP3')
    for field in ('available', 'support', 'peers')] + ['sprint_target']
QUALIFYING_PRACTICE_EXTRA_COLUMNS = (QUALIFYING_PRACTICE_SCORE_COLUMNS
                                     + QUALIFYING_PRACTICE_CONTEXT_COLUMNS)
QUALIFYING_PRACTICE_MODEL_COLUMNS = MODEL_COLUMNS + QUALIFYING_PRACTICE_EXTRA_COLUMNS
QUALIFYING_PRACTICE_MAX_DEPTH = 3


# SQ gets one practice session, so retain compound and format-specific evidence
# separately instead of relying on the single pooled practice/history anchor.
SPRINT_QUALIFYING_HISTORY_COLUMNS = [
    'sq_history_score', 'sq_team_history_score', 'sq_peak_score',
    'sq_gp_peak_score', 'sq_history_available', 'sq_history_weekends']
SPRINT_QUALIFYING_PACE_COLUMNS = ['sq_fp1_score', 'sq_fp1_gap',
    'sq_fp1_available', 'sq_fp1_support', 'sq_fp1_peers'] + [
    f'sq_{compound}_{field}'
    for compound in ('soft', 'medium', 'hard', 'intermediate', 'wet')
    for field in ('score', 'gap', 'available', 'support', 'peers')]
SPRINT_QUALIFYING_MODEL_COLUMNS = (MODEL_COLUMNS + SPRINT_QUALIFYING_HISTORY_COLUMNS
    + ['sprint_target'] + SPRINT_QUALIFYING_PACE_COLUMNS)
SPRINT_QUALIFYING_LEARNED_SHARE = 1.0
SPRINT_QUALIFYING_TRAINING_RATIO = 4.0
SPRINT_QUALIFYING_MAX_DEPTH = 3

# The Sprint specialist only assigns the existing front-eight score values.
# It does not change the base race model, score spread or raw lap admission.
SPRINT_BASE_COLUMNS = ['sp_' + c for c in RACE_MODEL_COLUMNS]
SPRINT_HISTORY_COLUMNS = [f'sp_{code}_{field}' for code in ('s', 'r')
    for field in ('score', 'team_score', 'peak_score', 'available', 'weekends')]
SPRINT_PACE_COLUMNS = [f'sp_{label}_{field}' for label in ('short', 'long')
    for field in ('score', 'gap', 'available', 'support', 'peers')]
SPRINT_CONTEXT_COLUMNS = ['sprint_target', 'sp_grid_delta', 'sp_qual_score',
                          'sp_qual_gap', 'sp_qual_available', 'sp_trend']
SPRINT_MODEL_COLUMNS = (SPRINT_BASE_COLUMNS + SPRINT_HISTORY_COLUMNS
                       + SPRINT_PACE_COLUMNS + SPRINT_CONTEXT_COLUMNS)
SPRINT_FRONT_SIZE = 8
SPRINT_LEARNED_SHARE = .5


def sprint_specialist_mode(target, inputs):
    return target == 'S' and bool(inputs.get('FP1')) and bool(inputs.get('SQ'))


def _sprint_form(history, target_round, code, who, value, window=3, peak=False):
    prior = [r for r in history if r['session'] == code
             and int(r['round']) < target_round
             and np.isfinite(numeric(r.get('position_norm')))]
    keep = set(sorted({int(r['round']) for r in prior})[-window:])
    weekends = {}
    for record in prior:
        if record.get(who) == value and int(record['round']) in keep:
            weekends.setdefault(int(record['round']), []).append(numeric(record['position_norm']))
    means = {rnd: float(np.mean(values)) for rnd, values in weekends.items()}
    if peak:
        return min(means.values(), default=np.nan)
    return weighted_available([(v, 2 ** (-(target_round-rnd)/HISTORY_HALF_LIFE))
                               for rnd, v in means.items()])


def sprint_specialist_context(driver, team, target_round, history, inputs):
    """Separate earlier Sprint/GP form and existing FP1 comparisons, not telemetry."""
    out = {}
    for code in ('S', 'R'):
        prefix = 'sp_' + code.lower()
        value = _sprint_form(history, target_round, code, 'driver', driver)
        out.update({prefix + '_score': value,
            prefix + '_team_score': _sprint_form(history, target_round, code, 'team', team),
            prefix + '_peak_score': _sprint_form(history, target_round, code, 'driver', driver, peak=True),
            prefix + '_available': float(np.isfinite(value)),
            prefix + '_weekends': len({int(r['round']) for r in history
                if r['session'] == code and int(r['round']) < target_round
                and r['driver'] == driver and np.isfinite(numeric(r.get('position_norm')))})})
    out['sp_s_recent2_score'] = _sprint_form(history, target_round, 'S', 'driver', driver, window=2)
    data = inputs.get('FP1') or {}
    for label in ('short', 'long'):
        records = matched_pace_evidence(driver, data, (label,))
        prefix = 'sp_' + label
        out.update({prefix + '_score': weighted_available([(r['score'], 1.) for r in records]),
            prefix + '_gap': weighted_available([(r['gap_pct']/PACE_MAX_GAP_PCT, 1.) for r in records]),
            prefix + '_available': float(bool(records)),
            prefix + '_support': float(sum(r['samples'] for r in records)),
            prefix + '_peers': float(np.mean([r['peers'] for r in records])) if records else 0.})
    participant = next((d for d in data.get('drivers', []) if d['driver'] == driver), {})
    out['sp_trend'] = numeric(participant.get('long_trend_pct_per_lap'))
    return out


def sprint_specialist_inputs(row, target):
    """Recompute after grid application; base influence columns are untouched."""
    values = {c: numeric(row.get(c)) for c in RACE_MODEL_COLUMNS}
    if target == 'S':
        history = numeric(row.get('sp_s_recent2_score'))
        if not np.isfinite(history):
            history = values['influence_history']
        sources = dict(history=history, grid=numeric(row.get('start_grid_norm')),
            qualifying=numeric(row.get('qualifying_grid_norm')),
            practice=numeric(row.get('practice_evidence_score')))
        values['influence_anchor'] = weighted_available([
            (sources['history'], .20), (sources['grid'], .375),
            (sources['qualifying'], .375), (sources['practice'], .05)])
        for name in SOURCE_NAMES:
            value = sources.get(name, np.nan)
            values['influence_' + name] = value if np.isfinite(value) else history
        qualifying = sources['qualifying']
        values['influence_grid_qualifying'] = qualifying if np.isfinite(qualifying) else history
    q = 'SQ' if target == 'S' else 'Q'
    qualifying = numeric(row.get(q + '_position_norm'))
    return {'sp_' + c: values[c] for c in RACE_MODEL_COLUMNS} | {
        'sp_grid_delta': numeric(row.get('start_grid_norm')) - numeric(row.get('qualifying_grid_norm')),
        'sp_qual_score': qualifying, 'sp_qual_gap': numeric(row.get(q + '_gap_pct'))/PACE_MAX_GAP_PCT,
        'sp_qual_available': float(np.isfinite(qualifying))}


def reorder_sprint_front(scores, specialist, drivers):
    return _reorder_front(scores, specialist, drivers, SPRINT_FRONT_SIZE)


def sprint_specialist_policy(active=False, trained=False):
    return dict(active=bool(active), trained=bool(active and trained),
        scope='Sprint after loaded FP1 and SQ; other session modes and stages retain the base model',
        front_size=SPRINT_FRONT_SIZE, outside_candidates=False,
        score_rule='specialist reorders only the base top eight; same score multiset and base sigma',
        model_features=list(SPRINT_MODEL_COLUMNS) if active else [],
        learned_share=SPRINT_LEARNED_SHARE if active and trained else 0.,
        history='auxiliary Sprint anchor uses driver mean over the last two earlier Sprint weekends, decayed by calendar rounds; falls back to the original GP/Sprint history when unavailable',
        separate_history='driver/team mean and driver best from last three earlier Sprint and GP weekends; no current-weekend history',
        practice='existing FP1 short/long comparisons; minimum two matched laps per compound and four comparison drivers; unknown remains missing',
        training='pooled earlier GP/Sprint regression, depth three, original per-weekend totals and six-round training half-life',
        early_fallback='no learned model before four usable earlier weekends; auxiliary evidence anchor may reorder the base top eight',
        auxiliary_anchor='20% history, 37.5% starting grid, 37.5% SQ position, 5% admitted FP1 pace; redistribute absent evidence',
        model_file='sprint_model.json' if active else None,
        audit_file='sprint_front.csv' if active else None,
        limits='six inspected historical Sprint tests and one complete ranking; fuel, traffic, strategy and incident causes remain unmodelled; simulation shares are uncalibrated')


def sprint_qualifying_mode(target, inputs):
    return target == 'SQ' and bool(inputs.get('FP1'))


def sprint_qualifying_context(driver, team, target_round, history, inputs):
    """Only earlier Q/SQ history and the SQ-specific FP1 comparison sidecar.

    All validated tyre ages are compared within the usual age/time bands. An
    absent sidecar is unknown; never silently substitute the low-age summary.
    """
    data = inputs.get('FP1') or {}
    comparison = {'drivers': [dict(driver=d['driver'], **{
        key.removeprefix('sq_'): value for key, value in d.items()
        if key.startswith('sq_short_')}) for d in data.get('drivers', [])]}
    records = matched_pace_evidence(driver, comparison, ('short',))
    out = {}
    def describe(rows, prefix):
        out[prefix + '_score'] = weighted_available([(r['score'], 1.) for r in rows])
        out[prefix + '_gap'] = weighted_available([(r['gap_pct'] / PACE_MAX_GAP_PCT, 1.) for r in rows])
        out[prefix + '_available'] = float(bool(rows))
        out[prefix + '_support'] = float(sum(r['samples'] for r in rows))
        out[prefix + '_peers'] = float(np.mean([r['peers'] for r in rows])) if rows else 0.
    describe(records, 'sq_fp1')
    for compound in ('soft', 'medium', 'hard', 'intermediate', 'wet'):
        describe([r for r in records if r['compound'] == compound], 'sq_' + compound)
    prior = [r for r in history if r['session'] in {'Q', 'SQ'}
             and int(r['round']) < target_round and np.isfinite(numeric(r.get('position_norm')))]
    sq_rounds = set(sorted({int(r['round']) for r in prior if r['session'] == 'SQ'})[-3:])
    def form(who, value, code='SQ', peak=False):
        rows = [r for r in prior if r.get(who) == value and r['session'] == code
                and (int(r['round']) in sq_rounds if code == 'SQ'
                     else target_round - HISTORY_WEEKENDS <= int(r['round']))]
        weekends = {}
        for r in rows:
            weekends.setdefault(int(r['round']), []).append(numeric(r['position_norm']))
        means = {rnd: float(np.mean(values)) for rnd, values in weekends.items()}
        if peak:
            return min(means.values(), default=np.nan)
        return weighted_available([(v, 2 ** (-(target_round-rnd)/HISTORY_HALF_LIFE))
                                   for rnd, v in means.items()])
    out['sq_history_score'] = form('driver', driver)
    out['sq_team_history_score'] = form('team', team)
    out['sq_peak_score'] = form('driver', driver, peak=True)
    out['sq_gp_peak_score'] = form('driver', driver, code='Q', peak=True)
    out['sq_history_available'] = float(np.isfinite(out['sq_history_score']))
    out['sq_history_weekends'] = len({int(r['round']) for r in prior
                                     if r['driver'] == driver and r['session'] == 'SQ'})
    return out


def sprint_qualifying_policy(active=False, trained=False):
    return dict(active=bool(active), trained=bool(active and trained),
        model_columns=list(SPRINT_QUALIFYING_MODEL_COLUMNS) if active and trained else [],
        learned_share=SPRINT_QUALIFYING_LEARNED_SHARE if active and trained else None,
        maximum_tree_depth=SPRINT_QUALIFYING_MAX_DEPTH if active and trained else None,
        practice='FP1 only; all validated tyre ages, matched within compound, freshness proxy, three-lap age bands and ten-minute windows',
        minimum_matched_laps_per_compound=2, minimum_comparison_drivers=4,
        history='current season only; separate driver/team mean and driver best over the last three earlier SQ weekends; driver best GP Q over four earlier calendar rounds',
        training='pooled earlier Q/SQ; SQ rows receive four times the within-weekend weight of Q rows, then each weekend total is restored',
        missing='native missing handling for separate scores/gaps; unavailable is not slow or zero',
        early_fallback='existing 85% history / 15% admitted FP1 evidence anchor, redistributed when missing; no fitted model before four usable history weekends',
        limits='FP1 programmes, fuel and engine settings remain unknown; learned scores and simulation shares are uncalibrated')


# Auxiliary GP pole model. Order is fixed to reproduce the validated native
# model: six explicit scores, per-session inputs, compound inputs, history, flag.
POLE_BASE_COLUMNS = ['pole_' + c for c in MODEL_COLUMNS]
POLE_FP_COLUMNS = [f'pole_{code}_{field}' for code in ('FP1', 'FP2', 'FP3')
                   for field in ('score', 'gap_norm', 'available', 'support', 'peers')]
POLE_COMPOUND_COLUMNS = [f'pole_{compound}_{field}'
    for compound in ('soft', 'medium', 'hard', 'intermediate', 'wet')
    for field in ('score', 'gap_norm', 'available', 'support', 'peers')]
POLE_HISTORY_COLUMNS = [f'pole_{code}_{field}' for code in ('q', 'sq')
    for field in ('score', 'team_score', 'peak_score', 'available', 'weekends', 'team_available')]
POLE_MODEL_COLUMNS = (POLE_BASE_COLUMNS + POLE_FP_COLUMNS + POLE_COMPOUND_COLUMNS
                      + POLE_HISTORY_COLUMNS + ['sprint_target'])
POLE_LEARNED_SHARE = .5
POLE_FRONT_SIZE = 5


def qualifying_front_mode(target, inputs):
    return target == 'Q' and bool(inputs.get('FP3')) and not any(inputs.get(c) for c in ('SQ', 'S'))


def qualifying_pole_context(driver, team, target, target_round, history, inputs, row):
    """Own one-sample evidence is auxiliary; the base model keeps two samples.

    All observations still obey the original raw quality/matching filters. No
    current qualifying result enters these fields. Missing values stay unknown.
    """
    out = {}
    admitted, practice, support = [], [], 0.
    for weight, code in enumerate(('FP1', 'FP2', 'FP3'), 1):
        data = inputs.get(code)
        records = matched_pace_evidence(driver, data, ('short',), minimum=1) if data else []
        admitted.extend((r, weight) for r in records)
        score = weighted_available([(r['score'], 1.) for r in records])
        practice.append((score, weight))
        support += sum(r['samples'] for r in records)
        out.update(_pole_describe([(r, 1.) for r in records], 'pole_' + code))
    for compound in ('soft', 'medium', 'hard', 'intermediate', 'wet'):
        out.update(_pole_describe([(r, w) for r, w in admitted if r['compound'] == compound],
                                  'pole_' + compound))
    prior = [r for r in history if r['session'] in {'Q', 'SQ'}
             and int(r['round']) < target_round and np.isfinite(numeric(r.get('position_norm')))]
    for code, label in (('Q', 'q'), ('SQ', 'sq')):
        keep = set(sorted({int(r['round']) for r in prior if r['session'] == code})[-3:])
        records = [r for r in prior if r['session'] == code and int(r['round']) in keep]
        def form(who, value, peak=False):
            weekends = {}
            for r in records:
                if r.get(who) == value:
                    weekends.setdefault(int(r['round']), []).append(numeric(r['position_norm']))
            means = {rnd: float(np.mean(values)) for rnd, values in weekends.items()}
            return (min(means.values(), default=np.nan) if peak else weighted_available([
                (v, 2 ** (-(target_round-rnd)/HISTORY_HALF_LIFE)) for rnd, v in means.items()]))
        driver_form, team_form = form('driver', driver), form('team', team)
        out.update({f'pole_{label}_score': driver_form, f'pole_{label}_team_score': team_form,
                    f'pole_{label}_peak_score': form('driver', driver, peak=True),
                    f'pole_{label}_available': float(np.isfinite(driver_form)),
                    f'pole_{label}_team_available': float(np.isfinite(team_form)),
                    f'pole_{label}_weekends': len({int(r['round']) for r in prior
                        if r['session'] == code and r['driver'] == driver})})
    out.update({'pole_' + c: row[c] for c in MODEL_COLUMNS})
    competitive = any(np.isfinite(numeric(row.get(c))) for c in ('sprint_evidence_score', 'SQ_position_norm'))
    if target == 'Q' and not competitive:
        value = weighted_available(practice)
        share = .5 * min(1., support / 8.) if np.isfinite(value) else 0.
        historical = row['influence_history']
        out.update({'pole_influence_' + name: value if name == 'practice' and np.isfinite(value)
                    else historical for name in SOURCE_NAMES})
        out['pole_influence_anchor'] = ((1-share)*historical
            + share*(value if np.isfinite(value) else historical))
    return out


def _pole_describe(records, prefix):
    return {prefix + '_score': weighted_available([(r['score'], w) for r, w in records]),
            prefix + '_gap_norm': weighted_available([(r['gap_pct']/PACE_MAX_GAP_PCT, w) for r, w in records]),
            prefix + '_available': float(bool(records)),
            prefix + '_support': float(sum(r['samples'] for r, _ in records)),
            prefix + '_peers': weighted_available([(r['peers'], w) for r, w in records]) if records else 0.}


def reorder_qualifying_front(scores, specialist, drivers):
    """Permute the base top five's existing values; keep the score distribution."""
    return _reorder_front(scores, specialist, drivers, POLE_FRONT_SIZE)


def _reorder_front(scores, specialist, drivers, size):
    scores, specialist, drivers = np.asarray(scores, float), np.asarray(specialist, float), np.asarray(drivers)
    if not (len(scores) == len(specialist) == len(drivers)) or len(set(drivers)) != len(drivers):
        raise ValueError('Front correction requires unique, matching driver fields.')
    if not (np.isfinite(scores).all() and np.isfinite(specialist).all()):
        raise ValueError('Front correction requires finite component scores.')
    if np.ptp(specialist) < 1e-9:
        return scores.copy()
    base = np.lexsort((drivers, scores))
    members = base[:min(size, len(base))]
    ordered = members[np.lexsort((drivers[members], specialist[members]))]
    result = scores.copy()
    result[ordered] = np.sort(scores[members])
    return result


def qualifying_front_policy(active=False, trained=False):
    return dict(active=bool(active), trained=bool(active and trained),
        scope='GP qualifying after loaded FP3 without SQ/Sprint anchors; other stages retain the base model',
        front_size=POLE_FRONT_SIZE, outside_candidates=False,
        score_rule='reorder only the base top five using the pole specialist; retain the same score multiset and base sigma',
        model_features=list(POLE_MODEL_COLUMNS) if active else [],
        model_file='pole_model.json' if active else None, audit_file='qualifying_front.csv' if active else None,
        learned_share=POLE_LEARNED_SHARE if active and trained else 0.,
        training='strictly earlier current-season Q/SQ; native binary original-P1 target; class balance restored within each unchanged weekend budget',
        base_minimum_matched_laps=2, specialist_minimum_matched_laps=1, minimum_comparison_drivers=4,
        limitations='one-lap evidence remains programme-dependent; simulations and classifier scores are uncalibrated')


def qualifying_practice_mode(target, inputs):
    """Use separate FP inputs for GP Q's practice-only stage, after practice."""
    return (target == 'Q' and not any(inputs.get(c) for c in ('SQ', 'S'))
            and any(inputs.get(c) for c in ('FP1', 'FP2', 'FP3')))


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


def matched_pace_evidence(driver, data, labels, *, minimum=2, minimum_drivers=4):
    """Rank usable matched comparisons; a compound needs four usable drivers.

    These remain programme-dependent observations. Sparse or extreme gaps
    cannot become a performance penalty just because a cache contains them.
    """
    evidence = []
    for label in labels:
        for compound in ('soft', 'medium', 'hard', 'intermediate', 'wet'):
            key = f'{label}_{compound}_gap_pct'
            peers = [d for d in data.get('drivers', [])
                     if numeric(d.get(f'{label}_{compound}_matched_laps', d.get(label + '_matched_laps'))) >= minimum
                     and pace_reason(d.get(key)) is None]
            me = next((d for d in peers if d['driver'] == driver), None)
            if me is None or len(peers) < minimum_drivers:
                continue
            gaps = np.array([numeric(d[key]) for d in peers])
            gap = numeric(me[key])
            # Tied observations share the mean of their occupied ranks.
            rank = (np.sum(gaps < gap) + (np.sum(gaps == gap)-1)/2) / (len(peers)-1)
            count = numeric(me.get(f'{label}_{compound}_matched_laps', me.get(label + '_matched_laps')))
            evidence.append(dict(score=float(rank), gap_pct=float(gap), compound=compound,
                                 label=label, peers=len(peers), samples=count))
    return evidence


def pace_score(driver, data, labels):
    return weighted_available([(r['score'], 1.) for r in matched_pace_evidence(driver, data, labels)])


def qualifying_practice_context(driver, inputs):
    """Describe existing comparisons, including the selected qualifying inputs.

    Missing scores/gaps stay missing. Raw matched counts can be one while the
    eligible flag stays false: production pace still needs two per compound and
    four comparison drivers. No fuel correction or cross-compound lap estimate.
    """
    out = {}
    available_weight = 0.
    used_weight = 0.
    for weight, code in enumerate(('FP1', 'FP2', 'FP3'), 1):
        data = inputs.get(code)
        records = matched_pace_evidence(driver, data, ('short',)) if data else []
        participant = next((d for d in data.get('drivers', []) if d['driver'] == driver), {}) if data else {}
        eligible = {r['compound'] for r in records}
        if data:
            available_weight += weight
        if records:
            used_weight += weight
        out[f'qual_{code}_score'] = weighted_available([(r['score'], 1.) for r in records])
        out[f'qual_{code}_matched_gap_pct'] = weighted_available([(r['gap_pct'], 1.) for r in records])
        out[f'qual_{code}_gap_norm'] = out[f'qual_{code}_matched_gap_pct'] / PACE_MAX_GAP_PCT
        out[f'qual_{code}_peers'] = float(np.mean([r['peers'] for r in records])) if records else 0.
        out[f'qual_{code}_available'] = float(bool(records))
        out[f'qual_{code}_support'] = float(sum(r['samples'] for r in records))
        for compound in ('soft', 'medium', 'hard', 'intermediate', 'wet'):
            value = participant.get(f'short_{compound}_matched_laps', participant.get('short_matched_laps', 0))
            out[f'qual_{code}_{compound}_matched_laps'] = numeric(value) if participant else np.nan
            out[f'qual_{code}_{compound}_eligible'] = float(compound in eligible)
    out['qual_practice_coverage'] = used_weight / available_weight if available_weight else 0.
    return out


def qualifying_diagnostic_policy(active=False):
    return dict(model_inputs=bool(active),
                active_model_columns=list(QUALIFYING_PRACTICE_EXTRA_COLUMNS) if active else [],
                maximum_tree_depth=QUALIFYING_PRACTICE_MAX_DEPTH if active else None,
                missing_model_scores='native missing-value handling; no history imputation for per-session scores/gaps',
                minimum_matched_laps_per_compound=2,
                minimum_comparison_drivers=4, missing_scores='unknown, not zero or an estimated lap time',
                matched_laps='per-compound counts; unknown when the session or driver is absent',
                score='normalized rank within admitted comparisons; lower is better',
                matched_gap_pct='mean of existing eligible compound comparison gaps, not an estimated qualifying lap',
                support='matched samples in admitted comparisons; not total laps driven',
                gap_norm='matched_gap_pct divided by the 5% comparable-gap limit; not a lap-time estimate',
                peers='mean admitted comparison-field size across compounds, including this driver',
                constraints='worse speed scores/gaps cannot improve the native prediction; support/missingness context has no forced speed direction',
                coverage='share of available FP1:FP2:FP3 weights 1:2:3 with an admitted driver score; diagnostic only',
                limits='Programme/fuel effects remain unobserved. Selected fields enter the practice-only GP qualifying model; matching, support thresholds and explicit scoring budgets are unchanged.')


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


def policy(*, sprint_qualifying=False, trained=True):
    return dict(history_weekends=HISTORY_WEEKENDS, history_half_life_rounds=HISTORY_HALF_LIFE,
                learned_adjustment_share=(SPRINT_QUALIFYING_LEARNED_SHARE if sprint_qualifying else LEARNED_SHARE) if trained else 0.,
                default_learned_adjustment_share=LEARNED_SHARE, source_weights=SOURCE_WEIGHTS,
                sprint_qualifying=sprint_qualifying_policy(active=sprint_qualifying, trained=trained),
                race_model_columns=RACE_MODEL_COLUMNS,
                qualifying_practice_model_columns=QUALIFYING_PRACTICE_MODEL_COLUMNS,
                qualifying_practice_scope='GP qualifying after practice, without completed SQ/Sprint evidence; other stages retain their original model',
                qualifying_practice_max_depth=QUALIFYING_PRACTICE_MAX_DEPTH,
                grid_qualifying_share=GRID_QUALIFYING_SHARE,
                grid_budget='Race/Sprint split the existing grid budget equally between starting grid and Q/SQ position when both are available; otherwise retain the existing anchor',
                grid_weight_diagnostics='influence_weight_grid is the combined budget; influence_weight_starting_grid and influence_weight_grid_qualifying are its applied components',
                comparable_pace_gap_range_pct=[0, PACE_MAX_GAP_PCT],
                maximum_clean_lap_ratio=PACE_MAX_LAP_SPREAD,
                pace_minimum_driver_samples=2, pace_minimum_field_drivers=4,
                missing_source='redistribute anchor weight across available sources; learned input uses history',
                historical_session_weight='one budget per weekend; GP/Q 3, Sprint/SQ 1',
                limits='Source budgets describe the explicit anchor, not fitted feature importance. SQ after FP1 uses a dedicated learned model once enough history exists. Practice remains affected by fuel/traffic/programmes. Simulation shares are uncalibrated.')
