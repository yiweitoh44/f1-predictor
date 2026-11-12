"""Saved forecast copies and result scoring. This module never downloads data."""
from pathlib import Path
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
import hashlib
import io
import json
import os
import platform
import shutil
import tempfile

import numpy as np
import pandas as pd

REQUIRED_FILES = ('predictions.csv', 'run_info.json', 'source_manifest.json',
                  'model.json', 'influence_policy.json', 'starting_grid.csv')
FILES = REQUIRED_FILES + ('prediction_inputs.csv', 'simulation_details.csv',
                          'backtest.csv', 'backtest_comparisons.csv')
SESSIONS = {'q': 'qualifying', 'qualifying': 'qualifying',
            'sq': 'sprint-qualifying', 'sprint-qualifying': 'sprint-qualifying',
            's': 'sprint', 'sprint': 'sprint', 'r': 'race', 'race': 'race'}
AUXILIARY_FILES = {
    'sprint_calibration': ('sprint_calibration.npz',),
    'qualifying_front': ('pole_model.json', 'qualifying_front.csv'),
    'sprint_specialist': ('sprint_model.json', 'sprint_front.csv'),
    'race_specialist': ('race_score_model.json', 'race_long_model.json', 'race_front.csv')}


def active_files(info):
    return FILES + tuple(name for flag, names in AUXILIARY_FILES.items()
                         if info.get(flag, {}).get('active') for name in names)


def outcome_metrics(predicted_drivers, predicted_positions, actual_drivers, actual_positions, expected_size=None):
    """Score published positions without renumbering partial classifications.

    Winner/podium identities can be known even when full-field error cannot be
    measured. Each metric has its own explicit eligibility flag.
    """
    if (not len(predicted_drivers) or len(predicted_drivers) != len(predicted_positions)
            or len(actual_drivers) != len(actual_positions)):
        raise ValueError('Scoring needs aligned nonempty drivers and positions.')
    ranks = np.asarray(predicted_positions, dtype=float)
    if not np.isfinite(ranks).all() or set(ranks) != set(range(1, len(ranks)+1)):
        raise ValueError('Predicted order must contain every unique position.')
    prediction = dict(zip(predicted_drivers, predicted_positions))
    truth = dict(zip(actual_drivers, actual_positions))
    if len(prediction) != len(predicted_drivers) or len(truth) != len(actual_drivers):
        raise ValueError('Scoring needs unique driver identities.')
    values = np.asarray(list(truth.values()), dtype=float)
    if (not np.isfinite(values).all() or np.any(values < 1)
            or np.any(values != np.floor(values)) or len(set(values)) != len(values)):
        raise ValueError('Scoring needs unique positive published integer positions.')
    pick = min(prediction, key=lambda d: (prediction[d], str(d)))
    winner = next((d for d, p in truth.items() if p == 1), None)
    full = (set(prediction) == set(truth) and len(truth) >= 3
            and (expected_size is None or len(truth) == expected_size)
            and set(values) == set(range(1, len(truth) + 1)))
    podium = {1, 2, 3}.issubset(values)
    predicted_top = set(sorted(prediction, key=lambda d: (prediction[d], str(d)))[:3])
    return dict(pick=pick, actual=winner, correct=bool(pick == winner) if winner is not None else None,
                winner_eligible=winner is not None, top3_eligible=podium, ranking_eligible=full,
                top3_hits=len(predicted_top & {d for d, p in truth.items() if p <= 3}) if podium else None,
                mean_rank_error=float(np.mean([abs(prediction[d] - truth[d]) for d in truth])) if full else None,
                reason='' if full else 'Incomplete driver field or published positions; full-field error omitted.'
                       if winner is not None else 'Published P1 is missing; winner and full-field error omitted.')


def utc(value):
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError('Forecast timestamps need an explicit timezone.')
    return result.tz_convert('UTC')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(latest, code_root):
    """Copy completed output; later `latest` runs cannot overwrite this copy."""
    latest, code_root = Path(latest), Path(code_root)
    parent = latest.parent / 'forecasts'
    parent.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc)
    pending = Path(tempfile.mkdtemp(prefix='.pending-', dir=parent))
    destination = parent / (created.strftime('%Y%m%dT%H%M%S%fZ') + '_' + pending.name[-8:])
    try:
        info = json.loads((latest / 'run_info.json').read_text(encoding='utf-8'))
        files = active_files(info)
        for name in files:
            shutil.copyfile(latest / name, pending / name)
        versions = {'python': platform.python_version()}
        for package in ('numpy', 'pandas', 'xgboost'):
            try:
                versions[package] = version(package)
            except PackageNotFoundError:
                versions[package] = 'not installed'
        code = ('predict_anytime.py', 'forecast_validation.py', 'openf1_data.py',
                'session_influence.py', 'grid_sources.py', 'circuit_sections.py',
                'tracks.csv', 'requirements.txt')
        manifest = dict(schema=1, created_utc=created.isoformat(),
                        files={name: digest(pending / name) for name in files},
                        code_sha256={name: digest(code_root / name) for name in code},
                        runtime_versions=versions)
        (pending / 'forecast_manifest.json').write_text(
            json.dumps(manifest, indent=2), encoding='utf-8')
        os.replace(pending, destination)
    except Exception:
        shutil.rmtree(pending)
        raise
    return destination


def score(folder, actual_file, now=None):
    """Score only a verified saved copy and an explicitly supplied outcome CSV."""
    folder, actual_file = Path(folder), Path(actual_file)
    manifest_bytes = (folder / 'forecast_manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    if (not isinstance(manifest, dict) or manifest.get('schema') != 1
            or not isinstance(manifest.get('files'), dict)
            or not set(REQUIRED_FILES).issubset(manifest['files'])
            or not set(manifest['files']).issubset(set(FILES).union(*AUXILIARY_FILES.values()))):
        raise ValueError('Unsupported or incomplete forecast manifest.')
    saved = {name: (folder / name).read_bytes() for name in manifest['files']}
    for name in saved:
        if hashlib.sha256(saved[name]).hexdigest() != manifest['files'][name]:
            raise ValueError('Saved forecast has changed: ' + name)
    meta = json.loads(saved['run_info.json'])
    if not set(manifest['files']).issubset(active_files(meta)) or not set(active_files(meta)[len(FILES):]).issubset(saved):
        raise ValueError('Active specialist files are missing or inactive files were included.')
    created, start = utc(manifest['created_utc']), utc(meta['target_start_utc'])
    end = utc(meta.get('target_end_utc') or meta['target_start_utc'])
    if end < start:
        raise ValueError('Invalid target session schedule.')
    scoring_time = utc(now) if now is not None else pd.Timestamp.now(tz='UTC')
    if scoring_time < end:
        raise ValueError('Target session has not reached its scheduled end; score after completion.')
    if utc(meta['as_of']) >= start or utc(meta['as_of']) > created or created > scoring_time:
        raise ValueError('Invalid forecast creation/input cutoff.')
    predicted = pd.read_csv(io.BytesIO(saved['predictions.csv']))
    actual_bytes = actual_file.read_bytes()
    actual = pd.read_csv(io.BytesIO(actual_bytes), encoding='utf-8-sig')
    required = {'year', 'round', 'session', 'driver', 'position'}
    if not required.issubset(actual.columns) or len(actual) < 3:
        raise ValueError('Actual-results CSV needs year,round,session,driver,position and at least three drivers.')
    for key in ('year', 'round'):
        values = pd.to_numeric(actual[key], errors='coerce')
        if not values.eq(int(meta[key])).all():
            raise ValueError('Actual results belong to a different ' + key + '.')
    sessions = actual.session.astype(str).str.strip().str.lower().map(SESSIONS)
    if not sessions.eq(meta['target']).all():
        raise ValueError('Actual results belong to a different session.')
    for table, position in ((actual, 'position'), (predicted, 'predicted_position')):
        if 'driver' not in table or position not in table or len(table) < 3 or table.driver.isna().any():
            raise ValueError('Results/forecast need at least three identified drivers and positions.')
        table['driver'] = table.driver.astype(str).str.strip().str.upper()
        values = pd.to_numeric(table[position], errors='coerce')
        if (table.driver.duplicated().any() or not table.driver.str.fullmatch(r'[A-Z0-9]{2,4}').all()
                or not np.isfinite(values).all() or values.duplicated().any()
                or not values.ge(1).all() or not values.eq(np.floor(values)).all()
                or position == 'predicted_position' and set(values) != set(range(1, len(table) + 1))):
            raise ValueError('Results/forecast need unique drivers and positive integer positions; forecast order must be complete.')
        table[position] = values.map(int)
    if 1 not in set(actual.position):
        raise ValueError('Actual results must include published P1; do not renumber missing classifications.')
    missing_forecast = sorted(set(actual.driver) - set(predicted.driver))
    missing_actual = sorted(set(predicted.driver) - set(actual.driver))
    eligible = (not missing_forecast and not missing_actual
                and set(actual.position) == set(range(1, len(actual) + 1)))
    top3_eligible = {1, 2, 3}.issubset(actual.position)
    result = dict(year=int(meta['year']), round=int(meta['round']), event=meta['event'],
                  session=meta['target'], forecast_created_utc=created.isoformat(),
                  target_start_utc=start.isoformat(), scoring_utc=scoring_time.isoformat(),
                  input_cutoff_utc=utc(meta['as_of']).isoformat(),
                  used_sessions=meta.get('used_sessions', []),
                  timing='pre-session saved forecast' if created < start else 'historical replay',
                  forecast_method=meta.get('forecast_method', 'unknown'), eligible=eligible,
                  ranking_eligible=eligible,
                  top3_eligible=top3_eligible,
                  missing_forecast_drivers=missing_forecast, missing_actual_drivers=missing_actual,
                  actual_file_sha256=hashlib.sha256(actual_bytes).hexdigest(),
                  forecast_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                  scoring_code_sha256=digest(Path(__file__)))
    # Keep the actual published P1/top-three identities, even when another
    # entrant is absent. Never re-rank an intersection into a false pole hit.
    pick = str(predicted.loc[predicted.predicted_position.idxmin(), 'driver'])
    winner = str(actual.loc[actual.position.idxmin(), 'driver'])
    result.update(pick=pick, actual=winner, correct=pick == winner,
                  top3_hits=(len(set(predicted[predicted.predicted_position <= 3].driver)
                                 & set(actual[actual.position <= 3].driver)) if top3_eligible else None),
                  drivers=len(predicted), classified_actual_drivers=len(actual),
                  mean_rank_error=None)
    result.update(outcome_metrics(predicted.driver.tolist(), predicted.predicted_position.tolist(),
                                 actual.driver.tolist(), actual.position.tolist()))
    if eligible:
        joined = predicted.merge(actual[['driver', 'position']], on='driver', validate='one_to_one')
        result['mean_rank_error'] = float((joined.predicted_position - joined.position).abs().mean())
    else:
        result['reason'] = 'Driver fields or published positions are incomplete; full-field ranking error omitted.'
    # Leave the saved copy untouched. Different outcomes/revisions get distinct records.
    parent = folder.parent.parent / 'evaluations'
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / (folder.name + '_' + result['actual_file_sha256'][:16]
                            + '_' + result['scoring_code_sha256'][:12] + '.json')
    destination.write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result, destination
