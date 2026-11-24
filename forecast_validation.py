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
    'driver_explanations': ('driver_evidence.json',),
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


def probability_metrics(simulation, predicted, actual):
    """Score saved simulation shares against already validated outcome rows.

    The winner score is the sum of squared categorical errors (range 0..2).
    Top-three membership uses a sum of binary squared errors (range 0..6).
    Missing tail classifications do not prevent either score when P1/P2/P3
    identities are known. A missing forecast entrant gets zero probability;
    never renormalize over the intersection with the published results.
    """
    if simulation is None or 'first_pct' not in simulation:
        return dict(available=False, reason='Saved first-place simulation shares unavailable.')
    if 'driver' not in simulation:
        raise ValueError('Saved probability table needs driver identities.')
    simulation = simulation.copy()
    simulation['driver'] = simulation.driver.astype(str).str.strip().str.upper()
    if simulation.driver.duplicated().any() or set(simulation.driver) != set(predicted.driver):
        raise ValueError('Saved probability drivers must match the complete forecast roster.')
    simulation = simulation.set_index('driver').loc[predicted.driver]
    first = pd.to_numeric(simulation.first_pct, errors='coerce').to_numpy(float) / 100.
    if (not np.isfinite(first).all() or np.any(first < 0) or np.any(first > 1)
            or not np.isclose(first.sum(), 1., rtol=0., atol=1e-8)):
        raise ValueError('Saved first-place shares must be finite, within 0..100 and sum to 100.')
    n = len(first)
    winners = actual.loc[actual.position.eq(1), 'driver'].tolist()
    result = dict(available=True, basis='saved Monte Carlo shares; not calibrated real-world probabilities',
                  forecast_drivers=n, winner_eligible=bool(winners), top3_eligible=False)
    if winners:
        winner = winners[0]
        target = predicted.driver.eq(winner).to_numpy(float)
        outside = float(winner not in set(predicted.driver))
        chance = float(first @ target)
        result.update(actual_winner_share=chance,
                      winner_brier_sum=float(np.sum((first-target)**2) + outside),
                      uniform_winner_brier_sum=float(np.sum((np.full(n, 1./n)-target)**2) + outside),
                      winner_log_loss=float(-np.log(chance)) if chance > 0 else None,
                      winner_log_loss_infinite=chance == 0,
                      actual_winner_missing_from_forecast=bool(outside))
    if 'top3_pct' in simulation:
        top = pd.to_numeric(simulation.top3_pct, errors='coerce').to_numpy(float) / 100.
        if (not np.isfinite(top).all() or np.any(top < 0) or np.any(top > 1)
                or np.any(top + 1e-8 < first)
                or not np.isclose(top.sum(), 3., rtol=0., atol=1e-8)):
            raise ValueError('Saved top-three shares must be valid marginals and sum to 300.')
        if {1, 2, 3}.issubset(actual.position):
            podium = set(actual.loc[actual.position.le(3), 'driver'])
            target = predicted.driver.isin(podium).to_numpy(float)
            outside = len(podium-set(predicted.driver))
            result.update(top3_eligible=True,
                          top3_brier_sum=float(np.sum((top-target)**2) + outside),
                          uniform_top3_brier_sum=float(np.sum((np.full(n, 3./n)-target)**2) + outside),
                          podium_drivers_missing_from_forecast=outside)
    return result


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


def read_saved_forecast(folder):
    """Read and verify immutable artifacts without scoring or downloading data."""
    folder = Path(folder)
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
    return saved, manifest, manifest_bytes


def score(folder, actual_file, now=None):
    """Score only a verified saved copy and an explicitly supplied outcome CSV."""
    folder, actual_file = Path(folder), Path(actual_file)
    saved, manifest, manifest_bytes = read_saved_forecast(folder)
    meta = json.loads(saved['run_info.json'])
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
    simulation = (pd.read_csv(io.BytesIO(saved['simulation_details.csv']))
                  if 'simulation_details.csv' in saved else None)
    result['probability_scores'] = probability_metrics(simulation, predicted, actual)
    # Leave the saved copy untouched. Different outcomes/revisions get distinct records.
    parent = folder.parent.parent / 'evaluations'
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / (folder.name + '_' + result['actual_file_sha256'][:16]
                            + '_' + result['scoring_code_sha256'][:12] + '.json')
    destination.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    return result, destination
