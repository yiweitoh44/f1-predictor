"""Forecasts, earlier-session observations and FIA declarations, not pace bonuses.

All caches belong under the user's external F1Data directory. A pinned weather
model initialization is not a verified publication timestamp: historical runs
use an explicitly reported 12-hour availability assumption.
"""
from __future__ import annotations
import hashlib
import io
import json
import re
from pathlib import Path
from urllib.parse import urlencode, urljoin

import numpy as np
import pandas as pd
import requests
from pypdf import PdfReader
from pypdf.errors import PyPdfError
import grid_sources as grids

ARCHIVE_START = pd.Timestamp('2026-04-02T00:00Z')
RUN_DELAY_HOURS = 12
ERRORS = (ValueError, KeyError, TypeError, OSError, requests.RequestException, PyPdfError)


def finite(value, low, high):
    try:
        value = float(value)
        return value if np.isfinite(value) and low <= value <= high else None
    except (ValueError, TypeError):
        return None


def pinned_run(cutoff):
    return grids.utc(cutoff).floor('6h') - pd.Timedelta(hours=RUN_DELAY_HOURS)


def summarize_forecast(data, latitude, longitude, start, end):
    """Rain quantities describe preceding hours, including partial-hour overlap.

    Require every overlapping hour. Missing/null rain is unknown, never dry.
    Temperature is air temperature at the hour boundary, not track temperature.
    """
    if not isinstance(data, dict):
        raise ValueError('Invalid forecast response.')
    if (finite(data.get('latitude'), -90, 90) is None
            or finite(data.get('longitude'), -180, 180) is None
            or abs(data['latitude'] - latitude) > 1
            or abs(data['longitude'] - longitude) > 1
            or data.get('utc_offset_seconds') != 0):
        raise ValueError('Forecast coordinates/timezone do not match the request.')
    units = data['hourly_units']
    if not isinstance(units, dict) or not isinstance(data.get('hourly'), dict):
        raise ValueError('Invalid forecast hourly data/units.')
    if units.get('temperature_2m') != '°C' or units.get('precipitation') != 'mm':
        raise ValueError('Unexpected forecast units.')
    hourly = data['hourly']
    if not isinstance(hourly.get('time'), list):
        raise ValueError('Invalid forecast hour list.')
    times = pd.to_datetime(hourly['time'], utc=True)
    if times.hasnans or times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError('Forecast hours are duplicated or unordered.')
    required = pd.date_range(start.floor('h') + pd.Timedelta(hours=1), end.ceil('h'), freq='h')
    if not len(required) or not required.isin(times).all():
        raise ValueError('Forecast does not cover the whole assumed session window.')
    fields = {'temperature_2m': (-90, 65), 'precipitation': (0, 500)}
    if 'precipitation_probability' in hourly:
        if units.get('precipitation_probability') != '%':
            raise ValueError('Unexpected precipitation probability units.')
        fields['precipitation_probability'] = (0, 100)
    for field in fields:
        if len(hourly[field]) != len(times):
            raise ValueError('Forecast field lengths disagree.')
    rows = []
    for time in required:
        i = times.get_loc(time)
        row = {'interval_end_utc': time.isoformat()}
        row.update({key: finite(hourly[key][i], *bounds) for key, bounds in fields.items()})
        rows.append(row)
    rain = [r['precipitation'] for r in rows]
    temp = [r['temperature_2m'] for r in rows if r['temperature_2m'] is not None]
    probs = [r.get('precipitation_probability') for r in rows]
    complete = all(v is not None for v in rain)
    peak = max(rain) if complete else None
    # This is a coarse forecast description, not a prediction of tyre choice.
    state = ('rain' if peak >= .2 else 'dry' if peak == 0 else 'uncertain') if complete else 'unknown'
    return dict(hours=rows, rain_signal=state, peak_hourly_precipitation_mm=peak,
                air_temperature_range_c=[min(temp), max(temp)] if len(temp) == len(rows) else None,
                highest_hourly_rain_probability_pct=max(probs) if all(v is not None for v in probs) else None,
                probability_note='Highest hourly probability, not whole-session rain probability.',
                rain_note='Rain forecast near the circuit; does not establish a wet racing surface. Partial boundary hours are included.')


class ContextStore:
    def __init__(self, root, offline=False, refresh=False):
        self.root = Path(root) / 'weather'
        self.offline, self.refresh = offline, refresh
        self.http = grids.GridStore(root, offline, refresh)
        self.http.root = self.root

    def forecast(self, track, start, cutoff, target, live=False):
        latitude = finite(track.get('latitude'), -90, 90)
        longitude = finite(track.get('longitude'), -180, 180)
        if latitude is None or longitude is None:
            raise ValueError('Circuit forecast coordinates unavailable.')
        start, cutoff = grids.utc(start), grids.utc(cutoff)
        end = start + pd.Timedelta(hours=2 if target == 'R' else 1)
        identity = dict(latitude=latitude, longitude=longitude, start=start.isoformat(), end=end.isoformat())
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        folder = self.root / 'forecast_snapshots' / key
        saved = []
        for path in folder.glob('*.json'):
            try:
                record = json.loads(path.read_text())
                if record['identity'] == identity and grids.utc(record['observed_utc']) <= cutoff:
                    # Only genuinely captured live snapshots use observation time as their gate.
                    if record['kind'] == 'captured_live_forecast':
                        record['summary'] = summarize_forecast(record['data'], latitude, longitude, start, end)
                        saved.append(record)
            except ERRORS:
                continue
        saved.sort(key=lambda r: r['observed_utc'], reverse=True)
        if saved and (not live or self.offline or
                      (not self.refresh and cutoff-grids.utc(saved[0]['observed_utc']) < pd.Timedelta(minutes=30))):
            return saved[0]
        params = dict(latitude=latitude, longitude=longitude, timezone='UTC',
                      hourly='temperature_2m,precipitation', temperature_unit='celsius', precipitation_unit='mm')
        if live:
            base = 'https://api.open-meteo.com/v1/forecast'
            params.update(hourly=params['hourly']+',precipitation_probability', forecast_days=16)
            kind, run = 'captured_live_forecast', None
        else:
            run = pinned_run(cutoff)
            if run < ARCHIVE_START:
                raise ValueError('Pinned GFS forecast archive starts 2 April 2026; no substitute observed weather used.')
            base = 'https://single-runs-api.open-meteo.com/v1/forecast'
            params.update(models='gfs_global', run=run.strftime('%Y-%m-%dT%H:%M'), forecast_days=16)
            kind = 'archived_forecast_with_assumed_release_delay'
        url = base + '?' + urlencode(params)
        # Pinned runs are immutable; do not re-download them on every replay.
        path = self.http.root / 'http' / (hashlib.sha256(url.encode()).hexdigest()+'.json')
        original_offline = self.http.offline
        try:
            if not live and path.exists() and not self.refresh:
                self.http.offline = True
            blob, seen = self.http.get(url)
        finally:
            self.http.offline = original_offline
        if live and seen >= start:
            raise ValueError('Forecast was first observed after the target started.')
        if live and not self.offline and pd.Timestamp.now(tz='UTC') - seen > pd.Timedelta(hours=3):
            raise ValueError('Cached live forecast is over three hours old; no fresh forecast available.')
        data = json.loads(blob)
        summary = summarize_forecast(data, latitude, longitude, start, end)
        if run is not None and pd.to_datetime(data['hourly']['time'][0], utc=True) != run:
            raise ValueError('Archived forecast does not begin at the requested model initialization.')
        record = dict(identity=identity, kind=kind, source='Open-Meteo / '+('best-match models' if live else 'NOAA GFS'),
                      url=url, observed_utc=seen.isoformat(), model_initialization_utc=run.isoformat() if run is not None else None,
                      assumed_release_delay_hours=RUN_DELAY_HOURS if run is not None else None,
                      publication_time_verified=False,
                      timing=('Captured before target start; model publication time not supplied.' if live else
                              'Initialization + 12 hours <= cutoff; actual historical publication time is not verified.'),
                      summary=summary, data=data)
        if live:
            grids.save_json(folder/(seen.strftime('%Y%m%dT%H%M%S%f')+'.json'), record)
        return record

    def observations(self, year, event, current, cutoff):
        """Only sessions already admitted by the predictor; FastF1 fields win."""
        result = {}; sessions = None
        for code, data in current.items():
            weather = data.get('weather', {})
            limits = {'AirTemp': (-90, 65), 'TrackTemp': (-90, 100), 'Rainfall': (0, 1)}
            values = {k: finite(weather.get(k), *bounds) for k, bounds in limits.items()}
            if values['Rainfall'] not in (0, 1):
                values['Rainfall'] = None
            sources = {k: 'FastF1' for k, v in values.items() if v is not None}
            record = dict(values=values, field_sources=sources, warnings=[])
            result[code] = record
            if all(v is not None for v in values.values()):
                continue
            try:
                if sessions is None:
                    blob, _ = self.http.get(f'https://api.openf1.org/v1/sessions?year={year}')
                    sessions = json.loads(blob)
                if not isinstance(sessions, list) or not all(isinstance(r, dict) for r in sessions):
                    raise ValueError('Invalid OpenF1 session list.')
                names = {**grids.SESSION_NAMES, 'FP1': 'Practice 1', 'FP2': 'Practice 2', 'FP3': 'Practice 3'}
                starts = [pd.Timestamp(event[f'Session{i}DateUtc']) for i in range(1, 6)
                          if event.get(f'Session{i}') == names[code]]
                if len(starts) != 1:
                    raise ValueError('No unique scheduled observation session.')
                start = starts[0]
                start = start.tz_localize('UTC') if start.tzinfo is None else start.tz_convert('UTC')
                matches = [r for r in sessions if r.get('session_name') == names[code]
                           and grids.normalized(r.get('location')) == grids.normalized(event['Location'])
                           and abs((grids.utc(r['date_start'])-start).total_seconds()) <= 600
                           and grids.utc(r['date_end']) <= cutoff]
                if len(matches) != 1:
                    raise ValueError('No uniquely matched completed OpenF1 session.')
                session = matches[0]; key = int(session['session_key'])
                url = f'https://api.openf1.org/v1/weather?session_key={key}'
                blob, seen = self.http.get(url); samples = json.loads(blob)
                if not isinstance(samples, list) or not all(isinstance(r, dict) for r in samples):
                    raise ValueError('Invalid OpenF1 weather response.')
                for field, remote in [('AirTemp', 'air_temperature'), ('TrackTemp', 'track_temperature'), ('Rainfall', 'rainfall')]:
                    usable = [finite(r.get(remote), *limits[field]) for r in samples
                              if r.get('session_key') == key and r.get('meeting_key') == session['meeting_key']
                              and grids.utc(session['date_start']) <= grids.utc(r['date']) <= min(grids.utc(session['date_end']), cutoff)]
                    usable = [v for v in usable if v is not None and (field != 'Rainfall' or v in (0, 1))]
                    if values[field] is None and usable:
                        values[field] = max(usable) if field == 'Rainfall' else float(np.median(usable))
                        sources[field] = 'OpenF1'
                record.update(url=url, observed_utc=seen.isoformat())
            except ERRORS as exc:
                record['warnings'].append(str(exc)[:180])
        return result

    def upgrades(self, year, event, target, cutoff):
        root = 'https://www.fia.com'; page, _ = self.http.page(root+'/documents')
        seasons = [v for v, label in page.options if grids.normalized(label) == f'season{year}']
        if len(seasons) != 1:
            raise ValueError('FIA season could not be identified.')
        page, _ = self.http.page(urljoin(root, seasons[0]))
        name = grids.FIA_EVENT_NAMES.get(event['EventName'], event['EventName'])
        events = [v for v, label in page.options if grids.normalized(label) == grids.normalized(name)]
        if len(events) != 1:
            raise ValueError('FIA event could not be identified.')
        page, _ = self.http.page(urljoin(root, events[0])); candidates = []
        for doc in page.documents:
            text = ' '.join(doc['text'].split())
            if 'car presentation submissions' not in text.lower() or 'recalled' in text.lower():
                continue
            match = re.search(r'Published on (\d{2}\.\d{2}\.\d{2} \d{2}:\d{2})', text)
            if match and doc.get('url'):
                published = pd.to_datetime(match[1], format='%d.%m.%y %H:%M').tz_localize('Europe/Paris').tz_convert('UTC')
                if published <= cutoff:
                    candidates.append((published, urljoin(root, doc['url'])))
        if not candidates:
            raise ValueError('No dated FIA car-presentation submission before cutoff.')
        published, url = max(candidates)
        if not url.startswith(root+'/system/files/') or not url.endswith('.pdf'):
            raise ValueError('Unexpected FIA submission URL.')
        blob, seen = self.http.get(url)
        reader = PdfReader(io.BytesIO(blob))
        if not 1 <= len(reader.pages) <= 40:
            raise ValueError('Unexpected submission page count.')
        pages = [(p.extract_text() or '').replace('\xa0', ' ') for p in reader.pages]
        cover = ' '.join(pages[0].split())
        if (str(year) not in cover or 'carpresentationsubmissions' not in grids.normalized(cover)
                or grids.normalized(name) not in grids.normalized(cover)):
            raise ValueError('FIA submission event/year/title mismatch.')
        issued = re.search(r'Date\s+(\d{1,2} \w+ \d{4})\s+Time\s+(\d{2}:\d{2})', cover)
        if not issued:
            raise ValueError('FIA submission issue time missing.')
        local = grids.session_time(event, target, local=True)
        if local.tzinfo is None:
            raise ValueError('Local offset missing for FIA submission date.')
        issue = pd.to_datetime(' '.join(issued.groups()), format='%d %B %Y %H:%M').tz_localize(local.tzinfo).tz_convert('UTC')
        known = max(published, issue)
        if known > cutoff or any(abs((t-grids.session_time(event, target)).total_seconds()) > 7*86400
                                for t in (issue, published)):
            raise ValueError('FIA submission outside the prediction cutoff/weekend.')
        return dict(source='FIA car presentation submissions', url=url, published_utc=published.isoformat(),
                    document_issued_utc=issue.isoformat(), known_utc=known.isoformat(), observed_utc=seen.isoformat(),
                    declarations=[dict(page=i+1, text=text.strip()) for i, text in enumerate(pages[1:], 1) if text.strip()],
                    interpretation='Team-declared changes and purposes, not independently measured gains. Missing text does not mean no upgrade.',
                    measured_performance_gain=None, score_effect=0)


def build_context(root, year, event, target, start, cutoff, track, current,
                  offline=False, refresh=False, live=False, enabled=True, air_temp=None, wet=None):
    context = dict(enabled=enabled, score_effect='none; forecasts and declarations are context only',
                   scenario={'air_temperature_c': air_temp, 'wet': wet}, forecast=None, upgrades=None,
                   observations={}, warnings=[], observation_note='Completed-session observations can contain later published revisions; rain flag means some rain, not a wholly wet session.')
    if air_temp is not None and finite(air_temp, -90, 65) is None:
        raise ValueError('--air-temp must be a finite Celsius value between -90 and 65.')
    if not enabled:
        return context
    store = ContextStore(root, offline, refresh)
    context['observations'] = store.observations(year, event, current, cutoff)
    for key, call in [('forecast', lambda: store.forecast(track, start, cutoff, target, live)),
                      ('upgrades', lambda: store.upgrades(year, event, target, cutoff))]:
        try:
            context[key] = call()
            if key == 'forecast':
                context[key].pop('data', None)  # Full provider response stays in the external cache.
        except ERRORS as exc:
            context['warnings'].append(key+': '+str(exc)[:180])
    context['warnings'].extend(store.http.warnings)
    return context
