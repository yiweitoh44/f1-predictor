"""Published grids with source identity, time gates and explicit pit-lane status.

Only grid documents are read: never the target race's results or running order.
Raw responses and immutable grid snapshots live in the external data cache.
"""
from __future__ import annotations
import hashlib
import io
import json
import os
import re
import tempfile
import unicodedata
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import numpy as np
import pandas as pd
import requests
from pypdf.errors import PyPdfError

PRIORITY = {'fia': 0, 'formula1': 1, 'openf1': 2}
SESSION_NAMES = {'R': 'Race', 'S': 'Sprint', 'Q': 'Qualifying', 'SQ': 'Sprint Qualifying'}
FIA_EVENT_NAMES = {'Barcelona Grand Prix': 'Barcelona-Catalunya Grand Prix'}
F1_SLUGS = dict(zip(
    ['Melbourne','Shanghai','Suzuka','Miami Gardens','Montréal','Monte Carlo','Barcelona',
     'Spielberg','Silverstone','Spa-Francorchamps','Budapest','Zandvoort','Monza','Madrid',
     'Baku','Kuala Lumpur','Marina Bay','Austin','Mexico City','São Paulo','Las Vegas','Lusail','Yas Marina'],
    ['australia','china','japan','miami','canada','monaco','barcelona-catalunya','austria',
     'great-britain','belgium','hungary','netherlands','italy','spain','azerbaijan','bahrain',
     'singapore','united-states','mexico','brazil','las-vegas','qatar','abu-dhabi']))


def normalized(value):
    return re.sub(r'[^a-z0-9]', '', unicodedata.normalize('NFKD', str(value)).encode('ascii', 'ignore').decode().lower())


def save_json(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,delete=False) as file:
        json.dump(value,file,sort_keys=True);temporary=file.name
    os.replace(temporary,path)


def utc(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError('Grid timestamps require an explicit UTC offset.')
    return stamp.tz_convert('UTC')


def session_time(event, code, local=False):
    for i in range(1, 6):
        if event.get(f'Session{i}') == SESSION_NAMES[code]:
            value = pd.Timestamp(event.get(f'Session{i}Date' + ('' if local else 'Utc')))
            if pd.isna(value):
                break
            return value if local else (value.tz_localize('UTC') if value.tzinfo is None else value.tz_convert('UTC'))
    raise ValueError(f'No dated {code} session for grid identity.')


class Page(HTMLParser):
    def __init__(self, text):
        super().__init__(); self.links=[]; self.options=[]; self.documents=[]; self.rows=[]; self.notes=[]
        self.option=None; self.doc=None; self.row=None; self.cell=None; self.para=None
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs=dict(attrs)
        if tag == 'a':
            self.links.append(attrs.get('href',''))
            if self.doc is not None: self.doc['url']=attrs.get('href','')
        if tag == 'option': self.option=[attrs.get('value',''),[]]
        if tag == 'li' and 'document-row' in attrs.get('class',''): self.doc={'text':[]}
        if tag == 'tr': self.row=[]
        if tag in {'td','th'} and self.row is not None: self.cell=[]
        if tag == 'p': self.para=[]

    def handle_data(self, data):
        if self.option is not None: self.option[1].append(data)
        if self.doc is not None: self.doc['text'].append(data)
        if self.cell is not None: self.cell.append(data)
        if self.para is not None: self.para.append(data)

    def handle_endtag(self, tag):
        if tag == 'option' and self.option is not None:
            self.options.append((self.option[0], ' '.join(self.option[1]).strip())); self.option=None
        if tag == 'li' and self.doc is not None:
            self.documents.append({**self.doc,'text':' '.join(self.doc['text'])}); self.doc=None
        if tag in {'td','th'} and self.cell is not None:
            self.row.append(' '.join(' '.join(self.cell).split())); self.cell=None
        if tag == 'tr' and self.row is not None:
            self.rows.append(self.row); self.row=None
        if tag == 'p' and self.para is not None:
            text=' '.join(' '.join(self.para).split())
            if text.startswith('Note'): self.notes.append(text)
            self.para=None


def identify_name(text, roster):
    compact=normalized(text)
    full=[d['driver'] for d in roster if normalized(d.get('name', d['driver'])) in compact]
    if len(full)==1: return full[0]
    # Published tables sometimes shorten given names. Only accept unique surnames.
    last=[d['driver'] for d in roster if len(d.get('name','').split())>1
          and normalized(d['name'].split()[-1]) in compact]
    if len(last)!=1: raise ValueError(f'Grid driver name is missing or ambiguous: {text[:70]}')
    return last[0]


def validate_rows(rows, roster):
    names={d['driver'] for d in roster}; found=[r['driver'] for r in rows]
    if len(found)!=len(set(found)) or set(found)!=names:
        raise ValueError('Published grid does not contain every roster driver exactly once.')
    positions=[]
    for row in rows:
        if row.get('pit_lane') is not None and not isinstance(row['pit_lane'],bool):
            raise ValueError('Invalid pit-lane status.')
        if row.get('pit_lane') is True:
            if row.get('position') is not None: raise ValueError('Pit-lane starter must not have a grid slot.')
            continue
        if isinstance(row['position'],bool): raise ValueError('Invalid grid position.')
        position=float(row['position'])
        if not np.isfinite(position) or position!=int(position) or not 1<=position<=len(names):
            raise ValueError('Invalid grid position.')
        row['position']=int(position); positions.append(int(position))
    if len(set(positions))!=len(positions): raise ValueError('Duplicate grid slots.')
    return rows


def parse_fia_pdf(blob, roster, event, target, published):
    from pypdf import PdfReader
    reader=PdfReader(io.BytesIO(blob))
    if len(reader.pages)>10: raise ValueError('Grid document has an unexpected page count.')
    text='\n'.join(page.extract_text() or '' for page in reader.pages).replace('\xa0',' ')
    flat=' '.join(text.split())
    title='Sprint Starting Grid' if target=='S' else 'Starting Grid'
    if title.lower() not in flat.lower() or (target=='R' and 'sprint starting grid' in flat.lower()):
        raise ValueError('FIA document is for a different session.')
    expected_names={event['EventName'],FIA_EVENT_NAMES.get(event['EventName'],event['EventName'])}
    if not any(normalized(name) in normalized(flat) for name in expected_names):
        raise ValueError('FIA grid event name does not match.')
    if str(session_time(event,target).year) not in flat:
        raise ValueError('FIA grid season does not match.')
    issued=re.search(r'Date\s+(\d{1,2} \w+ \d{4})\s+Time\s+(\d{2}:\d{2})',text)
    known=utc(published)
    if issued:
        local=session_time(event,target,local=True)
        if local.tzinfo is None: raise ValueError('Local event offset unavailable for FIA document time.')
        issue=pd.to_datetime(' '.join(issued.groups()),format='%d %B %Y %H:%M').tz_localize(local.tzinfo).tz_convert('UTC')
        if abs((known-issue).total_seconds())>7200:
            raise ValueError('FIA publication and document times disagree by over two hours.')
        known=max(known,issue)
    if abs((known-session_time(event,target)).total_seconds())>4*86400:
        raise ValueError('FIA document date does not match the weekend.')
    normal, _, pit=text.partition('DRIVERS REQUIRED TO START FROM THE PIT LANE')
    rows=[]
    for match in re.finditer(r'(?m)^\s*(\d{1,2})\s+(\d{1,3})\s+([A-Za-z][^\n]+)',normal):
        position,number,name=match.groups()
        rows.append(dict(driver=identify_name(name,roster),number=int(number),position=int(position),pit_lane=False))
    if pit:
        pit=re.split(r'\bNOTES\b|\* PENALTIES|The Stewards',pit)[0]
        for match in re.finditer(r'(?m)^\s*(\d{1,3})\s+([A-Za-z][^\n]+)',pit):
            number,name=match.groups()
            rows.append(dict(driver=identify_name(name,roster),number=int(number),position=None,pit_lane=True))
    return validate_rows(rows,roster),known


def parse_f1(text, roster):
    page=Page(text); rows=[]
    for cells in page.rows:
        if len(cells)!=5 or not cells[0].isdigit(): continue
        code=cells[2].split()[-1]
        rows.append(dict(driver=code,number=int(cells[1]),position=int(cells[0]),pit_lane=False))
    # Only explicit start-from-pit-lane statements change status. Unknown syntax
    # rejects this source rather than inventing a grid slot for a pit-lane start.
    for note in page.notes:
        for sentence in re.split(r'(?<=[.!?])\s+',note):
            if 'pit lane' not in sentence.lower(): continue
            match=re.search(r'(.+?)\s+(?:required to|will) start from (?:the )?pit lane',sentence,re.I)
            if not match: raise ValueError('Unrecognised pit-lane note in official grid.')
            prefix=match.group(1).removeprefix('Note - ')
            for name in re.split(r',|\band\b',prefix):
                driver=identify_name(name.strip(),roster)
                row=next((r for r in rows if r['driver']==driver),None)
                if row is None: raise ValueError('Pit-lane note driver absent from grid.')
                row.update(position=None,pit_lane=True)
    return validate_rows(rows,roster)


def manual_grid(path, roster, cutoff):
    frame=pd.read_csv(path,encoding='utf-8-sig')
    if not {'driver','grid'}.issubset(frame): raise ValueError('Grid CSV requires driver,grid columns.')
    rows=[]
    for data in frame.to_dict('records'):
        value=str(data['grid']).strip().upper()
        flag=str(data.get('pit_lane','')).strip().lower()
        pit=value in {'PIT','PL','PIT LANE'} or flag in {'true','1','yes'}
        if flag not in {'','nan','true','false','1','0','yes','no'}: raise ValueError('Invalid pit_lane value in grid CSV.')
        if pit and value not in {'PIT','PL','PIT LANE','','NAN','0','0.0'}:
            raise ValueError('Pit-lane rows need grid=PIT, empty or zero.')
        if pd.notna(data.get('known_utc')) and 'known_utc' in data:
            if utc(data['known_utc'])>cutoff: raise ValueError('Manual grid was published after the prediction cutoff.')
        rows.append(dict(driver=str(data['driver']).strip().upper(),position=None if pit else float(value),pit_lane=pit))
    return dict(source='manual',url=str(path),rows=validate_rows(rows,roster),
                known_utc=None,timing='user-supplied pre-session grid; timestamp checked when supplied')


class GridStore:
    def __init__(self, root, offline=False, refresh=False):
        self.root=Path(root)/'grids'; self.offline=offline; self.refresh=refresh; self.warnings=[]

    def get(self,url):
        path=self.root/'http'/(hashlib.sha256(url.encode()).hexdigest()+'.json')
        now=pd.Timestamp.now(tz='UTC'); saved=None
        if path.exists():
            try:
                saved=json.loads(path.read_text())
                if not isinstance(saved,dict) or saved.get('url')!=url: saved=None
                if saved:
                    utc(saved['fetched_utc']);bytes.fromhex(saved['hex'])
            except (ValueError,KeyError,TypeError,OSError): saved=None
        if saved and (self.offline or (not self.refresh and now-utc(saved['fetched_utc'])<pd.Timedelta(minutes=5))):
            return bytes.fromhex(saved['hex']),utc(saved['fetched_utc'])
        if self.offline: raise ValueError('grid source is not cached')
        try:
            response=requests.get(url,timeout=(5,15)); response.raise_for_status()
            requested,received=urlsplit(url),urlsplit(response.url)
            fia_catalog=(requested.hostname==received.hostname=='www.fia.com'
                         and requested.path.startswith('/documents') and received.path.startswith('/documents'))
            if response.url!=requests.Request('GET',url).prepare().url and not fia_catalog:
                raise ValueError('Grid source redirected to a different resource.')
        except requests.RequestException:
            if saved:
                self.warnings.append('Grid download failed; checking cached response: '+url)
                return bytes.fromhex(saved['hex']),utc(saved['fetched_utc'])
            raise
        if len(response.content)>10_000_000: raise ValueError('Grid response unexpectedly large.')
        now=pd.Timestamp.now(tz='UTC')
        saved=dict(url=url,fetched_utc=now.isoformat(),hex=response.content.hex())
        save_json(path,saved)
        return response.content,now

    def page(self,url):
        data,seen=self.get(url);return Page(data.decode('utf-8')),seen

    def fia(self,year,event,target,roster,cutoff):
        root='https://www.fia.com'; page,_=self.page(root+'/documents')
        seasons=[value for value,label in page.options if normalized(label)==f'season{year}']
        if len(seasons)!=1: raise ValueError('FIA season could not be identified.')
        page,_=self.page(urljoin(root,seasons[0]))
        event_name=FIA_EVENT_NAMES.get(event['EventName'],event['EventName'])
        events=[value for value,label in page.options if normalized(label)==normalized(event_name)]
        if len(events)!=1: raise ValueError('FIA event could not be identified.')
        page,_=self.page(urljoin(root,events[0])); records=[]
        for doc in page.documents:
            text=' '.join(doc['text'].split()); lower=text.lower()
            if 'starting grid' not in lower or ('sprint' in lower)!=(target=='S'): continue
            if 'recalled' in lower or not doc.get('url'): continue
            match=re.search(r'Published on (\d{2}\.\d{2}\.\d{2} \d{2}:\d{2})',text)
            if not match: continue
            # FIA labels its Paris wall clock "CET" throughout the year.
            published=pd.to_datetime(match[1],format='%d.%m.%y %H:%M').tz_localize('Europe/Paris').tz_convert('UTC')
            if published>cutoff: continue
            url=urljoin(root,doc['url']);data,seen=self.get(url)
            try:
                rows,known=parse_fia_pdf(data,roster,event,target,published)
                records.append(dict(source='fia',url=url,rows=rows,known_utc=known.isoformat(),observed_utc=seen.isoformat(),
                                    timing='dated FIA publication, cross-checked against document issue time',title=text.split('Published on')[0]))
            except (ValueError,KeyError,PyPdfError) as exc: self.warnings.append('FIA grid rejected: '+str(exc))
        if not records: raise ValueError('no usable dated FIA grid before cutoff')
        return records

    def formula1(self,year,event,target,roster):
        page,_=self.page(f'https://www.formula1.com/en/results/{year}/races')
        slug=F1_SLUGS.get(event.get('Location'))
        paths=set(l for l in page.links if re.fullmatch(fr'/en/results/{year}/races/\d+/{re.escape(slug or "UNKNOWN")}/race-result',l))
        if len(paths)!=1: raise ValueError('F1 event could not be identified unambiguously.')
        path=paths.pop().rsplit('/',1)[0] + ('/sprint-grid' if target=='S' else '/starting-grid')
        url='https://www.formula1.com'+path;data,seen=self.get(url)
        return [dict(source='formula1',url=url,rows=parse_f1(data.decode('utf-8'),roster),
                     known_utc=seen.isoformat(),observed_utc=seen.isoformat(),timing='first observed; page has no publication timestamp')]

    def openf1(self,year,event,target,roster):
        q='SQ' if target=='S' else 'Q';start=session_time(event,q)
        url=f'https://api.openf1.org/v1/sessions?year={year}'
        blob,_=self.get(url);sessions=json.loads(blob)
        if not isinstance(sessions,list) or not all(isinstance(r,dict) for r in sessions):
            raise ValueError('Invalid OpenF1 session list.')
        matches=[r for r in sessions if r.get('session_name')==SESSION_NAMES[q]
                 and abs((utc(r['date_start'])-start).total_seconds())<=600
                 and normalized(r.get('location'))==normalized(event['Location'])]
        if len(matches)!=1: raise ValueError('OpenF1 qualifying session could not be identified.')
        key=matches[0]['session_key'];meeting=matches[0]['meeting_key']
        drivers,_=self.get(f'https://api.openf1.org/v1/drivers?session_key={key}')
        drivers=json.loads(drivers)
        if not isinstance(drivers,list) or not all(isinstance(r,dict) for r in drivers):
            raise ValueError('Invalid OpenF1 driver list.')
        mapping={d['driver_number']:d['name_acronym'] for d in drivers
                 if d['session_key']==key and d['meeting_key']==meeting}
        url=f'https://api.openf1.org/v1/starting_grid?session_key={key}'
        blob,seen=self.get(url);data=json.loads(blob);rows=[]
        if not isinstance(data,list) or not all(isinstance(r,dict) for r in data):
            raise ValueError('Invalid OpenF1 starting grid.')
        for d in data:
            if d['session_key']!=key or d['meeting_key']!=meeting: raise ValueError('OpenF1 grid identity mismatch.')
            rows.append(dict(driver=mapping[d['driver_number']],number=d['driver_number'],position=d['position'],pit_lane=None))
        return [dict(source='openf1',url=url,rows=validate_rows(rows,roster),known_utc=seen.isoformat(),
                     observed_utc=seen.isoformat(),timing='first observed; API has no publication timestamp; pit-lane status unknown')]

    def resolve(self,year,event,target,roster,cutoff,live=False):
        records=[];folder=self.root/'snapshots'/str(year)/f"{int(event['RoundNumber']):02d}_{target}"
        for path in sorted(folder.glob('*.json')):
            try:
                record=json.loads(path.read_text())
                if record['year']!=year or record['event']!=event['EventName'] or record['target']!=target: continue
                validate_rows(record['rows'],roster);utc(record['known_utc'])
                records.append(record)
            except (ValueError,KeyError,TypeError,OSError): self.warnings.append('Invalid cached grid: '+str(path))
        for source in ('fia','formula1','openf1'):
            try:
                fresh=(self.fia(year,event,target,roster,cutoff) if source=='fia'
                       else getattr(self,source)(year,event,target,roster))
                for record in fresh:
                    record.update(year=year,event=event['EventName'],target=target)
                    serialized=json.dumps(record,sort_keys=True);digest=hashlib.sha256(serialized.encode()).hexdigest()
                    save_json(folder/(digest+'.json'),record)
                    records.append(record)
            except (requests.RequestException,ValueError,KeyError,TypeError,OSError,PyPdfError) as exc:
                self.warnings.append(f'{source}: {str(exc)[:180]}')
        if live:
            # A normal live request can observe a newly fetched grid during the
            # request, but never a record observed after the target starts.
            cutoff=min(pd.Timestamp.now(tz='UTC'),session_time(event,target)-pd.Timedelta(nanoseconds=1))
        eligible=[r for r in records if utc(r['known_utc'])<=cutoff and r['source'] in PRIORITY]
        if not eligible:
            if records: self.warnings.append('Undated grids first observed after cutoff were excluded.')
            return None
        eligible.sort(key=lambda r:(PRIORITY[r['source']],-utc(r['known_utc']).value))
        chosen=eligible[0];chosen['conflicts']=[];chosen['selection_cutoff_utc']=cutoff.isoformat()
        latest={}
        for record in eligible: latest.setdefault(record['source'],record)
        selected={r['driver']:r for r in chosen['rows']}
        for source,record in latest.items():
            if source==chosen['source']: continue
            different=[]
            for row in record['rows']:
                other=selected[row['driver']]
                if other['pit_lane'] is True and row['pit_lane'] is None: continue
                if row['position']!=other['position'] or (row['pit_lane'] is not None and row['pit_lane']!=other['pit_lane']):
                    different.append(row['driver'])
            if different: chosen['conflicts'].append(dict(source=source,drivers=different,url=record['url']))
        return chosen


def apply_grid(inputs, drivers, record):
    mapping={r['driver']:r for r in record['rows']};n=len(drivers)
    values=[]
    for d in drivers:
        row=mapping[d['driver']]
        # Pit-lane drivers share a back-of-field score; their actual exit order
        # and delay are unknown. Do not assign fictitious grid positions.
        values.append(1. if row['pit_lane'] is True else (row['position']-1)/max(1,n-1))
    inputs['start_grid_norm']=values
    inputs['start_pit_lane']=[mapping[d['driver']]['pit_lane'] for d in drivers]
    return inputs
