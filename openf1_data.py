"""OpenF1 schedules, classifications and lap evidence. No FastF1 dependency."""
from pathlib import Path
from datetime import timezone
from collections import Counter
import hashlib,json,time,re,os,tempfile
import numpy as np
import pandas as pd
import requests

NAMES={'Practice 1':'FP1','Practice 2':'FP2','Practice 3':'FP3','Sprint Qualifying':'SQ','Sprint Shootout':'SQ','Qualifying':'Q','Sprint':'S','Race':'R'}
_LAST_REQUEST=0.

def save(path,data):
 path.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile('w',dir=path.parent,delete=False,encoding='utf-8') as f:json.dump(data,f,sort_keys=True,separators=(',',':'),allow_nan=False);name=f.name
 os.replace(name,path)

def stamp(value):
 t=pd.Timestamp(value)
 if pd.isna(t) or t.tzinfo is None:raise ValueError('OpenF1 timestamp requires an offset')
 return t.tz_convert('UTC')

class Client:
 def __init__(self,root,offline=False,refresh=False):self.root=Path(root)/'openf1';self.offline=offline;self.refresh=refresh;self.session_ends={};self.provenance={};self.cache_events=[]
 def validate_record(self,record,url,params):
  if not isinstance(record,dict) or record.get('url')!=url:raise ValueError('OpenF1 cache identity mismatch')
  fetched=record.get('fetched_unix');data=record.get('data')
  if isinstance(fetched,bool) or not isinstance(fetched,(int,float)) or not np.isfinite(fetched) or fetched<0:raise ValueError('Invalid OpenF1 cache timestamp')
  if not isinstance(data,list) or not all(isinstance(row,dict) for row in data):raise ValueError('Invalid OpenF1 cached response')
  if 'session_key' in params and any(row.get('session_key')!=params['session_key'] for row in data):raise ValueError('OpenF1 session identity mismatch')
  json.dumps(record,allow_nan=False)
  return record
 def recover(self,url,params):
  """Recover only hash-verified observations for this exact query."""
  index=self.root/'index'/(hashlib.sha256(url.encode()).hexdigest()+'.json')
  try:
   digest=json.loads(index.read_text())['sha256']
   candidates=[self.root/'snapshots'/(digest+'.json')] if isinstance(digest,str) and re.fullmatch('[a-f0-9]{64}',digest) else []
  except (OSError,ValueError,KeyError,TypeError):candidates=[]
  # Older installations have snapshots without an index. Scan only when
  # recovering, not during ordinary successful cache reads/downloads.
  def read(path):
   raw=path.read_bytes()
   if hashlib.sha256(raw).hexdigest()!=path.stem:raise ValueError('Snapshot hash mismatch')
   return self.validate_record(json.loads(raw),url,params)
  for path in candidates:
   try:return read(path)
   except (OSError,ValueError,KeyError,TypeError):pass
  found=[]
  for path in (self.root/'snapshots').glob('*.json'):
   try:found.append(read(path))
   except (OSError,ValueError,KeyError,TypeError):continue
  return max(found,key=lambda record:record['fetched_unix']) if found else None
 def remember(self,record,used=True):
  """Keep immutable observed responses, not a claim of historical availability."""
  encoded=json.dumps(record,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
  digest=hashlib.sha256(encoded).hexdigest()
  relative=Path('snapshots')/(digest+'.json')
  snapshot=self.root/relative
  if not snapshot.exists() or hashlib.sha256(snapshot.read_bytes()).hexdigest()!=digest:save(snapshot,record)
  index=self.root/'index'/(hashlib.sha256(record['url'].encode()).hexdigest()+'.json')
  try:previous=json.loads(index.read_text())
  except (OSError,ValueError):previous={}
  previous_time=previous.get('fetched_unix') if isinstance(previous,dict) else None
  try:valid_index=isinstance(previous_time,(int,float)) and not isinstance(previous_time,bool) and np.isfinite(previous_time) and isinstance(previous.get('sha256'),str) and re.fullmatch('[a-f0-9]{64}',previous['sha256'])
  except (TypeError,ValueError,OverflowError):valid_index=False
  if not valid_index or previous_time<=record['fetched_unix']:
   save(index,dict(sha256=digest,fetched_unix=record['fetched_unix']))
  if used:
   self.provenance[record['url']]=dict(url=record['url'],fetched_unix=record['fetched_unix'],
       records=len(record['data']),sha256=digest,snapshot=str(relative))
  return record['data']
 def register_session(self,session):
  if session.get('date_end'):self.session_ends[session['session_key']]=stamp(session['date_end']).timestamp()
 def get(self,endpoint,**params):
  global _LAST_REQUEST
  url=requests.Request('GET','https://api.openf1.org/v1/'+endpoint,params=sorted(params.items())).prepare().url
  path=self.root/'http'/(hashlib.sha256(url.encode()).hexdigest()+'.json')
  cached=None
  try:
   if path.exists():cached=self.validate_record(json.loads(path.read_text()),url,params)
  except (OSError,ValueError,KeyError,TypeError):pass
  if cached is None and (self.offline or path.exists() or (self.root/'index'/(path.name)).exists()):
   cached=self.recover(url,params)
   if cached is not None:
    save(path,cached);self.cache_events.append(dict(endpoint=endpoint,params=params,recovered=True,fetched_unix=cached['fetched_unix']))
    print(f'  OpenF1 restored verified local snapshot: {endpoint} {params}',flush=True)
  if cached is not None and self.refresh and not self.offline:self.remember(cached,used=False)
  if cached is not None and (self.offline or not self.refresh):
   # Nonempty is not proof of completeness. Recent sessions must refresh
   # while timing/tyre corrections arrive; older data is also not immortal.
   end=self.session_ends.get(params.get('session_key'))
   settled=end is not None and cached['fetched_unix'] >= end + 2*86400
   ttl=7*86400 if settled and cached['data'] and endpoint not in {'sessions','meetings'} else 900
   if self.offline or time.time()-cached['fetched_unix']<ttl:
    return self.remember(cached)
   self.remember(cached,used=False)
  if self.offline:raise ValueError('OpenF1 response not cached: '+endpoint)
  print(f"  OpenF1 download: {endpoint} {params}", flush=True)
  for attempt in range(4):
   time.sleep(max(0,2.1-(time.monotonic()-_LAST_REQUEST)));_LAST_REQUEST=time.monotonic()
   r=requests.get(url,timeout=(5,25))
   if r.status_code==429:time.sleep(min(30,max(5,float(r.headers.get('Retry-After','10')))));continue
   if r.status_code==404:
    record=dict(url=url,fetched_unix=time.time(),data=[]);save(path,record);return self.remember(record)
   r.raise_for_status();data=r.json()
   if not isinstance(data,list) or not all(isinstance(x,dict) for x in data):raise ValueError('Invalid OpenF1 response')
   if 'session_key' in params and any(x.get('session_key')!=params['session_key'] for x in data):raise ValueError('OpenF1 session identity mismatch')
   record=dict(url=url,fetched_unix=time.time(),data=data);save(path,record);return self.remember(record)
  raise ValueError('OpenF1 rate limit: retry later')
 def schedule(self,year):
  meetings=self.get('meetings',year=year);sessions=self.get('sessions',year=year);events=[]
  self.session_ends.update({s['session_key']:stamp(s['date_end']).timestamp() for s in sessions if s.get('date_end')})
  meetings=sorted([m for m in meetings if not m.get('is_cancelled',False) and 'testing' not in m['meeting_name'].lower()],key=lambda m:m['date_start'])
  for rnd,m in enumerate(meetings,1):
   found=sorted([s for s in sessions if s['meeting_key']==m['meeting_key'] and not s.get('is_cancelled',False) and s['session_name'] in NAMES],key=lambda s:s['date_start'])
   if len(found)>5:raise ValueError('Ambiguous OpenF1 weekend sessions')
   event=dict(RoundNumber=rnd,EventName=m['meeting_name'],Location=m['location'],Country=m['country_name'],meeting_key=m['meeting_key'])
   for i,s in enumerate(found,1):
    start=stamp(s['date_start']);offset=pd.Timedelta(s['gmt_offset']);event.update({f'Session{i}':s['session_name'],f'Session{i}DateUtc':start,f'Session{i}Date':start.tz_convert(timezone(offset.to_pytimedelta()))})
    event[NAMES[s['session_name']]+'_openf1']=s
   events.append(event)
  return pd.DataFrame(events)


DELETION_CLOCK_MARGIN = pd.Timedelta(seconds=1)


def deletion_evidence(laps, drivers, control, cutoff):
 """Resolve explicit IDs/times; never guess the nearest lap from message time.

 An unresolved notice only makes laps already started at that notice unknown.
 One second covers the race-control timestamp's whole-second precision and
 observed lap-start jitter. Later laps still need every normal quality check.
 """
 deleted = {}
 unknown_before = {}
 numbers = {d['driver_number'] for d in drivers}
 for message in sorted(control, key=lambda item: stamp(item['date'])):
  when = stamp(message['date'])
  if when > cutoff:
   continue
  text = message.get('message', '').upper()
  restored = bool(re.search(r'\bREINSTATED\b', text))
  if not restored and not re.search(r'\bDELETED\b', text):
   continue
  car = re.search(r'\b(?:CAR|DRIVER)\s*#?\s*(\d+)\b', text)
  number = message.get('driver_number')
  if number is None and car:
   number = int(car[1])
  if car and number is not None and int(car[1]) != number:
   # Conflicting identity cannot validate a reinstatement.
   if not restored:
    for candidate in {number, int(car[1])} & numbers:
     unknown_before[candidate] = max(unknown_before.get(candidate, when), when + DELETION_CLOCK_MARGIN)
   continue
  # race_control.lap_number is the lap when the message was issued,
  # not necessarily the driver's deleted lap. Use the explicit message only.
  explicit = re.search(r'\bLAP\s+(\d+)\b', text)
  lap_id = int(explicit[1]) if explicit else None
  if number in numbers and isinstance(lap_id, (int, float)) and lap_id >= 1 and lap_id == int(lap_id):
   deleted[number, int(lap_id)] = not restored
   continue
  time_match = re.search(r'\bTIME\s+(\d+):(\d+\.\d+)\b', text)
  candidates = []
  if number in numbers and time_match:
   duration = 60 * int(time_match[1]) + float(time_match[2])
   for lap in laps:
    value = lap.get('lap_duration')
    if lap['driver_number'] != number or not isinstance(value, (int, float)) or not np.isfinite(value) or abs(value-duration) > .0005001:
     continue
    if lap.get('is_pit_out_lap') is True:
     # Outlap duration can include garage time before date_start. It cannot
     # supply a crossing timestamp, but a unique time still identifies the
     # already-excluded outlap without making all earlier laps unknown.
     eligible = not lap.get('date_start') or stamp(lap['date_start']) <= when + DELETION_CLOCK_MARGIN
    else:
     eligible = bool(lap.get('date_start')) and stamp(lap['date_start']) + pd.Timedelta(seconds=value) <= when + DELETION_CLOCK_MARGIN
    if eligible:
     candidates.append((number, lap['lap_number']))
  if len(candidates) == 1 or candidates and not restored:
   # A repeated identical time can exclude every candidate, but cannot
   # reinstate every candidate without uniquely identifying the lap.
   for candidate in candidates:
    deleted[candidate] = not restored
  elif not restored:
   affected = {number} if number in numbers else numbers
   for candidate in affected:
    unknown_before[candidate] = max(unknown_before.get(candidate, when), when + DELETION_CLOCK_MARGIN)
 return deleted, unknown_before


def pit_laps(laps, pits, cutoff):
 """Pit endpoint lap can identify either entry or exit. Use outlap evidence.

 An exit lap n excludes n-1/n; an entry lap n followed by an exit n+1
 excludes n/n+1, not the preceding clean lap. Unresolved records retain the
 conservative n-1/n exclusion. Never use a future pit report at this cutoff.
 """
 outs={(r['driver_number'],r['lap_number']) for r in laps if r.get('is_pit_out_lap') is True and (not r.get('date_start') or stamp(r['date_start'])<=cutoff)}
 excluded=set(outs)
 excluded.update((num,n-1) for num,n in outs)
 for pit in pits:
  if pit.get('date') and stamp(pit['date'])>cutoff:
   continue
  n=pit.get('lap_number');num=pit['driver_number']
  if n is None:
   continue
  if (num,n) in outs:
   excluded.update({(num,n-1),(num,n)})
  elif (num,n+1) in outs:
   excluded.update({(num,n),(num,n+1)})
  else:
   excluded.update({(num,n-1),(num,n)})
 return excluded


def whole_laps(value,minimum=0):
 """A source lap count is a whole number, never a boolean or fraction."""
 if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float,np.integer,np.floating)):return False
 try:return bool(np.isfinite(value) and value>=minimum and value==int(value))
 except (TypeError,ValueError,OverflowError):return False

def stint_evidence(stints,number,lap):
 matches=[]
 for stint in stints:
  if stint.get('driver_number')!=number:continue
  first=stint.get('lap_start');last=stint.get('lap_end')
  if whole_laps(first,1) and whole_laps(last,1) and last<first:return {},'invalid stint','reversed stint bounds'
  if not whole_laps(first,1) or not whole_laps(last,1):
   # A malformed interval cannot validate an overlapping lap. Known bounds
   # can still establish that a broken record is unrelated to this lap.
   if whole_laps(last,1) and last<lap or whole_laps(first,1) and first>lap:continue
   return {},'invalid stint','missing, reversed or noninteger stint bounds'
  if first<=lap<=last:matches.append(stint)
 if not matches:return {},'missing stint','no reported stint covers this lap'
 if len(matches)>1:return {},'ambiguous stint','overlapping reported stints'
 stint=matches[0]
 if not whole_laps(stint.get('stint_number'),1):return stint,'invalid stint','invalid stint number'
 if not whole_laps(stint.get('tyre_age_at_start')):return stint,'unknown tyre','tyre age is not a nonnegative whole lap count'
 if stint.get('compound') not in {'SOFT','MEDIUM','HARD','INTERMEDIATE','WET'}:return stint,'unknown tyre','unknown compound'
 return stint,None,None

def quality_laps(laps,stints,drivers,control,pits,start,cutoff,policy='strict'):
 """Derived validity, not an official IsAccurate flag; unknown stays rejected."""
 mapping={d['driver_number']:d['name_acronym'] for d in drivers};deleted,unknown_before=deletion_evidence(laps,drivers,control,cutoff);flags={};blocked=[];neutral_blocks=[];last=start;vsc_ending=False;sc_ending=False
 for msg in sorted(control,key=lambda x:stamp(x['date'])):
  t=stamp(msg['date'])
  if t>cutoff:continue
  if t>last and flags:
   blocked.append((last,t))
   if {'SC','VSC'} & set(flags.values()):neutral_blocks.append((last,t))
  last=t;flag=msg.get('flag');scope=(msg.get('scope'),msg.get('sector'))
  if flag in {'YELLOW','DOUBLE YELLOW','RED'}:flags[scope]=flag
  elif flag in {'GREEN','CLEAR'}:
   if msg.get('scope')=='Track':
    if flag=='GREEN':flags={};vsc_ending=False;sc_ending=False
    else:
     # Track clear alone is not a red-flag restart or an SC withdrawal.
     flags={k:v for k,v in flags.items() if v in {'RED','SC','VSC'}}
     if vsc_ending:flags.pop(('VirtualNeutralisation',None),None);vsc_ending=False
     if sc_ending:flags.pop(('Neutralisation',None),None);sc_ending=False
   else:flags.pop(scope,None)
  text=msg.get('message','').upper().strip()
  if text in {'VSC DEPLOYED','VIRTUAL SAFETY CAR DEPLOYED'}:
   flags[('VirtualNeutralisation',None)]='VSC';vsc_ending=False
  if text in {'VSC ENDING','VIRTUAL SAFETY CAR ENDING'}:vsc_ending=True
  if text=='SESSION STARTED':flags={k:v for k,v in flags.items() if v!='RED'}
  if text=='SAFETY CAR DEPLOYED':flags[('Neutralisation',None)]='SC';sc_ending=False
  if text=='SAFETY CAR IN THIS LAP':sc_ending=True
  if text=='SAFETY CAR IN PIT LANE':flags.pop(('Neutralisation',None),None)

 if flags:
  blocked.append((last,cutoff))
  if {'SC','VSC'} & set(flags.values()):neutral_blocks.append((last,cutoff))
 pitset=pit_laps(laps,pits,cutoff)
 ordered={(r['driver_number'],r['lap_number']):r for r in laps};duplicates=Counter((r['driver_number'],r['lap_number']) for r in laps);accepted=[];audit=[]
 for r in laps:
  num=r['driver_number'];n=r['lap_number'];pair=(num,n);reason=None;t=None;finish=None
  duration=r.get('lap_duration')
  if duplicates[pair]>1:reason='duplicate lap'
  elif r.get('date_start') is None or not isinstance(duration,(int,float)) or not np.isfinite(duration) or duration<=0:reason='missing timing'
  else:
   t=stamp(r['date_start']);finish=t+pd.Timedelta(seconds=duration)
   if t<start or finish>cutoff:reason='outside cutoff'
   elif pair in pitset:reason='pit boundary'
   elif deleted.get(pair,False):reason='deleted lap'
   elif num in unknown_before and t<=unknown_before[num]:reason='unresolved earlier deletion'
   elif any(t<b and finish>x for x,b in blocked):reason='flag/neutralisation interval'
   else:
    sectors=[r.get('duration_sector_'+str(i)) for i in [1,2,3]]
    tolerance=.0500001 if policy=='proxy' else .0030001
    if not all(isinstance(x,(int,float)) and np.isfinite(x) and x>0 for x in sectors):reason='missing sector timing'
    elif abs(sum(sectors)-duration)>tolerance:reason='sector sum mismatch'
    elif policy=='strict':
     prev=ordered.get((num,n-1))
     if prev and prev.get('date_start') and isinstance(prev.get('lap_duration'),(int,float)):
      p=stamp(prev['date_start']);pend=p+pd.Timedelta(seconds=prev['lap_duration'])
      if abs((t-pend).total_seconds())>.0030001:reason='consecutive lap timing mismatch'
      elif any(p<b and pend>x for x,b in neutral_blocks):reason='previous lap affected by neutralisation'
  tyre,tyre_reason,tyre_issue=stint_evidence(stints,num,n)
  if reason is None:reason=tyre_reason
  state='valid_for_pace' if reason is None else 'unverified' if reason in {'missing timing','missing sector timing','unknown tyre','missing stint','ambiguous stint','invalid stint','unresolved earlier deletion','consecutive lap timing mismatch'} else 'excluded_from_pace'
  initial_age=tyre.get('tyre_age_at_start')
  age_known=whole_laps(initial_age) and whole_laps(tyre.get('lap_start'),1)
  audit.append(dict(driver=mapping.get(num,str(num)),lap_number=n,accepted=reason is None,quality_state=state,reason=reason or 'passed derived checks',
      compound=tyre.get('compound'),stint=tyre.get('stint_number'),
      tyre_age_at_start=initial_age if age_known else None,
      derived_tyre_life=initial_age+n-tyre['lap_start']+1 if age_known else None,
      freshness_basis='age-at-start proxy; explicit new/used flag unavailable' if age_known else 'unknown',
      tyre_evidence_issue=tyre_issue,
      scheduled_elapsed_seconds=(t-start).total_seconds() if t is not None else None))
  if reason is None:
   accepted.append(dict(Driver=mapping[num],Compound=tyre['compound'],TyreLife=initial_age+n-tyre['lap_start']+1,FreshTyre=initial_age==0,LapNumber=n,Stint=tyre['stint_number'],LapStartTime=t-start,lap_seconds=duration))
 return pd.DataFrame(accepted,columns=['Driver','Compound','TyreLife','FreshTyre','LapNumber','Stint','LapStartTime','lap_seconds']),audit

class Store:
 def __init__(self,client,year,now,pace_function,policy='standard'):
  self.client=client;self.year=year;self.now=now;self.pace_function=pace_function;self.policy=policy;self.unavailable=[];self.warnings=[];self.memory={}
 def get(self,event,code):
  rnd=int(event['RoundNumber']);identity=(rnd,code)
  if identity in self.memory:return self.memory[identity]
  session=event.get(code+'_openf1')
  if not isinstance(session,dict) or stamp(session['date_start'])>=self.now:return None
  if stamp(session['date_end'])>=self.now:return None
  key=session['session_key'];start=stamp(session['date_start']);self.client.register_session(session)
  def get(endpoint):
   data=self.client.get(endpoint,session_key=key)
   if any(r.get('meeting_key')!=session['meeting_key'] for r in data):raise ValueError('OpenF1 meeting identity mismatch')
   return data
  try:
   drivers=get('drivers');results=[] if code.startswith('FP') else get('session_result')
   if len(drivers)<3:raise ValueError('Incomplete OpenF1 driver roster')
   n=len(drivers)
   # Preserve full finishing-order labels, including retired/not-classified
   # entries, only when the source's reported sequence is internally ordered.
   if code in {'R','S'} and len(results)==n and len({r['driver_number'] for r in results})==n:
    prefix=all(r.get('position') is None or r['position']==i for i,r in enumerate(results,1))
    nulls=[r for r in results if r.get('position') is None]
    laps=[r.get('number_of_laps') for r in nulls]
    ordered_laps=all(isinstance(x,(int,float)) for x in laps) and all(x>=y for x,y in zip(laps,laps[1:]))
    if prefix and ordered_laps:
     results=[dict(r,reported_position=r.get('position'),position=i,derived_order=r.get('position') is None) for i,r in enumerate(results,1)]
   positions={r['driver_number']:r for r in results}
   competitive=code in {'R','S','Q','SQ'}
   valid=[r.get('position') for r in results if isinstance(r.get('position'),(int,float)) and 1<=r['position']<=n]
   if competitive and (len(valid)<max(2,n//2) or len(valid)!=len(set(valid))):raise ValueError('Incomplete OpenF1 classification')
   grid={}
   if code in {'R','S'}:
    q=event.get(('Q' if code=='R' else 'SQ')+'_openf1')
    if isinstance(q,dict):grid={r['driver_number']:r.get('position') for r in self.client.get('starting_grid',session_key=q['session_key'])}
   pace={};lapdata=False;audit=[]
   if code.startswith('FP') or code=='S':
    control=get('race_control')
    if not any(r.get('flag')=='CHEQUERED' and start<=stamp(r['date'])<self.now for r in control) and self.now<start+pd.Timedelta(days=1):raise ValueError('No observed chequered flag before cutoff; conservative 24-hour fallback not reached')
    clean,audit=quality_laps(get('laps'),get('stints'),drivers,control,get('pit'),start,self.now,self.policy)
    if clean.empty and code.startswith('FP'):raise ValueError('No valid OpenF1 practice laps')
    pace=self.pace_function(clean,code);lapdata=True
   race_laps=max([r.get('number_of_laps') or 0 for r in results],default=0)
   records=[]
   for d in drivers:
    num=d['driver_number'];result=positions.get(num,{});pos=result.get('position');pos=pos if isinstance(pos,(int,float)) and 1<=pos<=n else None
    row=dict(reported_position=result.get('reported_position',pos),derived_finishing_order=result.get('derived_order',False),driver=d['name_acronym'],name=d.get('full_name') or d['name_acronym'],team=d.get('team_name') or 'Unknown',position=pos,position_norm=(pos-1)/(n-1) if pos else None,grid=grid.get(num),gap_pct=None,laps=result.get('number_of_laps'),sections=[],**pace.get(d['name_acronym'],{}))
    if code in {'R','S'}:
     finished=None if not all(k in result for k in ['dnf','dns','dsq']) else float(not any(result[k] for k in ['dnf','dns','dsq']))
     row.update(finished=finished,laps_completed_fraction=(result.get('number_of_laps') or 0)/race_laps if race_laps else None,finish_gap_pct=None,grid_gain_norm=(grid[num]-pos)/(n-1) if grid.get(num) and pos else None,neutralized_lap_fraction=None)
    records.append(row)
   observed=dict(AirTemp=None,TrackTemp=None,Rainfall=None)
   try:
    weather=get('weather') if code.startswith('FP') or code=='S' else []
    for local,remote in [('AirTemp','air_temperature'),('TrackTemp','track_temperature'),('Rainfall','rainfall')]:
     vals=[r[remote] for r in weather if r.get(remote) is not None and start<=stamp(r['date'])<self.now]
     observed[local]=float(max(vals) if local=='Rainfall' else np.median(vals)) if vals else None
   except (ValueError,KeyError,TypeError,OSError,requests.RequestException) as exc:
    # Weather is diagnostic. Its outage must not discard validated lap pace.
    observed=dict(AirTemp=None,TrackTemp=None,Rainfall=None)
    self.warnings.append(dict(round=rnd,session=code,endpoint='weather',reason=str(exc)))
    print(f'  R{rnd} {code}: weather unavailable; retaining validated pace.',flush=True)
   data=dict(code=code,drivers=records,weather=observed,lap_data_loaded=lapdata,pace_summary_version=2,section_analysis={'enabled':False},source='OpenF1',quality_policy=self.policy,lap_quality_audit=audit)
   self.memory[identity]=data;return data
  except (ValueError,KeyError,TypeError,OSError,requests.RequestException) as exc:
   self.unavailable.append(dict(round=rnd,session=code,reason=str(exc)));self.memory[identity]=None
   print(f'  R{rnd} {code} unavailable: {exc}',flush=True);return None
