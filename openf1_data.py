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
 with tempfile.NamedTemporaryFile('w',dir=path.parent,delete=False,encoding='utf-8') as f:json.dump(data,f,allow_nan=False);name=f.name
 os.replace(name,path)

def stamp(value):
 t=pd.Timestamp(value)
 if pd.isna(t) or t.tzinfo is None:raise ValueError('OpenF1 timestamp requires an offset')
 return t.tz_convert('UTC')

class Client:
 def __init__(self,root,offline=False,refresh=False):self.root=Path(root)/'openf1';self.offline=offline;self.refresh=refresh
 def get(self,endpoint,**params):
  global _LAST_REQUEST
  url=requests.Request('GET','https://api.openf1.org/v1/'+endpoint,params=sorted(params.items())).prepare().url
  path=self.root/'http'/(hashlib.sha256(url.encode()).hexdigest()+'.json')
  if path.exists() and (self.offline or not self.refresh):
   cached=json.loads(path.read_text())
   if cached['url']!=url:raise ValueError('OpenF1 cache identity mismatch')
   if self.offline or time.time()-cached['fetched_unix']<900 or cached['data'] and endpoint not in {'sessions','meetings'} and 'session_key' in params:
    return cached['data']
  if self.offline:raise ValueError('OpenF1 response not cached: '+endpoint)
  print(f"  OpenF1 download: {endpoint} {params}", flush=True)
  for attempt in range(4):
   time.sleep(max(0,2.1-(time.monotonic()-_LAST_REQUEST)));_LAST_REQUEST=time.monotonic()
   r=requests.get(url,timeout=(5,25))
   if r.status_code==429:time.sleep(min(30,max(5,float(r.headers.get('Retry-After','10')))));continue
   if r.status_code==404:
    save(path,dict(url=url,fetched_unix=time.time(),data=[]));return []
   r.raise_for_status();data=r.json()
   if not isinstance(data,list) or not all(isinstance(x,dict) for x in data):raise ValueError('Invalid OpenF1 response')
   if 'session_key' in params and any(x.get('session_key')!=params['session_key'] for x in data):raise ValueError('OpenF1 session identity mismatch')
   save(path,dict(url=url,fetched_unix=time.time(),data=data));return data
  raise ValueError('OpenF1 rate limit: retry later')
 def schedule(self,year):
  meetings=self.get('meetings',year=year);sessions=self.get('sessions',year=year);events=[]
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
 for message in sorted(control, key=lambda item: item['date']):
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


def quality_laps(laps,stints,drivers,control,pits,start,cutoff,policy='strict'):
 """Derived validity, not an official IsAccurate flag; unknown stays rejected."""
 mapping={d['driver_number']:d['name_acronym'] for d in drivers};deleted,unknown_before=deletion_evidence(laps,drivers,control,cutoff);flags={};blocked=[];neutral_blocks=[];last=start
 for msg in sorted(control,key=lambda x:x['date']):
  t=stamp(msg['date'])
  if t>cutoff:continue
  if t>last and flags:
   blocked.append((last,t))
   if 'SC' in flags.values():neutral_blocks.append((last,t))
  last=t;flag=msg.get('flag');scope=(msg.get('scope'),msg.get('sector'))
  if flag in {'YELLOW','DOUBLE YELLOW','RED'}:flags[scope]=flag
  elif flag in {'GREEN','CLEAR'}:
   if msg.get('scope')=='Track':flags={}
   else:flags.pop(scope,None)
  text=msg.get('message','')
  if 'SAFETY CAR DEPLOYED' in text:flags[('Neutralisation',None)]='SC'
  if 'SAFETY CAR IN PIT LANE' in text:flags.pop(('Neutralisation',None),None)

 if flags:
  blocked.append((last,cutoff))
  if 'SC' in flags.values():neutral_blocks.append((last,cutoff))
 pitset={(r['driver_number'],n) for r in pits if r.get('lap_number') is not None for n in [r['lap_number']-1,r['lap_number']]}
 outs={(r['driver_number'],r['lap_number']) for r in laps if r.get('is_pit_out_lap') is True}
 ordered={(r['driver_number'],r['lap_number']):r for r in laps};duplicates=Counter((r['driver_number'],r['lap_number']) for r in laps);accepted=[];audit=[]
 for r in laps:
  num=r['driver_number'];n=r['lap_number'];pair=(num,n);reason=None;t=None;finish=None
  duration=r.get('lap_duration')
  if duplicates[pair]>1:reason='duplicate lap'
  elif r.get('date_start') is None or not isinstance(duration,(int,float)) or not np.isfinite(duration) or duration<=0:reason='missing timing'
  else:
   t=stamp(r['date_start']);finish=t+pd.Timedelta(seconds=duration)
   if t<start or finish>cutoff:reason='outside cutoff'
   elif pair in pitset or pair in outs or (num,n+1) in outs:reason='pit boundary'
   elif deleted.get(pair,False):reason='deleted lap'
   elif num in unknown_before and t<=unknown_before[num]:reason='unresolved earlier deletion'
   elif any(t<b and finish>x for x,b in blocked):reason='flag/neutralisation interval'
   else:
    sectors=[r.get('duration_sector_'+str(i)) for i in [1,2,3]]
    tolerance=.0500001 if policy=='proxy' else .0030001
    if not all(isinstance(x,(int,float)) and np.isfinite(x) and x>0 for x in sectors) or abs(sum(sectors)-duration)>tolerance:reason='sector sum mismatch'
    elif policy=='strict':
     prev=ordered.get((num,n-1))
     if prev and prev.get('date_start') and isinstance(prev.get('lap_duration'),(int,float)):
      p=stamp(prev['date_start']);pend=p+pd.Timedelta(seconds=prev['lap_duration'])
      if abs((t-pend).total_seconds())>.0030001:reason='consecutive lap timing mismatch'
      elif any(p<b and pend>x for x,b in neutral_blocks):reason='previous lap affected by neutralisation'
  matches=[s for s in stints if s['driver_number']==num and s.get('lap_start') is not None and s.get('lap_end') is not None and s['lap_start']<=n<=s['lap_end']]
  if reason is None:
   if len(matches)!=1:reason='ambiguous stint'
   else:
    s=matches[0];age=s.get('tyre_age_at_start')
    if not isinstance(age,(int,float)) or age<0 or s.get('compound') not in {'SOFT','MEDIUM','HARD','INTERMEDIATE','WET'}:reason='unknown tyre'
  state='valid_for_pace' if reason is None else 'unverified' if reason in {'missing timing','unknown tyre','ambiguous stint','unresolved earlier deletion','consecutive lap timing mismatch'} else 'excluded_from_pace'
  audit.append(dict(driver=mapping.get(num,str(num)),lap_number=n,accepted=reason is None,quality_state=state,reason=reason or 'passed derived checks'))
  if reason is None:
   accepted.append(dict(Driver=mapping[num],Compound=s['compound'],TyreLife=age+n-s['lap_start']+1,FreshTyre=age==0,LapNumber=n,Stint=s['stint_number'],LapStartTime=t-start,lap_seconds=duration))
 return pd.DataFrame(accepted,columns=['Driver','Compound','TyreLife','FreshTyre','LapNumber','Stint','LapStartTime','lap_seconds']),audit

class Store:
 def __init__(self,client,year,now,pace_function,policy='standard'):
  self.client=client;self.year=year;self.now=now;self.pace_function=pace_function;self.policy=policy;self.unavailable=[];self.memory={}
 def get(self,event,code):
  rnd=int(event['RoundNumber']);identity=(rnd,code)
  if identity in self.memory:return self.memory[identity]
  session=event.get(code+'_openf1')
  if not isinstance(session,dict) or stamp(session['date_start'])>=self.now:return None
  if stamp(session['date_end'])>=self.now:return None
  key=session['session_key'];start=stamp(session['date_start'])
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
   weather=get('weather') if code.startswith('FP') or code=='S' else [];observed={}
   for local,remote in [('AirTemp','air_temperature'),('TrackTemp','track_temperature'),('Rainfall','rainfall')]:
    vals=[r[remote] for r in weather if r.get(remote) is not None and start<=stamp(r['date'])<self.now]
    observed[local]=float(max(vals) if local=='Rainfall' else np.median(vals)) if vals else None
   data=dict(code=code,drivers=records,weather=observed,lap_data_loaded=lapdata,pace_summary_version=2,section_analysis={'enabled':False},source='OpenF1',quality_policy=self.policy,lap_quality_audit=audit)
   self.memory[identity]=data;return data
  except (ValueError,KeyError,TypeError,OSError,requests.RequestException) as exc:
   self.unavailable.append(dict(round=rnd,session=code,reason=str(exc)));self.memory[identity]=None
   print(f'  R{rnd} {code} unavailable: {exc}',flush=True);return None
