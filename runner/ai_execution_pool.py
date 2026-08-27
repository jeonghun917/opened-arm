#!/usr/bin/env python3
from __future__ import annotations
import argparse, copy, json, re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA='project-ai-execution-pools-v0'
TASK_TYPES={'CODING_WORKER','AI_REVIEW'}
TASK_STATES={'QUEUED','RUNNING','COMPLETED','FAILED','BLOCKED'}
PROJECT_MODES={'ENFORCED','OBSERVE_ONLY'}
ID_RE=re.compile(r'^[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}$')
MAX_TASKS=500; MAX_RECEIPTS=2000
class PoolError(ValueError): pass

def _id(v:Any,f:str)->str:
    if not isinstance(v,str) or not ID_RE.fullmatch(v.strip()): raise PoolError(f'{f} is invalid')
    return v.strip()
def _n(v:Any,f:str,none=False)->int|None:
    if v is None and none:return None
    if isinstance(v,bool) or not isinstance(v,int) or v<0 or v>9007199254740991:raise PoolError(f'{f} must be a non-negative safe integer')
    return v
def _p(v:Any,f:str,none=False)->int|None:
    x=_n(v,f,none)
    if x is not None and x<1:raise PoolError(f'{f} must be at least 1')
    return x
def _iso(v:Any,f:str)->str:
    if not isinstance(v,str) or not v.strip():raise PoolError(f'{f} must be a timestamp')
    try:d=datetime.fromisoformat(v.strip().replace('Z','+00:00'))
    except ValueError as e:raise PoolError(f'{f} must be ISO-8601-like') from e
    if d.tzinfo is None:d=d.replace(tzinfo=timezone.utc)
    return d.astimezone(timezone.utc).isoformat().replace('+00:00','Z')
def _now()->str:return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def empty_state():return {'schema':SCHEMA,'projects':{},'tasks':[],'receipts':[]}

def project_policy(pid,raw):
    pid=_id(pid,'projectId')
    if not isinstance(raw,dict):raise PoolError('project policy must be an object')
    mode=raw.get('mode','OBSERVE_ONLY')
    if mode not in PROJECT_MODES:raise PoolError('project mode is invalid')
    return {'projectId':pid,'mode':mode,'slotCount':_p(raw.get('slotCount'),'slotCount',True),'budgetUsdMicros':_n(raw.get('budgetUsdMicros'),'budgetUsdMicros',True)}
def task(raw):
    if not isinstance(raw,dict):raise PoolError('task must be an object')
    tt=raw.get('taskType'); st=raw.get('state','QUEUED')
    if tt not in TASK_TYPES:raise PoolError('taskType is invalid')
    if st not in TASK_STATES:raise PoolError('task state is invalid')
    if bool(raw.get('automaticRetry',False)):raise PoolError('automaticRetry must remain false')
    paid=bool(raw.get('paid',False)); est=_n(raw.get('estimatedCostUsdMicros'),'estimatedCostUsdMicros',True)
    if not paid and est not in (None,0):raise PoolError('free task cannot reserve paid-model cost')
    candidate=None if raw.get('candidateRef') is None else _id(raw.get('candidateRef'),'candidateRef'); authority=None if raw.get('authorityRef') is None else _id(raw.get('authorityRef'),'authorityRef')
    if authority is None:raise PoolError('authorityRef is required for every execution task')
    if tt=='AI_REVIEW' and candidate is None:raise PoolError('AI_REVIEW requires an exact candidateRef')
    return {'taskId':_id(raw.get('taskId'),'taskId'),'projectId':_id(raw.get('projectId'),'projectId'),'workstreamId':_id(raw.get('workstreamId'),'workstreamId'),'runId':_id(raw.get('runId'),'runId'),'taskType':tt,'state':st,'createdAt':_iso(raw.get('createdAt'),'createdAt'),'paid':paid,'explicitApproval':bool(raw.get('explicitApproval',False)),'automaticRetry':False,'estimatedCostUsdMicros':est,'candidateRef':candidate,'authorityRef':authority,'blockedReason':None if raw.get('blockedReason') is None else str(raw.get('blockedReason'))[:500],'startedAt':None if raw.get('startedAt') is None else _iso(raw.get('startedAt'),'startedAt'),'completedAt':None if raw.get('completedAt') is None else _iso(raw.get('completedAt'),'completedAt')}
def receipt(raw):
    if not isinstance(raw,dict):raise PoolError('receipt must be an object')
    tt=raw.get('taskType'); result=raw.get('result'); auth=raw.get('usageAuthority','PROVIDER_REPORTED')
    if tt not in TASK_TYPES:raise PoolError('receipt taskType is invalid')
    if result not in {'SUCCESS','FAILURE'}:raise PoolError('receipt result is invalid')
    if auth not in {'PROVIDER_REPORTED','DETERMINISTIC','ESTIMATE_ONLY'}:raise PoolError('usageAuthority is invalid')
    model=raw.get('modelId'); model=None if model is None else _id(model,'modelId'); calls=_n(raw.get('modelCalls',0),'modelCalls')
    if calls and model is None:raise PoolError('modelId is required when modelCalls > 0')
    return {'receiptId':_id(raw.get('receiptId'),'receiptId'),'taskId':_id(raw.get('taskId'),'taskId'),'projectId':_id(raw.get('projectId'),'projectId'),'workstreamId':_id(raw.get('workstreamId'),'workstreamId'),'runId':_id(raw.get('runId'),'runId'),'taskType':tt,'provider':_id(raw.get('provider'),'provider'),'modelId':model,'result':result,'inputTokens':_n(raw.get('inputTokens',0),'inputTokens'),'outputTokens':_n(raw.get('outputTokens',0),'outputTokens'),'modelCalls':calls,'estimatedCostUsdMicros':_n(raw.get('estimatedCostUsdMicros'),'estimatedCostUsdMicros',True),'authoritativeCostUsdMicros':_n(raw.get('authoritativeCostUsdMicros'),'authoritativeCostUsdMicros',True),'retryCount':_n(raw.get('retryCount',0),'retryCount'),'usageAuthority':auth,'sourceRef':_id(raw.get('sourceRef'),'sourceRef'),'startedAt':_iso(raw.get('startedAt'),'startedAt'),'completedAt':_iso(raw.get('completedAt'),'completedAt'),'resultAuthority':'HYPOTHESIS_ONLY' if tt=='AI_REVIEW' else 'CANDIDATE_ONLY','mayMerge':False,'mayWidenAuthority':False}
def state(raw):
    if not isinstance(raw,dict) or raw.get('schema')!=SCHEMA:raise PoolError(f'state schema must be {SCHEMA}')
    ps=raw.get('projects',{}); ts=raw.get('tasks',[]); rs=raw.get('receipts',[])
    if not isinstance(ps,dict):raise PoolError('projects must be an object')
    if not isinstance(ts,list) or len(ts)>MAX_TASKS:raise PoolError('tasks must be a bounded array')
    if not isinstance(rs,list) or len(rs)>MAX_RECEIPTS:raise PoolError('receipts must be a bounded array')
    out={'schema':SCHEMA,'projects':{k:project_policy(k,v) for k,v in ps.items()},'tasks':[task(v) for v in ts],'receipts':[receipt(v) for v in rs]}
    ids=[x['taskId'] for x in out['tasks']]; rids=[x['receiptId'] for x in out['receipts']]
    if len(ids)!=len(set(ids)):raise PoolError('duplicate taskId')
    if len(rids)!=len(set(rids)):raise PoolError('duplicate receiptId')
    return out

def budget_usage(s,pid):
    done=sum((r['authoritativeCostUsdMicros'] if r['authoritativeCostUsdMicros'] is not None else (r['estimatedCostUsdMicros'] or 0)) for r in s['receipts'] if r['projectId']==pid)
    running=sum((t['estimatedCostUsdMicros'] or 0) for t in s['tasks'] if t['projectId']==pid and t['state']=='RUNNING' and t['paid'])
    return {'completedChargeUsdMicros':done,'runningReservationUsdMicros':running,'committedUsdMicros':done+running}
def start_decision(s,t):
    p=s['projects'].get(t['projectId'])
    if p is None:return 'CONFIG_REQUIRED','Project execution policy is not configured.'
    if p['mode']=='OBSERVE_ONLY':return 'CONFIG_REQUIRED','Project is observation-only.'
    if p['slotCount'] is None:return 'CONFIG_REQUIRED','Project slotCount is not configured.'
    running=sum(1 for x in s['tasks'] if x['projectId']==t['projectId'] and x['state']=='RUNNING')
    if running>=p['slotCount']:return 'WAIT','All project-local execution slots are occupied.'
    if t['paid']:
        if not t['explicitApproval']:return 'DENY','Paid model execution requires explicit approval.'
        if p['budgetUsdMicros'] is None:return 'CONFIG_REQUIRED','Paid execution budget is not configured.'
        if t['estimatedCostUsdMicros'] is None:return 'CONFIG_REQUIRED','Paid execution requires a pre-run cost estimate.'
        if budget_usage(s,t['projectId'])['committedUsdMicros']+t['estimatedCostUsdMicros']>p['budgetUsdMicros']:return 'DENY','Projected project spend exceeds configured budget.'
    return 'ALLOW','Project-local slot and resource policy allow execution.'
def allocate_once(raw,now=None):
    s=state(copy.deepcopy(raw)); at=_iso(now or _now(),'now'); started=[]; decisions=[]
    for pid in sorted({t['projectId'] for t in s['tasks'] if t['state']=='QUEUED'}):
        q=sorted((t for t in s['tasks'] if t['projectId']==pid and t['state']=='QUEUED'),key=lambda x:(x['createdAt'],x['taskId']))
        for t in q:
            d,reason=start_decision(s,t)
            if d=='ALLOW':t['state']='RUNNING';t['startedAt']=at;t['blockedReason']=None;started.append(t['taskId']);continue
            decisions.append({'taskId':t['taskId'],'decision':d,'reason':reason})
            if d in {'DENY','CONFIG_REQUIRED'}:t['state']='BLOCKED';t['blockedReason']=reason;t['completedAt']=at;continue
            break
    return s,{'startedTaskIds':started,'decisions':decisions}
def record_receipt(raw,rr):
    s=state(copy.deepcopy(raw)); r=receipt(rr)
    if any(x['receiptId']==r['receiptId'] for x in s['receipts']):raise PoolError('duplicate receiptId')
    t=next((x for x in s['tasks'] if x['taskId']==r['taskId']),None)
    if t is None:raise PoolError('receipt task does not exist')
    if t['state']!='RUNNING':raise PoolError('receipt may only close a RUNNING task')
    for f in ('projectId','workstreamId','runId','taskType'):
        if t[f]!=r[f]:raise PoolError(f'receipt {f} does not match task authority identity')
    if t['startedAt'] and r['startedAt']!=t['startedAt']:raise PoolError('receipt startedAt does not match task')
    if r['completedAt']<r['startedAt']:raise PoolError('receipt completedAt precedes startedAt')
    if t['paid'] and r['modelCalls']<1:raise PoolError('paid model task must report at least one model call')
    if t['paid'] and r['estimatedCostUsdMicros']!=t['estimatedCostUsdMicros']:raise PoolError('receipt estimate must equal the pre-run budget reservation')
    if not t['paid'] and ((r['authoritativeCostUsdMicros'] or 0)>0 or (r['estimatedCostUsdMicros'] or 0)>0):raise PoolError('free task cannot report paid model cost')
    s['receipts'].append(r);t['state']='COMPLETED' if r['result']=='SUCCESS' else 'FAILED';t['completedAt']=r['completedAt'];return s
def ledger_summary(raw):
    s=state(raw);tot={'inputTokens':0,'outputTokens':0,'modelCalls':0,'estimatedCostUsdMicros':0,'authoritativeCostUsdMicros':0,'successCount':0,'failureCount':0,'retryCount':0};by={k:{'executions':0,'modelCalls':0} for k in sorted(TASK_TYPES)}
    for r in s['receipts']:
        for f in ('inputTokens','outputTokens','modelCalls','retryCount'):tot[f]+=r[f]
        tot['estimatedCostUsdMicros']+=r['estimatedCostUsdMicros'] or 0;tot['authoritativeCostUsdMicros']+=r['authoritativeCostUsdMicros'] or 0;tot['successCount' if r['result']=='SUCCESS' else 'failureCount']+=1;by[r['taskType']]['executions']+=1;by[r['taskType']]['modelCalls']+=r['modelCalls']
    return {'schema':SCHEMA,'commonLedger':True,'totals':tot,'byTaskType':by}
def _task(tid,pid,wid,rid,tt,created,paid=False,approval=False,estimate=None,candidate=None):
    return {'taskId':tid,'projectId':pid,'workstreamId':wid,'runId':rid,'taskType':tt,'state':'QUEUED','createdAt':created,'paid':paid,'explicitApproval':approval,'automaticRetry':False,'estimatedCostUsdMicros':estimate,'candidateRef':candidate,'authorityRef':'authority:test'}
def self_test():
    s=empty_state();s['projects']={'a':{'mode':'ENFORCED','slotCount':1,'budgetUsdMicros':1000},'b':{'mode':'ENFORCED','slotCount':1,'budgetUsdMicros':1000},'c':{'mode':'ENFORCED','slotCount':2,'budgetUsdMicros':1000},'u':{'mode':'ENFORCED','slotCount':1,'budgetUsdMicros':None}}
    z='2026-08-27T00:00:00Z';s['tasks']=[_task('a-1','a','ws-a','run-a1','CODING_WORKER',z),_task('a-2','a','ws-a','run-a2','AI_REVIEW','2026-08-27T00:00:01Z',candidate='sha:a2'),_task('b-1','b','ws-b','run-b1','CODING_WORKER',z),_task('c-r1','c','ws-c','run-c1','AI_REVIEW',z,True,True,10,'sha:abc'),_task('c-r2','c','ws-c','run-c2','AI_REVIEW','2026-08-27T00:00:01Z',True,True,10,'sha:abc'),_task('u-1','u','ws-u','run-u1','AI_REVIEW',z,True,True,10,'sha:u1')]
    s,r=allocate_once(s,'2026-08-27T00:01:00Z');started=set(r['startedTaskIds']);assert {'a-1','b-1','c-r1','c-r2'}<=started and 'a-2' not in started;assert next(t for t in s['tasks'] if t['taskId']=='u-1')['state']=='BLOCKED';assert next(t for t in s['tasks'] if t['taskId']=='a-2')['state']=='QUEUED'
    s=record_receipt(s,{'receiptId':'ra','taskId':'a-1','projectId':'a','workstreamId':'ws-a','runId':'run-a1','taskType':'CODING_WORKER','provider':'LOCAL_TEST','modelId':None,'result':'SUCCESS','modelCalls':0,'inputTokens':0,'outputTokens':0,'estimatedCostUsdMicros':0,'authoritativeCostUsdMicros':0,'retryCount':0,'usageAuthority':'DETERMINISTIC','sourceRef':'test:a1','startedAt':'2026-08-27T00:01:00Z','completedAt':'2026-08-27T00:02:00Z'})
    s=record_receipt(s,{'receiptId':'rc','taskId':'c-r1','projectId':'c','workstreamId':'ws-c','runId':'run-c1','taskType':'AI_REVIEW','provider':'AWS_BEDROCK_QWEN','modelId':'qwen.qwen3-coder-30b-a3b-v1:0','result':'SUCCESS','modelCalls':1,'inputTokens':100,'outputTokens':20,'estimatedCostUsdMicros':10,'authoritativeCostUsdMicros':None,'retryCount':0,'usageAuthority':'PROVIDER_REPORTED','sourceRef':'test:c1','startedAt':'2026-08-27T00:01:00Z','completedAt':'2026-08-27T00:02:00Z'})
    rr=next(x for x in s['receipts'] if x['receiptId']=='rc');assert rr['resultAuthority']=='HYPOTHESIS_ONLY' and rr['mayMerge'] is False;summary=ledger_summary(s);assert summary['commonLedger'] and summary['byTaskType']['CODING_WORKER']['executions']==1 and summary['byTaskType']['AI_REVIEW']['executions']==1 and summary['totals']['modelCalls']==1
    s2,r2=allocate_once(s,'2026-08-27T00:03:00Z');assert 'a-2' in r2['startedTaskIds']
    bad={'receiptId':'bad','taskId':'a-2','projectId':'b','workstreamId':'ws-a','runId':'run-a2','taskType':'AI_REVIEW','provider':'LOCAL_TEST','modelId':None,'result':'SUCCESS','modelCalls':0,'inputTokens':0,'outputTokens':0,'estimatedCostUsdMicros':0,'authoritativeCostUsdMicros':0,'retryCount':0,'usageAuthority':'DETERMINISTIC','sourceRef':'test:bad','startedAt':'2026-08-27T00:03:00Z','completedAt':'2026-08-27T00:04:00Z'}
    try:record_receipt(s2,bad)
    except PoolError:pass
    else:raise AssertionError('identity mismatch must fail closed')
    return {'schema':SCHEMA,'status':'PASS','checks':{'projectQueuesIndependent':True,'missingBudgetPaidExecutionBlocked':True,'parallelAiReviewSupported':True,'commonAllocatorAndLedger':True,'reviewHasNoMergeAuthority':True,'receiptIdentityBound':True,'automaticRetryForbidden':True}}
def _load(p):return json.loads(Path(p).read_text())
def _dump(p,d):Path(p).write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
def main(argv=None):
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='cmd',required=True);sub.add_parser('self-test');a=sub.add_parser('allocate');a.add_argument('state');a.add_argument('output');r=sub.add_parser('record');r.add_argument('state');r.add_argument('receipt');r.add_argument('output');m=sub.add_parser('summary');m.add_argument('state');x=p.parse_args(argv)
    if x.cmd=='self-test':print(json.dumps(self_test(),sort_keys=True));return 0
    if x.cmd=='allocate':s,report=allocate_once(_load(x.state));_dump(x.output,s);print(json.dumps(report,sort_keys=True));return 0
    if x.cmd=='record':s=record_receipt(_load(x.state),_load(x.receipt));_dump(x.output,s);print(json.dumps(ledger_summary(s),sort_keys=True));return 0
    if x.cmd=='summary':print(json.dumps(ledger_summary(_load(x.state)),sort_keys=True));return 0
    raise AssertionError('unreachable')
if __name__=='__main__':raise SystemExit(main())
