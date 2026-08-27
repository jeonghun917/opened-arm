#!/usr/bin/env python3
from __future__ import annotations
import argparse, copy, json
from pathlib import Path
from typing import Any
from ai_execution_pool import PoolError, ledger_summary, record_receipt, state

SEMANTIC_SCHEMA='semantic-review-pool-v0'

def _n(v:Any,field:str)->int:
    if isinstance(v,bool) or not isinstance(v,int) or v<0: raise PoolError(f'{field} must be a non-negative integer')
    return v

def semantic_receipt(raw_state:dict[str,Any],task_id:str,pool:dict[str,Any],receipt_id:str,source_ref:str,completed_at:str)->dict[str,Any]:
    s=state(copy.deepcopy(raw_state))
    task=next((x for x in s['tasks'] if x['taskId']==task_id),None)
    if task is None: raise PoolError('semantic review task does not exist')
    if task['state']!='RUNNING' or task['taskType']!='AI_REVIEW': raise PoolError('semantic review adapter requires a RUNNING AI_REVIEW task')
    if not task['paid'] or not task['explicitApproval']: raise PoolError('semantic review adapter requires an explicitly approved paid task')
    if task['estimatedCostUsdMicros'] is None: raise PoolError('semantic review adapter requires the allocator cost reservation')
    if not isinstance(pool,dict) or pool.get('schema')!=SEMANTIC_SCHEMA: raise PoolError('semantic review pool schema mismatch')
    if pool.get('task_id')!=task['taskId']: raise PoolError('semantic review pool task identity mismatch')
    if pool.get('authority')!='HYPOTHESIS_ONLY': raise PoolError('semantic review authority must remain HYPOTHESIS_ONLY')
    if pool.get('production_pass_fail_authority') is not False: raise PoolError('semantic review may not gain PASS/FAIL authority')
    if pool.get('automatic_retry') is not False: raise PoolError('semantic review automatic retry must remain false')
    budget=_n(pool.get('review_budget'),'review_budget'); completed=_n(pool.get('completed_reviews'),'completed_reviews')
    if budget<1 or completed>budget: raise PoolError('semantic review counts are invalid')
    usage=pool.get('usage')
    if not isinstance(usage,dict): raise PoolError('semantic review usage is missing')
    input_tokens=_n(usage.get('input_tokens'),'input_tokens'); output_tokens=_n(usage.get('output_tokens'),'output_tokens')
    model_id=pool.get('model_id')
    if not isinstance(model_id,str) or not model_id.strip(): raise PoolError('semantic review model_id is missing')
    result='SUCCESS' if completed==budget else 'FAILURE'
    return {
      'receiptId':receipt_id,'taskId':task['taskId'],'projectId':task['projectId'],'workstreamId':task['workstreamId'],'runId':task['runId'],'taskType':'AI_REVIEW',
      'provider':'AWS_BEDROCK','modelId':model_id.strip(),'result':result,'inputTokens':input_tokens,'outputTokens':output_tokens,'modelCalls':budget,
      'estimatedCostUsdMicros':task['estimatedCostUsdMicros'],'authoritativeCostUsdMicros':None,'retryCount':0,'usageAuthority':'PROVIDER_REPORTED',
      'sourceRef':source_ref,'startedAt':task['startedAt'],'completedAt':completed_at,
    }

def apply_semantic_result(raw_state,task_id,pool,receipt_id,source_ref,completed_at):
    return record_receipt(raw_state,semantic_receipt(raw_state,task_id,pool,receipt_id,source_ref,completed_at))

def self_test():
    raw={'schema':'project-ai-execution-pools-v0','projects':{'p':{'mode':'ENFORCED','slotCount':2,'budgetUsdMicros':1000}},'tasks':[{'taskId':'review-1','projectId':'p','workstreamId':'ws','runId':'run-1','taskType':'AI_REVIEW','state':'RUNNING','createdAt':'2026-08-27T00:00:00Z','startedAt':'2026-08-27T00:01:00Z','paid':True,'explicitApproval':True,'automaticRetry':False,'estimatedCostUsdMicros':100,'candidateRef':'sha:abc','authorityRef':'authority:test'}],'receipts':[]}
    pool={'schema':'semantic-review-pool-v0','task_id':'review-1','model_id':'qwen.qwen3-coder-30b-a3b-v1:0','authority':'HYPOTHESIS_ONLY','production_pass_fail_authority':False,'automatic_retry':False,'review_budget':2,'completed_reviews':2,'usage':{'input_tokens':120,'output_tokens':30}}
    out=apply_semantic_result(raw,'review-1',pool,'receipt-1','GitHubRun:test','2026-08-27T00:02:00Z')
    r=out['receipts'][0]; assert r['projectId']=='p' and r['workstreamId']=='ws' and r['runId']=='run-1'; assert r['resultAuthority']=='HYPOTHESIS_ONLY' and r['mayMerge'] is False; assert ledger_summary(out)['totals']['modelCalls']==2
    bad=dict(pool);bad['automatic_retry']=True
    raw2=copy.deepcopy(raw)
    try: semantic_receipt(raw2,'review-1',bad,'receipt-2','GitHubRun:test2','2026-08-27T00:02:00Z')
    except PoolError: pass
    else: raise AssertionError('automatic retry must fail closed')
    return {'schema':'semantic-review-pool-adapter-v0','status':'PASS','checks':{'identityBound':True,'commonLedger':True,'hypothesisOnly':True,'noMergeAuthority':True,'noAutomaticRetry':True}}

def main(argv=None):
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='cmd',required=True); sub.add_parser('self-test'); a=sub.add_parser('apply');
    for name in ('state','task_id','pool_result','receipt_id','source_ref','completed_at','output'): a.add_argument(name)
    x=p.parse_args(argv)
    if x.cmd=='self-test': print(json.dumps(self_test(),sort_keys=True)); return 0
    raw=json.loads(Path(x.state).read_text()); pool=json.loads(Path(x.pool_result).read_text()); out=apply_semantic_result(raw,x.task_id,pool,x.receipt_id,x.source_ref,x.completed_at); Path(x.output).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps(ledger_summary(out),sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
