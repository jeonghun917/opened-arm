from __future__ import annotations
import json,sys
from pathlib import Path
CONDITIONS=["B","C","G","M","X"]
IDENTITY={"stimuli_index_git_blob_sha1":"95b2371ac7bb9416ad5b9a7d33f2df4de40d4d4c","evaluator_git_blob_sha1":"55d28e6564396408f293f67ebabfc258e41ae9ef","external_trace_git_blob_sha1":"fb056e547feaedfead6b2bb67f87e31449d21466","evaluator_sha256":"2bcca90657a39eb67b14b1015f617b15fdd9e588a5620144fffa591cb1e1b92f","external_trace_sha256":"e901b4e5b343d567816fd72bfd8c6b11e314ddbb442578da9999f55e8a26db69"}
HIST={"source_candidate":"f7516276f5d4bc9b05f3a226334473bdb98869d8","run":34722990921,"artifact":10306803736,"M_minus_X_correct_count":0,"M_minus_X_margin_sum":5.005261647049338,"episode5_M_minus_X_margin":2.523124316241592,"episode6_M_minus_X_margin":2.4821373308077455}
def classify(r,equal,ident):
 if not equal or not ident or not all(bool(r[n].get("protocol_ok")) for n in CONDITIONS) or any(any(bool(v) for v in r[n].get("proposal_oracle_used",())) for n in CONDITIONS) or not (r["M"].get("bank_binding_ok") and r["X"].get("bank_binding_ok")): return "PROTOCOL_BLOCKED"
 m,x,g=r["M"],r["X"],r["G"]; mc,xc,gc=int(m["proposal_correct_count"]),int(x["proposal_correct_count"]),int(g["proposal_correct_count"]); a,b=m["proposal_correctness"],x["proposal_correctness"]
 if mc>xc and int(bool(m["task_completed"]))>=int(bool(x["task_completed"])) and int(a[4])>=int(b[4]) and int(a[5])>=int(b[5]) and mc>=gc: return "SUPPORTED"
 mm,xm=m["expected_margins"],x["expected_margins"]; d=float(m["expected_margin_sum"])-float(x["expected_margin_sum"])
 if mc==xc and d>0 and float(mm[4])-float(xm[4])>0 and float(mm[5])-float(xm[5])>0: return "MIXED"
 if mc>xc: return "MIXED"
 return "NOT_SUPPORTED"
def signature(x):
 keys=("provider_mutation_count","provider_receipt_count","provider_request_ids","source_reads","usable_memory_ids","decision_prompt_trace","decision_options_trace","decision_prompt_bytes","proposal_units","proposal_unit_kinds","event_trace","external_trace_git_blob_sha1")
 return tuple(json.dumps(x[k],sort_keys=True) for k in keys)
def delta(r,l,q,i): return {"correct":int(r[l]["proposal_correctness"][i])-int(r[q]["proposal_correctness"][i]),"margin":float(r[l]["expected_margins"][i])-float(r[q]["expected_margins"][i])}
def main():
 root=Path(sys.argv[1] if len(sys.argv)>1 else "."); shards={}
 for c in CONDITIONS:
  m=list(root.rglob(f"condition_result_{c}.json"))
  if len(m)!=1: raise RuntimeError(f"expected exactly one shard for {c}, got {m}")
  d=json.loads(m[0].read_text())
  if d.get("status")!="P-R2-02_CONDITION_RESULT" or d.get("condition")!=c or not d.get("scientific_evidence") or d.get("input_identity")!=IDENTITY: raise RuntimeError(f"invalid shard {c}")
  shards[c]=d
 commits={d.get("source_commit") for d in shards.values()}
 if len(commits)!=1 or None in commits: raise RuntimeError(f"source commit mismatch {commits}")
 runtimes=[d["runtime"] for d in shards.values()]
 if any(x!=runtimes[0] for x in runtimes[1:]): raise RuntimeError("runtime identity differs across shards")
 scales={}
 for scale in ["0.4B","2.9B"]:
  r={c:shards[c]["scales"][scale] for c in CONDITIONS}; sig=signature(r["B"]); equal=all(signature(r[c])==sig for c in CONDITIONS[1:]); m,x=r["M"],r["X"]; pri={"episode5_security":{"M":m["prior_selected_state_sha256_trace"][4],"X":x["prior_selected_state_sha256_trace"][4]},"episode6_runtime":{"M":m["prior_selected_state_sha256_trace"][5],"X":x["prior_selected_state_sha256_trace"][5]}}; ident=pri["episode5_security"]["M"]!=pri["episode5_security"]["X"] and pri["episode6_runtime"]["M"]!=pri["episode6_runtime"]["X"]; status=classify(r,equal,ident)
  scales[scale]={"conditions":r,"equal_external_workflow":equal,"route_identifiability_ok":ident,"return_prior_state_digests":pri,"hypothesis_status":status,"protocol_ok":all(r[c]["protocol_ok"] for c in CONDITIONS) and equal and ident,"primary":{"M_minus_X_correct_count":int(m["proposal_correct_count"])-int(x["proposal_correct_count"]),"M_minus_X_margin_sum":float(m["expected_margin_sum"])-float(x["expected_margin_sum"]),"security_return_episode5_M_minus_X":delta(r,"M","X",4),"runtime_return_episode6_M_minus_X":delta(r,"M","X",5)},"secondary":{"M_minus_G_correct_count":int(m["proposal_correct_count"])-int(r["G"]["proposal_correct_count"]),"M_minus_G_margin_sum":float(m["expected_margin_sum"])-float(r["G"]["expected_margin_sum"]),"correction_episode4_M_minus_C":delta(r,"M","C",3),"security_return_episode5_M_minus_G":delta(r,"M","G",4),"runtime_return_episode6_M_minus_G":delta(r,"M","G",5)}}
 a,b=scales["0.4B"]["primary"],scales["2.9B"]["primary"]
 out={"status":"P-R2-02_RESULT","scientific_evidence":True,"scientific_matrix_run_ordinal":1,"source_commit":next(iter(commits)),"runtime":runtimes[0],"input_identity":IDENTITY,"shard_conditions":CONDITIONS,"scales":scales,"frozen_historical_0p4b_reference":HIST,"scale_interaction":{"matched_gpu_M_minus_X_hard_delta_2p9b_minus_0p4b":int(b["M_minus_X_correct_count"])-int(a["M_minus_X_correct_count"]),"matched_gpu_M_minus_X_margin_delta_2p9b_minus_0p4b":float(b["M_minus_X_margin_sum"])-float(a["M_minus_X_margin_sum"]),"matched_gpu_ep5_M_minus_X_margin_delta_2p9b_minus_0p4b":float(b["security_return_episode5_M_minus_X"]["margin"])-float(a["security_return_episode5_M_minus_X"]["margin"]),"matched_gpu_ep6_M_minus_X_margin_delta_2p9b_minus_0p4b":float(b["runtime_return_episode6_M_minus_X"]["margin"])-float(a["runtime_return_episode6_M_minus_X"]["margin"])},"primary_hypothesis_status":scales["2.9B"]["hypothesis_status"],"protocol_ok":bool(scales["0.4B"]["protocol_ok"] and scales["2.9B"]["protocol_ok"]),"claim_boundary":"Inherited P-R2-01 classifier applies only to 2.9B primary. Matched-GPU 0.4B is calibration/scale-interaction control; historical CPU 0.4B is preserved separately."}
 Path("scientific_result.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n"); print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
