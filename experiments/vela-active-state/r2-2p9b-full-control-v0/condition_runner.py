from __future__ import annotations
import gc,importlib.metadata,json,os,sys,time,traceback
from pathlib import Path
from science_common import *
def main():
 c=os.environ.get("VELA_CONDITION","").strip().upper()
 if c not in CONDITIONS: raise RuntimeError(f"invalid VELA_CONDITION={c!r}")
 ensure_runtime(); import numpy as np,torch
 if torch.__version__!=EXPECTED_TORCH: raise RuntimeError(f"torch drift {torch.__version__}")
 if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
 arches=list(torch.cuda.get_arch_list()); cap=tuple(int(x) for x in torch.cuda.get_device_capability(0))
 if cap!=(6,0) or "sm_60" not in arches: raise RuntimeError(f"frozen backend requires P100 sm_60; device={torch.cuda.get_device_name(0)} cap={cap} arches={arches}")
 os.environ["RWKV_V7_ON"]="1"; os.environ["RWKV_JIT_ON"]="1"; os.environ["RWKV_CUDA_ON"]="0"
 from huggingface_hub import hf_hub_download
 from rwkv.model import RWKV
 from rwkv.utils import PIPELINE
 root=Path(__file__).resolve().parent; stimuli,idx=load_stimuli(root); evdoc=load_git(root,"EVALUATOR.json",EVALUATOR_BLOB,EVALUATOR_SHA256); external=load_git(root,"EXTERNAL_TRACE.json",EXTERNAL_BLOB,EXTERNAL_SHA256); evaluator=evdoc["rows"]
 reports={}; model_runtime={}
 for spec in MODELS:
  p=Path(hf_hub_download(repo_id=spec["repo"],filename=spec["file"],revision=spec["revision"]))
  if p.stat().st_size!=spec["size"] or sha256_file(p)!=spec["sha256"]: raise RuntimeError(f"model identity mismatch {spec['scale']}")
  torch.cuda.reset_peak_memory_stats(); t0=time.monotonic(); model=RWKV(model=str(p)[:-4],strategy=STRATEGY); pipe=PIPELINE(model,"rwkv_vocab_v20230424"); load_s=time.monotonic()-t0; t1=time.monotonic(); rep=run_scale(model,pipe,spec["scale"],c,stimuli,evaluator,external); torch.cuda.synchronize(); elapsed=time.monotonic()-t1
  rep["task_completed"]=rep["final_manifest"]==evdoc["expected_final_manifest"]; rep["provider_complete"]=rep["task_completed"]; reports[spec["scale"]]=rep; model_runtime[spec["scale"]]={"model":spec,"load_seconds":load_s,"condition_seconds":elapsed,"peak_cuda_bytes":int(torch.cuda.max_memory_allocated())}; del pipe,model; gc.collect(); torch.cuda.empty_cache()
 result={"status":"P-R2-02_CONDITION_RESULT","scientific_evidence":True,"scientific_matrix_run_ordinal":1,"condition":c,"source_commit":os.environ.get("VELA_SOURCE_COMMIT"),"runtime":{"python":sys.version.split()[0],"torch":torch.__version__,"torch_cuda_arch_list":arches,"rwkv":importlib.metadata.version("rwkv"),"tokenizers":importlib.metadata.version("tokenizers"),"numpy":np.__version__,"cuda":torch.version.cuda,"device":torch.cuda.get_device_name(0),"device_capability":list(cap),"strategy":STRATEGY,"rwkv_cuda_kernel":False},"input_identity":{"stimuli_index_git_blob_sha1":INDEX_BLOB,"evaluator_git_blob_sha1":EVALUATOR_BLOB,"external_trace_git_blob_sha1":EXTERNAL_BLOB,"evaluator_sha256":EVALUATOR_SHA256,"external_trace_sha256":EXTERNAL_SHA256},"model_runtime":model_runtime,"scales":reports}
 Path(f"condition_result_{c}.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n"); print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
if __name__=="__main__":
 try: main()
 except BaseException as e:
  c=os.environ.get("VELA_CONDITION","").strip().upper() or "UNKNOWN"; Path(f"condition_result_{c}.json").write_text(json.dumps({"status":"CONDITION_ERROR","scientific_evidence":False,"condition":c,"error_type":type(e).__name__,"error":str(e),"traceback_tail":traceback.format_exc().splitlines()[-100:]},indent=2)+"\n"); raise
