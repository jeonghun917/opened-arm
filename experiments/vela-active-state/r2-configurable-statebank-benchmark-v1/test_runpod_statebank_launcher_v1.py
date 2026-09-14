#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import signal
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

SOURCE = "d" * 40
MODEL = "rwkv7-g1i-2.9b-20260805"
RETRIEVAL = "OFF"
PACK = "experiments/vela-active-state/r2-configurable-statebank-benchmark-v1/problem_packs/VELA_R2_StateBank_ProblemPack_v1/PROBLEM_SOURCE_V1.json"
CONFIG = "experiments/vela-active-state/r2-configurable-statebank-benchmark-v1/EXECUTION_CONFIG_V1.json"

def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("vela_statebank_launcher_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod

class Harness:
    def __init__(self, root: Path, mode: str, mod):
        self.root = root
        self.evidence = root / "evidence"
        self.evidence.mkdir()
        self.mode = mode
        self.mod = mod
        self.delete_count = 0
        self.get_count = 0
        self.poll_count = 0
        self.create_command = None
        self.raw_result = (json.dumps({"schema":"vela-statebank-benchmark-result:v1","status":"COMPLETE","scientific_evidence":True,"source_commit":SOURCE,"model_profile":{"id":MODEL},"retrieval_mode":RETRIEVAL},separators=(",", ":"),sort_keys=True).encode()+b"\n")
    def fake_sh(self, cmd, **kw):
        if len(cmd)>=2 and cmd[0]==sys.executable and str(cmd[1]).endswith("runpod_select_a40_capacity_v1.py"):
            Path(cmd[3]).write_text(json.dumps({"status":"PASS","paid_mutation":False,"gpuId":"NVIDIA A40","displayName":"A40","memoryInGb":48,"secureCloud":True,"securePricePerHr":0.49,"inventoryStockStatus":"Low","selectedDataCenter":{"dataCenterId":"CA-MTL-1","stockStatus":"Low"}}));return SimpleNamespace(returncode=0,stdout="",stderr="")
        if cmd[:2]==["runpodctl","user"]: return SimpleNamespace(returncode=0,stdout="{}\n",stderr="")
        if cmd[:3]==["runpodctl","gpu","list"]: return SimpleNamespace(returncode=0,stdout='[{"gpuId":"NVIDIA A40"}]\n',stderr="")
        if cmd[:3]==["runpodctl","pod","create"]:
            self.create_command=list(cmd);env_json=json.loads(cmd[cmd.index("--env")+1]);env_list=[f"VELA_RESULT_TOKEN={env_json['VELA_RESULT_TOKEN']}",f"VELA_SUPERVISOR_B64={env_json['VELA_SUPERVISOR_B64']}"];return SimpleNamespace(returncode=0,stdout=json.dumps({"id":"testpod123","env":env_list}),stderr="")
        if cmd[:3]==["runpodctl","pod","get"]:
            self.get_count+=1;uptime=[30,35,4][min(self.get_count-1,2)] if self.mode=="restart" else 20+self.get_count*3;return SimpleNamespace(returncode=0,stdout=json.dumps({"id":"testpod123","runtimeStatus":"running","uptimeSeconds":uptime,"env":{"VELA_RESULT_TOKEN":"secret-token","VELA_SUPERVISOR_B64":"secret-supervisor"}}),stderr="")
        if cmd[:3]==["runpodctl","pod","logs"]: return SimpleNamespace(returncode=0,stdout="container diagnostic\n",stderr="")
        if cmd[:3]==["runpodctl","pod","delete"]:
            self.delete_count+=1;return SimpleNamespace(returncode=0,stdout='{"deleted":true}\n',stderr="")
        raise AssertionError(f"unexpected command: {cmd}")
    def request_json(self,_url):
        self.poll_count+=1
        if self.mode=="cancel": raise self.mod.ControllerSignal(signal.SIGTERM)
        if self.mode=="restart": return 0,{}
        if self.poll_count==1:return 200,{"schema":"vela-statebank-proxy-status:v1","state":"RUNNING","phase":"run_benchmark","boot_id":"x","source_commit":SOURCE,"model_profile_id":MODEL,"retrieval_mode":RETRIEVAL}
        return 200,{"schema":"vela-statebank-proxy-status:v1","state":"COMPLETE","phase":"result_ready","boot_id":"x","source_commit":SOURCE,"model_profile_id":MODEL,"retrieval_mode":RETRIEVAL,"result_bytes":len(self.raw_result),"result_sha256":hashlib.sha256(self.raw_result).hexdigest()}
    def fetch_result(self,_url,path): path.write_bytes(self.raw_result);return 200,self.raw_result

def run_case(mode: str):
    launcher=Path(__file__).with_name("runpod_statebank_launch_v1.py");mod=load_module(launcher);root=Path(tempfile.mkdtemp(prefix=f"vela-statebank-launcher-{mode}-"));h=Harness(root,mode,mod);mod.sh=h.fake_sh;mod.request_json=h.request_json;mod.fetch_result=h.fetch_result;mod.image_ref=lambda evidence:mod.PINNED_IMAGE;mod.time.sleep=lambda _seconds:None
    old_env=os.environ.copy();old_argv=sys.argv[:];os.environ.update({"RUNPOD_API_KEY":"x","VELA_SOURCE_COMMIT":SOURCE,"VELA_MODEL_PROFILE_ID":MODEL,"VELA_RETRIEVAL_MODE":RETRIEVAL,"VELA_PROBLEM_PACK_REL":PACK,"VELA_EXECUTION_CONFIG_REL":CONFIG,"VELA_PAID_EXECUTION_AUTHORIZED":"true","VELA_MAX_RUNTIME_SECONDS":"900","VELA_EVIDENCE_DIR":str(h.evidence)});sys.argv=[str(launcher)]
    try:
        try:mod.main()
        except SystemExit as exc:code=int(exc.code or 0)
        else:code=0
    finally:os.environ.clear();os.environ.update(old_env);sys.argv=old_argv
    return mod,h,code

def main():
    _,success,code=run_case("success");assert code==0;assert success.delete_count==1;envelope=json.loads((success.evidence/"provider-envelope.json").read_text());assert envelope["status"]=="PASS" and envelope["pod_deleted"] is True;assert "--terminate-after" not in success.create_command;create_text=(success.evidence/"pod-create.json").read_text();assert "secret-token" not in create_text and "secret-supervisor" not in create_text;pod_get_text=(success.evidence/"last-pod-get.json").read_text();assert "secret-token" not in pod_get_text and "secret-supervisor" not in pod_get_text
    _,restart,code=run_case("restart");assert code==1 and restart.delete_count==1;envelope=json.loads((restart.evidence/"provider-envelope.json").read_text());assert envelope["terminal_reason"]=="CONTAINER_RESTART_DETECTED_BY_CONTROL_PLANE" and envelope["pod_deleted"] is True
    _,cancelled,code=run_case("cancel");assert code==128+signal.SIGTERM and cancelled.delete_count==1;envelope=json.loads((cancelled.evidence/"provider-envelope.json").read_text());assert envelope["terminal_reason"]=="CONTROLLER_CANCELLED" and envelope["pod_deleted"] is True and envelope["scientific_evidence"] is False
    print("STATEBANK_LAUNCHER_SYNTHETIC_PASS")
if __name__=="__main__":main()
