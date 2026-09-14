#!/usr/bin/env python3
from __future__ import annotations
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

HERE=Path(__file__).resolve().parent
OLD_TEST=HERE.parent/'r2-2p9b-full-control-v0'/'test_runpod_science_launcher_v4.py'

def load(path:Path,name:str):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise RuntimeError(f'cannot load {path}')
    mod=importlib.util.module_from_spec(spec); sys.modules[name]=mod; spec.loader.exec_module(mod); return mod

def main()->None:
    adapter=load(HERE/'runpod_science_condition_launch_v5.py','_vela_r203_launcher_adapter')
    old_test=load(OLD_TEST,'_vela_pr202_launcher_test')
    out=Path(tempfile.mkstemp(prefix='vela-r203-adapted-',suffix='.sh')[1])
    try:
        out.write_text(adapter.adapted_source(),encoding='utf-8'); out.chmod(0o700)
        subprocess.run(['bash','-n',str(out)],check=True)
        src=out.read_text(encoding='utf-8')
        for token in ['MAX_RUNTIME_SECONDS=900','WORKLOAD_TIMEOUT_SECONDS=840','--container-disk-in-gb 50','experiments/vela-active-state/r2-7p2b-capacity-followup-v0','name="vela-p-r2-03-science-','RUNPOD_HTTPS_PROXY_V1','CONTAINER_RESTART_DETECTED_BY_CONTROL_PLANE']:
            assert token in src,token
        assert 'pod logs' not in src
        old_test.test_success(out)
        old_test.test_restart_fail_closed(out)
    finally:
        out.unlink(missing_ok=True)
    print('RUNPOD_SCIENCE_LAUNCHER_V5_SYNTHETIC_PASS')

if __name__=='__main__':
    main()
