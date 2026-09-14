from __future__ import annotations
import hashlib, importlib.util, os, tempfile, urllib.request
from pathlib import Path

BASE_SCIENCE_COMMON_BLOB = "77cee99a8ff131b31165521fa8c9cfb76a7ce174"
BASE_RELATIVE_PATH = "experiments/vela-active-state/r2-2p9b-full-control-v0/science_common.py"
LOCAL_BASE_PATH = Path(__file__).resolve().parent.parent / "r2-2p9b-full-control-v0" / "science_common.py"

def _gitblob(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()

def _load_frozen_base() -> tuple[Path, bytes]:
    if LOCAL_BASE_PATH.exists():
        raw = LOCAL_BASE_PATH.read_bytes()
        if _gitblob(raw) != BASE_SCIENCE_COMMON_BLOB:
            raise RuntimeError(f"frozen P-R2-02 science_common drift: {_gitblob(raw)}")
        return LOCAL_BASE_PATH, raw
    source = os.environ.get("VELA_SOURCE_COMMIT", "")
    if len(source) != 40:
        raise RuntimeError("VELA_SOURCE_COMMIT required to resolve frozen science core inside RunPod")
    url = f"https://raw.githubusercontent.com/jeonghun917/opened-arm/{source}/{BASE_RELATIVE_PATH}"
    raw = urllib.request.urlopen(url, timeout=30).read()
    if _gitblob(raw) != BASE_SCIENCE_COMMON_BLOB:
        raise RuntimeError(f"remote frozen P-R2-02 science_common drift: {_gitblob(raw)}")
    fd, name = tempfile.mkstemp(prefix="vela-pr202-science-common-", suffix=".py")
    os.close(fd)
    p = Path(name)
    p.write_bytes(raw)
    return p, raw

_BASE_SCIENCE_COMMON_PATH, _raw = _load_frozen_base()
_spec = importlib.util.spec_from_file_location("_vela_pr202_science_common", _BASE_SCIENCE_COMMON_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError("cannot load frozen P-R2-02 science_common")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

for _name in dir(_base):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_base, _name)

MATCHED_PROFILE = "g1i-20260805"
MODELS = [
    {
        "scale": "2.9B",
        "family": "g1i",
        "repo": "BlinkDL/rwkv7-g1",
        "revision": "ede85bf8ab2e59aff7d7ca909fbbc73317866d89",
        "file": "rwkv7-g1i-2.9b-20260805-ctx16384.pth",
        "size": 5896273469,
        "sha256": "ac1ae23d0e65c1d35ba523eacd81a2a4dacb7b886479909bbff34f312e766320",
    },
    {
        "scale": "7.2B",
        "family": "g1i",
        "repo": "BlinkDL/rwkv7-g1",
        "revision": "ede85bf8ab2e59aff7d7ca909fbbc73317866d89",
        "file": "rwkv7-g1i-7.2b-20260805-ctx16384.pth",
        "size": 14400007869,
        "sha256": "0d09d8961448032501c4d432c33a224c66356d43c10174386ea86b0da2b127d8",
    },
]
_BASE_RUN_SCALE = _base.run_scale

def run_scale(model, pipe, scale, condition, stimuli, evaluator, external):
    out = _BASE_RUN_SCALE(model, pipe, scale, condition, stimuli, evaluator, external)
    out["state_mode"] = f"rwkv7-g1i-{scale}-matched-gpu:{STRATEGY}"
    out["matched_model_profile"] = MATCHED_PROFILE
    return out
