from __future__ import annotations
import importlib
import json

CANDIDATES = ["Python_sml_ClientInterface", "soar_sml"]
loaded = None
errors = {}
for name in CANDIDATES:
    try:
        importlib.import_module(name)
        loaded = name
        break
    except Exception as exc:
        errors[name] = f"{type(exc).__name__}: {exc}"

report = {
    "candidate": "Soar 9.6.5",
    "role": "integrated_cognitive_architecture",
    "python_binding_import": loaded is not None,
    "module": loaded,
    "errors": errors,
    "claim_boundary": "Import/runtime feasibility only; working-memory and checkpoint semantics are not yet tested.",
}
print(json.dumps(report, indent=2))
if loaded is None:
    raise SystemExit(1)
