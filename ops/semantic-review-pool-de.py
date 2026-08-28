#!/usr/bin/env python3
import importlib.util
from pathlib import Path

POOL = Path(__file__).resolve().parents[1] / 'infra' / 'aws' / 'semantic-review' / 'pool.py'
spec = importlib.util.spec_from_file_location('semantic_review_pool_base', POOL)
if spec is None or spec.loader is None:
    raise SystemExit('semantic_review_pool_import_failed')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
if module.REVIEW_COUNT != 2:
    raise SystemExit('D/E wrapper requires SEMANTIC_REVIEW_COUNT=2')
module.REVIEWERS = [
    ('D', 'Prioritize API contracts, persistence semantics, idempotency, serialization, and cross-module invariants.'),
    ('E', 'Prioritize resource limits, performance traps, lifecycle cleanup, observability, and operational failure modes.'),
]
raise SystemExit(module.main())
