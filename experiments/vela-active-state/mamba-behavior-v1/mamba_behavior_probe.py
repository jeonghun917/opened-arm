from __future__ import annotations

import json
import os
import tempfile
import traceback
from pathlib import Path

import torch

MODEL_ID = "state-spaces/mamba-130m-hf"

FIXTURES = [
    {
        "id": "F1_CODEWORD",
        "state": "The active codeword is ALPHA. BETA is not the active codeword.",
        "probe": " If the active codeword is ALPHA answer A. If it is BETA answer B. Answer:",
        "expected": " A",
    },
    {
        "id": "F2_CORRECTION",
        "state": "The color was RED. Correction: the final color is BLUE, not RED.",
        "probe": " If the final color is BLUE answer A. If it is RED answer B. Answer:",
        "expected": " A",
    },
    {
        "id": "F3_SCOPE",
        "state": "Only the Mars record is in scope. The Venus record is explicitly out of scope.",
        "probe": " If Mars is in scope answer A. If Venus is in scope answer B. Answer:",
        "expected": " A",
    },
    {
        "id": "F4_HYPOTHESIS",
        "state": "Hypothesis one is supported by the current evidence. Hypothesis two remains unresolved.",
        "probe": " If hypothesis one is the supported one answer A. Otherwise answer B. Answer:",
        "expected": " A",
    },
    {
        "id": "F5_PLAN",
        "state": "The plan is collect evidence, then commit. Evidence collection is complete.",
        "probe": " If the next plan step is commit answer A. If it is collect answer B. Answer:",
        "expected": " A",
    },
]


def write_report(report: dict) -> None:
    result_path = os.environ.get("VELA_RESULT_PATH")
    if result_path:
        p = Path(result_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def run_cached_probe(model, probe_ids, cache, start_pos: int):
    out = None
    for j in range(probe_ids.shape[1]):
        token = probe_ids[:, j : j + 1]
        pos = torch.tensor([start_pos + j], dtype=torch.long, device=token.device)
        out = model(
            token,
            cache_params=cache,
            cache_position=pos,
            use_cache=True,
            return_dict=True,
        )
        cache = out.cache_params
    if out is None:
        raise RuntimeError("probe tokenization unexpectedly empty")
    return out.logits[:, -1].detach().float().cpu(), cache


def run():
    transformers_version = None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version

        device = "cpu"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(device).eval()

        candidate_texts = [" A", " B"]
        candidate_ids = {}
        for text in candidate_texts:
            ids = tokenizer(text, add_special_tokens=False).input_ids
            if len(ids) != 1:
                raise RuntimeError(f"candidate {text!r} is not one token: {ids}")
            candidate_ids[text] = ids[0]

        rows = []
        max_restore_diff = 0.0
        max_replay_diff = 0.0
        restored_match_count = 0
        replay_match_count = 0
        state_influence_count = 0
        acc = {"native": 0, "restored": 0, "replay": 0, "fresh": 0}
        total_prefix_tokens = 0

        with torch.no_grad():
            for fx in FIXTURES:
                state_ids = tokenizer(
                    fx["state"], return_tensors="pt", add_special_tokens=False
                ).input_ids.to(device)
                probe_ids = tokenizer(
                    fx["probe"], return_tensors="pt", add_special_tokens=False
                ).input_ids.to(device)
                total_prefix_tokens += int(state_ids.shape[1])

                pre = model(state_ids, use_cache=True, return_dict=True)
                cache = pre.cache_params
                if cache is None:
                    raise RuntimeError("Mamba prefill returned no cache")

                with tempfile.TemporaryDirectory() as td:
                    cp = Path(td) / "cache.pt"
                    torch.save(cache, cp)

                    native_logits, _ = run_cached_probe(
                        model, probe_ids, cache, int(state_ids.shape[1])
                    )
                    restored_cache = torch.load(cp, map_location="cpu", weights_only=False)
                    restored_logits, _ = run_cached_probe(
                        model, probe_ids, restored_cache, int(state_ids.shape[1])
                    )

                replay_ids = torch.cat([state_ids, probe_ids], dim=1)
                replay_logits = model(replay_ids, use_cache=False, return_dict=True).logits[:, -1].detach().float().cpu()
                fresh_logits = model(probe_ids, use_cache=False, return_dict=True).logits[:, -1].detach().float().cpu()

                restore_diff = float((native_logits - restored_logits).abs().max())
                replay_diff = float((native_logits - replay_logits).abs().max())
                max_restore_diff = max(max_restore_diff, restore_diff)
                max_replay_diff = max(max_replay_diff, replay_diff)
                if restore_diff <= 1e-5:
                    restored_match_count += 1
                if replay_diff <= 1e-4:
                    replay_match_count += 1

                conds = {}
                for name, logits in [
                    ("native", native_logits),
                    ("restored", restored_logits),
                    ("replay", replay_logits),
                    ("fresh", fresh_logits),
                ]:
                    scores = {c: float(logits[0, tid]) for c, tid in candidate_ids.items()}
                    chosen = max(scores, key=scores.get)
                    expected = fx["expected"]
                    correct = chosen == expected
                    if correct:
                        acc[name] += 1
                    other = " B" if expected == " A" else " A"
                    conds[name] = {
                        "scores": scores,
                        "chosen": chosen,
                        "correct": correct,
                        "expected_margin": scores[expected] - scores[other],
                    }

                native_margin = conds["native"]["expected_margin"]
                fresh_margin = conds["fresh"]["expected_margin"]
                if abs(native_margin - fresh_margin) > 1e-5:
                    state_influence_count += 1

                rows.append(
                    {
                        "id": fx["id"],
                        "state_tokens": int(state_ids.shape[1]),
                        "probe_tokens": int(probe_ids.shape[1]),
                        "restore_max_abs_diff": restore_diff,
                        "replay_max_abs_diff": replay_diff,
                        "native_minus_fresh_margin": native_margin - fresh_margin,
                        "conditions": conds,
                    }
                )

        n = len(FIXTURES)
        report = {
            "status": "M1_BEHAVIORAL_PROBE_ONLY",
            "model": MODEL_ID,
            "device": device,
            "torch_version": torch.__version__,
            "transformers_version": transformers_version,
            "fixture_count": n,
            "accuracy": {k: v / n for k, v in acc.items()},
            "max_native_vs_restored_logit_diff": max_restore_diff,
            "max_native_vs_replay_logit_diff": max_replay_diff,
            "restored_matches_native_fixtures": restored_match_count,
            "full_replay_matches_native_fixtures": replay_match_count,
            "fixtures_with_state_influence_on_expected_margin": state_influence_count,
            "prefix_tokens_avoided_by_checkpoint_per_full_pass": total_prefix_tokens,
            "fixtures": rows,
            "claim_boundary": (
                "Simple forced-choice probe on Mamba-1 130M. It tests behavioral preservation of serialized causal state, "
                "not VELA identity, general reasoning quality, or Mamba-3."
            ),
        }
        write_report(report)
        if restored_match_count != n:
            raise SystemExit(1)
    except BaseException as exc:
        if isinstance(exc, SystemExit) and exc.code == 0:
            raise
        report = {
            "status": "MAMBA_BEHAVIOR_PROBE_ERROR",
            "model": MODEL_ID,
            "torch_version": torch.__version__,
            "transformers_version": transformers_version,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-20:],
            "claim_boundary": "Experiment/runtime failure only; do not treat this as a Mamba architecture verdict.",
        }
        write_report(report)
        raise


if __name__ == "__main__":
    run()
