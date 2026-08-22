from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

import torch

MODEL_ID = "RWKV/rwkv-4-169m-pile"

FIXTURES = [
    {
        "id": "F1_CODEWORD",
        "prefix": "Codeword: ALPHA.\n",
        "suffix": "Codeword:",
        "candidates": [" ALPHA", " BETA"],
        "expected": " ALPHA",
    },
    {
        "id": "F2_CORRECTION",
        "prefix": "Initial codeword: RED.\nCorrection: codeword is BLUE.\n",
        "suffix": "Current codeword:",
        "candidates": [" BLUE", " RED"],
        "expected": " BLUE",
    },
    {
        "id": "F3_SCOPE",
        "prefix": "Project A token: MARS.\nProject B token: VENUS.\n",
        "suffix": "Project A token:",
        "candidates": [" MARS", " VENUS"],
        "expected": " MARS",
    },
    {
        "id": "F4_HYPOTHESIS",
        "prefix": "Option one status: ACTIVE.\nOption two status: REJECTED.\n",
        "suffix": "Active option:",
        "candidates": [" one", " two"],
        "expected": " one",
    },
    {
        "id": "F5_PLAN",
        "prefix": "Step one: collect.\nStep two: verify.\nStep three: commit.\nCompleted: collect, verify.\n",
        "suffix": "Next step:",
        "candidates": [" commit", " collect"],
        "expected": " commit",
    },
]


def clone_state(state, device):
    if state is None:
        return None
    return [x.detach().to(device).clone() for x in state]


def state_to_cpu(state):
    return [x.detach().cpu().clone() for x in state]


def score_candidate(model, tokenizer, device, initial_state, suffix: str, candidate: str) -> float:
    suffix_ids = tokenizer(suffix, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    cand_ids = tokenizer(candidate, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    if suffix_ids.shape[1] < 1 or cand_ids.shape[1] < 1:
        raise ValueError("fixture tokenization produced an empty sequence")

    # To score m candidate tokens, feed suffix + first m-1 candidate tokens.
    if cand_ids.shape[1] > 1:
        feed = torch.cat([suffix_ids, cand_ids[:, :-1]], dim=1)
    else:
        feed = suffix_ids

    with torch.no_grad():
        out = model(
            feed,
            state=clone_state(initial_state, device),
            use_cache=True,
            return_dict=True,
        )
        logp = torch.log_softmax(out.logits.detach().float(), dim=-1)

    n = suffix_ids.shape[1]
    total = 0.0
    for j in range(cand_ids.shape[1]):
        pos = n - 1 + j
        tok = int(cand_ids[0, j])
        total += float(logp[0, pos, tok])
    return total


def score_condition(model, tokenizer, device, state, fixture):
    scores = {
        cand: score_candidate(model, tokenizer, device, state, fixture["suffix"], cand)
        for cand in fixture["candidates"]
    }
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    chosen = ranked[0][0]
    expected = fixture["expected"]
    other_best = max(v for k, v in scores.items() if k != expected)
    margin = scores[expected] - other_best
    return {
        "scores": scores,
        "chosen": chosen,
        "expected": expected,
        "correct": chosen == expected,
        "expected_margin": margin,
    }


def run():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device).eval()

    rows = []
    max_restore_score_diff = 0.0
    max_replay_score_diff = 0.0
    saved_prefix_tokens = 0

    for fixture in FIXTURES:
        prefix_ids = tokenizer(
            fixture["prefix"], return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device)
        saved_prefix_tokens += int(prefix_ids.shape[1])

        with torch.no_grad():
            first = model(prefix_ids, use_cache=True, return_dict=True)
            base_state_cpu = state_to_cpu(first.state)
            replay = model(prefix_ids, use_cache=True, return_dict=True)
            replay_state_cpu = state_to_cpu(replay.state)

        with tempfile.TemporaryDirectory() as td:
            cp = Path(td) / "state.pt"
            torch.save(base_state_cpu, cp)
            restored_state_cpu = torch.load(cp, map_location="cpu")

            conditions = {
                "native": score_condition(model, tokenizer, device, base_state_cpu, fixture),
                "restored": score_condition(model, tokenizer, device, restored_state_cpu, fixture),
                "replay": score_condition(model, tokenizer, device, replay_state_cpu, fixture),
                "fresh": score_condition(model, tokenizer, device, None, fixture),
            }

        for cand in fixture["candidates"]:
            max_restore_score_diff = max(
                max_restore_score_diff,
                abs(conditions["native"]["scores"][cand] - conditions["restored"]["scores"][cand]),
            )
            max_replay_score_diff = max(
                max_replay_score_diff,
                abs(conditions["native"]["scores"][cand] - conditions["replay"]["scores"][cand]),
            )

        rows.append(
            {
                "id": fixture["id"],
                "prefix_tokens": int(prefix_ids.shape[1]),
                "conditions": conditions,
                "native_minus_fresh_margin": (
                    conditions["native"]["expected_margin"] - conditions["fresh"]["expected_margin"]
                ),
            }
        )

    accuracy = {
        cond: sum(int(r["conditions"][cond]["correct"]) for r in rows) / len(rows)
        for cond in ["native", "restored", "replay", "fresh"]
    }
    state_influence_count = sum(
        int(abs(r["native_minus_fresh_margin"]) > 1e-5) for r in rows
    )
    restore_equivalent = max_restore_score_diff <= 1e-5
    replay_equivalent = max_replay_score_diff <= 1e-5

    report = {
        "status": "M1_BEHAVIORAL_PROBE_ONLY",
        "model": MODEL_ID,
        "device": device,
        "dtype": str(dtype),
        "fixture_count": len(rows),
        "accuracy": accuracy,
        "max_native_vs_restored_score_diff": max_restore_score_diff,
        "max_native_vs_replay_score_diff": max_replay_score_diff,
        "restored_matches_native": restore_equivalent,
        "full_prefix_replay_matches_native": replay_equivalent,
        "fixtures_with_state_influence_on_expected_margin": state_influence_count,
        "prefix_tokens_avoided_by_checkpoint_per_full_pass": saved_prefix_tokens,
        "fixtures": rows,
        "interpretation": (
            "This is a deterministic next-token choice probe. It tests whether serialized RWKV recurrent state "
            "reproduces the same choice scores as native continuation, whether full prefix replay reconstructs "
            "the same state, and whether omitting pre-cut state changes the score margin."
        ),
        "claim_boundary": (
            "The fixtures are intentionally simple and the 169M base model is not treated as a VELA-quality "
            "cognitive engine. PASS is operational/behavioral state evidence, not cognitive continuity, identity, "
            "or engine-selection evidence."
        ),
    }

    result_path = os.environ.get("VELA_RESULT_PATH")
    if result_path:
        p = Path(result_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    # Gate only the matched-state mechanics, not whether this tiny base model solves every fixture.
    if not (restore_equivalent and replay_equivalent and state_influence_count > 0):
        raise SystemExit(1)
    return report


if __name__ == "__main__":
    run()
