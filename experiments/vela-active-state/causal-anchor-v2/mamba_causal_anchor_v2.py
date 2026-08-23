from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import traceback
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parents[1]
V4_PATH = BASE / "learned-upgrade-v4" / "mamba_upgrade_decision_fidelity.py"
FA_PATH = BASE / "functional-approx-v1" / "mamba_functional_approx.py"

spec = importlib.util.spec_from_file_location("vela_v4_anchor", V4_PATH)
v4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v4)
spec2 = importlib.util.spec_from_file_location("vela_fa_anchor", FA_PATH)
fa = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(fa)
v3 = v4.v3

INTERVALS = [4, 8, 16, 32]

FIXTURES = [
    {
        "id": "early",
        "project": "Orion",
        "old": "ALPHA", "new": "BETA",
        "segments": [
            "Project Orion remains active. ",
            "The old codeword was ALPHA. ",
            "Correction: the current codeword is BETA, not ALPHA. ",
            "Unrelated telemetry packet 17 was archived. ",
            "Verification is incomplete. ",
            "A historical memo mentions ALPHA but is obsolete. ",
            "Hypothesis one is weakened. ",
            "External action remains blocked."
        ],
        "critical_segment": 2,
    },
    {
        "id": "middle",
        "project": "Helios",
        "old": "BLUE", "new": "RED",
        "segments": [
            "Project Helios remains active. ",
            "The old codeword was BLUE. ",
            "Unrelated telemetry packet 21 was archived. ",
            "Verification is incomplete. ",
            "Correction: the current codeword is RED, not BLUE. ",
            "A historical memo mentions BLUE but is obsolete. ",
            "Hypothesis one is weakened. ",
            "External action remains blocked."
        ],
        "critical_segment": 4,
    },
    {
        "id": "late",
        "project": "Icarus",
        "old": "LOW", "new": "HIGH",
        "segments": [
            "Project Icarus remains active. ",
            "The old codeword was LOW. ",
            "Unrelated telemetry packet 31 was archived. ",
            "Verification is incomplete. ",
            "Hypothesis one is weakened. ",
            "A historical memo mentions LOW but is obsolete. ",
            "Correction: the current codeword is HIGH, not LOW. ",
            "External action remains blocked."
        ],
        "critical_segment": 6,
    },
    {
        "id": "distractor_heavy",
        "project": "Juno",
        "old": "EAST", "new": "WEST",
        "segments": [
            "Project Juno remains active. ",
            "The old codeword was EAST. ",
            "A note says weather moved eastward; this is unrelated to the codeword. ",
            "Verification is incomplete. ",
            "A historical memo mentions EAST but is obsolete. ",
            "Two unrelated sensors were recalibrated. ",
            "Correction: the current codeword is WEST, not EAST. ",
            "Hypothesis one is weakened. ",
            "External action remains blocked."
        ],
        "critical_segment": 6,
    },
]


def write_report(report):
    pth = os.environ.get("VELA_RESULT_PATH")
    if pth:
        p = Path(pth); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def save_xproj(model):
    return {name: p.detach().clone() for name, p in model.named_parameters() if ".mixer.x_proj.weight" in name}


def load_xproj(model, snap):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in snap:
                p.copy_(snap[name])


def train_upgrade(model, tok):
    trainable = []
    for p in model.parameters(): p.requires_grad_(False)
    for name, p in model.named_parameters():
        if ".mixer.x_proj.weight" in name:
            p.requires_grad_(True); trainable.append(p)
    if not trainable: raise RuntimeError("no x_proj weights found")
    opt = torch.optim.AdamW(trainable, lr=v3.LR, weight_decay=0.0)
    order = list(range(len(v3.TRAIN))); losses = []
    for epoch in range(v3.EPOCHS):
        random.Random(v3.SEED + epoch).shuffle(order); model.train(); total = 0.0
        for idx in order:
            prompt, gold = v3.TRAIN[idx]
            opt.zero_grad(set_to_none=True)
            loss = v3.train_loss(model, tok, prompt, gold)
            loss.backward(); torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step(); total += float(loss.detach())
        losses.append(total / len(order))
    model.eval(); return losses


def boundaries(tok, segments):
    starts, ends = [], []
    for i in range(len(segments)):
        pre = "".join(segments[:i])
        through = "".join(segments[:i+1])
        starts.append(int(tok(pre, return_tensors="pt", add_special_tokens=False).input_ids.shape[1]) if pre else 0)
        ends.append(int(tok(through, return_tensors="pt", add_special_tokens=False).input_ids.shape[1]))
    return starts, ends


def prefix_cache(model, ids, pos):
    if pos == 0: return None
    with torch.no_grad():
        return v4.clone_cache(model(ids[:, :pos], use_cache=True, return_dict=True).cache_params)


def run_slice(model, ids, start, end, cache):
    if end <= start: return v4.clone_cache(cache), None
    with torch.no_grad():
        c, out = v3.run_tokens_with_cache(model, ids[:, start:end], v4.clone_cache(cache), start)
    return v4.clone_cache(c), out


def probe_specs(fx):
    return [
        ("codeword", "\nCurrent codeword:", [" " + fx["new"], " " + fx["old"]], " " + fx["new"]),
        ("verification", "\nVerification status:", [" incomplete", " complete"], " incomplete"),
        ("project", f"\nProject {fx['project']} status:", [" active", " paused"], " active"),
        ("action", "\nExternal action is:", [" blocked", " allowed"], " blocked"),
    ]


def evaluate_migration(model, tok, ids, start_cache, start_pos, native, T, specs):
    migrated, _ = run_slice(model, ids, start_pos, T, start_cache)
    native_rows = fa.score_probe_specs(model, tok, native, T, specs)
    rows = fa.score_probe_specs(model, tok, migrated, T, specs)
    return {
        "anchor_pos": start_pos,
        "replayed_tokens": T - start_pos,
        "state_error_vs_w2_native": v3.cache_distance(migrated, native),
        "functional_vs_w2_native": fa.compare_rows(rows, native_rows),
        "probe_rows": rows,
    }


def run():
    transformers_version = None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version
        torch.manual_seed(v3.SEED); random.seed(v3.SEED)
        tok = AutoTokenizer.from_pretrained(v3.MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(v3.MODEL_ID, torch_dtype=torch.float32).cpu().eval()

        # Freeze actual W1 checkpoint material before any learning.
        fixture_data = []
        all_needed_positions = {}
        for fx in FIXTURES:
            text = "".join(fx["segments"])
            ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
            T = int(ids.shape[1]); starts, ends = boundaries(tok, fx["segments"])
            positions = set(starts + [T])
            for k in INTERVALS:
                positions.update(range(0, T + 1, k)); positions.add(T)
            all_needed_positions[fx["id"]] = sorted(p for p in positions if 0 <= p <= T)
            caches = {p: prefix_cache(model, ids, p) for p in all_needed_positions[fx["id"]]}
            fixture_data.append({"fx": fx, "text": text, "ids": ids, "T": T, "starts": starts, "ends": ends, "w1_caches": caches})

        baseline = v3.evaluate(model, tok)
        w1_weights = save_xproj(model)
        losses = train_upgrade(model, tok)
        w2_weights = save_xproj(model)
        after = v3.evaluate(model, tok)

        rows = []
        detector_hits = 0
        sparse_success = {str(k): 0 for k in INTERVALS}
        for data in fixture_data:
            fx, ids, T = data["fx"], data["ids"], data["T"]
            starts, ends, w1_caches = data["starts"], data["ends"], data["w1_caches"]

            load_xproj(model, w2_weights)
            with torch.no_grad():
                native = v4.clone_cache(model(ids, use_cache=True, return_dict=True).cache_params)
            native_probe = fa.score_probe_specs(model, tok, native, T, probe_specs(fx))

            # Upgrade-sensitivity detector: each segment is run twice from the SAME actual W1 pre-event state,
            # once with W1 transition dynamics and once with W2. No W2-native prefix is used for scoring.
            sensitivity = []
            for i, (s, e) in enumerate(zip(starts, ends)):
                start_cache = w1_caches[s]
                load_xproj(model, w1_weights)
                old_after, old_out = run_slice(model, ids, s, e, start_cache)
                load_xproj(model, w2_weights)
                new_after, new_out = run_slice(model, ids, s, e, start_cache)
                state_rms = v3.cache_distance(old_after, new_after)["rms"]
                logit_rms = None
                if old_out is not None and new_out is not None:
                    a = old_out.logits[:, -1].detach().float(); b = new_out.logits[:, -1].detach().float()
                    logit_rms = float(torch.sqrt(torch.mean((a-b)*(a-b))))
                score = state_rms * (1.0 + math.log1p(max(logit_rms or 0.0, 0.0)))
                sensitivity.append({"segment": i, "start": s, "end": e, "tokens": e-s, "state_rms": state_rms, "terminal_logit_rms": logit_rms, "score": score, "text": fx["segments"][i]})
            sensitivity.sort(key=lambda x: x["score"], reverse=True)
            predicted = sensitivity[0]["segment"]
            detector_hit = predicted == fx["critical_segment"]
            detector_hits += int(detector_hit)
            detected_pos = starts[predicted]
            oracle_pos = starts[fx["critical_segment"]]

            load_xproj(model, w2_weights)
            exact_detected = evaluate_migration(model, tok, ids, w1_caches[detected_pos], detected_pos, native, T, probe_specs(fx))
            oracle = evaluate_migration(model, tok, ids, w1_caches[oracle_pos], oracle_pos, native, T, probe_specs(fx))

            sparse = {}
            for k in INTERVALS:
                saved = list(range(0, T + 1, k))
                anchor = max([p for p in saved if p <= detected_pos], default=0)
                if anchor not in w1_caches:
                    raise RuntimeError(f"missing W1 checkpoint fixture={fx['id']} k={k} anchor={anchor}")
                res = evaluate_migration(model, tok, ids, w1_caches[anchor], anchor, native, T, probe_specs(fx))
                res["checkpoint_interval"] = k
                res["stored_checkpoint_count"] = len(saved)
                res["detector_anchor_pos"] = detected_pos
                sparse[str(k)] = res
                if res["functional_vs_w2_native"]["decision_agreement"] == 1.0:
                    sparse_success[str(k)] += 1

            rows.append({
                "fixture": fx["id"], "history_tokens": T,
                "critical_segment_oracle": fx["critical_segment"],
                "detected_segment": predicted, "detector_hit": detector_hit,
                "top_sensitivity": sensitivity[:4],
                "w2_native_probe": native_probe,
                "exact_detected_anchor": exact_detected,
                "oracle_anchor": oracle,
                "uniform_sparse_checkpoints": sparse,
            })

        n = len(rows)
        report = {
            "status": "VELA_CAUSAL_ANCHOR_SPARSITY_AND_DETECTOR_V2",
            "model": v3.MODEL_ID,
            "torch_version": torch.__version__, "transformers_version": transformers_version,
            "capability": {
                "baseline_accuracy": baseline["accuracy"], "after_accuracy": after["accuracy"],
                "baseline_correction_accuracy": baseline["correction_accuracy"], "after_correction_accuracy": after["correction_accuracy"],
                "baseline_control_accuracy": baseline["control_accuracy"], "after_control_accuracy": after["control_accuracy"],
                "epoch_loss": losses,
            },
            "detector": {
                "definition": "Rank each event by W1-vs-W2 transition divergence when both process that event from the same actual W1 pre-event checkpoint.",
                "top1_critical_event_accuracy": detector_hits / n,
            },
            "sparsity": {
                "intervals_tokens": INTERVALS,
                "full_functional_agreement_rate_by_interval": {k: sparse_success[k] / n for k in sparse_success},
            },
            "fixtures": rows,
            "success_definition": "A useful protocol should identify upgrade-sensitive causal events without consulting W2-native prefix state, then recover W2-native functional decisions from sparse actual W1 checkpoints with bounded replay.",
            "claim_boundary": "Small synthetic Mamba-130M correction experiment. Oracle critical-event labels are used only for evaluation, not detector scoring. Functional agreement on four probes is not identity proof or broad reasoning evidence.",
        }
        write_report(report)
    except BaseException as exc:
        write_report({"status":"VELA_CAUSAL_ANCHOR_V2_ERROR","model":getattr(v3,"MODEL_ID",None),"torch_version":torch.__version__,"transformers_version":transformers_version,"error_type":type(exc).__name__,"error":str(exc),"traceback_tail":traceback.format_exc().splitlines()[-40:]}); raise

if __name__ == "__main__": run()
