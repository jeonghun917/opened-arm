from __future__ import annotations

import json
import tempfile
from pathlib import Path

import soar_sml as sml


def marker_value(agent):
    value = agent.GetInputLink().GetParameterValue("vela-marker")
    return None if value is None else str(value)


def shutdown(kernel):
    if kernel is not None:
        try:
            kernel.Shutdown()
        except Exception:
            pass


def run() -> dict:
    k1 = k2 = k3 = None
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        rule_file = td_path / "vela-rule.soar"
        save_file = td_path / "saved-agent.soar"
        rule_file.write_text(
            "sp {vela*scope-marker\n"
            "   (state <s> ^superstate nil)\n"
            "-->\n"
            "   (<s> ^vela-rule-loaded yes)\n"
            "}\n",
            encoding="utf-8",
        )

        try:
            # A. Same-process reference.
            k1 = sml.Kernel.CreateKernelInNewThread()
            a1 = k1.CreateAgent("native")
            if not a1.LoadProductions(str(rule_file)):
                raise RuntimeError("failed to load fixture production")
            a1.GetInputLink().CreateStringWME("vela-marker", "active")
            native_marker = marker_value(a1)
            save_output = a1.ExecuteCommandLine(f"save agent {save_file}")
            save_created = save_file.exists() and save_file.stat().st_size > 0

            # B. Fresh runtime restored only from Soar's ordinary save-agent artifact.
            k2 = sml.Kernel.CreateKernelInNewThread()
            a2 = k2.CreateAgent("restored")
            source_output = a2.ExecuteCommandLine(f"source {save_file}") if save_created else ""
            saved_agent_marker = marker_value(a2)
            production_print = a2.ExecuteCommandLine("print vela*scope-marker")
            production_restored = "vela*scope-marker" in production_print

            # C. Fresh runtime with matched external working-state replay.
            k3 = sml.Kernel.CreateKernelInNewThread()
            a3 = k3.CreateAgent("replayed")
            if save_created:
                a3.ExecuteCommandLine(f"source {save_file}")
            a3.GetInputLink().CreateStringWME("vela-marker", "active")
            replay_marker = marker_value(a3)

            report = {
                "candidate": "Soar 9.6.5",
                "test": "save-agent scope vs live working memory",
                "native_working_marker": native_marker,
                "save_agent_file_created": save_created,
                "saved_agent_restores_production": production_restored,
                "saved_agent_restores_working_marker": saved_agent_marker == "active",
                "transcript_replay_restores_working_marker": replay_marker == "active",
                "expected_scope_pattern": False,
                "save_output_tail": save_output[-300:] if save_output else "",
                "source_output_tail": source_output[-300:] if source_output else "",
                "interpretation": (
                    "Soar save-agent preserves at least procedural agent content, but the live input-link "
                    "working-memory marker is absent in a fresh runtime unless that working state is replayed."
                ),
                "claim_boundary": "State-scope evidence only; not a full Soar checkpoint test or VELA continuity proof.",
            }
            report["expected_scope_pattern"] = bool(
                native_marker == "active"
                and save_created
                and production_restored
                and saved_agent_marker != "active"
                and replay_marker == "active"
            )
            print(json.dumps(report, indent=2, ensure_ascii=False))
            if not report["expected_scope_pattern"]:
                raise SystemExit(1)
            return report
        finally:
            shutdown(k1)
            shutdown(k2)
            shutdown(k3)


if __name__ == "__main__":
    run()
