#!/usr/bin/env python3
"""Fail-closed admission for the Qwen2.5-1.5B P32/D2 1030-kernel TileGraph.

This tool accepts only the measured execution window: Prefill has 388 native
kernels, Decode1 has 321, and Decode2 has 321.  A native kernel is eligible
for typed TileGraph construction only when its graph record reports both an
existing complete native memory program and a typed kernel-operand ABI.

The result is an admission receipt, not a simulator result.  It contains no
memory trace and never reports timing, traffic, or hardware accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_KERNELS = {
    "Measured/Prefill": 388,
    "Measured/Decode1": 321,
    "Measured/Decode2": 321,
}
EXPECTED_MEMORY_APIS = {
    "Measured/Prefill": 20,
    "Measured/Decode1": 6,
    "Measured/Decode2": 6,
}
EXPECTED_CASE_ID = "qwen25_1p5b-p32-d2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fail(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def counter_by_phase(nodes: list[dict[str, Any]], kind: str) -> dict[str, int]:
    return dict(
        Counter(
            node.get("phase")
            for node in nodes
            if node.get("kind") == kind and node.get("phase") is not None
        )
    )


def evaluate(graph_path: Path) -> dict[str, Any]:
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    fail(graph.get("schema") == "TILEGEN_QWEN_MEASURED_ONLY_TILEGRAPH_V1",
         "unsupported measured TileGraph schema")
    contract = graph.get("input_contract")
    fail(isinstance(contract, dict), "missing input contract")
    fail(contract.get("case_id") == EXPECTED_CASE_ID,
         "wrong workload identity; expected Qwen2.5-1.5B B1/P32/D2")
    nodes = graph.get("nodes")
    fail(isinstance(nodes, list), "nodes must be an array")
    ids = [node.get("id") for node in nodes]
    fail(all(isinstance(item, str) and item for item in ids), "invalid node id")
    fail(len(ids) == len(set(ids)), "duplicate node id")

    kernel_counts = counter_by_phase(nodes, "native_kernel")
    api_counts = counter_by_phase(nodes, "memory_api_submission")
    fail(kernel_counts == EXPECTED_KERNELS,
         f"wrong 1030-kernel phase denominator: {kernel_counts}")
    fail(api_counts == EXPECTED_MEMORY_APIS,
         f"wrong measured memory-API denominator: {api_counts}")
    native_kernels = [node for node in nodes if node.get("kind") == "native_kernel"]
    fail(len(native_kernels) == sum(EXPECTED_KERNELS.values()),
         "native kernel total is not 1030")

    untyped: list[dict[str, Any]] = []
    complete_programs = 0
    typed_operands = 0
    for node in native_kernels:
        program = node.get("memory_program")
        views = node.get("module_view_binding")
        complete = isinstance(program, dict) and program.get("complete_native_memory_program") is True
        typed = isinstance(views, dict) and views.get("kernel_operand_ABI_typed") is True
        complete_programs += int(complete)
        typed_operands += int(typed)
        if not (complete and typed):
            untyped.append(
                {
                    "id": node["id"],
                    "phase": node["phase"],
                    "memory_program_status": program.get("status") if isinstance(program, dict) else None,
                    "memory_program_kind": program.get("kind") if isinstance(program, dict) else None,
                    "kernel_operand_ABI_typed": typed,
                }
            )

    status = (
        "ADMITTED_FOR_TYPED_TILEGRAPH_CONSTRUCTION"
        if not untyped
        else "REJECTED_UNTYPED_NATIVE_KERNELS"
    )
    return {
        "schema": "TILEGEN_QWEN1030_FULL_COSIM_ADMISSION_V1",
        "status": status,
        "claim_boundary": (
            "This is a pre-construction admission receipt. It neither executes "
            "GTSim/HBFSim nor establishes timing, traffic, cache, or hardware accuracy."
        ),
        "source_graph": {
            "path": str(graph_path.resolve()),
            "bytes": graph_path.stat().st_size,
            "sha256": sha256(graph_path),
        },
        "input_contract": {
            "case_id": contract["case_id"],
            "model_key": contract.get("model_key"),
            "batch_size": contract.get("batch_size"),
            "prefill_length": contract.get("prefill_length"),
            "decode_steps": contract.get("decode_steps"),
        },
        "denominator": {
            "native_kernels_by_phase": kernel_counts,
            "native_kernels_total": len(native_kernels),
            "memory_api_submissions_by_phase": api_counts,
            "memory_api_submissions_total": sum(api_counts.values()),
            "allocation_metadata_nodes": sum(
                node.get("kind") == "allocation_API_observation" for node in nodes
            ),
            "warmup_or_external_native_kernels_in_execution": 0,
        },
        "typed_evidence": {
            "complete_native_memory_programs": complete_programs,
            "kernel_operand_ABI_typed": typed_operands,
            "missing_or_partial_native_kernel_count": len(untyped),
            "first_missing_or_partial_native_kernels": untyped[:16],
        },
        "intermediate_trace_retained": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True,
                        help="measured-only TileGraph JSON")
    parser.add_argument("--output", type=Path, required=True,
                        help="new admission-receipt JSON; an existing file is rejected")
    args = parser.parse_args()
    try:
        fail(args.graph.is_file(), "graph file is missing")
        fail(not args.output.exists(), "refusing to overwrite admission receipt")
        receipt = evaluate(args.graph)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 0 if receipt["status"] == "ADMITTED_FOR_TYPED_TILEGRAPH_CONSTRUCTION" else 2
    except Exception as error:
        print(json.dumps({"status": "REJECTED_INVALID_ADMISSION_INPUT", "reason": str(error)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
