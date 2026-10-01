# Canonical ProgramGraph-to-HBFSim workflow

This document defines branch `codex/qwen15-programgraph-hbfsim-20261001`.
It is a TileGen workflow and is intentionally independent of the TileGraph
co-simulation branch.

## Scope and terminology

**ProgramGraph** is the captured native workload contract: kernel launches,
CUDA memory API submissions, allocation observations, host submission order,
and exact provider bindings (process, function, code digest, raw-argument
digest, shape, and phase). It is not a GPU scheduler trace and it does not
contain every SASS instruction or lane address.

**Provider** is a checked, family-specific source program. Given a bound
kernel and its shape, it emits source memory effects to the TileGen runner.
The provider is constructed from the graph/registry and static kernel
semantics; a complete NVBit memory trace is not a production input.

**FunctionalCache** is TileGen's deterministic L1/L2 functional model. It
consumes source effects in serial caller order, applies the selected Ada
profile and `sector32` policy, and emits post-cache memory transactions.
There is no GPU issue timing, warp scheduling, MSHR timing, or stall feedback.

**Post-cache range** is a contiguous, same-cause aggregation of adjacent 32 B
records. The emitter never deduplicates traffic or invents bytes. It may emit
32/64/96/128 B reads and 32/64/96/128 B dirty writes, with 32 B alignment and
no 128 B line crossing.

## Execution path

```text
graph.json + registry.json + provider runtime
        | complete preflight (all six phases and bindings)
        v
selected timeline (complete or measured-only)
        | provider emits effects
        v
TileGen FunctionalCache (L1/L2, sector32)
        | RangeEmitter; no trace file
        v
FIFO owned by TileGen adapter
        v
programgraph_hbfsim_service -> existing HBFSim backend
```

`--phase-scope measured-only` still validates the complete graph, but sends
only all allocation metadata plus `Measured/Prefill`, `Measured/Decode1`, and
`Measured/Decode2` kernel/API nodes. The cache is cold at the beginning of
this selected execution. For Qwen2.5-1.5B P32D2 this is 388 + 321 + 321 =
**1030 measured kernels**; the 1030 warmup kernels are not sent.

The HBFSim upstream source and public interface are unchanged. The only
32 B-read admission change is in TileGen's vendored
`source/work/tilegen-hbf-drain-native-r1/native_backend.h`, where the adapter
accepts aligned 32/64/96/128 B reads just as it already accepts sector32
writes. HBFSim still receives ordinary `L2DramRequest` objects through its
existing backend.

## Reproducible commands

Build the TileGen-owned service and runner in a disposable build directory:

```bash
cmake -S . -B /tmp/tilegen-cmake -DBUILD_TESTING=OFF
cmake --build /tmp/tilegen-cmake --target programgraph_hbfsim_service
c++ -std=c++20 -O3 -DTINY_SHA_PORTABLE \
  -Illm/executor-r1 -Isource \
  -Isource/work/tilegen-full-r1/core-native-copy-r2/include \
  -Isource/work/gddr6-support/delivery-stage/sources/third_party/gtsim-prepared/third_party \
  llm/executor-r1/main.cpp -o /tmp/source-cache-runner
```

Generate the TileGen logical address map from allocation observations (this is
not a physical GPU VA mapping):

```bash
python3 llm/tools/programgraph_service_map.py \
  --graph <graph.json> --output <service-map.json>
```

Run without retaining a post-cache trace:

```bash
python3 llm/tools/run_programgraph_hbfsim.py \
  --graph <graph.json> --registry <registry.json> --runtime <extend.py> \
  --runner /tmp/source-cache-runner \
  --service /tmp/tilegen-cmake/programgraph_hbfsim_service \
  --config <hbf-config-with-hbm-replicate-symmetric-pseudo-channels=false> \
  --service-map <service-map.json> --output <run-dir> \
  --phase-scope measured-only --fast-gemv --fast-gemm
```

The wrapper uses a FIFO, writes `hbfsim-result.json`, and removes only that
run's FIFO. It must not be changed to a regular range-output file for a formal
run; a file would be a retained post-cache trace.

## Evidence and limits

The result proves that the selected ProgramGraph effects were accepted by the
TileGen FunctionalCache and that the resulting range stream was consumed by
the HBFSim backend. It does **not** prove GPU timing accuracy: there is no
closed-loop SM/warp scheduler, no MSHR model in the FunctionalCache, and no
stall feedback to the producer. HBFSim timing is a post-cache memory-service
simulation under its supplied (possibly uncalibrated) memory configuration.

The service map is a packed logical address map derived from allocation
observations. It is provenance-preserving for this run, but it is not a claim
that packed HBFSim addresses equal the RTX 4000 Ada physical hash.
