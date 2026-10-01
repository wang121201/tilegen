# P32D2 sector32 ProgramGraph-to-HBFSim acceptance

## Identity and scope

This record belongs to TileGen branch
`codex/qwen15-programgraph-hbfsim-20261001`, commit `34558d3`. The workload
is Qwen2.5-1.5B, batch 1, `Prefill=32`, `Decode=2`, bfloat16, native eager,
with the complete six-phase graph validated but only the three Measured phases
executed. “Measured-only” means no Warmup kernel/API effects are sent to the
cache runner; allocation metadata is retained so the cache address contract is
valid. The expected measured kernel count is 388 + 321 + 321 = **1030**.

Run directory:

`/home/xmu/nvidiagds/codex-runs/qwen-programgraph-sector32-20261001/canonical-1030-hbfsim-r8`

The post-cache stream used a FIFO. No complete post-cache trace file was
retained (`trace_retained=false`, `postcache_trace_retained=false`).

## Results

| Check | Observed |
|---|---:|
| ProgramGraph measured kernels | 1030 |
| Allocation metadata records | 105 |
| CUDA memory API records | 32 |
| Source command records | 621,173 |
| FunctionalCache DRAM read requests / bytes | 288,090,411 / 9,218,893,152 B |
| FunctionalCache dirty-write requests / bytes | 2,018,245 / 64,583,840 B |
| HBFSim submitted merged ranges | 203,405,057 |
| HBFSim raw sector records represented by ranges | 290,108,656 |
| HBFSim physical read bytes | 9,218,893,152 B |
| HBFSim physical write bytes | 64,583,840 B |
| Producer return code | 0 |
| HBFSim service return code | 0 |

The byte equality between FunctionalCache and HBFSim is an interface/stream
closure check. It does not establish RTX 4000 Ada timing accuracy. The HBFSim
configuration is an uncalibrated GDDR6-targeted memory-service model, and the
result explicitly claims no GPU stall feedback or SM/warp timing.

## Historical HTML relationship

The supplied `qwen1p5b-r4-prefill-sweep` HTML lists the historical P32/D2
write-error columns as Full **-6.88%**, Prefill **-7.87%**, Decode1 **-14.01%**,
and Decode2 **-8.30%**. Those columns are the historical r4 comparison and are
not HBFSim physical-byte measurements; the HTML itself states that object-level
DRAM attribution and HBFSim co-simulation were unavailable. Therefore this
acceptance record must not convert those percentages into a hardware timing or
bandwidth error for the new HBFSim run.
