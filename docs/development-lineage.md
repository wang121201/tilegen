# Qwen / R4 / TileGraph 开发谱系

## 唯一活动开发分支

Qwen2.5-1.5B（Qwen1.5B）工作只在 `codex/qwen15-tilegraph-cosim-20261001` 开发。除非长期工作无法在此分支共存并先在本文说明，不创建新的 Qwen branch。短期试验使用带 SHA-256 manifest 的结果目录，不以 branch 命名。

P 是 prefill token 数；D 是 decode step 数。ProgramGraph 是原生 kernel/API/allocation 提交图和 provider 注册表；FunctionalCache 是按有序 read/write effect 转移 cache 状态的非时序模型；TileGraph 是含 CTA（线程块）、warp（32 线程束）、tile 和依赖的细粒度执行图。

## 分支和归档

| 名称 | 定位 | 状态 |
| --- | --- | --- |
| `codex/qwen15-tilegraph-cosim-20261001` | Qwen TileGraph importer/input check、ProgramGraph 文档、当前 sector32 FunctionalCache | 唯一活动分支 |
| `codex/qwen15-full-cosim-20260929-r1` @ `eab70a9` | 活动分支直接祖先，2026-09-29 full-cosim 门禁 | 归档 |
| `codex/gtsim-main-dev-20261001` 与 `codex/gtsim-qwen-p32d2-1030-20261001` @ `45a996c` | 两名称同一提交，public-release/README 原型 | 归档，不再分叉 |
| `codex-runs/gtsim-ada-r4-*-20260922-r1/` | 固定 P/D 的 ProgramGraph + FunctionalCache 实验 snapshot | 不是 Git branch，只读证据 |
| `stable/ada-sector32-line` @ `be1da38` | 32 B sector cache policy 基线 | 策略祖先，不是 Qwen 结果分支 |

“R4 分支”不是有效名称。R4 的版本是 case 的 graph、registry、provider、executor、binary、policy、source-stream 和 summary 的 SHA-256 pin 闭包，而不是某一个 Git commit。

## 历史 r4 扫描报告

两份 r4 报告均为：

```text
native capture → ProgramGraph + provider → effect stream → FunctionalCache → DRAM counters
```

不是 TileGraph。它们使用 128 B read-fill/RFO、32 B dirty writeback 和 `SERIAL_CALLER_STREAM_ORDER_NOT_GPU_TIMING`；不建模 GPU issue/completion、warp overlap、MSHR 或端到端时延。

| 报告 case | 历史结果根 |
| --- | --- |
| P128/D2、D4、D8、D16 | `codex-runs/gtsim-ada-r4-p128-20260922-r1/` |
| P64/D2、P256/D2、P512/D2 | `codex-runs/gtsim-ada-r4-prefill-20260922-r1/` |
| P32/D2 | `codex-runs/gtsim-ada-sector32-20260922-r1/cases/p32-old128/` |

状态 `PASS_COMPLETE_NATIVE_GRAPH_CACHE_EXECUTION` 只表示 ProgramGraph/provider 覆盖和 FunctionalCache 会计闭合，不表示 TileGraph/GTSim 调度闭环或硬件时序通过。

## R4 的精确版本

R4 不是一个 Git commit，而是固定模型/runtime/P/D/capture contract、`graph.json`、`registry.json`、builder、`extend.py`、provider、`whole_stream.py`、`runner.h`、`candidate_direct_cache.h`、runner binary、cache policy 和结果 digest 的闭包。

P64/D2 使用 `builds/p64-r1/repo/llm/executor-r1/` snapshot；P256/P512 使用 `builds/p256-p512-r3/repo/llm/executor-r1/`；P128 前端源码在其 R4 归档中，runner binary 独立 pin。历史 snapshot 通常没有 `.git`，相同文件名不保证同版本。不得用当前 branch 代码替换 snapshot 并称为 R4 重现。

## 当前活动分支新增内容

当前活动分支从 `eab70a9` 派生，新增 Qwen P32/D2 的 TileGraph admission、compiler-IR input、非 trace 依赖 gate、KernelBinding input check、TileGraph 到 TileGen/HBFSim 开发接口和 per-kernel memory descriptor。FunctionalCache 是继承组件，不是当前分支新发明的核心；历史 R4 已使用同类 ProgramGraph 功能 cache 路径。

## 规则

1. 新 Qwen 提交只进入唯一活动分支。
2. 新 case 在 `codex-runs/` manifest 中记录 branch commit（若适用）和外部 source pin。
3. 历史 R4 manifest 必须写 `source_snapshot_not_git_branch=true`。
4. 祖先和原型以 annotated tag 固定；未获明确删除授权不得删除分支或 worktree。
5. 新建 branch 前须在本文写明无法在活动分支完成的原因、寿命、归档目标和 owner。
6. 不 reset、checkout 或覆盖用户未提交修改。

入口：ProgramGraph 见 `docs/programgraph-functional-workflow.md`；TileGraph 见 `docs/qwen15-tilelang-tilegraph.md` 和 `docs/qwen15-full-cosim-tilegraph-workflow.md`；R4 从对应 `cases/.../status.json` 的 SHA-256 pin 追溯。
