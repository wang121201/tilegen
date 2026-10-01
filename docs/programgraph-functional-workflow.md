# ProgramGraph 功能流量工作流

## 1. 目的、范围与结论

本文是分支 `codex/qwen15-tilegraph-cosim-20261001` 的当前入口说明。它定义固定模型、固定 prefill/decode（P/D）工作负载如何经 ProgramGraph 生成 TileGen 的**功能性**内存流量，并把该路径与 pure GTSim TileGraph 执行严格分开。

P 是 prefill token count（前填 token 数）；D 是 decode step count（连续解码步数）。ProgramGraph 是一次固定会话的 host-side kernel/memory-API 提交索引，加上已验证 kernel family 的全局访存程序注册表。effect stream 是 provider 按该索引产生的逻辑 read/write effects。FunctionalCache 是按明确 cache policy 和调用顺序消费 effect stream 的非时序 cache 投影。TileGraph 是含 CTA（CUDA cooperative thread array，线程块）、warp（32 线程束）、tile、数据依赖和循环依赖的细粒度图。GTSim 是 TileGen 中消费 TileGraph、建模节点 issue/completion 的执行器。

**当前结论：** ProgramGraph 路径可以稳定地回答“在声明的 functional cache 假设下，固定 P/D 会产生多少 cache-projected DRAM read/write bytes”；它不能回答端到端 GPU latency、warp overlap、带宽利用率或硬件精度。TileGraph 路径才保留这些执行语义，但当前 Qwen P32D2 仅有 importer/input-check 证据，未完成全量耦合执行。

## 2. 两条路径及其边界

```text
固定模型、运行时、P/D
        │
        ├─ native capture → ProgramGraph + provider registry
        │                    → effect stream → FunctionalCache → counters
        │                                                  └→ optional post-cache trace
        │
        └─ TileLang/TIR templates + descriptors → TileGraph → GTSim fine execution
                                                        → L1/L2 → optional memory completion feedback
```

| 项目 | ProgramGraph 功能流量路径 | pure GTSim / TileGraph 路径 |
| --- | --- | --- |
| 输入语义 | 已观测 launch、API/allocation 生命周期、严格绑定的 provider | tile/CTA/warp 节点、依赖、资源和 memory descriptor |
| 执行顺序 | `SERIAL_CALLER_STREAM_ORDER_NOT_GPU_TIMING` | 由依赖、驻留、issue 和 completion 决定 |
| 保留 | read/write effect、已选 cache policy、cache admission/eviction、可选 cache 后请求序列 | ProgramGraph 所保留内容，加上 CTA/warp 调度、依赖、在途请求和 completion feedback |
| 简化/移除 | GPU issue/completion、warp overlap、MSHR/in-flight 行为、GPU 时钟、计算 stall | 无法自动从 host capture 推导的 native kernel 算法仍需要 TileGraph 模板/IR |
| 合法输出 | 功能性 bytes/counters；有条件的 post-cache trace | 依赖/调度模型结果；若后端闭环运行，可报告模型周期，但仍非自动硬件精度 |
| 禁止声称 | GPU timing、真实 bandwidth、端到端 inference time、硬件准确 | 即使运行完成，也须独立校准与硬件验证 |

两条路径应共享模型/算子语义、内存对象与地址物化器；不得声称因 cache 名称相同而产生逐条相同的流量。固定 caller order 与实际 issue order 不同，因而 LRU、合并和写回可不同。

## 3. ProgramGraph 的规范构建与准入

1. **固定契约。** Pin 模型权重、runtime、batch、P/D、采样、CUDA 环境和 ROI（region of interest，测量区间）。
2. **闭合 capture。** 收集 kernel launch、function ID/code hash、raw argument hash、CUDA memory API、allocation/free 生命周期及静态元数据。capture 证明观察到的调用，不自动证明 operand/address 语义。
3. **构造结构图。** 只生成 `native_kernel`、`memory_api_submission`、`allocation_API_observation` 和 host/context/stream submission-order 边；不得伪造 GPU completion、issue-cycle、stall 或数据依赖边。
4. **绑定 provider。** 每一 in-scope kernel 必须与 process、function ID、code SHA-256、raw-argument SHA-256、shape 和对象 binding 严格匹配。没有 provider 必须 fail closed，不能估计或静默跳过。
5. **integrate 与 pin。** 仅在全覆盖后登记 `callable_current_native_global_program`；对 graph、registry、runtime、executor 和结果记录 path、bytes、SHA-256。integrated 表示“捕获图已接到 provider”，不表示已构成 CTA/warp TileGraph。
6. **执行和封存。** `provider.events()` 产生 effect stream，FunctionalCache 输出 counters；若导出 trace，须同时记录其格式、policy、顺序和最后 dirty flush 策略。通过条件是无 fallback、全覆盖、输入/输出 pin 和会计守恒。

这是一条**系统化而非自动泛化**的流程。任意新模型或新 P/D 都必须重新 capture 并重新检查 kernel 集合、ABI、shape、对象生命周期及 provider 覆盖；新 kernel family 通常要求新增/审计 provider。

## 4. 当前项目组织

| 位置 | 职责 | 当前读法 |
| --- | --- | --- |
| `llm/executor-r1/whole_stream.py` | ProgramGraph/registry preflight 和 effect-stream driver | ProgramGraph 路径的入口，不是 GTSim scheduler |
| `llm/executor-r1/runner.h`、`main.cpp` | FunctionalCache 配置与 counters/可选 trace 输出 | 明示 serial caller order 和非 timing 范围 |
| `llm/tools/qwen15_tilelang_tilegraph.py` | Qwen P32D2 TileGraph builder | 生成细粒度图工件，不从 native capture 自动反推 |
| `source/tilegraph_input_check.cpp` | TileGraph schema/import smoke check | 当前仅验证 importer 接受；不是 GPU/HBFSim 完整运行 |
| `docs/qwen15-full-cosim-tilegraph-workflow.md` | 历史 full-cosim 证据 | 不应当作当前 ProgramGraph 的默认指南 |
| `docs/ada-sector32-stable-release-20260924.md` | sector32 cache 组件及合同历史 | 组件/策略参考，非 P64D2 新结果 |

## 5. 32 B 事务状态

当前 `llm/executor-r1/runner.h` 的默认 `TILEGEN_L2_DATA_POLICY=sector32` 配置为：L2 read fill / read-for-ownership（RFO）为 **32 B**，writeback request 为 **32 B**，并采用 lazy sector known-byte store policy。这里的 32 B 是 FunctionalCache 的请求粒度；它不是已恢复的硬件 memory-controller transaction，也不自动保证与 NCU counter 一致。

**重要限制：** 现有 post-cache trace exporter 仍要求 `TILEGEN_L2_DATA_POLICY=old128`，因为该 trace schema 尚不能表达 32 B read fills。故“FunctionalCache 默认 32 B”与“已经有 32 B post-cache trace/HBFSim replay”是两件不同的事。任何以旧 trace 得到的 DRAM 结果都必须标为 128 B read-fill / 32 B writeback，不能冒充新的 sector32 结果。

## 6. P64D2：历史 strongest evidence 与硬件对照

历史 P64D2 完整 ProgramGraph 功能 cache 执行的收据为 `PASS_COMPLETE_NATIVE_GRAPH_CACHE_EXECUTION`。其输入契约是 Qwen2.5-1.5B、B1、BF16、P64D2，且有 graph/registry/runtime/executor 的 SHA-256 pin。它不是 current sector32 run：其实际 cache 配置为 **128 B read fill/RFO、32 B writeback、serial caller order、no terminal dirty flush**。

同一固定 P64D2 全范围的硬件 NCU（NVIDIA Nsight Compute）ROI 与该历史功能 cache 投影可作**方向性流量对照**。两者都以字节为单位；NCU 的 `dram__bytes_*` 是真实硬件计数器，模型值是 FunctionalCache 结果，不能比较为 latency 或带宽精度。

| Full ROI | 历史 FunctionalCache | NCU hardware | 模型 − 硬件 | 相对硬件 |
| --- | ---: | ---: | ---: | ---: |
| DRAM read bytes | 9,375,131,520 | 9,252,321,280 | +122,810,240 | +1.33% |
| DRAM write bytes | 147,275,296 | 151,476,096 | −4,200,800 | −2.77% |
| Read + write bytes | 9,522,406,816 | 9,403,797,376 | +118,609,440 | +1.26% |

这些数字说明该**历史 128 B-read policy**的 aggregate byte projection 与同工作负载 NCU 的全范围字节接近；它们不证明 32 B policy 也同样接近。当前没有已封存的 P64D2 sector32 ProgramGraph run，因此不能把上表升级为“最新 32 B 结果”。要建立这个结论，需要在现有 graph/registry 的严格 preflight 后，以 sector32 重跑，不导出旧格式 trace，记录新的 counters 与 pin，再与同一 NCU ROI 比较。

## 7. 可复现操作顺序

1. 先执行 graph/registry/runtime preflight；确认 coverage 为完整且所有 pin 一致。
2. 选择唯一 policy：只做最新流量时使用 `TILEGEN_L2_DATA_POLICY=sector32`；需要旧 post-cache trace 回放时显式选择 `old128`，并在结果中写明该限制。
3. 运行 `whole_stream.py`，保存 result/status/cache summary；默认不保留全 effect trace。需要 trace 时只在新 schema 已支持 32 B read fill 后启用。
4. 将模型 counters 与同 P/D、同 ROI、同软件/硬件工作负载契约的 NCU `dram__bytes_read.sum` / `dram__bytes_write.sum` 分方向比较；不得把 Prefill、D1、D2 等独立 ROI 相加伪造 Full。
5. 若目标转为调度、stall 或 end-to-end 时间，停止把 ProgramGraph 结果当作答案，构建并执行 TileGraph/GTSim 闭环。

## 8. 为什么此路径有方法学依据

ProgramGraph 不是把 GTSim 任意删减后得到的临时工程捷径。它属于 **trace/effect-driven functional cache simulation（由访存事件驱动的功能 cache 仿真）**：先把目标明确限定为“指定 cache machine 对固定工作负载会输出哪些流量”，再只保留这一问题所需的状态。

对一个确定性的 FunctionalCache，设初始 cache 状态、cache policy 和按序输入的全部 cache-relevant memory effects 都相同，则其状态转移、hit/miss、fill、eviction、dirty writeback 与 counters 都相同。换言之，有序 effect stream 是这个 **已声明 cache 抽象** 的充分输入；该结论不等价于真实 GPU 或端到端执行等价。

经典 storage-hierarchy 方法已将 address/reference string 驱动 cache 状态机作为计算访问统计的基础；Mattson 等人讨论了把典型程序地址序列在模拟存储层次上运行以得到各层访问统计的方法。[Mattson et al., “Evaluation Techniques for Storage Hierarchies,” 1970](https://doi.org/10.1147/SJ.92.0078)。现代 GPU 模拟框架同样区分 execution-driven 与 trace-driven 前端，并以硬件 counter 的逐项对照来验证模型，而非把 trace 模式自动当作硬件真值；见 [Accel-Sim](https://mkhairy.github.io/Docs/Accel-Sim.pdf)。

因此，该路径的严谨性来自一个带条件的 refinement（精化）关系：

```text
已验证 kernel 语义 + binding
        → ordered read/write effects
        → FunctionalCache state transition
        → cache-projected DRAM requests/counters
```

若 provider 对目标 workload 的每一个 in-scope kernel 都产生了与抽象目标相同的 effect，且顺序、初态、policy、粒度相同，则 FunctionalCache 的结果对该抽象是精确的。反之，code SHA、kernel 名称或 launch journal 只能防止 provider 被错用，**不能单独证明 provider 的地址语义正确**。新 family、shape、runtime 或 P/D 必须重新完成语义验证。

### 8.1 为什么 effect 必须带 read/write

effect 不是“已经发往 DRAM 的事务”，而是算子语义层的 memory action：

```text
read effect  = object/address range + size + read
write effect = object/address range + size + write
```

读和写会驱动不同的 cache 状态转移。read 可能触发 lookup/miss/read fill；write 还会涉及 allocation 或 bypass、dirty-sector 更新、read-for-ownership（RFO）和随后 writeback。因此不能只保存地址而丢掉方向。当前 sector32 policy 正是在 FunctionalCache 消费 effect 后，决定 32 B read fill/RFO 与 32 B writeback；effect 本身不应被误称为 DRAM transaction。

### 8.2 精度、简化与速度的四层门禁

| 层级 | 必须成立的条件 | 可得到的结论 | 不能得到的结论 |
| --- | --- | --- | --- |
| 身份层 | workload、capture、code/ABI、object binding 和输入 pin 一致 | provider 没有被明显错配 | 地址语义正确 |
| 语义层 | provider 有独立公式/oracle、全覆盖和 fail-closed 验证 | effect 对目标抽象可信 | GPU issue/time 正确 |
| cache 层 | 顺序、初态、policy、粒度、dirty flush 与会计守恒明确 | cache-projected counters 对该 cache machine 正确 | 硬件 counters 自动正确 |
| 硬件层 | 同 P/D、同 ROI、同 workload 契约下与 NCU directionally 对照 | 特定 metric 的外部误差 | 时延、带宽、全部场景泛化 |

当前 P64D2 只在历史 128 B-read / 32 B-write policy 下有强的功能 cache 与 Full-ROI NCU byte 对照。它证明该具体条件模型的 aggregate bytes 接近，不证明最新 sector32 policy、GPU timing 或任意 P/D 同样准确。

### 8.3 与 pure GTSim 的关系

ProgramGraph 路径与 pure GTSim 不是同一输入的无损互转，而是面向不同问题的两个 frontend：

- ProgramGraph 通过消去 CTA/warp DAG 构建、scheduler、in-flight request、completion feedback 与 GPU cycle 状态，换取较低的 host-side 执行成本和可批量固定 P/D 流量研究能力。
- TileGraph/GTSim 保留 CTA/warp/tile、依赖、驻留、issue 与 completion 语义；它是研究 overlap、stall、调度和模型周期所必需的路径。
- 两者可共享已审计的算子语义、对象 binding 和地址物化器，但不能假定 cache output 逐条相同，因为 caller order 与实际 issue order、合并和 LRU 状态不同。

GPU 动态插桩可提供 function/ISA/运行时观察等证据，但并不会自动替代 provider 的语义证明；NVIDIA NVBit 的设计正说明此类工具可用于动态 instruction/memory-reference instrumentation，而不是把 kernel 身份自动变为完整访存语义。[NVBit](https://research.nvidia.com/publication/2019-10_nvbit-dynamic-binary-instrumentation-framework-nvidia-gpus)
