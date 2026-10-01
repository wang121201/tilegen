# 历史 Qwen2.5-1.5B TileGen + HBFSim 全耦合与 TileGraph 构建、校准流程

> **定位（2026-10-01）。** 本文保留 `codex/qwen15-full-cosim-20260929-r1` 的历史 full-cosimulation 证据与术语；它不是当前 `codex/qwen15-tilegraph-cosim-20261001` 的默认执行路径。当前 worktree 对固定 P/D 的可批量执行路径是 ProgramGraph 的功能 cache 流量投影；入口、项目组织、32 B cache/trace 限制和 P64D2 对照见 [ProgramGraph 功能流量工作流](programgraph-functional-workflow.md)。

> **不要误读。** TileGraph 仍是 pure GTSim 语义所需的输入结构；当前 Qwen P32D2 TileGraph 已通过输入检查，但尚未以此文定义的完整方式执行 GPU、HBFSim 或硬件校准。因此本文中的 full-cosimulation 定义不能被套用于 ProgramGraph 的流量结果。

## 保留文档的原始范围

以下内容记录历史分支的实现与证据，未因当前分支的 ProgramGraph 路径而改写。

## 1. 文档目的与当前结论

本文是 XMU 服务器上的独立、自包含工作流，目标是为 Qwen2.5-1.5B、批量大小 1（Batch Size 1，B1）、BF16（Brain Floating Point 16-bit，16 位脑浮点格式）建立 TileGen 与 HBFSim 的耦合仿真证据，并规定如何构建、验证和校准可用于性能结论的 TileGraph。

本文中：

- P 表示 prefill token count，即一次前填阶段输入的 token 数；P32 表示 32 个前填 token。
- D 表示 decode step count，即前填后连续执行的解码步数；D2 表示执行 D1、D2 两个解码步。
- CTA（Cooperative Thread Array，协作线程数组）是 CUDA 线程块的调度单位。
- Warp（线程束）是本模型中 32 个线程的执行与依赖建模单位。
- DAG（Directed Acyclic Graph，有向无环图）表示节点及其依赖边构成的无环图。
- GPU（Graphics Processing Unit，图形处理器）是本工作流的目标处理器。
- CUDA 是 NVIDIA 的 GPU 并行计算平台；kernel 是由 CPU 主机发起、在 GPU 上执行的函数；launch 是一次 kernel 启动。
- PTX（Parallel Thread Execution，并行线程执行）是 NVIDIA 虚拟指令集；SASS（Shader Assembly，着色器汇编）是 NVIDIA 目标机器指令表示。
- TileGraph 是一个 kernel 的 CTA/warp 级 DAG，加上指令类别、执行管线、显式内存访问、地址身份、资源需求和完成语义。
- GTSim 是 TileGen 内的 GPU 时序/调度执行器。
- HBFSim 是本工作流使用的内存后端；当前实例用 HBFSim 的 HBM（High Bandwidth Memory，高带宽内存）类实现承载一个 GDDR6（Graphics Double Data Rate 6，第六代图形双倍数据率内存）意图配置，但不等于真实 GPU 的 GDDR6 控制器。
- L1/L2 分别表示一级和二级缓存。
- NCU（NVIDIA Nsight Compute）是硬件性能计数器采集器；其被插桩时长只作诊断，不能替代未插桩 CUDA Event 正式时长。
- ROI（Region of Interest，感兴趣区间）是 Prefill、D1…Dn 或 Full 等被单独统计的运行区间。
- full cosimulation（全耦合仿真）严格表示：同一连续会话中执行 GTSim 的 CTA 驻留、warp 调度、依赖、计算子操作发射与完成，经过 L1/L2，再将缺失和写回送入 HBFSim，并以真实返回完成解锁上游节点。它不自动意味着硬件准确。
- native transport 是完整封存的原生输入包，包含规范化控制 JSON（JavaScript Object Notation，JavaScript 对象表示法）和 8 个压缩证据 frame（帧）。
- kernel family 是共享同一种 TileGraph 构造规则的 kernel 类别。P28QKV 是当前 QKV（Query、Key、Value，查询、键、值）矩阵乘 kernel family 的固有名称，其中 P28 不表示 28 个 prefill token；GEMV（General Matrix-Vector Multiplication，通用矩阵向量乘）是解码矩阵向量乘 family。
- SHA-256（Secure Hash Algorithm 256-bit，256 位安全哈希算法）用于封存文件和结构身份；B 表示 byte（字节），GB/s 表示每秒十进制十亿字节，MHz 表示兆赫。
- descriptor（描述符）是携带张量/内存布局与访问元数据的运行时对象；implicit descriptor dependency 是尚未显式编码进 TileGraph 的描述符相关依赖。
- memory replay（内存重放）表示按已封存的地址级读写事务及顺序驱动缓存/内存后端；CUDA、PTX 或 SASS 文本本身不是 HBFSim 的输入。
- quiescent（静止）表示结束时调度器、缓存未决请求、MSHR（Miss Status Holding Register，缺失状态保持寄存器）、HBFSim 队列和完成队列均无未完成工作。

截至 2026-09-29：

1. P32D2 的 1138 个调用已通过全输入构造与一致性验证，但尚未执行 1138 调用全量耦合长跑。
2. P32D2 已有三类最终二进制耦合门禁：完整 48-CTA P28 QKV、每调用 1 CTA 的连续 27 调用前填家族覆盖、每调用 1 CTA 的 GEMV 与两个 Decode Attention 家族覆盖。
3. 当前结果证明耦合链路可运行并闭合，不证明 Qwen2.5-1.5B 的硬件时延准确。计算代价和地址仍含估计项，隐式 descriptor 依赖不完整，GPU 时钟、物理地址映射和 GDDR6 时序尚未校准。
4. P64D2、P128D2、P256D2、P512D2、P128D4、P128D8、P128D16 目前只有功能流量/NCU 对照材料，没有各自的完整 native transport，因此不得用 P32D2 TileGraph 替代，也不得报告为 full cosimulation 矩阵。

## 2. 唯一 XMU 工作位置与冻结身份

```text
branch: codex/qwen15-full-cosim-20260929-r1
base: stable/ada-sector32-line@be1da386f2432d127f41568e4bbc9edfdd7815f6
worktree: /home/xmu/nvidiagds/simulators/.codex-worktrees/qwen15-full-cosim-20260929-r1
binary: /home/xmu/nvidiagds/simulators/.codex-worktrees/qwen15-full-cosim-20260929-r1/build/qwen15-cosim/tilegen_native
binary SHA-256: c9e5fdec83979c0b8ffd698839fe99345fdcb661f6e864f7cee82f77f4e086b8
```

P32D2 完整 transport：

```text
path: /home/xmu/nvidiagds/simulators/tilegen/build/qwen-matrix-current/native-transport/qwen-p32d2.input
bytes: 24009556
SHA-256: 5c9271b420af9dd02da04d784666455b0e48e1c1fac61f0556049ea23d6c9c4b
```

物理源码闭包根目录：

```text
/home/xmu/nvidiagds/codex-runs/hbserve-memgen-gtsim-alignment/build-r1/mirror/work
```

运行时必须显式设置 `TILEGEN_XMU_SOURCE_ROOT`。transport 内旧的 `/Users/wgs/...` 前缀只保留为逻辑来源标识，永远不得直接打开。解析器要求最终文件是 XMU 根目录内的现存、非符号链接普通文件，并拒绝路径穿越和根外文件。未设置环境变量必须 fail closed，即立即拒绝。

HBFSim 身份：

```text
upstream commit: d7a2ca64614a6d9ce8d7a69beb77ce78b66df1a8
selected.cfg: /home/xmu/nvidiagds/codex-runs/hbserve-memgen-gtsim-alignment/build-r1/mirror/work/tilegen-cycle-overlap-r1/selected.cfg
selected.cfg bytes: 3578
selected.cfg SHA-256: b0b1098dd31828b92c11a8ec9d8eca8b61ccae5a2389e1f8612aba967a41a1ca
```

嵌入版本与上游提交有 32 个共同文件，其中 30 个逐字节相同。两个有界修改文件是 `src/physical/hbm/hbm_device.cpp` 和 `src/physical/hbm/hbm_device.hpp`；修改范围为独立 pseudo-channel（伪通道）drain 接口、保守历史 latch 和测试观察/空指针保护，原全局 service 方法保留，建模命令和状态更新不应改变。任何后续修改都必须重新做逐文件身份审计。

当前后端边界：

- 意图内存为 GDDR6；实现类型为 `hbfsim::physical::hbm::HbmDevice`。
- 地址映射 `pch-interleave-bg-rotate-v2` 仅是通用 service 映射，不是 GPU 物理 bank/channel 映射。
- 峰值数据通路参数为 360.0 GB/s，命令时钟 2250 MHz，burst 为 32 B。
- JEDEC（Joint Electron Device Engineering Council，联合电子器件工程委员会）控制器未验证、芯片身份未验证、硬件时序未校准、真实 GPU 地址映射未恢复。
- 因此 HBFSim 数字是结构化模型输出，不是硬件 oracle（硬件真值）。

## 3. 当前可重复门禁

结果根目录是 `/home/xmu/nvidiagds/simulators/.codex-runs/qwen15-full-cosim-20260929-r1`，不写回本地 Codex 工作区。

| 门禁 | 输入范围 | 结果 | 关键闭合量 |
| --- | --- | --- | --- |
| 完整输入验证 | 1138 调用、772512 CTA、4712768255 节点 | `VALIDATED_NOT_EXECUTED` | 0 unsupported；结果 SHA-256 `2f8422ee5ceb2155ae2cb1797bb7a28ade06b455decca331bff3452d20b0fcfa` |
| 完整 P28 门禁 | 前 17 调用；P28 为 48 CTA 完整网格 | `MODELED_BOUNDED_SUBSEQUENCE_EXECUTED` | 988860 周期；读 51588352 B；写 375936 B；静止；结果 SHA-256 `1e9c52a0af31e1d032696d0386198df8b131b121e141c40e37cca13ef32ac354` |
| 前填家族门禁 | 连续前 27 调用，每调用 1 CTA | `MODELED_BOUNDED_SUBSEQUENCE_EXECUTED` | 1765351 周期；读 8647296 B；写 0 B；静止；结果 SHA-256 `d1a9385596c92e45933a11da272073c8e0d66ebaf4139b9257b663a768933f6d` |
| 解码家族门禁 | GEMV、AttentionDecode1、AttentionDecode2，各 1 CTA | `MODELED_BOUNDED_SUBSEQUENCE_EXECUTED` | 14079 周期；读 94208 B；写 0 B；静止；省略 1135 调用且 launch gap=380；结果 SHA-256 `4ec5dae238da3db346612ce774d8c295faabdc5713aa53e546baf2a36a9408a7` |
| 单元/组件测试 | 9 个 CTest 测试 | 9/9 PASS | dirty32、cache、native trace、CTA validation、trace/stage replay、phase report、backend |

写 0 B 不表示没有写请求；连续会话默认不做最终 dirty cache 强制写回时，仍驻留在 L2 的脏数据不会计为 DRAM 写流量。跨实验比较必须固定并报告 final dirty flush 策略。

完整 P28 的 overlap（重叠）观测窗口为 968362 周期：

- compute inflight union 为 61161 周期；定义为实际 SIMD（Single Instruction Multiple Data，单指令多数据）、SFU（Special Function Unit，特殊函数单元）或 SHFL（warp shuffle，线程束交换）子操作从发射到计划完成的并集，不是执行单元 busy。
- memory outstanding union 为 3882 周期；定义为全局内存节点从第一次发射到实际最终完成的并集，包含 load/store、重试、cache 和返回，不是物理 DRAM busy。
- 两者同时为 2071 周期；compute-only 为 59090；memory-only 为 1811；neither 为 905390。
- overlap 可发生在不同 SM（Streaming Multiprocessor，流式多处理器）之间。neither 只表示上述两类均未被跟踪，不表示 GPU 空闲；shared-memory、barrier 和不支持的节点被排除。
- 支持的计算子操作和全局内存节点均完成，最终 in-flight 计数为 0，节点覆盖闭合。

## 4. XMU 运行命令

以下命令只在新的输出目录中写结果，不复用或覆盖已有结果目录。

```bash
WT=/home/xmu/nvidiagds/simulators/.codex-worktrees/qwen15-full-cosim-20260929-r1
BIN="$WT/build/qwen15-cosim/tilegen_native"
SRC=/home/xmu/nvidiagds/codex-runs/hbserve-memgen-gtsim-alignment/build-r1/mirror/work
INPUT=/home/xmu/nvidiagds/simulators/tilegen/build/qwen-matrix-current/native-transport/qwen-p32d2.input
OUT=/home/xmu/nvidiagds/simulators/.codex-runs/qwen15-full-cosim-20260929-r1/manual-$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"

env TILEGEN_XMU_SOURCE_ROOT="$SRC"   "$BIN" --mode=cosim --validate-only --full-workflow   < "$INPUT" > "$OUT/result.json" 2> "$OUT/stderr.log"

sha256sum "$BIN" "$INPUT" "$OUT/result.json"
```

`--validate-only --full-workflow` 只构造并验证 1138 个调用，不执行模型周期。真正的 1138 调用全量执行必须另建输出目录、先通过第 7 节所有门禁，并显式记录开始时间、进程号、输入/二进制哈希和资源预算；当前文档不把它列为已完成结果。

## 5. Accurate TileGraph 的构建流程

“accurate” 必须按层级使用，不能只因图可运行就称为准确。

### 5.1 封存 workload 身份

固定模型、层数、B、BF16、P、D、软件提交、编译器/运行时版本、kernel 名称和 launch 顺序。每个 P/D 组合单独封存，禁止由 P32D2 外推 P64D2 或 P128D16。

记录每次 launch 的 grid、block、动态 shared memory、stream、参数字节、张量形状/stride/dtype、allocation 生命周期和 kernel 代码身份。参数捕获必须能区分相同指针在不同 allocation epoch（分配世代）中的复用。

### 5.2 从编译/原生证据构造节点

优先从 TileLang 的 scheduled IR（scheduled Intermediate Representation，经调度中间表示）和目标端已排程指令建立节点；缺失时才使用经过逐算子验证的模板。每个节点至少包含：

- 稳定 node id、CTA id、warp id、程序计数器或等价位置；
- opcode 类别、执行管线、子操作数、延迟和吞吐约束；
- predicate/active mask（谓词/活跃 lane 掩码）；
- issue dependency（允许发射前必须满足的依赖）；
- completion dependency（只有前驱真正完成后才能满足的依赖）。

不得把运行时 `scheduler_cycle`、host 回调先后或容器遍历顺序冒充图的结构顺序。

### 5.3 建立地址和 memory sub-operation

地址身份至少为 `allocation_id + allocation_epoch + byte_offset`；仅保存裸虚拟地址不足以处理复用和别名。每个 memory sub-operation（内存子操作）记录读/写、字节数、lane 地址、sector/line 覆盖、cache policy、原子/一致性要求及节点内顺序。

逐层建立守恒账本：

```text
请求事务 = L1 命中 + L1 绕过/缺失
L2 接收 = L2 命中 + 合并到 pending fill + 新 DRAM fill + write path
HBFSim 外层接受 = 物理接受 = 物理完成 = 外层交付
运行结束时 live request = 0
```

读写分母必须分开；requested bytes、cache-line bytes、DRAM transferred bytes 不能混用。

### 5.4 单 CTA oracle 与全网格扩展

先对每个 kernel 选择至少一个代表 CTA，逐 warp 对比节点数、边数、PC/opcode、active mask、地址/字节、节点内顺序和最终 hash。任何差异先修 frontend，不得用调度参数吸收。

单 CTA 通过后，再按真实 grid 扩展，验证 CTA 到 SM 的放置、resident CTA 上限、register/shared-memory 资源、warp 数和完整 retirement。full-grid 门禁必须至少覆盖每一种 grid/资源类别，不能只运行 prefix 1 CTA。

### 5.5 连续会话、调度和重叠

Prefill 与 D1…Dn 必须在一个连续 session 中运行，L2 和 HBFSim 状态跨 kernel 保留；kernel 边界只做规定的 L1 flush 和 quiescence，不得重置 L2 或地址空间。

调度门禁记录实际 scheduler visit、候选拒绝原因、资源驻留和完成回调。compute/memory overlap 使用半开区间 `[issue, completion)`，明确纳入和排除节点。GPU union、per-SM pooled time 和 per-SP（Streaming Processor，流处理单元）pooled time 必须分开，不能相加。

### 5.6 transport 封存

每个 P/D transport 都要封存：

- control JSON 的规范化 SHA-256；
- 每个压缩 frame 的解压字节数和 SHA-256；
- launch 总数、顺序、family、grid/block 和参数身份；
- source pin 文件的 SHA-256、相对逻辑路径和 XMU 物理根；
- TileGraph 节点/边/内存记录的计数和 hash；
- 构建程序提交、编译器身份和命令。

任何 pin 缺失、hash 不符、根外路径、unsupported family、重复 JSON key 或 launch gap 未解释时都 fail closed。

## 6. 校准路线

校准分三层，不能跨层晋级。

### 6.1 功能/流量层

用硬件 NCU 的 `dram__bytes_read.sum` 与 `dram__bytes_write.sum` 检查同一 ROI、同一 P/D、同一软件身份的读写字节。报告硬件样本数、均值/离散度、模型字节和相对误差。

现有 P32/P64/P128/P256/P512 at D2 与 P128 at D2/D4/D8/D16 数据只属于此层；旧功能 adapter 的计算模型是零计算周期，不能生成模型带宽或 full-cosim 时延。

### 6.2 结构时序层

分别校准并验证：

1. 计算节点延迟、流水线吞吐、CTA 驻留和调度；
2. L1/L2 容量、line/sector、replacement、write-allocate 和 flush；
3. GPU 时钟与 DVFS（Dynamic Voltage and Frequency Scaling，动态电压频率调整）状态；
4. 地址到 channel/pseudo-channel/bank/row 的映射；
5. GDDR6 时序、queue、FR-FCFS（First Ready First Come First Serve，行命中优先的先就绪先服务）和 refresh；
6. kernel 间状态延续与最终 dirty flush。

训练组合与 held-out（留出验证）组合必须分离。可用 P32D2、P128D4、P512D2 做覆盖训练，用 P64D2、P256D2、P128D8、P128D16 做留出验证；这只是数据划分建议，不是对这些组合已经可运行的声明。

### 6.3 硬件时延层

正式时长使用未被 profiler 插桩的 CUDA Event，固定 warmup、重复次数、时钟/功耗模式并报告分位数。NCU/NSYS（NVIDIA Nsight Systems）被插桩时长只作诊断。每个 ROI 做 exact identity join（精确身份连接），至少匹配 model、B、dtype、P、D、software/kernel SHA、launch 序列和样本范围。

只有模型在未参与拟合的 P/D、kernel family 和 ROI 上达到预先写明的误差阈值，才能称为 hardware calibrated（硬件已校准）。阈值必须在看结果前固定，并同时报告分子、分母和失败样本，不能只给平均误差。

## 7. 晋级门禁与矩阵计划

以下是本文的证据等级，不是程序当前自动生成的 status：

- `GRAPH_STRUCTURALLY_VALIDATED`：TileGraph 身份、节点/边、地址、计数和单 CTA/full-grid oracle 通过。
- `COUPLED_MODELED_EXECUTED`：GTSim→L1/L2→HBFSim 连续执行并静止，守恒账本闭合。
- `TRAFFIC_VALIDATED`：同身份硬件 ROI 的读写流量通过预注册阈值。
- `HARDWARE_CALIBRATED`：计算、cache、地址映射、内存时序和未插桩端到端时长均通过留出验证。

当前 P32D2 只达到有界范围的 `COUPLED_MODELED_EXECUTED`，以及 1138 调用的结构输入验证；不具备 `HARDWARE_CALIBRATED` 资格。

| 轴 | 组合 | 每格必须新增的证据 |
| --- | --- | --- |
| Prefill sweep | P32/P64/P128/P256/P512，固定 D2 | 独立 native transport、完整 validate、family/full-grid 门禁、连续 full execution、同身份硬件流量与时长 |
| Decode sweep | P128，D2/D4/D8/D16 | 同上，并确保 D1…Dn 连续状态和 KV cache（Key-Value cache，键值缓存）生命周期准确 |

执行顺序：生成该格 transport → 身份/哈希审计 → 单 CTA oracle → 每 family full-grid → 连续有界序列 → 完整 validate → 资源评估 → 完整 execution → 流量校准 → 时延校准。前一门禁失败时停止该格，不得批量把未运行格标成失败。

## 8. 当前未完成项

- 尚未生成 P64D2、P128D2、P256D2、P512D2、P128D4、P128D8、P128D16 的完整 native transport。
- 尚未执行 P32D2 的 1138 调用全量耦合长跑。
- 计算与地址中的 estimated 标记尚未消除；implicit descriptor dependencies 尚未闭合。
- HBFSim 后端仍是未校准的 GDDR6 意图模型，真实 GPU 物理映射和 JEDEC 控制器行为尚未验证。
- 尚未完成 Qwen2.5-1.5B 的未插桩 CUDA Event 时长对齐与 held-out 验证。

下一步应优先建立“每个 P/D 独立的 compiler/native-derived TileGraph 与 transport 生成链”，先做 P128D4 作为同时覆盖较长 prefill 和多 decode 的中间格；在该格通过身份、单 CTA 与 full-grid 门禁前，不启动完整矩阵长跑。
