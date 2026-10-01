# TileGen：当前工作流入口

此 worktree 是 `codex/qwen15-tilegraph-cosim-20261001`。它保留两条**不能混同**的路径：

1. **ProgramGraph 功能流量路径（当前用于固定 P/D 的稳定路径）**：`ProgramGraph → provider.events() effect stream → FunctionalCache → counters / optional post-cache trace`。
2. **TileGraph 细粒度路径（pure GTSim 语义）**：`TileGraph → CTA/warp/tile dependency execution → L1/L2 → optional memory completion feedback`。

前者不执行 GPU issue、warp overlap、MSHR 时序或 GPU completion；它只能产出以明确 cache policy 和 serial caller order 为条件的功能性流量。后者保留这些调度/依赖语义，但当前 Qwen P32D2 工件只完成 importer/input check，不是已经完成的完整 GPU/HBFSim 运行。

请先读 [ProgramGraph 功能流量工作流、证据和 32 B 状态](docs/programgraph-functional-workflow.md)。其中给出了项目组织、可复现门禁、与 pure GTSim 的简化边界，以及 P64D2 的历史硬件对照。本文后续内容是历史 native-trace / HBFSim 融合记录，**不是本分支当前 Qwen ProgramGraph 的运行指南**。

下文保留历史融合分支记录。

最新衔接：GEMV、SiLU 的 direct binding 与精确 cosim Builder 已共用 cache 前 `PreparedMemory`，并减少等价 CTA 校验的主机分配。三组配对测试中引擎执行窗口为 **1.08×**，完整子进程 CPU 时间基本持平；不能据此声称端到端明显提速。实现、验收和范围见 [共享前端报告](docs/shared-frontend.md)。

最新增加：已有 direct trace 可通过独立 `replay.py` 直接回放到 HBFSIM，跳过计算和 GPU stall，保留内存队列/时序。Decode 有界子集的 591,652 条请求回放 CPU **0.022 分钟**；完整生成与回放合计约 **0.440 分钟**，与保留计算依赖的 cosim 口径不同。见 [纯访存回放说明](docs/memory-only-replay.md)。

现在可在这条 direct 路径导出 CTA 组计算 profile，再用 `replay.py --mode stage-overlap` 执行阶段级计算/访存重叠。地址流不变，计算成本取原 SourceNode 的静态资源需求，运行时跳过 warp/DAG 调度；窗口、计算尾部、kernel 屏障及近似范围见 [阶段重叠说明](docs/stage-overlap.md)。自动 profile 当前限九类原生 binding，不覆盖完整 1138-call 流程。

阶段回放现在支持配置独立的 GDDR6、HBM 和原生 HBF controller/NAND 后端，并输出阶段、kernel、CTA 组的读写与带宽。HBF 的请求字节、4 KB 页介质流量、控制器 HBM 流量和最终持久化尾部独立计数；它们不能混为同一个带宽。见 [多后端阶段报告](docs/multi-backend-stage.md)。

新目标为当前 SGLang native 的 **B1 / BF16 / 32 层 / P1024 / D32**，同源硬件采集包见 [采集说明](capture/p1024d32/README.md)。33 阶段输入与自然 CUDA event 已采集；正式 NCU 已完成 3 组 × 6 范围（Full、Prefill、D1、D8、D16、D32），保存原始报告、CSV 和来源校验记录。不同范围来自独立运行，不能用 Full−Prefill 推导 Decode 流量；NCU 时间与自然运行时间分别报告。

输入采集、NCU 测量与新形状 TileGen 地址模型的资格检查是不同步骤；**完整 P1024/D32 TileGen 尚未准入**，新 attention/GEMM/merge kernel 的参数、原生动态证据、地址绑定及阶段/容量适配见 [准入审计](docs/native-p1024d32-admission.md)。状态与正式 NCU 汇总见 [进度记录](validation/native-p1024d32-progress.json)，结果文档见 [阶段后端与新负载报告](/Users/wgs/Documents/Codex/2026-09-17/zhi/outputs/tilegen-stage-backends-20260918/index.html)。

## 独立分支与实现

- 基线提交 `0e21251f126510744d1b319f043e7e6b2dae5e1f`：抽取 22 个翻译单元、171 个实际源码依赖（约 4.19 MB），展开原 VFS overlay。原工作目录未修改，B8 候选仍在另一个仓库。
- `provenance/source-map.json` 逐文件记录逻辑路径、实际来源、SHA 和 include 改写。上游以封存文件组织，所以这是有来源记录的独立导入及集成提交，没有伪造 Git 合并祖先。
- `source/direct_native.h`：9 类复用原 native binding，其余 11 类通过原 Builder 按一个 CTA 提取访存并释放，没有新增模型地址公式。
- `source/direct_cache.h`：复用原 L1、dirty-mask 规则，使用立即完成的功能 L2 LRU。两种模式共用 `source/native_trace.h`。
- 原生 L2 把连续 dirty run 改为逐个 32 B 请求，有限待准入队列上界改为 `4(B+1)`。后端只增加可选观察器，成功准入后记录一次，重试不重记。
- 有界子集只构建实际选中 family 的 typed 预验证模型；八帧仍全部解压并验证声明的 SHA。预验证与执行模型分开持有，避免 full-grid 模板被 prefix-CTA 执行错误复用。完整 1138 流程仍包含全部 family，不能把这项子集收益外推给完整流程。

## 运行档

| 参数 | 执行方式 | 时序范围 |
|---|---|---|
| `replay.py` | 已有 direct cache 后 trace → HBFSIM；不重新建模/cache | 全部请求就绪，按原序尽快填充有限内存队列；无计算/依赖/GPU stall |
| `replay.py --mode stage-overlap` | direct `--phase-ctas 48` 导出的 trace + 静态计算 profile → HBFSIM | 阶段访存完成后计算，下一组可预取；跨 kernel 屏障，固定 direct cache，无 warp 依赖执行 |
| `--mode direct` | 原生地址；9 类免逐 CTA DAG，11 类单 CTA 投影；功能 L1/L2 | 固定 call / CTA / member 顺序；立即完成；不运行计算调度或 HBFSIM，时间字段未知 |
| `--mode cosim`（默认） | 原生计算/依赖图、已有精确主机加速、GTSim L1/L2、HBFSIM | 保留计算、访存停顿及重叠，有限队列/准入 |
| `--mode cosim-fast` | C 原 hybrid 路径与 fine fallback | **显式近似时序档**：q16、memory-phase16、epoch8、independent drain；不宣称与全 fine 周期相同 |

direct 与 cosim 共用地址规则，**过 cache 流量不保证逐条相同**：跳过计算和在途请求会改变跨 warp/CTA 顺序、MSHR 合并及 LRU。需要实际调度顺序的地址流时使用 `cosim --trace`。
硬件时序尚未校准；原有 estimated-address、依赖资格等限制保留。后端和阶段模型没有使用 NCU 流量或带宽拟合系数；P1024/D32 新采集的执行状态以独立收据为准。

## 构建与使用

要求 C++20、zlib、Python 3。默认直接 clang 构建，无须 CMake。

```sh
python3 build.py --output build/cache-alignment-r1 --jobs 2 --native --thin-lto
python3 run.py --input /absolute/path/workload.input --output build/direct-run --mode direct
python3 run.py --input /absolute/path/workload.input --output build/cosim-run --mode cosim --trace
python3 run.py --input /absolute/path/workload.input --output build/fast-run --mode cosim-fast
```

输出目录必须新建；已有二进制可以直接运行。`--binary` 可指定其他构建。
`run.py` 给出子进程真实 CPU 和 elapsed 的秒数、分钟数；编译、输入准备另计，结果核验用时另列。
零退出码还须通过结果身份和 trace SHA 核验才算成功。`build.py` 核对构建前后源码 SHA 不变。
另提供 CMake/CTest 配置；本次本机采用 clang 构建，没有执行本机 CTest。

固定输入为 B1、Llama3-8B BF16、P32/D2 的原生 1138-call 模型。
**源码/Git 独立，运行数据尚未搬迁封包**：transport 含八个压缩模板，模型仍按原 pin 读取本机封存的工作流、程序和配置。不是任意模型、shape 或 batch 的入口。
源码编译不再依赖原目录；运行输入仍为只读封存数据。

完整运行需完整输入和 `--full-workflow`。wrapper 默认 trace 配额 8 GiB；旧 full cosim 流量按当前 144 B 记录估算约 **48.12 GiB**，仅用于容量预算，不能当成本分支完整运行实测。
full 导出需显式设置 `--max-trace-bytes 68719476736`（64 GiB 上限）并准备空间。
超配额会失败并保留 `.partial`，只有读回校验完成才发布正式文件。两种模式都不做最终 dirty flush。

## 初次融合验收

以下为 `ad8afa3` 集成基线的记录。最新源码的独立原公式检查、CTA 校验负测、20-family trace 精确回归及配对性能结果见 [共享前端报告](docs/shared-frontend.md) 和 `validation/shared-frontend.json`。

- 32 B：15 种 dirty mask、跨行/重复/部分写、6 种容量、step/epoch1/4/8 背压；ASan/UBSan 三模式共 59,321 项检查，每 tick 核验 dirty-sector/byte 守恒。
- Trace：18,981 项检查；后端开关前后完成序列、周期、读写及物理统计一致，覆盖重试、损坏、截断、配额和文件发布。
- 9 类 binding 对原 Builder：19,816 条访存指令、542,588 个地址范围，方向、bypass、matrix、subop、lane、地址和宽度精确一致。
- 20 类 kernel 的实际运行、decode 子集性能和同二进制 trace 开关比较，见 `validation/qualification.json`、`docs/qualification.md`。

上述为代表性来源和有界 CTA 验证，未重跑 1138 个完整网格；不把 direct/cosim 流量相等列作验收条件。格式与地址含义见 `docs/native-trace.md`。

复跑入口（均为 CPU 工作）：

```sh
python3 tests/run_unit_tests.py --output build/unit-tests
python3 tests/run_unit_tests.py --output build/unit-tests-asan --sanitize address,undefined
python3 tests/prepare_inputs.py --source-runtime /absolute/path/canonical-full-runtime-r4 --output build/validation-inputs
python3 tests/run_direct_projection.py --build build/shared-frontend-r2 --input build/validation-inputs/families.input --output build/projection
python3 tests/run_direct_projection.py --test prepared-memory --build build/shared-frontend-r2 --input build/validation-inputs/decode-512.input --output build/prepared-memory-check
```

准备脚本复用原封存的 CPU lowering，记录准备时间和输入 SHA，不进行 GPU 采样。
该准备器沿用上游 Darwin 内存监控，数据准备仍依赖当前本机封存目录。

## 新增：XMU Accel-Sim Ada 配置入口

`ada_profile.py` 提供独立的 adaptive L1 / 32 B sector / lazy-write 功能缓存回放；配置来源、运行示例及与 structure-only GTSim 适配器的区别见 [Ada Accel-Sim 配置说明](docs/ada-accelsim-profile.md)。该入口不表示旧 cosim 已完整实现 Accel-Sim 时序，也未完成本配置的 NCU 精度验收。

### Ada 校准配置 r2 / r3

当前新增功能入口 `ada_profile.py` 默认使用 `r2-adaptive`，保留 `--profile tuner-v1`；r3 FIFO静态配置仅显式实验选择。详见 [校准配置、实际shared绑定与运行说明](docs/ada-calibrated-profile.md)。原 `run.py` direct/cosim默认未变。
