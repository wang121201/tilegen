# Qwen2.5-1.5B P32D2 的 TileLang TileGraph 输入

此文档定义本分支的当前输入边界。Qwen2.5-1.5B-Instruct 是一个 28 层 Transformer，隐藏维度为 1536，前馈中间维度为 8960，查询头数为 12，键值头数为 2，每头维度为 128。P32D2 表示批大小为 1、32 个 token 的 prefill（提示预填充）和两个单 token decode（自回归解码）步骤。

TileGraph 是 TileGen 的 tile-level 输入表示。一个模板包含节点、执行组、数据边和顺序边；节点是全局内存、标量计算、Tensor Core 或控制操作。TileLang 的 Tensor Intermediate Representation（TIR，张量中间表示）保留线程块、共享缓冲区、片段缓冲区、`T.Pipelined` 循环、`T.copy` 和 `T.gemm_py`。生成器把这些编译器可见的语义转写为可复用的 TileGraph 模板；非 GEMM 的向量/控制操作采用论文允许的直接节点和边构造。

生成器是 `llm/tools/qwen15_tilelang_tilegraph.py`。它只读取其自身的模型常数并调用 TileLang 编译器；不读取 CUDA trace、NVBit、NCU、硬件结果、校准参数或 SGLang 运行时输出。源码参考树为 `/home/xmu/nvidiagds/simulators/tilelang`；执行时应使用一个可导入 TileLang 的 Python 运行时。当前服务器使用 `/home/xmu/sgl/bin/python3`。源码树缺少本地 `build/lib`，因此它作为前端参考；实际 TIR 由已安装的 TileLang 编译运行时生成。

1030 是明确的 P32D2 场景基数合同，而非从真实 trace 推断的 runtime census：prefill 388，decode_1 321，decode_2 321。输出的 `cardinality_contract.meaning` 固定这一边界。每个 kernel 实例通过 `template_id` 指向一个 TIR 生成或直接构造的模板，并显式列出向后的数据和顺序依赖。`TILEGEN_TILELANG_TILEGRAPH_INPUT_V1` 和 `input_status=TILEGRAPH_INPUT_READY` 表示它可被 TileGen 的 TileGraph 前端消费；它不表示现有的 `tilegen_native` 已能读取该 JSON。该二进制目前只接受旧的压缩 native 程序传输，因此后续接入工作应是为既有 `KernelBinding` 添加这个格式的 reader，而不是采集或校准真实 trace。

示例（输出目录必须是新目录，避免覆盖任何历史证据）：

```bash
/home/xmu/sgl/bin/python3 llm/tools/qwen15_tilelang_tilegraph.py \
  --output /home/xmu/nvidiagds/.codex-runs/qwen15-tilegraph-cosim-20261001/tilegraph-qwen-p32d2.json
/home/xmu/sgl/bin/python3 llm/tools/qwen15_tilelang_tilegraph.py \
  --check /home/xmu/nvidiagds/.codex-runs/qwen15-tilegraph-cosim-20261001/tilegraph-qwen-p32d2.json
```
