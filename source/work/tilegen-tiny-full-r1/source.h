#pragma once
#include "../tilegen-packet-integration-r1/source_support.h"
#include "../tilegen-packet-runtime-r1/runtime.h"
#include <span>

namespace tiny_full {
using namespace native_sequence;
namespace rt = packet_runtime;

enum class Kind { Compute, Control, Tensor, Barrier, Global, Shared, AsyncCopy };
enum class Path { DirectGlobal, Shared, AsyncGlobalToShared };
struct SourceNode {
    unsigned id=0, warp=0, ordinal=0;
    U pc=0, source_ordinal=0, mask=0, compute_elements=0, tensor_fma=0;
    Kind kind=Kind::Compute;
    std::string pipeline;
    bool write=false;
    U memory_bytes=0, memory_offset=0;
    std::string memory_buffer;
    std::vector<unsigned> completion_dependencies, issue_dependencies;
};
struct MemoryDescriptor {
    Path path=Path::DirectGlobal;
    bool write=false, bypass_l1=false;
    U global_bytes=0, shared_bytes=0;
    // One original memory instruction, retaining lane/mask/range provenance.
    // Descriptors are owned by the active port operation, never by a DAGNode.
    std::vector<g::ExplicitMemorySubop> global_subops, shared_subops;
    std::vector<g::CacheLineKey> lines;
    g::ExplicitSRAMService shared_service{};
};
class KernelBinding {
public:
    virtual ~KernelBinding()=default;
    virtual U ctas() const=0;
    virtual unsigned warps(U cta) const=0;
    virtual unsigned resident_limit() const=0;
    // Original graph span, including nonuniform CTA template classes.
    virtual U first_node(U cta) const=0;
    virtual std::span<const SourceNode> nodes(U cta) const=0;
    virtual U template_class(U cta) const=0;
    virtual MemoryDescriptor memory(U cta,unsigned member) const=0;
    virtual J evidence() const=0;
    int warp_token(U cta,unsigned warp,unsigned sms) const {
        p::need(cta<ctas()&&warp<warps(cta),"tiny source placement range");
        return g::cta_placement::token(int(cta),int(warp),int(warps(cta)),int(sms),4,true);
    }
};
} // namespace tiny_full
