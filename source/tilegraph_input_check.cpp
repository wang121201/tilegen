#define main frozen_parent_unused_main
#include "work/tilegen-full-r1/driver-pooled-fusednorm-r1/streaming.cpp"
#undef main
#include "work/tilegen-tiny-full-r1/import.h"
#include "work/tilegen-tiny-full-r1/reporting.h"
#include "work/tilegen-hybrid-full-r1/fine_context.h"
#include <fstream>
#include <iostream>
#include <map>
#include <set>

namespace tg_input {
using J=nlohmann::json;
using namespace native_sequence;
using tiny_full::Kind;
using tiny_full::KernelBinding;
using tiny_full::MemoryDescriptor;
using tiny_full::SourceNode;
using tiny_full::U;

[[noreturn]] void reject(const std::string& message) {throw std::runtime_error(message);}
void need(bool condition,const std::string& message) {if(!condition)reject(message);}

Kind kind(const std::string& value) {
    if(value=="Compute")return Kind::Compute;
    if(value=="Control")return Kind::Control;
    if(value=="Tensor")return Kind::Tensor;
    if(value=="Barrier")return Kind::Barrier;
    if(value=="Global")return Kind::Global;
    if(value=="Shared")return Kind::Shared;
    if(value=="AsyncCopy")return Kind::AsyncCopy;
    reject("unknown TileGraph node kind: "+value);
}

class Binding final:public KernelBinding {
    std::vector<SourceNode> nodes_;
    unsigned warps_=0;
    U ctas_=1;
    unsigned resident_limit_=4;
    J evidence_;
    U address_base_=0;
    U cta_stride_=U(3)<<20;
    U max_local_address_=0;
    static U grid_product(const J& grid) {
        need(grid.is_object(),"launch grid object");
        const U x=p::natural(grid.at("x")),y=p::natural(grid.at("y")),z=p::natural(grid.at("z"));
        need(x>0&&y>0&&z>0,"launch grid must be positive");
        return p::multiply(p::multiply(x,y),z);
    }
public:
    explicit Binding(const J& templ, U address_base=0, const J* launch_override=nullptr,U cta_stride=U(3)<<20):address_base_(address_base),cta_stride_(cta_stride) {
        need(cta_stride_>0,"CTA address stride must be positive");
        const auto& local=templ.at("nodes");
        need(local.is_array()&&!local.empty(),"template has no nodes");
        const J launch=launch_override?*launch_override:templ.value("launch",J::object());
        const J grid=launch.value("grid",J({{"x",1},{"y",1},{"z",1}}));
        ctas_=grid_product(grid);
        const U threads=launch.value("threads",U(0));
        resident_limit_=unsigned(launch.value("resident_cta_limit",4));
        need(resident_limit_>=1&&resident_limit_<=32,"resident CTA limit");
        const auto iterations=templ.contains("loop")?p::natural(templ.at("loop").at("iterations")):U(1);
        need(iterations>0&&iterations<=4096,"template loop iterations out of bounded input range");
        for(std::size_t i=0;i<local.size();++i) {
            need(p::natural(local.at(i).at("id"))==i,"template node ids must be dense");
            warps_=std::max(warps_,unsigned(p::natural(local.at(i).at("execution_group"))+1));
        }
        if(threads)warps_=std::max(warps_,unsigned((threads+31)/32));
        need(warps_>0&&warps_<=32,"template execution-group range");
        need(local.size()<=std::size_t(UINT32_MAX)/iterations,"expanded source node overflow");
        std::vector<unsigned> ordinal(warps_);nodes_.reserve(local.size()*iterations);
        for(U iteration=0;iteration<iterations;++iteration) for(std::size_t local_id=0;local_id<local.size();++local_id) {
            const auto& input=local.at(local_id);SourceNode node;
            node.id=unsigned(nodes_.size());node.source_ordinal=local_id;node.warp=unsigned(p::natural(input.at("execution_group")));
            node.ordinal=ordinal.at(node.warp)++;node.kind=kind(input.at("kind").get<std::string>());node.pipeline=input.at("pipeline").get<std::string>();
            node.write=input.at("write").get<bool>();node.compute_elements=p::natural(input.value("compute_elements",U(0)));node.tensor_fma=p::natural(input.value("tensor_fma",U(0)));node.memory_bytes=p::natural(input.value("tile_bytes",U(0)));node.memory_offset=p::natural(input.value("memory_offset",U(0)));node.memory_buffer=input.value("memory_buffer",std::string{});nodes_.push_back(std::move(node));
        }
        for(const auto& edge:templ.at("edges")) {
            const auto from=unsigned(p::natural(edge.at("from"))),to=unsigned(p::natural(edge.at("to")));need(from<local.size()&&to<local.size(),"template edge endpoint range");
            const auto type=edge.at("type").get<std::string>();const bool issue=type=="order",loop=type=="loop_carried";need(type=="data"||issue||loop,"unknown TileGraph edge type");
            const U distance=loop?p::natural(edge.value("distance",U(1))):U(0);if(loop)need(distance>0&&distance<=iterations,"loop edge distance");
            for(U iteration=loop?distance:U(0);iteration<iterations;++iteration) {const U source_iteration=loop?iteration-distance:iteration;
                const auto source=unsigned(source_iteration*local.size()+from),target=unsigned(iteration*local.size()+to);need(source<target,"TileGraph expansion must be topological");
                (issue?nodes_.at(target).issue_dependencies:nodes_.at(target).completion_dependencies).push_back(source);}
        }
        for(auto& node:nodes_) {std::set<unsigned> seen;for(const auto* deps:{&node.completion_dependencies,&node.issue_dependencies}) for(const auto dep:*deps)need(seen.insert(dep).second,"duplicate typed/combined TileGraph edge");}
        for(const auto& node:nodes_)if(node.kind==Kind::Global) {const U bytes=node.memory_bytes?node.memory_bytes:128;max_local_address_=std::max(max_local_address_,p::add(p::add(node.memory_offset,p::multiply(U(node.id),128)),bytes-U(1)));}
        need(max_local_address_<cta_stride_,"CTA address stride does not cover expanded TileGraph template");
        evidence_={{"schema","TILELANG_TILEGRAPH_KERNEL_BINDING_V1"},{"template_id",templ.at("template_id")},{"construction",templ.at("construction")},{"source_template_nodes",local.size()},{"expanded_source_nodes",nodes_.size()},{"loop_iterations",iterations},{"execution_groups",warps_},{"grid_ctas",ctas_},{"resident_cta_limit",resident_limit_},{"address_cta_stride_bytes",cta_stride_},{"trace_used",false},{"hardware_calibration_used",false}};
    }
    U ctas() const override {return ctas_;}
    unsigned warps(U cta) const override {need(cta<ctas_,"CTA range");return warps_;}
    unsigned resident_limit() const override {return resident_limit_;}
    U first_node(U cta) const override {need(cta<ctas_,"CTA range");return p::multiply(cta,U(nodes_.size()));}
    std::span<const SourceNode> nodes(U cta) const override {need(cta<ctas_,"CTA range");return nodes_;}
    U template_class(U cta) const override {need(cta<ctas_,"CTA range");return 0;}
    U max_source_address() const {return p::add(p::add(address_base_,p::multiply(ctas_-U(1),cta_stride_)),max_local_address_);}
    MemoryDescriptor memory(U cta,unsigned member) const override {
        need(cta<ctas_&&member<nodes_.size(),"memory descriptor range");const auto& node=nodes_.at(member);
        need(node.kind==Kind::Global||node.kind==Kind::Shared,"not a memory node");MemoryDescriptor descriptor;descriptor.write=node.write;
        descriptor.path=node.kind==Kind::Global?tiny_full::Path::DirectGlobal:tiny_full::Path::Shared;
        if(node.kind==Kind::Global) {
            const U bytes=node.memory_bytes?node.memory_bytes:128;
            const U offset=address_base_+p::multiply(cta,cta_stride_)+node.memory_offset+U(node.id)*128;descriptor.global_bytes=bytes;
            for(U line=0;line<bytes;line+=128) descriptor.lines.push_back(g::CacheLineKey{1,offset+line});g::ExplicitMemorySubop sub;
            sub.modeled_member_start=int(node.id);sub.source_member_ordinals.push_back(int(node.id));sub.requested_bytes=bytes;
            sub.ranges.push_back({int(node.id),offset,bytes});descriptor.global_subops.push_back(std::move(sub));
        }
        return descriptor;
    }
    J evidence() const override {return evidence_;}
};

J read_json(const char* path) {std::ifstream stream(path);need(bool(stream),"cannot open TileGraph input");return J::parse(stream);}
int run(const char* path) {
    const J graph=read_json(path);need(graph.at("schema")=="TILEGEN_TILELANG_TILEGRAPH_INPUT_V1","TileGraph schema");need(graph.at("input_status")=="TILEGRAPH_INPUT_READY","TileGraph input status");
    for(const char* forbidden:{"trace","ncu","hardware","calibration","measured"})need(!graph.contains(forbidden),std::string("forbidden input dependency: ")+forbidden);
    std::map<std::string,const J*> templates;for(const auto& item:graph.at("templates"))need(templates.emplace(item.at("template_id").get<std::string>(),&item).second,"unique template id");need(templates.size()==18,"expected 18 source templates");
    std::map<std::string,std::shared_ptr<const packet_runtime::Program>> programs;auto config=GTSim::make_rtx4000_ada_footprint_reference_config();config.silence_mode=true;U expanded_nodes=0;
    const J template_validation_launch=J::object();
    for(const auto& item:templates) {Binding binding(*item.second,0,&template_validation_launch);auto program=tiny_full::import_program(binding,0,1,config);need(program->logical_members==binding.nodes(0).size(),"TileGen imported node census");need(!program->groups.empty(),"TileGen imported program is empty");expanded_nodes+=binding.nodes(0).size();programs.emplace(item.first,std::move(program));}
    const auto& kernels=graph.at("kernels");need(kernels.size()==1030,"expected 1030 kernels");std::map<std::string,U> phases;
    for(std::size_t index=0;index<kernels.size();++index) {const auto& kernel=kernels.at(index);need(p::natural(kernel.at("kernel_id"))==index,"dense kernel id");need(programs.contains(kernel.at("template_id").get<std::string>()),"kernel template exists");for(const char* field:{"data_dependencies","order_dependencies"}) for(const auto& dep:kernel.at(field))need(p::natural(dep)<index,"kernel dependency is backward");++phases[kernel.at("phase").get<std::string>()];}
    need(phases==std::map<std::string,U>{{"decode_1",321},{"decode_2",321},{"prefill",388}},"phase kernel census");std::cout<<J({{"schema","TILEGEN_TILEGRAPH_INPUT_CHECK_V1"},{"status","PASS_TILEGEN_ACCEPTS_TILEGRAPH_INPUT"},{"kernels",kernels.size()},{"templates",templates.size()},{"template_expanded_nodes",expanded_nodes},{"trace_used",false},{"hardware_calibration_used",false},{"HBFSIM_executed",false},{"GPU_executed",false}}).dump()<<'\n';return 0;
}

int run_full(const char* graph_path,const char* control_path) {
    const J graph=read_json(graph_path),control=read_json(control_path);
    need(graph.at("schema")=="TILEGEN_TILELANG_TILEGRAPH_INPUT_V1"&&graph.at("kernels").size()==1030,"full TileGraph contract");
    need(control.contains("memory_model")&&control.contains("service_address_map"),"full run memory control");
    const auto& layout=graph.at("address_layout");need(layout.at("schema")=="TILEGRAPH_PACKED_SOURCE_ADDRESS_LAYOUT_V1","full packed source address layout");
    const U kernel_stride=p::natural(layout.at("kernel_stride_bytes")),cta_stride=p::natural(layout.at("cta_stride_bytes"));need(kernel_stride>0&&cta_stride>0,"full packed source address strides");
    std::map<std::string,const J*> templates;for(const auto& item:graph.at("templates"))templates.emplace(item.at("template_id").get<std::string>(),&item);
    const auto source_covered=[&](U address) {for(const auto& span:control.at("service_address_map").at("spans")){const U base=p::natural(span.at("source_base")),bytes=p::natural(span.at("bytes"));if(address>=base&&address-base<bytes)return true;}return false;};
    for(std::size_t index=0;index<graph.at("kernels").size();++index) {const auto& kernel=graph.at("kernels").at(index);need(p::natural(kernel.at("address_base"))==p::multiply(U(index),kernel_stride),"full packed kernel address base");need(kernel.contains("launch"),"full-grid kernel launch contract");const auto& templ=*templates.at(kernel.at("template_id").get<std::string>());Binding binding(templ,p::natural(kernel.at("address_base")),&kernel.at("launch"),cta_stride);need(source_covered(binding.max_source_address()),"full TileGraph source address outside service map");}
    auto cfg=GTSim::make_rtx4000_ada_footprint_reference_config();cfg.silence_mode=true;cfg.workload_type=g::WorkloadType::Llama3Elementwise;
    native_sequence::Mapper source_mapper;coupling::Runtime memory(control,cfg,source_mapper);need(bool(memory.backend),"full run requires native HBF backend");
    auto l2=tiny_full::make_l2(cfg,memory);g::Cycle cycle=0;hybrid_full::FineContext session(cfg,*l2,cycle);
    U total_read=0,total_write=0,total_nodes=0,total_ctas=0;const auto started=std::chrono::steady_clock::now();
    for(std::size_t index=0;index<graph.at("kernels").size();++index) {
        const auto& kernel=graph.at("kernels").at(index);need(kernel.contains("launch"),"full-grid kernel launch contract");
        const auto& templ=*templates.at(kernel.at("template_id").get<std::string>());
        Binding binding(templ,p::natural(kernel.at("address_base")),&kernel.at("launch"),cta_stride);
        need(binding.ctas()<=U(INT_MAX),"CTA count exceeds CtaGraphStore domain");
        const auto nodes=binding.nodes(0);const U ctas=binding.ctas();std::vector<g::CtaGraphStore::Span> spans;spans.reserve(ctas);
        for(U c=0;c<ctas;++c)spans.push_back({int(binding.first_node(c)),int(nodes.size())});
        g::DAG dag;std::unique_ptr<g::CtaGraphStore> store;
        g::CtaGraphStore::Limits limits;limits.max_nodes_per_cta=nodes.size();limits.max_explicit_ranges_per_cta=nodes.size();
        limits.max_live_nodes=std::max<std::size_t>(nodes.size(),std::size_t(std::min<U>(ctas,96))*nodes.size());
        limits.max_live_explicit_ranges=limits.max_live_nodes;
        g::CtaGraphStore::Spec spec{};spec.cta_count=int(ctas);spec.warps_per_cta=int(binding.warps(0));spec.sm_count=cfg.num_sms;
        spec.total_nodes=nodes.size()*ctas;spec.spans=spans;spec.resident_cta_limit_per_sm=int(binding.resident_limit());spec.per_sm_warp_placement=true;
        spec.allow_declared_tensor_work=true;spec.allow_abstract_async_copy=false;spec.allow_observed_async_shared_service=false;
        store=std::make_unique<g::CtaGraphStore>(spec,[&](int c){
            g::CtaGraphStore::Owned out;const auto local=binding.nodes(U(c));const int first=int(binding.first_node(U(c)));out.reserve(local.size());
            for(const auto& source:local) {
                std::vector<int> completion;for(const auto dep:source.completion_dependencies)completion.push_back(first+int(dep));
                std::string pipeline=source.pipeline,op="compute";int elements=int(source.compute_elements);g::DataType dtype=g::DataType::FP16;
                if(source.kind==Kind::Global){pipeline=source.write?"ST":"LD";op=source.write?"st.reg2dram":"ld.dram2reg";elements=128;}
                else if(source.kind==Kind::Tensor){pipeline="Tensor";op="mma";elements=16384;}
                else if(source.kind==Kind::Control){pipeline="SIMD";op="compute";elements=1;}
                else if(source.kind==Kind::Barrier){pipeline="BARRIER";op="barrier";elements=1;}
                g::Tile tile(0,0,1,std::max(1,elements));auto node=std::make_unique<g::DAGNode>(first+int(source.id),"tilegraph-c"+std::to_string(c)+"-n"+std::to_string(source.id),pipeline,op,
                    g::cta_placement::token(c,source.warp,binding.warps(U(c)),cfg.num_sms,4,true),completion,0,tile,dtype);
                node->sm_id=c%cfg.num_sms;node->thread_block_id=c;node->compiler_cta_id=c;node->source_ordinal=int(source.source_ordinal);
                for(const auto dep:source.issue_dependencies)node->issue_depends_on.push_back(first+int(dep));
                if(source.kind==Kind::Global){const auto descriptor=binding.memory(U(c),source.id);node->matrix_id=1;node->memory_access_granularity_bytes=1;node->memory_coalesce_bytes=128;node->explicit_memory_subops=descriptor.global_subops;}
                if(source.kind==Kind::Tensor)node->declare_tensor_fma_work(int(source.tensor_fma/16384));
                out.push_back(std::move(node));
            }
            return out;
        },[&](int,const std::vector<g::DAGNode*>&,g::Cycle){} ,limits);
        dag.cta_graph_store=store.get();session.set_next_kernel_resident_cta_limit(int(binding.resident_limit()));
        auto result=session.run_kernel(&dag,g::Cycle(2000000000),false,nullptr,nullptr,g::Cycle(10000000));
        need(store->telemetry().at("live_nodes")==0&&store->telemetry().at("live_ctas")==0,"resident CTA graph not reclaimed");
        dag.cta_graph_store=nullptr;store.reset();
        total_ctas+=ctas;total_nodes+=nodes.size()*ctas;
        for(U c=0;c<ctas;++c)for(const auto& source:nodes)if(source.kind==Kind::Global){const auto d=binding.memory(c,source.id);if(source.write)total_write+=d.global_bytes;else total_read+=d.global_bytes;}
        if((index+1)%10==0)std::cerr<<J({{"progress_completed_kernels",index+1},{"total_kernels",graph.at("kernels").size()},{"completed_ctas",total_ctas},{"cycle",cycle},{"kernel_cycles",result.kernel_end_cycle-result.start_cycle}}).dump()<<'\n';
    }
    need(l2->is_quiescent(),"full TileGraph run left L2 non-quiescent");memory.backend->finalize();const auto physical=memory.backend->physical_statistics();const auto admission=memory.backend->admission_statistics();
    std::cout<<J({{"schema","TILEGEN_TILEGRAPH_FULL_RESULT_V1"},{"status","PASS_TILEGEN_HBFSIM_FULLGRID_1030"},{"kernels",1030},{"ctas",total_ctas},{"nodes",total_nodes},
        {"resident_cta_limit",4},{"cycle",cycle},{"logical_read_bytes",total_read},{"logical_write_bytes",total_write},{"physical_read_bytes",physical.read_bytes},{"physical_write_bytes",physical.write_bytes},
        {"admitted_requests",admission.accepted},{"completed_requests",admission.completed},{"peak_live",admission.peak_live},{"trace_used",false},{"hardware_calibration_used",false},
        {"GPU_executed",false},{"HBFSIM_executed",true},{"standard_full_grid",true},{"host_seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()}}).dump()<<'\n';return 0;
}
} // namespace tg_input
int main(int argc,char** argv) {try {if(argc==2)return tg_input::run(argv[1]);if(argc==4&&std::string(argv[1])=="--full")return tg_input::run_full(argv[2],argv[3]);throw std::runtime_error("usage: tilegraph_input_check TILEGRAPH.json | tilegraph_input_check --full TILEGRAPH.json MEMORY_CONTROL.json");}catch(const std::exception& error) {std::cerr<<nlohmann::json({{"status","REJECTED"},{"reason",error.what()}}).dump()<<'\n';return 2;}}
