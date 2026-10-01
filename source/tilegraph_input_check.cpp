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
    J evidence_;
    U address_base_=0;
public:
    explicit Binding(const J& templ, U address_base=0):address_base_(address_base) {
        const auto& local=templ.at("nodes");
        need(local.is_array()&&!local.empty(),"template has no nodes");
        const auto iterations=templ.contains("loop")?p::natural(templ.at("loop").at("iterations")):U(1);
        need(iterations>0&&iterations<=4096,"template loop iterations out of bounded input range");
        for(std::size_t i=0;i<local.size();++i) {
            need(p::natural(local.at(i).at("id"))==i,"template node ids must be dense");
            warps_=std::max(warps_,unsigned(p::natural(local.at(i).at("execution_group"))+1));
        }
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
        evidence_={{"schema","TILELANG_TILEGRAPH_KERNEL_BINDING_V1"},{"template_id",templ.at("template_id")},{"construction",templ.at("construction")},{"source_template_nodes",local.size()},{"expanded_source_nodes",nodes_.size()},{"loop_iterations",iterations},{"execution_groups",warps_},{"trace_used",false},{"hardware_calibration_used",false}};
    }
    U ctas() const override {return 1;}
    unsigned warps(U cta) const override {need(cta==0,"CTA range");return warps_;}
    unsigned resident_limit() const override {return 1;}
    U first_node(U cta) const override {need(cta==0,"CTA range");return 0;}
    std::span<const SourceNode> nodes(U cta) const override {need(cta==0,"CTA range");return nodes_;}
    U template_class(U cta) const override {need(cta==0,"CTA range");return 0;}
    MemoryDescriptor memory(U cta,unsigned member) const override {
        need(cta==0&&member<nodes_.size(),"memory descriptor range");const auto& node=nodes_.at(member);
        need(node.kind==Kind::Global||node.kind==Kind::Shared,"not a memory node");MemoryDescriptor descriptor;descriptor.write=node.write;
        descriptor.path=node.kind==Kind::Global?tiny_full::Path::DirectGlobal:tiny_full::Path::Shared;
        if(node.kind==Kind::Global) {
            const U bytes=node.memory_bytes?node.memory_bytes:128;
            const U offset=address_base_+node.memory_offset+U(node.id)*128;descriptor.global_bytes=bytes;
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
    for(const auto& item:templates) {Binding binding(*item.second);auto program=tiny_full::import_program(binding,0,1,config);need(program->logical_members==binding.nodes(0).size(),"TileGen imported node census");need(!program->groups.empty(),"TileGen imported program is empty");expanded_nodes+=binding.nodes(0).size();programs.emplace(item.first,std::move(program));}
    const auto& kernels=graph.at("kernels");need(kernels.size()==1030,"expected 1030 kernels");std::map<std::string,U> phases;
    for(std::size_t index=0;index<kernels.size();++index) {const auto& kernel=kernels.at(index);need(p::natural(kernel.at("kernel_id"))==index,"dense kernel id");need(programs.contains(kernel.at("template_id").get<std::string>()),"kernel template exists");for(const char* field:{"data_dependencies","order_dependencies"}) for(const auto& dep:kernel.at(field))need(p::natural(dep)<index,"kernel dependency is backward");++phases[kernel.at("phase").get<std::string>()];}
    need(phases==std::map<std::string,U>{{"decode_1",321},{"decode_2",321},{"prefill",388}},"phase kernel census");std::cout<<J({{"schema","TILEGEN_TILEGRAPH_INPUT_CHECK_V1"},{"status","PASS_TILEGEN_ACCEPTS_TILEGRAPH_INPUT"},{"kernels",kernels.size()},{"templates",templates.size()},{"template_expanded_nodes",expanded_nodes},{"trace_used",false},{"hardware_calibration_used",false},{"HBFSIM_executed",false},{"GPU_executed",false}}).dump()<<'\n';return 0;
}

int run_full(const char* graph_path,const char* control_path) {
    const J graph=read_json(graph_path),control=read_json(control_path);
    need(graph.at("schema")=="TILEGEN_TILELANG_TILEGRAPH_INPUT_V1"&&graph.at("kernels").size()==1030,"full TileGraph contract");
    need(control.contains("memory_model")&&control.contains("service_address_map"),"full run memory control");
    std::map<std::string,const J*> templates;for(const auto& item:graph.at("templates"))templates.emplace(item.at("template_id").get<std::string>(),&item);
    auto cfg=GTSim::make_rtx4000_ada_footprint_reference_config();cfg.silence_mode=true;cfg.workload_type=g::WorkloadType::Llama3Elementwise;
    native_sequence::Mapper source_mapper;coupling::Runtime memory(control,cfg,source_mapper);need(bool(memory.backend),"full run requires native HBF backend");
    auto l2=tiny_full::make_l2(cfg,memory);g::Cycle cycle=0;hybrid_full::FineContext session(cfg,*l2,cycle);
    U total_read=0,total_write=0,total_nodes=0;const auto started=std::chrono::steady_clock::now();
    for(std::size_t index=0;index<graph.at("kernels").size();++index) {
        const auto& kernel=graph.at("kernels").at(index);Binding binding(*templates.at(kernel.at("template_id").get<std::string>()),p::natural(kernel.at("address_base")));g::DAG dag;
        const auto nodes=binding.nodes(0);std::vector<std::unique_ptr<g::DAGNode>> owners;owners.reserve(nodes.size());
        for(const auto& source:nodes) {
            std::vector<int> completion;for(const auto dep:source.completion_dependencies)completion.push_back(int(dep));
            std::string pipeline=source.pipeline,op="compute";int elements=int(source.compute_elements);g::DataType dtype=g::DataType::FP16;
            if(source.kind==Kind::Global){pipeline=source.write?"ST":"LD";op=source.write?"st.reg2dram":"ld.dram2reg";elements=128;}
            else if(source.kind==Kind::Tensor){pipeline="Tensor";op="mma";elements=16384;}
            else if(source.kind==Kind::Control){pipeline="SIMD";op="compute";elements=1;}
            else if(source.kind==Kind::Barrier){pipeline="BARRIER";op="barrier";elements=1;}
            g::Tile tile(0,0,1,std::max(1,elements));auto node=std::make_unique<g::DAGNode>(int(source.id),"tilegraph-n"+std::to_string(source.id),pipeline,op,
                g::cta_placement::token(0,source.warp,binding.warps(0),cfg.num_sms,4,true),completion,0,tile,dtype);
            node->sm_id=0;node->thread_block_id=0;node->compiler_cta_id=0;node->source_ordinal=int(source.source_ordinal);
            for(const auto dep:source.issue_dependencies)node->issue_depends_on.push_back(int(dep));
            if(source.kind==Kind::Global){const auto descriptor=binding.memory(0,source.id);node->matrix_id=1;node->memory_access_granularity_bytes=1;node->memory_coalesce_bytes=128;node->explicit_memory_subops=descriptor.global_subops;}
            if(source.kind==Kind::Tensor)node->declare_tensor_fma_work(int(source.tensor_fma/16384));
            dag.add_node(node.release());
        }
        dag.build_dependency_graph();auto result=session.run_kernel(&dag,g::Cycle(2000000000),false,nullptr,nullptr,g::Cycle(10000000));
        total_nodes+=nodes.size();for(const auto& source:nodes)if(source.kind==Kind::Global){const auto d=binding.memory(0,source.id);if(source.write)total_write+=d.global_bytes;else total_read+=d.global_bytes;}
        if((index+1)%10==0)std::cerr<<J({{"progress_completed_kernels",index+1},{"total_kernels",graph.at("kernels").size()},{"cycle",cycle},{"kernel_cycles",result.kernel_end_cycle-result.start_cycle}}).dump()<<'\n';
    }
    need(l2->is_quiescent(),"full TileGraph run left L2 non-quiescent");memory.backend->finalize();const auto physical=memory.backend->physical_statistics();const auto admission=memory.backend->admission_statistics();
    std::cout<<J({{"schema","TILEGEN_TILEGRAPH_FULL_RESULT_V1"},{"status","PASS_TILEGEN_HBFSIM_FULL_1030"},{"kernels",1030},{"nodes",total_nodes},
        {"cycle",cycle},{"logical_read_bytes",total_read},{"logical_write_bytes",total_write},{"physical_read_bytes",physical.read_bytes},{"physical_write_bytes",physical.write_bytes},
        {"admitted_requests",admission.accepted},{"completed_requests",admission.completed},{"peak_live",admission.peak_live},{"trace_used",false},{"hardware_calibration_used",false},
        {"GPU_executed",false},{"HBFSIM_executed",true},{"host_seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()}}).dump()<<'\n';return 0;
}
} // namespace tg_input
int main(int argc,char** argv) {try {if(argc==2)return tg_input::run(argv[1]);if(argc==4&&std::string(argv[1])=="--full")return tg_input::run_full(argv[2],argv[3]);throw std::runtime_error("usage: tilegraph_input_check TILEGRAPH.json | tilegraph_input_check --full TILEGRAPH.json MEMORY_CONTROL.json");}catch(const std::exception& error) {std::cerr<<nlohmann::json({{"status","REJECTED"},{"reason",error.what()}}).dump()<<'\n';return 2;}}
