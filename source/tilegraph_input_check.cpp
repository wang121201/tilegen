#define main frozen_parent_unused_main
#include "work/tilegen-full-r1/driver-pooled-fusednorm-r1/streaming.cpp"
#undef main
#include "work/tilegen-tiny-full-r1/import.h"
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
public:
    explicit Binding(const J& templ) {
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
            node.write=input.at("write").get<bool>();node.compute_elements=p::natural(input.value("compute_elements",U(0)));node.tensor_fma=p::natural(input.value("tensor_fma",U(0)));nodes_.push_back(std::move(node));
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
    MemoryDescriptor memory(U cta,unsigned member) const override {need(cta==0&&member<nodes_.size(),"memory descriptor range");const auto& node=nodes_.at(member);need(node.kind==Kind::Global||node.kind==Kind::Shared,"not a memory node");MemoryDescriptor descriptor;descriptor.write=node.write;descriptor.path=node.kind==Kind::Global?tiny_full::Path::DirectGlobal:tiny_full::Path::Shared;return descriptor;}
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
} // namespace tg_input
int main(int argc,char** argv) {try {if(argc!=2)throw std::runtime_error("usage: tilegraph_input_check TILEGRAPH.json");return tg_input::run(argv[1]);}catch(const std::exception& error) {std::cerr<<nlohmann::json({{"status","REJECTED"},{"reason",error.what()}}).dump()<<'\n';return 2;}}
