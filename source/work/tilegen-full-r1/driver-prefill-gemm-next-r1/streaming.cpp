#include "cta_graph_store.h"
#include "simulator_session.h"
#include "session_mode.h"
#include "scheduler_observer.h"
#include "cycle_overlap.h"
#include "integration.h"
#include "backend_reporting.h"
#include "retry_host_memo.h"
#include "../driver-pooled-fusednorm-r1/sha256.h"
#include "../driver-pooled-fusednorm-r1/l2_event_audit.h"
#include <array>
#include <chrono>
#include <iostream>
#include <set>
#include <map>
#include <sys/resource.h>
#define tiny_sha compact_output_sha
#include "../observation-output-compact-r1/compact_cycle_overlap.h"
#undef tiny_sha

namespace prefill_gemm {
using J=nlohmann::json;using U=std::uint64_t;namespace g=GTSim;using Clock=std::chrono::steady_clock;
void need(bool v,const std::string& s){if(!v)throw std::runtime_error("Prefill GEMM bridge: "+s);}
U integer(const J& x,U max=UINT64_MAX){need(x.is_number_integer()&&!x.is_boolean(),"integer type");need(!x.is_number_integer()||x.is_number_unsigned()||x.get<std::int64_t>()>=0,"nonnegative integer");U v=x.get<U>();need(v<=max,"integer bound");return v;}
U add(U a,U b){need(b<=UINT64_MAX-a,"integer addition overflow");return a+b;}
U mul(U a,U b){need(!a||b<=UINT64_MAX/a,"integer multiplication overflow");return a*b;}
#include "statistics.h"
#include "role_profiles.h"
constexpr U CHILD_RSS_CAP=12ULL<<30;
U peak_rss(){struct rusage r{};need(getrusage(RUSAGE_SELF,&r)==0,"own child RSS read");return U(r.ru_maxrss);}
void host_bound(){need(peak_rss()<=CHILD_RSS_CAP,"own child 12GiB peak RSS bound");}
struct Mapper final:g::L2DramAddressMapper{U map(const g::CacheLineKey& k)const override{need(k.matrix_id==1,"single source process VA domain");return k.line_addr;}};
struct Global {int lane,role;U offset,width;};
struct Shared {int lane;U offset,width;};
struct Object {std::string role,logical;U pointer,bytes,stride;};
struct Node {int old,warp,ordinal,pc,elements;U effective;std::string kind,pipe,opcode,memop;U width=0,source_mask=0,fma=0;std::vector<int>deps,issues;std::vector<Global>global;std::vector<Shared>shared;};
struct Model {
 Profile p;int ctas;U live_nodes,live_ranges;std::vector<Node> nodes;std::vector<Object> objects;std::vector<g::CtaGraphStore::Span> spans;U ranges=0,read=0,write=0,shared_read=0,shared_write=0,copies=0,zero_copies=0,tensors=0;
 explicit Model(const J& j):p(profile(j.at("role"))),ctas(int(integer(j.at("ctas"),p.grid))),live_nodes(U(std::min(ctas,96))*p.perwarp*4),live_ranges(U(std::min(ctas,96))*p.ranges){
  need(ctas>=1&&j.at("schema")=="PREFILL_GEMM_ROLE_RAM_TEMPLATE_V1"&&integer(j.at("declared_grid"))==U(p.grid),"finite closed single-call schema");
  need(integer(j.at("warps"))==4&&integer(j.at("nodes_per_warp"))==U(p.perwarp)&&integer(j.at("native_resident_cta_cap"))==1,"native warp/resource identity");
  need(integer(j.at("native_static_shared_bytes"))==49152&&integer(j.at("native_dynamic_shared_bytes"))==U(p.shared-49152),"separate native shared fields");
  const auto& ob=j.at("objects");need(ob.is_array()&&ob.size()==3,"three typed objects");
  const std::array<std::string,3> roles={"weight","input","output"};const std::array<U,3> sizes=p.sizes,strides=p.strides;
  for(int i=0;i<3;++i){const auto&o=ob[i];Object v{o.at("role"),o.at("logical_identity"),integer(o.at("pointer")),integer(o.at("bytes")),integer(o.at("cta_stride"))};need(v.role==roles[i]&&v.bytes==sizes[i]&&v.stride==strides[i]&&!v.logical.empty()&&v.pointer%128==0,"closed role geometry");(void)add(v.pointer,v.bytes);objects.push_back(v);}
  for(int i=0;i<3;++i)for(int k=i+1;k<3;++k)need(add(objects[i].pointer,objects[i].bytes)<=objects[k].pointer||add(objects[k].pointer,objects[k].bytes)<=objects[i].pointer,"disjoint call roles");
  const auto& ns=j.at("nodes");need(ns.is_array()&&ns.size()==U(4*p.perwarp),"complete observed CTA instruction count");nodes.reserve(ns.size());std::set<int> oldids;std::set<std::pair<int,int>> visits;std::array<int,4>last{{-1,-1,-1,-1}};std::array<std::set<int>,4>memord;
  const std::set<std::string> kinds={"compute","control","async_copy","async_commit","async_wait","shared_matrix","tensor","barrier","shared","global"};
  for(std::size_t i=0;i<ns.size();++i){const auto&r=ns[i];Node n;n.old=int(integer(r.at("original_id"),U(4*p.perwarp-1)));n.warp=int(integer(r.at("warp"),3));n.ordinal=int(integer(r.at("program_ordinal"),(p.perwarp-1)));n.pc=int(integer(r.at("pc"),U(p.pcmax)));n.effective=integer(r.at("effective_mask"),UINT32_MAX);n.kind=r.at("kind");n.pipe=r.at("pipeline").is_null()?"":r.at("pipeline").get<std::string>();n.opcode=r.at("opcode");n.elements=int(integer(r.at("elements"),2048));n.width=integer(r.at("width"),16);n.memop=r.at("memory_op").is_null()?"":r.at("memory_op").get<std::string>();
   need(oldids.insert(n.old).second&&visits.emplace(n.warp,n.ordinal).second&&kinds.count(n.kind)&&n.elements>=1,"complete distinct source node identity");need(n.old==p.perwarp*n.warp+n.ordinal,"original warp-major node identity");need(n.ordinal>last[n.warp],"topological mapping preserves observed warp issue order");last[n.warp]=n.ordinal;
   std::set<int> edges;for(const auto&d:r.at("depends_on")){int v=int(integer(d,U(4*p.perwarp-1)));need(v<int(i)&&edges.insert(v).second,"topological unique completion edge");n.deps.push_back(v);}for(const auto&d:r.at("issue_depends_on")){int v=int(integer(d,U(4*p.perwarp-1)));need(v<int(i)&&edges.insert(v).second,"topological unique disjoint issue edge");n.issues.push_back(v);}
   std::set<int> global_lanes;for(const auto&v:r.at("global_ranges")){need(v.is_array()&&v.size()==4,"global lane tuple");Global x{int(integer(v[0],31)),int(integer(v[1],2)),integer(v[2]),integer(v[3],16)};need(x.width==n.width&&(x.width==4||x.width==8||x.width==16)&&global_lanes.insert(x.lane).second,"actual global lane width/identity");need((n.effective>>x.lane)&1,"global lane active");const auto&o=objects[x.role];need(add(add(x.offset,mul(U(ctas-1),o.stride)),x.width)<=o.bytes,"whole selected grid width inside own role");n.global.push_back(x);}
   std::set<int> shared_lanes;for(const auto&v:r.at("shared_ranges")){need(v.is_array()&&v.size()==3,"shared lane tuple");Shared x{int(integer(v[0],31)),integer(v[1],p.shared),integer(v[2],16)};need((x.width==4||x.width==16)&&x.offset%4==0&&add(x.offset,x.width)<=U(p.shared)&&shared_lanes.insert(x.lane).second,"actual shared lane extent");need((n.effective>>x.lane)&1,"shared lane active");n.shared.push_back(x);}
   if(n.kind=="async_copy"||n.kind=="global"){
    need(r.contains("memory_record_ordinal"),"source memory ordinal present");int o=int(integer(r.at("memory_record_ordinal"),p.memcounts[n.warp]-1));need(memord[n.warp].insert(o).second,"one-to-one memory ordinal");need(n.memop=="R"||n.memop=="W","memory direction");U b=0;for(auto&x:n.global)b+=x.width;if(n.memop=="R")read+=b;else write+=b;
   }else need(n.global.empty()&&!r.contains("memory_record_ordinal"),"no fabricated memory event");
   if(n.kind=="async_copy"){
    ++copies;need(n.opcode=="LDGSTS.E.BYPASS.LTC128B.128.CONSTANT"&&n.width==16&&n.memop=="R"&&n.shared.size()==std::size_t(__builtin_popcount(unsigned(n.effective))),"exact observed native copy form/destination");n.source_mask=integer(r.at("source_read_mask"),UINT32_MAX);U actual=0;for(auto&x:n.global)actual|=U(1)<<x.lane;need(actual==n.source_mask&&(n.source_mask&~n.effective)==0,"independent source-read mask");need(integer(r.at("zero_fill_mask"),UINT32_MAX)==(n.effective&~n.source_mask),"explicit model zero-fill scope");zero_copies+=!n.source_mask;for(auto&x:n.shared){need(x.width==16&&x.offset%16==0,"copy destination16B");shared_write+=x.width;}
   }else if(n.kind=="shared"||n.kind=="shared_matrix"){
    need(!n.shared.empty()&&(n.memop=="R"||n.memop=="W"),"real shared payload");if(n.kind=="shared_matrix")need(n.opcode=="LDSM.16.M88.4"&&n.memop=="R"&&n.width==16,"closed modeled LDSM contract");for(auto&x:n.shared)if(n.memop=="R")shared_read+=x.width;else shared_write+=x.width;
   }else need(n.shared.empty(),"shared service only explicit supported nodes");
   if(n.kind=="tensor"){++tensors;n.fma=integer(r.at("declared_fma_work"));need(n.opcode=="HMMA.16816.F32.BF16"&&n.fma==2048,"actual per-instruction Tensor work model");}
   else if(n.kind=="compute"||n.kind=="control"||n.kind=="async_commit"||n.kind=="async_wait")need(n.pipe=="SIMD"||n.pipe=="SFU"||n.pipe=="SHFL","existing compute pipeline only");
   ranges+=n.global.size()+n.shared.size();nodes.push_back(std::move(n));
  }
  for(int w=0;w<4;++w){need(last[w]==(p.perwarp-1)&&memord[w].size()==std::size_t(p.memcounts[w]),"whole warp global/program coverage");int o=0;for(int n:memord[w])need(n==o++,"gap-free original memory ordinal");}
  need(read==p.read&&write==p.write&&copies==p.copies&&zero_copies==p.zero&&tensors==p.tensors,"actual single CTA operation/traffic census");
  const auto&e=j.at("template_evidence");need(integer(e.at("ranges_per_cta"))==ranges&&integer(e.at("requested_read_bytes_per_cta"))==read&&integer(e.at("requested_write_bytes_per_cta"))==write&&integer(e.at("shared_read_logical_bytes_per_cta"))==shared_read&&integer(e.at("shared_write_logical_bytes_per_cta"))==shared_write,"independent bridge template census");
  need(ranges==p.ranges&&live_ranges<=40820736&&live_nodes<=4190592,"finite declared graph capacity");
  for(int c=0;c<ctas;++c)spans.push_back({int(c*nodes.size()),int(nodes.size())});
 }
 J service_map()const{std::vector<std::pair<U,U>> ss;for(auto&o:objects)ss.emplace_back(o.pointer,(o.pointer+o.bytes+127)/128*128);std::sort(ss.begin(),ss.end());std::vector<std::pair<U,U>> merged;for(auto v:ss){if(!merged.empty()&&v.first<=merged.back().second)merged.back().second=std::max(v.second,merged.back().second);else merged.push_back(v);}J out=J::array();U base=0;for(auto v:merged){out.push_back({{"source_base",v.first},{"bytes",v.second-v.first},{"service_base",base}});base+=v.second-v.first;}return {{"schema","SG_SOURCE_TO_SERVICE_MAP_V1"},{"qualification","PACKED_SOURCE_GENERATED_128B_LINES_NOT_PHYSICAL_HARDWARE_ADDRESSES"},{"spans",out}};}
};
struct Builder {
 const Model&m;U built=0,retired=0,retired_ctas=0,completion_edges=0,issue_edges=0,read=0,write=0,copy_done=0,zero_done=0,tensor_work=0;std::vector<std::string>digests;tiny_sha::Sha256 addresses;
 explicit Builder(const Model& model):m(model),digests(model.ctas){}
 void word(U v){std::array<char,8>b{};for(int i=0;i<8;++i)b[i]=char(v>>(8*i));addresses.add(b.data(),b.size());}
 g::CtaGraphStore::Owned build(int c){host_bound();g::CtaGraphStore::Owned out;out.reserve(m.nodes.size());int first=m.spans[c].first_node;
  for(std::size_t i=0;i<m.nodes.size();++i){const auto&t=m.nodes[i];if(i%4096==0)host_bound();std::vector<int>deps;for(int x:t.deps)deps.push_back(first+x);std::string pipe=t.pipe,op="compute";
   if(t.kind=="async_copy"){pipe="LD";op="cp.dram2sram_ldgsts";}
   else if(t.kind=="global"){pipe=t.memop=="R"?"LD":"ST";op=t.memop=="R"?"ld.dram2reg":"st.reg2dram";}
   else if(t.kind=="shared"||t.kind=="shared_matrix"){pipe=t.memop=="R"?"LD":"ST";op=t.memop=="R"?"ld.sram2reg":"st.reg2sram";}
   else if(t.kind=="tensor"){pipe="Tensor";op="mma";}else if(t.kind=="barrier"){pipe="BARRIER";op="barrier";}
   g::Tile tile(0,0,1,t.elements);if(t.kind=="tensor")tile=g::Tile(0,0,16,8);
   auto n=std::make_unique<g::DAGNode>(first+int(i),"prefill_gemm."+std::to_string(first+i),pipe,op,g::cta_placement::token(c,t.warp,4,48,4,true),deps,0,tile,t.kind=="tensor"?g::DataType::BF16:g::DataType::FP32);
   n->sm_id=c%48;n->thread_block_id=c;n->compiler_cta_id=c;n->source_ordinal=t.old;n->graph_node_id=std::to_string(t.warp)+":"+std::to_string(t.ordinal)+":"+std::to_string(t.pc);n->semantic_role=t.opcode;n->matrix_id=1;
   for(int x:t.issues)n->issue_depends_on.push_back(first+x);
   if(t.kind=="global"||t.kind=="async_copy"){
    g::ExplicitMemorySubop sub;for(auto&x:t.global){const auto&o=m.objects[x.role];U addr=add(o.pointer,add(x.offset,mul(U(c),o.stride)));sub.ranges.push_back({x.lane,addr,x.width});sub.source_member_ordinals.push_back(x.lane);sub.requested_bytes+=x.width;word(U(c));word(i);word(U(x.lane));word(addr);word(x.width);}n->explicit_memory_subops.push_back(std::move(sub));n->memory_access_granularity_bytes=int(t.width);n->memory_coalesce_bytes=128;
   }
   if(t.kind=="shared"||t.kind=="shared_matrix"||t.kind=="async_copy"){
    g::ExplicitMemorySubop sub;for(auto&x:t.shared){sub.ranges.push_back({x.lane,x.offset,x.width});sub.source_member_ordinals.push_back(x.lane);sub.requested_bytes+=x.width;}
    if(t.kind=="async_copy"){n->explicit_async_shared_service_v1=true;n->async_copy_bypass_l1=true;n->async_copy_shared_subops.push_back(std::move(sub));}
    else{n->explicit_sram_bank_service_v1=true;n->explicit_memory_subops.push_back(std::move(sub));n->memory_access_granularity_bytes=int(t.width);}
   }
   if(t.kind=="tensor")n->declare_tensor_fma_work(16);
   out.push_back(std::move(n));++built;
  }return out;
 }
 void retire(int c,const std::vector<g::DAGNode*>& ns,g::Cycle cy){need(ns.size()==m.nodes.size(),"CTA retirement count");tiny_sha::Sha256 h;const int first=m.spans[c].first_node;
  for(std::size_t i=0;i<ns.size();++i){const auto*n=ns[i];const auto&t=m.nodes[i];need(n->id==first+int(i)&&n->finished&&n->issue_done&&n->issue_deps_resolved&&n->remaining_deps==0,"node identity/completion closed");need(n->pending_transactions==(n->op_type==g::OpType::BARRIER?1:0)&&n->end<=cy,"transactions/retirement cycle closed");
   for(int d:n->depends_on){need(n->start>=ns.at(d-first)->end+1,"completion edge timing");++completion_edges;}for(int d:n->issue_depends_on){need(n->start>=ns.at(d-first)->issue_cycle+1,"issue edge timing");++issue_edges;}
   if(t.kind=="async_copy"){need(n->async_copy_phase==3&&n->tma_issue_complete_cycle>=n->issue_cycle&&n->end>n->tma_issue_complete_cycle,"copy global acknowledgement then SRAM destination completion");++copy_done;zero_done+=!t.source_mask;}
   if(t.kind=="tensor")tensor_work+=U(n->tensor_fma_work_per_subpartition());
   if(t.kind=="global"||t.kind=="async_copy")for(auto&x:t.global){if(t.memop=="R")read+=x.width;else write+=x.width;}
   std::array<g::Cycle,14>values{{n->id,t.old,t.warp,t.ordinal,t.pc,n->start,n->end,n->issue_cycle,n->ready_cycle,n->total_transactions,n->next_transaction_index,n->pending_transactions,n->remaining_deps,n->tma_issue_complete_cycle}};std::array<char,14*8>bytes{};for(std::size_t k=0;k<values.size();++k)for(int b=0;b<8;++b)bytes[k*8+b]=char(U(values[k])>>(8*b));h.add(bytes.data(),bytes.size());++retired;
  }digests[c]=h.hex();++retired_ctas;std::cerr<<J({{"progress_completed_CTAs",retired_ctas},{"total_CTAs",m.ctas},{"role",m.p.name},{"continuous_cycle",cy}}).dump()<<std::endl;
 }
 J receipt(){tiny_sha::Sha256 h;for(auto&s:digests){need(s.size()==64,"all CTA audit hashes");h.add(s+"\n");}return {{"built_nodes",built},{"retired_nodes",retired},{"retired_CTAs",retired_ctas},{"completion_edges_checked",completion_edges},{"issue_edges_checked",issue_edges},{"requested_read_bytes",read},{"requested_write_bytes",write},{"async_copy_nodes_completed",copy_done},{"zero_source_copy_nodes_completed",zero_done},{"declared_tensor_fma_work",tensor_work},{"per_CTA_node_semantics_sha256",digests},{"ordered_CTA_node_semantics_sha256",h.hex()},{"modeled_address_materialization_sha256",addresses.hex()}};}
};
J run(J input,bool validate,bool eager,bool events){auto start=Clock::now();Model m(input);need(input.at("service_address_map")==m.service_map(),"single call typed object service map");input.erase("nodes");
 host_bound();auto cfg=g::make_rtx4000_ada_footprint_reference_config();cfg.silence_mode=true;cfg.workload_type=g::WorkloadType::Llama3Elementwise;Mapper mapper;coupling::Runtime memory(input,cfg,mapper);need(bool(memory.backend),"original external HBFSIM required");
 J result={{"schema","PREFILL_GEMM_SINGLE_ROLE_EXPLICIT_MODEL_RESULT_V1"},{"status",validate?"VALIDATED_NOT_EXECUTED":"PREFILL_GEMM_EXPLICIT_STRUCTURAL_MODEL_EXECUTED"},{"executed_CTAs",m.ctas},{"role",m.p.name},{"declared_grid_CTAs",m.p.grid},{"nodes_per_CTA",m.nodes.size()},{"full_grid",m.ctas==m.p.grid},{"same_source_sample_CTA0",m.ctas==1 && input.at("template_evidence").at("source_process")==input.at("template_evidence").at("selected_call_process") && input.at("template_evidence").at("source_launch_key")==input.at("template_evidence").at("selected_call_key")},{"template_evidence",input.at("template_evidence")},{"objects",input.at("objects")},{"service_address_map",input.at("service_address_map")},{"materialization",eager?"bounded_eager_reference":"resident_CTA"},{"event_mode",events},{"native_hardware_timing_qualified",false},{"implicit_register_dependencies_complete",false},{"full_LLM_state",false},{"full_LLM_time_ns",nullptr},{"full_LLM_bandwidth_GBps",nullptr},{"same_scope_NCU_measurement_available",false},{"hardware_error_percent",nullptr},{"cache_history","single selected source or structurally estimated call from cold model cache; intermediate workflow omitted"},{"boundary_policy",{{"quiescence",true},{"final_dirty_flush",false}}},{"resources",{{"native_resident_CTA_cap",1},{"native_static_shared_bytes",49152},{"native_dynamic_shared_bytes",m.p.shared-49152},{"registers",m.p.regs}}},{"graph_capacity",{{"nodes_per_CTA",m.nodes.size()},{"ranges_per_CTA",m.ranges},{"total_nodes",m.nodes.size()*m.ctas},{"total_ranges",m.ranges*m.ctas},{"maximum_live_nodes",m.live_nodes},{"maximum_live_ranges",m.live_ranges}}},{"no_expanded_trace_saved",true},{"copy_cache_policy",{{"known_BYPASS_L1_per_node",true},{"L2_policy_unchanged",true},{"CONSTANT_policy_recovered",false},{"unread_source_lanes_issue_global",false}}},{"inherited_compute_service",{{"tensor_width",cfg.tensor_core_width},{"tensor_latency_cycles",cfg.tensor_core_latency_cycles},{"declared_FMA_per_observed_HMMA",2048},{"issue_cycles_per_HMMA_warp",(2048+cfg.tensor_core_width-1)/cfg.tensor_core_width},{"hardware_calibrated",false}}}};
 result["host_capacity"]={{"DAGNode_sizeof",sizeof(g::DAGNode)},{"ExplicitMemorySubop_sizeof",sizeof(g::ExplicitMemorySubop)},{"Node_template_sizeof",sizeof(Node)},{"Global_range_sizeof",sizeof(Global)},{"Shared_range_sizeof",sizeof(Shared)},{"declared_live_nodes",m.live_nodes},{"declared_live_ranges",m.live_ranges},{"child_RSS_cap_bytes",CHILD_RSS_CAP},{"peak_child_RSS_bytes",peak_rss()}};if(validate)return result;
 need(!eager||m.ctas==1,"eager reference bounded to one actual CTA");g::ContinuousL2Session session(cfg,memory.backend.get(),memory.mapper.get());session.set_next_kernel_resident_cta_limit(1);tiny_runtime::sp_event_mode=events;Builder builder(m);g::DAG dag;std::unique_ptr<g::CtaGraphStore> store;
 g::CtaGraphStore::Limits limits;limits.max_nodes_per_cta=m.nodes.size();limits.max_explicit_ranges_per_cta=m.ranges;limits.max_live_nodes=m.live_nodes;limits.max_live_explicit_ranges=m.live_ranges;
 if(eager){for(auto&n:builder.build(0))dag.add_node(n.release());}else{g::CtaGraphStore::Spec spec{};spec.cta_count=m.ctas;spec.warps_per_cta=4;spec.sm_count=48;spec.total_nodes=m.nodes.size()*U(m.ctas);spec.spans=m.spans;spec.resident_cta_limit_per_sm=1;spec.per_sm_warp_placement=true;spec.allow_declared_tensor_work=true;spec.allow_abstract_async_copy=false;spec.allow_observed_async_shared_service=true;store=std::make_unique<g::CtaGraphStore>(spec,[&](int c){return builder.build(c);},[&](int c,const auto&ns,g::Cycle cy){builder.retire(c,ns,cy);},limits);dag.cta_graph_store=store.get();}
 bool observe=input.at("aggregate_observations").get<bool>();g::scheduler_observer::reset();g::scheduler_observer::set_enabled(observe);g::cycle_overlap::begin(48,4,0,observe);host_range_audit::L2Events audit;auto tick=Clock::now();const auto r=session.run_kernel(&dag,g::Cycle(integer(input.at("max_kernel_cycles"),100000000)),false,&audit,nullptr,10000000);double engine=std::chrono::duration<double>(Clock::now()-tick).count();J overlap=tilegen_observation_output::compact_cycle_overlap(g::cycle_overlap::finish(r.kernel_end_cycle));g::scheduler_observer::set_enabled(false);
 if(eager)builder.retire(0,dag.nodes,session.cycle());need(builder.built==builder.retired&&builder.retired==m.nodes.size()*U(m.ctas)&&builder.retired_ctas==U(m.ctas),"whole prefix node/CTA retirement");need(builder.read==m.read*U(m.ctas)&&builder.write==m.write*U(m.ctas)&&builder.copy_done==m.copies*U(m.ctas)&&builder.zero_done==m.zero_copies*U(m.ctas)&&builder.tensor_work==m.tensors*2048*U(m.ctas),"retired address/async/Tensor work count");need(session.is_quiescent()&&r.start_cycle==0&&r.quiescent_end_cycle==session.cycle(),"single session quiescence/cycle closure");
 auto s=session.statistics();need(s.pre_l1_reads==m.read/128*U(m.ctas) && s.pre_l1_writes==m.write/128*U(m.ctas) && s.l1_bypassed_transactions==s.pre_l1_reads+s.pre_l1_writes && s.l1_read_hits==0 && s.l1_read_misses==0 && s.l1_write_hits==0 && s.l1_write_misses==0,"all actual LDGSTS reads and direct stores use their configured L1 bypass paths");need(s.accepted_transactions==s.processed_transactions&&s.dram_fill_requests==s.dram_fill_completions&&s.dram_writeback_requests==s.dram_writeback_completions,"core memory completions");memory.backend->finalize();sg_hbf::require_external_only(session.l2());auto physical=memory.backend->physical_statistics();need(physical.read_bytes==s.dram_fill_requests*128&&physical.write_bytes==s.dram_writeback_requests*128,"native backend/core byte conservation");const auto& credits=memory.backend->admission_statistics();need(credits.accepted==credits.completed&&credits.reserved_bursts==0&&memory.backend->queue_depth()==0,"finite backend drain");
 const auto& clock=input.at("memory_model").at("clock");long double ns_per_cycle=clock.at("period_ps_numerator").get<long double>()/clock.at("period_ps_denominator").get<long double>()/1000;double ns=double(session.cycle()*ns_per_cycle);
 result["execution"]={{"start_cycle",r.start_cycle},{"kernel_end_cycle",r.kernel_end_cycle},{"quiescent_end_cycle",r.quiescent_end_cycle},{"window_cycles",session.cycle()},{"drain_cycles",r.drain_cycles},{"clock",clock},{"time_ns",ns},{"DRAM_read_bytes",physical.read_bytes},{"DRAM_write_bytes",physical.write_bytes},{"DRAM_GBps",double(physical.read_bytes+physical.write_bytes)/ns},{"before",p28::statistics(r.before)},{"after",p28::statistics(r.after)},{"node_audit",builder.receipt()},{"ordered_l2_events",audit.receipt()},{"scheduler_visits",g::scheduler_observer::snapshot_json()},{"cycle_overlap",overlap},{"end_is_quiescent",true}};
 if(store){auto tel=store->telemetry();need(tel.at("live_nodes")==0&&tel.at("live_ctas")==0&&tel.at("live_explicit_ranges")==0,"RAM graph reclaimed");result["execution"]["resident_graph"]=tel;}
 result["memory_backend"]={{"physical_statistics",tiny_hbf_reporting::native_statistics(physical)},{"memory_path",tiny_hbf_reporting::path_statistics(*memory.backend)},{"runtime_instances",1},{"continuous_sessions",1},{"finalize_calls",1}};host_bound();result["host_capacity"]["peak_child_RSS_bytes"]=peak_rss();result["host"]={{"engine_seconds",engine},{"total_seconds",std::chrono::duration<double>(Clock::now()-start).count()}};return result;
}
}
int main(int argc,char**argv){GTSim::retry_host::prefix_enabled=true;GTSim::retry_host::ready_front_enabled=true;try{bool validate=false,eager=false,events=true;std::string path;for(int i=1;i<argc;++i){std::string a=argv[i];if(a=="--validate-only")validate=true;else if(a=="--eager-reference")eager=true;else if(a=="--event-off")events=false;else{prefill_gemm::need(path.empty()&&a=="/dev/stdin","RAM input only");path=a;}}prefill_gemm::need(path=="/dev/stdin","stdin required");std::string raw;std::array<char,8192>b{};while(std::cin){std::cin.read(b.data(),b.size());raw.append(b.data(),std::cin.gcount());prefill_gemm::need(raw.size()<=128U<<20,"single CTA RAM envelope128MiB cap");}std::vector<std::set<std::string>>keys;auto cb=[&](int,nlohmann::json::parse_event_t e,nlohmann::json&v){if(e==nlohmann::json::parse_event_t::object_start)keys.emplace_back();if(e==nlohmann::json::parse_event_t::key)prefill_gemm::need(keys.back().insert(v.get<std::string>()).second,"duplicate JSON key");if(e==nlohmann::json::parse_event_t::object_end)keys.pop_back();return true;};auto input=nlohmann::json::parse(raw,cb);auto sha=tiny_sha::sha256(raw);raw.clear();raw.shrink_to_fit();auto out=prefill_gemm::run(std::move(input),validate,eager,events);out["input_sha256"]=sha;std::cout<<out.dump()<<'\n';return 0;}catch(const std::exception&e){std::cerr<<nlohmann::json({{"status","REJECTED"},{"reason",e.what()}}).dump()<<'\n';return 2;}}
