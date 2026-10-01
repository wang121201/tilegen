#pragma once
#include "candidate_direct_cache.h"
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <functional>
#include <ostream>
#include <set>

namespace source_cache {
using J=nlohmann::json;using U=std::uint64_t;
inline void need(bool v,const char* m){if(!v)throw std::invalid_argument(m);}
inline U number(const J& x,U lo=0,U hi=UINT64_MAX){need(x.is_number_integer()&&!x.is_boolean(),"unsigned integer required");if(!x.is_number_unsigned())need(x.get<std::int64_t>()>=0,"negative integer");U n=x.get<U>();need(n>=lo&&n<=hi,"integer outside range");return n;}
inline U add(U a,U b){need(a<=UINT64_MAX-b,"counter overflow");return a+b;}
inline U product(const J& x){need(x.is_array()&&x.size()==3,"xyz geometry");U n=1;for(const auto& v:x){U y=number(v,1,UINT32_MAX);need(n<=UINT64_MAX/y,"geometry overflow");n*=y;}return n;}
inline bool boolean(const J& x){need(x.is_boolean(),"boolean required");return x.get<bool>();}
inline GTSim::PerSmL1Config l1_config(){return llm_l1::config(llm_l1::assumed_shared());}
// Process-fixed cache policy; no tensor, address or NCU-dependent selection.
inline direct_native::DataPolicy selected_data_policy(){static const auto p=[](){const char* raw=std::getenv("TILEGEN_L2_DATA_POLICY");const std::string value=raw?raw:"sector32";need(value=="old128"||value=="sector32","TILEGEN_L2_DATA_POLICY must be old128 or sector32");return value=="sector32"?direct_native::DataPolicy::SECTOR32:direct_native::DataPolicy::OLD128;}();return p;}
inline int selected_ef_hit_rate(){static const int rate=[](){const char* raw=std::getenv("TILEGEN_EF_HIT_RATE");if(!raw)return 288;std::string v(raw);need(!v.empty()&&v.size()<=5&&v.find_first_not_of("0123456789")==std::string::npos,"TILEGEN_EF_HIT_RATE must be decimal 0..65536");unsigned n=std::stoul(v);need(n<=65536,"TILEGEN_EF_HIT_RATE outside 0..65536");return int(n);}();return rate;}
inline U selected_dirty_age(){static const U budget=[](){need(std::getenv("TILEGEN_L2_DIRTY_PRESSURE_MAX_LINES")==nullptr,"dirty age candidate does not combine pressure policy");const char* raw=std::getenv("TILEGEN_L2_DIRTY_AGE_ACCESSES");if(!raw)return U{0};std::string v(raw);need(!v.empty()&&v.size()<=20&&v.find_first_not_of("0123456789")==std::string::npos,"TILEGEN_L2_DIRTY_AGE_ACCESSES must be unsigned decimal");return U(std::stoull(v));}();return budget;}
inline direct_native::DirtyAgeClock selected_dirty_age_clock(){static const auto clock=[](){const char* raw=std::getenv("TILEGEN_L2_DIRTY_AGE_CLOCK");const std::string value=raw?raw:"global";need(value=="global"||value=="set","TILEGEN_L2_DIRTY_AGE_CLOCK must be global or set");return value=="set"?direct_native::DirtyAgeClock::SET_FORWARDED_LINE:direct_native::DirtyAgeClock::GLOBAL_FORWARDED_LINE;}();return clock;}
inline direct_native::DiagnosticOptions options(){direct_native::DiagnosticOptions o;o.ef_hit_rate=selected_ef_hit_rate();o.dirty_age_accesses=selected_dirty_age();o.dirty_age_clock=selected_dirty_age_clock();o.data_policy=selected_data_policy();return o;}
inline J configuration(){return {{"L1",llm_l1::description(l1_config(),llm_l1::assumed_shared(),llm_l1::initial_origin())},{"L2",{{"bytes",41943040},{"geometry","PAPER_ADA_L2_V1_20x1024x16"},{"data_policy",selected_data_policy()==direct_native::DataPolicy::SECTOR32?"sector32":"old128"},{"read_fill_RFO_bytes",selected_data_policy()==direct_native::DataPolicy::SECTOR32?32:128},{"write_allocate",true},{"partial_writeback_policy","MASKED_32B_REQUEST_NO_PREMERGE"},{"writeback_byte_mask_preserved",true},{"writeback_byte_mask_scope","functional_ledger_observer_and_companion_hash_not_native_trace_Record_or_HBFSIM_payload"},{"L2_hits_scope","resident_128B_tag_hits_including_sector_misses_not_NCU_hit_rate"},{"L2_sector_read_counters_scope","forwarded_read_and_atomic_sectors_only_not_NCU_hit_rate"},{"store_fetch_policy",selected_data_policy()==direct_native::DataPolicy::SECTOR32?"LAZY_SECTOR_KNOWN_BYTES":"FULL_LINE_RFO"},{"writeback_request_bytes",32},{"EF_hit_numerator",selected_ef_hit_rate()},{"EF_hit_denominator",65536},{"skip_store_RFO",selected_data_policy()==direct_native::DataPolicy::SECTOR32},{"dirty_age_accesses",selected_dirty_age()},{"dirty_age_clock",direct_native::dirty_age_clock_name(selected_dirty_age_clock())},{"dirty_age_budget_unit",direct_native::dirty_age_budget_unit(selected_dirty_age_clock())},{"dirty_age_group_count",20480},{"dirty_age_group_mapping","frozen L2GroupedLru.group(original_byte_VA); matrix namespace excluded"},{"dirty_age_whole_line_store_refresh",true},{"dirty_age_policy","LAST_STORE_L2_ACCESS_BUDGET_KEEP_CLEAN_TAG_LRU_UNCHANGED"}}},{"sm_policy","MODELED_CTA_LINEAR_ID_MOD_48_NOT_OBSERVED_SM"},{"issue_order","SERIAL_CALLER_STREAM_ORDER_NOT_GPU_TIMING"},{"DMA_model","L2_COHERENT_FUNCTIONAL_128B_CHUNKS_NOT_HARDWARE_QUALIFIED"},{"end_flush",false},{"cache_policy_modified",(selected_data_policy()==direct_native::DataPolicy::SECTOR32||selected_ef_hit_rate()!=288||selected_dirty_age()!=0)}};}
inline J difference(const J& a,const J& b){J r=J::object();for(auto i=b.begin();i!=b.end();++i)if(i.key().find("fnv")==std::string::npos&&i.key().find("resident")==std::string::npos&&i.key().find("capacity")==std::string::npos&&i.key()!="dirty_tail_bytes"&&i.value().is_number_unsigned()&&a.contains(i.key())&&a.at(i.key()).is_number_unsigned()){U x=a.at(i.key()).get<U>(),y=i.value().get<U>();need(y>=x,"cumulative counter regressed");r[i.key()]=y-x;}return r;}
struct Effect {std::string operation;U cta=0,warp=0,pc=0,width=0,effective=0,global_mask=0;std::array<U,32> addresses{};};
struct Policy {bool bypass=false,low_priority=false;std::string semantic="other";};
inline std::function<void(const native_trace::Record&)> postcache_stream_observer;
class Runner {
 std::ostream& out_;std::unique_ptr<direct_native::FunctionalCache> cache_;
 bool started_=false,ended_=false,open_=false,is_api_=false,current_write_=false,current_atomic_=false;
 U id_=0,ordinal_=0,kernels_=0,apis_=0,memory_events_=0,zero_events_=0,api_ranges_=0,api_chunks_=0;
 U kernel_payload_=0,api_payload_=0,read_effect_=0,write_effect_=0,atomic_events_=0,g2s_events_=0;
 U source_hash_=14695981039346656037ULL,post_hash_=14695981039346656037ULL;
 U grid_=0,warps_=0;std::string phase_,semantic_,api_;J before_,last_snapshot_,grid_xyz_,block_xyz_;std::map<std::string,std::array<U,2>> roles_;
 std::set<U> kernel_ids_,api_ids_;std::clock_t cpu0_=std::clock(),operation_cpu_=cpu0_,snapshot_cpu_=cpu0_;std::chrono::steady_clock::time_point wall0_=std::chrono::steady_clock::now();
 std::vector<GTSim::ExplicitMemorySubop> subops_{1};
 struct OperationOwner {std::string phase,semantic;};
 std::map<U,OperationOwner> operation_owners_;
 using OwnershipKey=std::array<std::string,7>;
 std::map<OwnershipKey,std::array<U,3>> writeback_owners_;
 const OperationOwner& owner(U ordinal)const{auto it=operation_owners_.find(ordinal);need(it!=operation_owners_.end(),"dirty owner ordinal lacks phase");return it->second;}
 void account_writeback(const direct_native::DirtyOwner& o,const direct_native::Context& trigger,direct_native::WritebackReason reason,std::uint32_t mask){
  const auto& first=owner(o.first);const auto& last=owner(o.last);const auto& cause=owner(trigger.call_index);
  auto& row=writeback_owners_[{first.phase,last.phase,cause.phase,direct_native::writeback_reason_name(reason),first.semantic,last.semantic,cause.semantic}];row[0]+=32;row[1]+=std::popcount(mask);row[2]+=mask!=UINT32_MAX;
 }
 J dirty_ownership()const{
  J wb=J::array(),tail=J::array();
  for(const auto& [key,v]:writeback_owners_)wb.push_back({{"first_writer_phase",key[0]},{"last_writer_phase",key[1]},{"trigger_phase",key[2]},{"reason",key[3]},{"first_writer_semantic",key[4]},{"last_writer_semantic",key[5]},{"trigger_semantic",key[6]},{"write_bytes",v[0]},{"enabled_write_byte_coverage",v[1]},{"masked_writeback_requests",v[2]},{"writeback_merge_read_bytes",0}});
  std::map<std::array<std::string,4>,U> carried;
  for(const auto& row:cache_->dirty_owner_tail()){const auto& first=owner(row.at("first_writer").get<U>());const auto& last=owner(row.at("last_writer").get<U>());carried[{first.phase,last.phase,first.semantic,last.semantic}]+=row.at("bytes").get<U>();}
  U total=0;for(const auto& [key,bytes]:carried){total+=bytes;tail.push_back({{"first_writer_phase",key[0]},{"last_writer_phase",key[1]},{"first_writer_semantic",key[2]},{"last_writer_semantic",key[3]},{"dirty_bytes",bytes}});}
  need(total==cache_->snapshot().at("resident_dirty_sectors").get<U>()*32,"phase dirty carry conservation");
  return {{"scope","first/last touch of dirty sector lifetime; not per-byte ownership"},{"operation_ordinal_phase_table_entries",operation_owners_.size()},{"writebacks_cumulative",wb},{"resident_dirty_carry",tail},{"resident_dirty_bytes",total}};
 }
 void hash(U& h,U n){for(unsigned i=0;i<8;++i){h^=(n>>(8*i))&255;h*=1099511628211ULL;}}
 void emit_summary(const J& j){out_<<j.dump()<<'\n';need(bool(out_),"snapshot output failed");}
 void account(const std::string& role,U read,U write){need(!role.empty()&&role.size()<=256,"bounded semantic label");need(roles_.count(role)||roles_.size()<512,"too many semantic labels");auto& r=roles_[role];r[0]=add(r[0],read);r[1]=add(r[1],write);read_effect_=add(read_effect_,read);write_effect_=add(write_effect_,write);}
 void invoke(const Effect& e,const Policy& policy,bool api_chunk){
  bool write=e.operation=="WRITE"||e.operation=="ATOMIC_RMW";bool rmw=e.operation=="ATOMIC_RMW";bool g2s=e.operation=="GLOBAL_TO_SHARED";
  need(write||e.operation=="READ"||g2s,"unknown memory operation");need(e.width>0&&e.width<=65536,"bounded source width");need((e.global_mask&~e.effective)==0,"global effect mask exceeds instruction-effective lanes");
  U op=e.operation=="READ"?0:e.operation=="WRITE"?1:rmw?2:3;for(U x:{op,e.cta,e.warp,e.pc,e.width,e.effective,e.global_mask})hash(source_hash_,x);for(U a:e.addresses)hash(source_hash_,a);
  auto& sub=subops_[0];sub.ranges.clear();sub.source_member_ordinals.clear();sub.requested_bytes=0;
  for(unsigned lane=0;lane<32;++lane)if(e.global_mask&(U{1}<<lane)){U a=e.addresses[lane];need(a<=UINT64_MAX-(e.width-1),"source address overflow");sub.ranges.push_back({int(lane),a,e.width});sub.source_member_ordinals.push_back(int(lane));sub.requested_bytes=add(sub.requested_bytes,e.width);}
  if(!api_chunk){++memory_events_;kernel_payload_=add(kernel_payload_,sub.requested_bytes);if(rmw)++atomic_events_;if(g2s)++g2s_events_;}else{++api_chunks_;api_payload_=add(api_payload_,sub.requested_bytes);}
  account(policy.semantic,(!write||rmw)?sub.requested_bytes:0,write?sub.requested_bytes:0);
  if(sub.ranges.empty()){++zero_events_;return;}
  direct_native::Context ctx;ctx.call_index=ordinal_;ctx.node_id=id_;ctx.cta=e.cta;ctx.warp=e.warp;ctx.pc=e.pc;ctx.sm_id=api_chunk?0:e.cta%48;ctx.atomic_rmw=rmw;ctx.low_priority=policy.low_priority;ctx.role_id=policy.semantic=="weights"?1:policy.semantic=="activation"?2:policy.semantic=="kv"||policy.semantic=="kvcache"?3:4;
  current_write_=write;current_atomic_=rmw;cache_->instruction(0,write,policy.bypass,subops_,ctx);
 }
public:
 explicit Runner(std::ostream& out):out_(out){cache_=std::make_unique<direct_native::FunctionalCache>(l1_config(),40ULL<<20,[](int matrix,U a){need(matrix==0,"global VA namespace");return a;},[this](const native_trace::Record& r){
   need(r.bytes==(r.cause==native_trace::Cause::DirtyWriteback?32:selected_data_policy()==direct_native::DataPolicy::SECTOR32?32:128),"cache transaction size changed");for(U n:native_trace::words(r))hash(post_hash_,n);if(postcache_stream_observer)postcache_stream_observer(r);
  },GTSim::L2GeometryConfig::paper_ada_l2_v1(),options(),direct_native::OwnerObserver{},[this](const auto& o,const auto& c,auto reason,std::uint32_t mask){account_writeback(o,c,reason,mask);});last_snapshot_=snapshot();}
 J dirty_distribution()const{J v=cache_->dirty_group_observation();v.erase("resident_dirty_lines_by_group");return v;}
 J snapshot()const{J s=cache_->snapshot();s["L1_adapter_observation"]=cache_->l1_observation();s["source_warp_events"]=memory_events_;s["zero_global_effect_events"]=zero_events_;s["source_kernel_payload_bytes_once"]=kernel_payload_;s["source_API_payload_bytes"]=api_payload_;s["source_read_effect_bytes"]=read_effect_;s["source_write_effect_bytes"]=write_effect_;s["source_atomic_RMW_events"]=atomic_events_;s["source_global_to_shared_events"]=g2s_events_;s["API_ranges"]=api_ranges_;s["API_128B_chunks"]=api_chunks_;s["kernel_boundaries"]=kernels_;s["API_boundaries_without_cache_flush"]=apis_;s["source_effect_projection_fnv1a64"]=source_hash_;s["postcache_record_fnv1a64"]=post_hash_;s["dirty_tail_bytes"]=s.at("resident_dirty_sectors").get<U>()*32;
  need(s.at("dirty_sector_creations").get<U>()*32==s.at("DRAM_write_bytes").get<U>()+s.at("dirty_tail_bytes").get<U>(),"dirty byte conservation");U reads=0,bytes=0;for(unsigned reason=0;reason<unsigned(direct_native::ReadReason::COUNT);++reason){const std::string name=direct_native::read_reason_name(direct_native::ReadReason(reason));reads+=s.at("DRAM_"+name+"_requests").get<U>();bytes+=s.at("DRAM_"+name+"_bytes").get<U>();}need(s.at("DRAM_read_requests").get<U>()==reads&&s.at("DRAM_read_bytes").get<U>()==bytes,"read reason conservation");return s;}
 void consume_effect(const Effect& e,const Policy& policy){need(started_&&!ended_&&open_&&!is_api_,"memory requires active kernel");need(e.cta<grid_&&e.warp<warps_,"source CTA/warp outside current geometry");invoke(e,policy,false);}
 void validate_program_binding(const J& b)const{need(open_&&!is_api_&&number(b.at("native_launch_id"))==id_&&b.at("phase")==phase_&&b.at("grid")==grid_xyz_&&b.at("block")==block_xyz_,"source program differs from active native kernel binding/geometry");}
 void command(const J& j){need(j.is_object()&&j.contains("type"),"command object/type required");auto type=j.at("type").get<std::string>();need(!ended_,"input after run_end");
  if(type=="run_begin"){need(!started_&&j.at("schema")=="TILEGEN_SOURCE_CACHE_STREAM_V1"&&j.at("sm_policy")=="cta_mod_48","explicit run schema/SM policy");started_=true;emit_summary({{"type","configuration"},{"configuration",configuration()},{"source_declaration",j}});return;}
  need(started_,"run_begin required");
  if(type=="allocation_metadata"){need(!open_,"allocation command inside active operation");cache_->allocation_metadata(j.at("node"));return;}
  if(type=="begin_kernel"||type=="begin_api"){
   need(!open_,"nested operation");id_=number(j.at("id"),0,UINT64_MAX-1);phase_=j.at("phase").get<std::string>();need(!phase_.empty()&&phase_.size()<=1024,"bounded phase");semantic_=j.value("semantic",std::string("other"));is_api_=type=="begin_api";need((is_api_?api_ids_:kernel_ids_).insert(id_).second,"duplicate operation id");before_=snapshot();operation_cpu_=std::clock();++ordinal_;operation_owners_.emplace(ordinal_,OperationOwner{phase_,semantic_});open_=true;
   if(is_api_){need(j.at("dma_model")=="L2_COHERENT_FUNCTIONAL_128B_CHUNKS","explicit DMA model required");api_=j.at("api").get<std::string>();++apis_;}else{grid_xyz_=j.at("grid");block_xyz_=j.at("block");grid_=product(grid_xyz_);U threads=product(block_xyz_);need(threads<=1024,"CTA threads");warps_=(threads+31)/32;cache_->begin_kernel(j);++kernels_;emit_summary({{"type","kernel_L1_configuration"},{"kernel_id",id_},{"phase",phase_},{"L1",cache_->l1_configuration()}});}return;
  }
  if(type=="memory"){
   const auto& e=j.at("event");const auto& q=j.at("policy");Effect x;x.operation=e.at("operation").get<std::string>();x.cta=number(e.at("cta_linear_id"));x.warp=number(e.at("cta_warp_id"));x.pc=number(e.at("pc"));x.width=number(e.at("width"),1,65536);x.global_mask=number(e.at("global_effective_mask"),0,UINT32_MAX);x.effective=e.contains("effective_mask")&&!e.at("effective_mask").is_null()?number(e.at("effective_mask"),0,UINT32_MAX):x.global_mask;
   need(e.value("domain",std::string("global"))=="global","only global effects accepted");Policy policy;policy.bypass=boolean(q.at("bypass_l1"));auto pri=q.at("l2_priority").get<std::string>();need(pri=="normal"||pri=="evict_first","explicit L2 priority");policy.low_priority=pri=="evict_first";policy.semantic=q.value("semantic",e.value("role",semantic_));
   const auto& addresses=e.at("lane_addresses");need(addresses.is_object()||addresses.is_array(),"lane-address object or32array");if(addresses.is_array()){need(addresses.size()==32,"32 lane address array");for(unsigned lane=0;lane<32;++lane)if(!addresses[lane].is_null())x.addresses[lane]=number(addresses[lane]);else need(!(x.global_mask&(U{1}<<lane)),"missing enabled lane");}else{U present=0;for(auto a=addresses.begin();a!=addresses.end();++a){need(!a.key().empty()&&a.key().size()<=2&&a.key().find_first_not_of("0123456789")==std::string::npos,"decimal lane key");unsigned lane=unsigned(std::stoul(a.key()));need(lane<32&&a.key()==std::to_string(lane),"canonical lane key");x.addresses[lane]=number(a.value());present|=U{1}<<lane;}need((x.global_mask&~present)==0,"missing global lane address");}
   consume_effect(x,policy);return;
  }
  if(type=="api_range"){
   need(open_&&is_api_,"api_range outside API");std::string op=j.at("operation").get<std::string>();need(op=="READ"||op=="WRITE","API ordinary read/write");U address=number(j.at("address")),bytes=number(j.at("bytes"));need(bytes==0||address<=UINT64_MAX-(bytes-1),"API range overflow");++api_ranges_;Policy policy{true,false,j.value("semantic",std::string("API_history"))};
   while(bytes){U n=std::min<U>(bytes,128-address%128);Effect e;e.operation=op;e.width=n;e.effective=e.global_mask=1;e.pc=native_trace::unknown;e.addresses[0]=address;invoke(e,policy,true);address+=n;bytes-=n;}return;
  }
  if(type=="end_kernel"||type=="end_api"){
   need(open_&&number(j.at("id"))==id_&&(type=="end_api")==is_api_,"operation end identity");J after=snapshot();emit_summary({{"type",type},{"id",id_},{"phase",phase_},{"api",is_api_?api_:""},{"semantic",semantic_},{"before",before_},{"after",after},{"delta",difference(before_,after)},{"CPU_minutes",double(std::clock()-operation_cpu_)/CLOCKS_PER_SEC/60.0},{"L2_flush",false}});open_=false;return;
  }
  if(type=="drain"){
   need(!open_,"drain inside active operation");const auto label=j.at("label").get<std::string>();const auto phase=j.at("phase").get<std::string>();need(!label.empty()&&label.size()<=256&&!phase.empty()&&phase.size()<=1024,"bounded drain label/phase");
   const auto before=snapshot();const auto owners_before=dirty_ownership();++ordinal_;operation_owners_.emplace(ordinal_,OperationOwner{phase,"explicit_drain"});direct_native::Context ctx;ctx.call_index=ordinal_;ctx.sm_id=0;
   cache_->drain(ctx);const auto after=snapshot();emit_summary({{"type","drain"},{"label",label},{"phase",phase},{"before",before},{"after",after},{"delta",difference(before,after)},{"dirty_ownership_before",owners_before},{"dirty_ownership_after",dirty_ownership()},{"diagnostic_not_natural_ROI",true},{"clean_tags_and_LRU_preserved",true},{"explicit_dirty_drain",true},{"L2_flush",false}});return;
  }
  if(type=="snapshot"){need(!open_,"snapshot inside operation");J s=snapshot();auto now=std::clock();emit_summary({{"type","snapshot"},{"label",j.at("label")},{"cumulative",s},{"dirty_ownership",dirty_ownership()},{"dirty_age_observation",cache_->dirty_age_observation()},{"dirty_group_observation",dirty_distribution()},{"delta_since_previous_snapshot",difference(last_snapshot_,s)},{"CPU_minutes_since_previous_snapshot",double(now-snapshot_cpu_)/CLOCKS_PER_SEC/60.0},{"CPU_minutes_since_run_start",double(now-cpu0_)/CLOCKS_PER_SEC/60.0},{"L2_flush",false}});last_snapshot_=s;snapshot_cpu_=now;return;}
  if(type=="run_end"){need(!open_,"run ended inside operation");cache_->verify_resident_ledger();ended_=true;emit_summary({{"type","run_end"},{"cumulative",snapshot()},{"dirty_ownership",dirty_ownership()},{"L2_flush",false}});return;}
  throw std::invalid_argument("unknown stream command");
 }
 bool ended()const{return ended_;}
 J summary()const{need(started_&&ended_&&!open_,"input missing explicit run_end");J roles=J::object();for(const auto& [name,v]:roles_)roles[name]={{"read_effect_bytes",v[0]},{"write_effect_bytes",v[1]}};return {{"status","PASS_STREAMED_SOURCE_FUNCTIONAL_CACHE_RUN"},{"configuration",configuration()},{"snapshot",snapshot()},{"dirty_ownership",dirty_ownership()},{"source_by_semantic",roles},{"EF_hit_throttle",cache_->ef_hit_throttle_observation()},{"dirty_age_observation",cache_->dirty_age_observation()},{"dirty_group_observation",dirty_distribution()},{"CPU_minutes",double(std::clock()-cpu0_)/CLOCKS_PER_SEC/60.0},{"wall_minutes",std::chrono::duration<double>(std::chrono::steady_clock::now()-wall0_).count()/60.0},{"expanded_source_records_retained",0},{"postcache_records_retained",0},{"actual_SM_or_issue_timing_claimed",false},{"no_end_flush",true},{"no_implicit_end_flush",true},{"explicit_drain_performed",snapshot().at("explicit_drain_count").get<U>()>0},{"whole_graph_input_qualification_owned_by_caller",true}};}
};
}
