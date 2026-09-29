#define main frozen_parent_unused_main
#include "../driver-pooled-fusednorm-r1/streaming.cpp"
#undef main
#include "../canonical-silu-driver-r1/modeled_builder.h"
#include "../canonical-fusednorm-driver-r1/modeled_builder.h"
#include "../canonical-mixed643-driver-r1/rotary_class_support.h"
#include "../canonical-rotary-driver-r1/modeled_builder.h"
#include "../canonical-norm-driver-r1/modeled_builder.h"
#include "../canonical-mixed643-driver-r1/index_adapter_support.h"
#include "../canonical-mixed643-driver-r1/copy_adapter_support.h"
#define main frozen_p28_unused_main
#include "../driver-gemm-p28-r2/streaming.cpp"
#undef main
#include "../canonical-mixed643-driver-r1/p28_support.h"
#include "../canonical-mixed643-driver-r1/model_dispatch.h"
#include "legacy_prepared.h"
#include "../canonical-gemv-host-r2/modeled_builder.h"
#define main frozen_next_gemm_unused_main
#include "../driver-prefill-gemm-next-r1/streaming.cpp"
#undef main
#include "frame_bridge.h"
#include "attention_support.h"
#include "helper_support.h"
#include "../runtime-compressed-frame-r1/compressed_frame.h"
#include "model_dispatch.h"
#include "../../tilegen-hybrid-full-r1/fine_context.h"
#include "../../tilegen-hybrid-cache-r1/bindings.h"
#include "../../tilegen-hybrid-cache-r1/tiny_context.h"
#include "../../tilegen-hybrid-full-r1/options.h"
#include "../../tilegen-hybrid-cache-r1/progress_summary.h"
#include "../../../native_trace.h"
#include "../../../cache_profile.h"
namespace unified_trace { inline native_trace::Writer* writer=nullptr; }
namespace hbf_drain_cli {
inline sg_hbf::NativeServicePolicy policy;
inline bool parse(const std::string& value){
    if(value=="--hbf-drain=global")policy.drain=sg_hbf::DrainMode::Global;
    else if(value=="--hbf-drain=independent")policy.drain=sg_hbf::DrainMode::Independent;
    else if(value=="--hbf-service-quantum=1")policy.quantum=1;
    else if(value=="--hbf-service-quantum=2")policy.quantum=2;
    else if(value=="--hbf-service-quantum=4")policy.quantum=4;
    else if(value=="--hbf-service-quantum=8")policy.quantum=8;
    else return false;
    return true;
}
}
namespace canonical_full {
using namespace native_sequence;
using P28Builder=canonical_mixed643::P28Builder;
J run(const J& in,compressed_frame::Cache&frames,bool validate,bool eager,bool events,bool oracle,bool full){
 auto start=Clock::now();auto model_owner=frames.with_decoded("Helpers",[&](const std::string&raw){return std::make_unique<Model>(in,raw);});Model&model=*model_owner;bool observe=in.at("aggregate_observations");
 std::vector<std::unique_ptr<Prepared>> prepared;for(const auto& t:model.calls){try{prepared.push_back(std::make_unique<Prepared>(model,t,eager));}catch(const std::exception&e){throw std::runtime_error("Prepared "+t.family+" "+t.call->at("source_launch_key").get<std::string>()+": "+e.what());}}std::cerr<<"FULL_RUNTIME_STAGE prepared"<<std::endl;
 p::need(!oracle,"mixed adapter has no new address oracle mode; immutable family generators already checked");
 U selected_nodes=0;for(const auto&k:prepared)selected_nodes+=k->total_nodes;p::need(!full||(prepared.size()==1138&&model.available.size()==1138&&model.prefix==0&&!eager&&events),"full workflow requires all1138 in sealed actual order, full grids and event mode");p::need(validate||full||(prepared.size()<=64&&selected_nodes<=15000000),"bounded integration cap without explicit full-workflow mode");
 const auto max_cycles=g::Cycle(p::natural(in.at("max_kernel_cycles"),2000000000));p::need(max_cycles>0,"positive max cycles");for(const auto&k:prepared)if(is_helper(k->family)){const auto&h=model.helpers->at(k->call.at("source_launch_key"));p::need(k->total_nodes==h.census(k->executed).nodes&&k->warps==h.warps&&k->resident==h.resident&&k->expected_read==h.census(k->executed).read&&k->expected_write==h.census(k->executed).write,"Helper Prepared exact typed model/resource closure");}auto cfg=g::make_rtx4000_ada_footprint_reference_config();cfg.silence_mode=true;cfg.workload_type=g::WorkloadType::Llama3Elementwise;Mapper mapper;coupling::Runtime memory(in,cfg,mapper,hbf_drain_cli::policy);p::need(bool(memory.backend),"external backend required");if(unified_trace::writer)memory.backend->set_trace_sink(unified_trace::writer);
 U omitted=0,prior=0;J gaps=J::array();for(const auto&t:model.calls){const auto*c=t.call;U launch=p::natural(c->at("native_launch_binding").at("native_launch_id"));if(prior){omitted+=launch-prior-1;gaps.push_back({{"after_native_launch_id",prior},{"before_native_launch_id",launch},{"omitted_launches",launch-prior-1}});}prior=launch;}
 J result={{"schema","CANONICAL_FULL_RUNTIME_COMPRESSED_RESULT_V4"},{"status",validate?"VALIDATED_NOT_EXECUTED":"MODELED_BOUNDED_SUBSEQUENCE_EXECUTED"},{"scope","EXPLICIT_REGISTERED_MODEL_ROUTES_BOUNDED_SELECTED_EXECUTION_UNMODELED_HELPERS_REJECTED"},{"estimated_compute",true},{"estimated_address",true},{"native_target_qualified",false},{"implicit_descriptor_dependencies_complete",false},{"IndexPut_semantics",{{"indexed_cbank_values_observed",false},{"branch_value_semantics_proved",false},{"constant_cache_timing_modeled",false}}},{"PrefillCopy_semantics",{{"observed_parameter_bytes_match_source",true},{"full_functor_bytes_source_transfer_qualified",false},{"branch_value_semantics_proved",false},{"constant_cache_timing_modeled",false}}},{"full_LLM_state",false},{"complete_SGLang_phase",false},{"full_LLM_time_ns",nullptr},{"full_LLM_bandwidth_GBps",nullptr},{"same_scope_NCU_measurement_available",false},{"hardware_error_percent",nullptr},{"selected_source_count",prepared.size()},{"canonical_total_omitted_calls",1138-prepared.size()},{"all_registered_calls_executed",false},{"registered_calls",model.available.size()},{"unsupported_calls",1138-model.available.size()},{"full_workflow_requested",full},{"prepared_node_count",selected_nodes},{"prepared_CTA_count",[&]{U n=0;for(const auto&k:prepared)n+=k->executed;return n;}()},{"source_launch_id_gap",omitted},{"source_launch_gaps",gaps},{"source_process",model.workflow.at("process")},{"workflow_file",seals().at("workflow_file")},{"family_template_scope","Frozen family model loaders/builders, original source VA never executed"},{"materialization",eager?"bounded_eager_reference":"resident_CTA"},{"event_mode",events},{"aggregate_observations",observe},{"event_hash_namespace",prepared.size()==1?"NONE_SINGLE_SOURCE":"KERNEL_INDEX_PREFIX"},{"cache_history_scope","initial cold plus only selected modeled calls; omitted native launches would change actual cache state"},{"cache_identity_qualification",{{"physical_model_key","same current PID/context raw 128B VA lines"},{"logical_objects_are_distinct_metadata",true},{"allocator_lifetime_proven",false},{"physical_address_reuse_proven",false},{"VA_reuse_assumed_for_selected_sequence_model",true},{"actual_cross_phase_cache_state_recovered",false}}},{"boundary_policy",{{"quiescence_per_kernel",true},{"L2_reset",false},{"backend_reset",false},{"final_dirty_flush",false},{"native_async_store_overlap_modeled",false}}},{"L1_policy",{{"mode",g::per_sm_l1_mode_name(cfg.per_sm_l1.mode)},{"persistence",g::per_sm_l1_persistence_name(cfg.per_sm_l1.persistence)}}},{"pipeline",J::array()}};
 for(const auto& k:prepared){try{J objects;const J views=object_views(k->call);for(const auto& item:views.items()){const auto&role=item.key();const auto&o=item.value();objects[role]={{"logical_identity",o.at("logical_identity")},{"pointer",o.at("pointer")},{"bytes",o.contains("bytes")?o.at("bytes"):o.at("span_bytes")},{"root_id",o.contains("root")?o.at("root").at("id"):J(nullptr)}};}result["pipeline"].push_back({{"family",k->family},{"source_launch_key",k->call.at("source_launch_key")},{"native_launch_binding",k->call.at("native_launch_binding")},{"phase",k->call.at("phase")},{"layer",k->call.at("layer")},{"target_function_id",k->call.at("function_id")},{"objects",objects},{"executed_ctas",k->executed},{"full_declared_target_grid",k->executed==p::natural(k->call.at("grid")[0])*p::natural(k->call.at("grid")[1])*p::natural(k->call.at("grid")[2])},{"estimated_compute",true},{"estimated_address",true},{"graph_budget",k->budget},{"warp_placement",placement_receipt(int(k->executed),k->warps,k->resident)}});}catch(const std::exception&e){throw std::runtime_error("Pipeline metadata "+k->family+" "+k->call.at("source_launch_key").get<std::string>()+": "+e.what());}}
 result["source_resolution"]={{"schema","TILEGEN_XMU_SOURCE_ROOT_V1"},{"configured_root",tilegen_source::configured_root().string()},{"environment_variable","TILEGEN_XMU_SOURCE_ROOT"},{"legacy_path_prefix",tilegen_source::legacy_root_prefix()},{"legacy_path_semantics","logical provenance only; never opened directly"},{"physical_reads_confined_to_root",true}};
 result["cache_configuration"]=native_cache::profile(cfg);
 auto shared_l2=tiny_full::make_l2(cfg,memory);g::Cycle shared_cycle=0;hybrid_full::FineContext session(cfg,*shared_l2,shared_cycle);hybrid_full::TinyContext tiny_context(cfg,*shared_l2,memory,shared_cycle);hybrid_full::Bindings tiny_bindings(model,frames,*memory.mapper);hybrid_full::options.validate(model);U tiny_call_count=0;J routes=J::array();const double model_setup_seconds=std::chrono::duration<double>(Clock::now()-start).count();frame_bridge::Active<GemmCatalog>active([&]{return session.is_quiescent();});J frame_validation=J::array();std::weak_ptr<const GemmCatalog>prior_owner;U frozen_encoded_bytes=frames.receipt().at("encoded_payload_bytes").get<U>();const auto*session_l2=&session.l2();const auto*backend_identity=memory.backend.get();
 // Verify every compressed frame, but build typed preflight models only when selected.
 // Keep preflight ownership separate: its full CTA count must never be reused
 // by an execution Active whose selected prefix can have a different count.
 for(const std::string f:{"P28QKV","GEMMO","GEMMGate","GEMMDown","AttentionPrefill","AttentionDecode1","AttentionDecode2"}){if(!validate&&std::none_of(prepared.begin(),prepared.end(),[&](const auto& k){return k->family==f;})){frames.with_decoded(f,[](const std::string&){return 0;});continue;}auto it=std::find_if(model.available.begin(),model.available.end(),[&](const auto&v){return v.second.family==f;});p::need(it!=model.available.end(),"all seven framed source routes");const auto&t=it->second;U count=is_attention(f)?8:p::natural(t.call->at("grid")[0]);auto lease=frames.with_decoded(f,[&](const std::string&raw){return std::make_shared<const GemmCatalog>(t,count,raw);});frame_validation.push_back({{"family",f},{"nodes",lease->nodes},{"typed_edges",lease->edges},{"ranges",lease->ranges},{"current_child_peak_RSS_bytes",prefill_gemm::peak_rss()}});}
 result["helper_class_validation"]=model.helpers->receipt();result["frame_validation"]=frame_validation;result["frame_validation_scope"]=validate?"all_seven_framed_families":"selected_framed_families; all_eight_frame_payloads_and_decoded_SHA_verified";result["compressed_frame_cache"]=frames.receipt();result["compressed_frame_cache"]["active_typed_GEMM_or_Attention_models_max"]=1;result["compressed_frame_cache"]["retained_JSON_DOM_copies_for_GEMM"]=0;result["host_capacity"]={{"current_child_peak_RSS_bytes",prefill_gemm::peak_rss()},{"SourceBundle_DOM_copies_per_source",1},{"registered_source_encoded_bytes",seals().at("registered_pool").at("encoded_source_bytes")}};
 if(validate){model.finish();result["source_pool"]=model.receipt();result["large_frame_typed_constructions"]=active.constructions;result["runtime_instances"]=1;result["continuous_sessions"]=1;return result;}
tiny_runtime::sp_event_mode=events;const auto initial=session.statistics();g::Cycle previous=0,sum_windows=0,sum_kernel=0,sum_drain=0;U sum_r=0,sum_w=0,prior_r=0,prior_w=0;tiny_sha::Sha256 workflow_digest;double engine_total=0;
 auto execute_one=[&](std::size_t index){memory.backend->set_trace_context(index);const auto& k=*prepared[index];std::cerr<<J({{"progress_started_call",index+1},{"progress_completed_calls",index},{"total_calls",prepared.size()},{"family",k.family},{"source_launch_key",k.call.at("source_launch_key")},{"total_CTAs",k.executed},{"continuous_cycle",session.cycle()}}).dump()<<std::endl;
  const bool use_tiny=hybrid_full::options.use_tiny(k.family,k.call.at("source_launch_key").get<std::string>());
  routes.push_back({{"index",index},{"source_launch_key",k.call.at("source_launch_key")},{"family",k.family},{"route",use_tiny?"tiny":"fine"}});
  if(use_tiny){
   p::need(!eager&&events,"tiny route requires resident/event mode");
   const auto binding_started=Clock::now();auto owner=tiny_bindings.bind(k);
   const double binding_seconds=std::chrono::duration<double>(Clock::now()-binding_started).count();
   hybrid_full::TinyPolicy policy;policy.q=hybrid_full::options.q;policy.silu_q1=hybrid_full::options.silu_q1;policy.phase_width=hybrid_full::options.phase_width;policy.prefill_compute_width=hybrid_full::options.prefill_compute_width;policy.memory_epoch=hybrid_full::options.memory_epoch;policy.max_kernel_cycles=U(max_cycles);
   const J entry={{"call",k.call},{"family",k.family}};
   auto row=tiny_context.run_kernel(*owner,entry,policy,binding_seconds);
   const U begin=row.at("start_cycle"),kc=row.at("kernel_cycles"),dc=row.at("drain_cycles"),wc=row.at("window_cycles");
   p::need(begin==U(previous)&&wc==kc+dc&&begin+wc==U(session.cycle()),"hybrid continuous tiny boundary");
   p::need(row.at("CTAs")==k.executed&&row.at("logical_nodes")==k.total_nodes,"hybrid original Prepared work census");
   const auto&front=row.at("frontend");p::need(front.at("logical_global_read_bytes")==k.expected_read&&front.at("logical_global_write_bytes")==k.expected_write,"hybrid original Prepared requested bytes");
   if(index)p::need(row.at("counter_before")==result["pipeline"][index-1].at("execution").at("after"),"hybrid cumulative counters never reset");
   U rd=row.at("DRAM_read_bytes"),wr=row.at("DRAM_write_bytes");auto physical=memory.backend->physical_statistics();
   p::need(physical.read_bytes-prior_r==rd&&physical.write_bytes-prior_w==wr,"hybrid native per-call byte deltas");prior_r=physical.read_bytes;prior_w=physical.write_bytes;
   const auto&a=memory.backend->admission_statistics();
   J e={{"start_cycle",begin},{"kernel_end_cycle",begin+kc},{"quiescent_end_cycle",begin+wc},{"kernel_cycles",kc},{"drain_cycles",dc},{"window_cycles",wc},
     {"before",row.at("counter_before")},{"after",row.at("counter_after")},{"counter_delta",row.at("counter_delta")},{"gauges_after",gauges(session.statistics())},
     {"DRAM_read_bytes",rd},{"DRAM_write_bytes",wr},{"requested_read_bytes",k.expected_read},{"requested_write_bytes",k.expected_write},
     {"retired_nodes",k.total_nodes},{"retired_ctas",k.executed},{"kernel_prefixed_per_node_sha256",nullptr},{"unprefixed_per_node_sha256",nullptr},{"ordered_l2_events",nullptr},
     {"cycle_overlap",nullptr},{"scheduler_visits",nullptr},{"modeled_materialization_address_sha256",nullptr},
     {"finite_credits",{{"accepted_cumulative",a.accepted},{"completed_cumulative",a.completed},{"peak_live_cumulative",a.peak_live},{"reserved_bursts",a.reserved_bursts},{"outer_queue",memory.backend->queue_depth()}}},
     {"host_engine_seconds",row.at("host_engine_seconds")},{"tiny",row}};
   result["pipeline"][index]["execution"]=std::move(e);engine_total+=row.at("host_engine_seconds").get<double>();
   sum_windows+=wc;sum_kernel+=kc;sum_drain+=dc;sum_r+=rd;sum_w+=wr;previous=session.cycle();++tiny_call_count;
   p::need(session_l2==&session.l2()&&backend_identity==memory.backend.get()&&frames.receipt().at("encoded_payload_bytes")==frozen_encoded_bytes&&frames.size()==8&&frames.receipt().at("live_decoded_bytes")==0,"hybrid fixed cache/backend/frames");
   tiny_full::require_closed(*shared_l2,memory);
   std::cerr<<J({{"progress_completed_calls",index+1},{"total_calls",prepared.size()},{"family",k.family},{"source_launch_key",k.call.at("source_launch_key")},{"route","tiny"},{"continuous_cycle",session.cycle()}}).dump()<<std::endl;
   return;
  }
  std::shared_ptr<const GemmCatalog>lease;bool gemm=is_framed(k.family);if(gemm){active.select(k.call.at("source_launch_key"),[&]{return frames.with_decoded(k.family,[&](const std::string&raw){return std::make_shared<const GemmCatalog>(k.t,k.executed,raw);});});p::need(prior_owner.expired(),"prior real factory/graph owner released before switch");lease=active.lease();prior_owner=lease;active.begin_call();}session.set_next_kernel_resident_cta_limit(k.resident);std::unique_ptr<AttentionBuilder>ab;std::unique_ptr<helper_bridge::Builder>hb;std::unique_ptr<P28Builder>pb;std::unique_ptr<NextBuilder>gb;std::unique_ptr<canonical_gemv::ModeledBuilder>vb;std::unique_ptr<canonical_silu::ModeledBuilder>sb;std::unique_ptr<canonical_fused::ModeledBuilder>fb;std::unique_ptr<canonical_norm::ModeledBuilder>nb;std::unique_ptr<canonical_rotary::ModeledBuilder>rb;std::unique_ptr<canonical_indexput::ModeledBuilder>ib;std::unique_ptr<canonical_prefillcopy::ModeledBuilder>cb;
  if(is_helper(k.family)){hb=std::make_unique<helper_bridge::Builder>(model.helpers->at(k.call.at("source_launch_key")),k.executed);}else if(is_attention(k.family)){ab=std::make_unique<AttentionBuilder>(*lease->attentionmodel);}else if(k.family=="P28QKV"){pb=std::make_unique<P28Builder>(*lease->p28model);}else if(gemm){gb=std::make_unique<NextBuilder>(*lease->nextmodel);}else if(k.family=="GEMV"){vb=std::make_unique<canonical_gemv::ModeledBuilder>(*model.gemv,k.call,*k.bundle,*memory.mapper,k.executed);}else if(k.family=="SiLU")sb=std::make_unique<canonical_silu::ModeledBuilder>(*model.legacy->silu,k.call,*k.bundle,*memory.mapper,k.executed);else if(k.family=="FusedNorm")fb=std::make_unique<canonical_fused::ModeledBuilder>(*model.legacy->fused,k.call,*k.bundle,*memory.mapper,k.executed);else if(k.family=="PlainNorm")nb=std::make_unique<canonical_norm::ModeledBuilder>(*model.legacy->norm,k.call,*k.bundle,*memory.mapper,k.executed);else if(k.family=="IndexPut")ib=std::make_unique<canonical_indexput::ModeledBuilder>(*model.legacy->indexput,k.call,*k.index_bundle,*memory.mapper,k.executed);else if(k.family=="PrefillCopy")cb=std::make_unique<canonical_prefillcopy::ModeledBuilder>(*model.legacy->copy,k.call,*k.copy_bundle,*memory.mapper,k.executed);else rb=std::make_unique<canonical_rotary::ModeledBuilder>(*model.legacy->rotary,k.call,*k.class_bundle,*memory.mapper,k.executed);
  auto build_cta=[&,lease](int c){if(hb)return hb->build(c);if(ab)return ab->build(c);if(gb)return gb->build(c);if(vb)return vb->build(c);if(pb)return pb->build(c);if(sb)return sb->build(c);if(fb)return fb->build(c);if(nb)return nb->build(c);if(ib)return ib->build(c);if(cb)return cb->build(c);return rb->build(c);};
  auto visit=[&](auto&&fn){if(hb)fn(*hb);else if(ab)fn(*ab);else if(gb)fn(*gb);else if(vb)fn(static_cast<Builder&>(*vb));else if(pb)fn(*pb);else if(cb)fn(*cb);else if(ib)fn(*ib);else if(rb)fn(static_cast<native_cta_sequence::ClassBuilder&>(*rb));else if(sb)fn(static_cast<Builder&>(*sb));else if(fb)fn(static_cast<Builder&>(*fb));else fn(static_cast<Builder&>(*nb));};
  visit([&](auto&builder){builder.fast_hash=true;builder.merge_ranges=true;builder.kernel_index=int(index);if(gemm)active.graph_enter();g::DAG dag;std::unique_ptr<g::CtaGraphStore> store;
  if(eager){for(U c=0;c<k.executed;++c)for(auto& n:build_cta(int(c)))dag.add_node(n.release());}else{g::CtaGraphStore::Spec spec{};spec.cta_count=int(k.executed);spec.warps_per_cta=k.warps;spec.sm_count=48;spec.total_nodes=k.total_nodes;spec.spans=builder.spans;spec.resident_cta_limit_per_sm=k.resident;spec.per_sm_warp_placement=true;spec.allow_declared_tensor_work=gemm;spec.allow_observed_async_shared_service=gemm;spec.allow_abstract_async_copy=false;store=std::make_unique<g::CtaGraphStore>(spec,[&,lease](int c){return build_cta(c);},[&,lease](int c,const std::vector<g::DAGNode*>& ns,g::Cycle cy){builder.retire(c,ns,cy);if(builder.retired_ctas%1024==0)std::cerr<<J({{"progress_call",index+1},{"source_launch_key",k.call.at("source_launch_key")},{"progress_completed_CTAs",builder.retired_ctas},{"total_CTAs",k.executed},{"continuous_cycle",cy}}).dump()<<std::endl;},k.limits);dag.cta_graph_store=store.get();}
  g::scheduler_observer::reset();g::scheduler_observer::set_enabled(observe);g::cycle_overlap::begin(cfg.num_sms,4,session.cycle(),observe);host_range_audit::L2Events audit;if(prepared.size()>1)audit.digest.add("kernel="+std::to_string(index)+"\n");auto tick=Clock::now();auto r=session.run_kernel(&dag,max_cycles,false,&audit,nullptr,10000000);double engine=std::chrono::duration<double>(Clock::now()-tick).count();engine_total+=engine;J overlap=tilegen_observation_output::compact_cycle_overlap(g::cycle_overlap::finish(r.kernel_end_cycle)),scheduler=g::scheduler_observer::snapshot_json();g::scheduler_observer::set_enabled(false);
  if(eager)for(U c=0;c<k.executed;++c){auto span=builder.spans[c];std::vector<g::DAGNode*> ns(dag.nodes.begin()+span.first_node,dag.nodes.begin()+span.first_node+span.node_count);builder.retire(int(c),ns,session.cycle());}
  p::need(builder.built_nodes==builder.retired_nodes&&builder.retired_nodes==k.total_nodes&&builder.retired_ctas==k.executed,"complete selected grid retirement");p::need(builder.read==k.expected_read&&builder.write==k.expected_write,"requested traffic closure");p::need(session.is_quiescent()&&r.start_cycle==previous&&r.quiescent_end_cycle==session.cycle(),"continuous cycle and quiescence closure");if(index)p::need(stats(r.before)==result["pipeline"][index-1].at("execution").at("after"),"cumulative counters reset between sources");
  auto d=delta(stats(r.after),stats(r.before));U rd=d.at("dram_fill_bytes").get<U>(),wr=d.at("dram_writeback_bytes").get<U>();auto physical=memory.backend->physical_statistics();p::need(physical.read_bytes-prior_r==rd&&physical.write_bytes-prior_w==wr,"per-source native/L2 byte delta");prior_r=physical.read_bytes;prior_w=physical.write_bytes;const auto& a=memory.backend->admission_statistics();p::need(a.accepted==a.completed&&a.reserved_bursts==0&&memory.backend->queue_depth()==0,"per-source finite credits not closed");p::need(r.after.accepted_transactions==r.after.processed_transactions&&r.after.dram_fill_requests==r.after.dram_fill_completions&&r.after.dram_writeback_requests==r.after.dram_writeback_completions,"per-source memory completion closure");
  const std::string digest=list_digest(builder.retired_kernel_semantics);workflow_digest.add(std::to_string(index)+":"+digest+"\n");J e={{"start_cycle",r.start_cycle},{"kernel_end_cycle",r.kernel_end_cycle},{"quiescent_end_cycle",r.quiescent_end_cycle},{"kernel_cycles",r.kernel_end_cycle-r.start_cycle},{"drain_cycles",r.drain_cycles},{"window_cycles",r.quiescent_end_cycle-r.start_cycle},{"before",stats(r.before)},{"after",stats(r.after)},{"counter_delta",d},{"gauges_after",gauges(r.after)},{"DRAM_read_bytes",rd},{"DRAM_write_bytes",wr},{"requested_read_bytes",builder.read},{"requested_write_bytes",builder.write},{"retired_nodes",builder.retired_nodes},{"retired_ctas",builder.retired_ctas},{"kernel_prefixed_per_node_sha256",digest},{"unprefixed_per_node_sha256",list_digest(builder.retired_semantics)},{"ordered_l2_events",audit.receipt()},{"cycle_overlap",overlap},{"scheduler_visits",scheduler},{"finite_credits",{{"accepted_cumulative",a.accepted},{"completed_cumulative",a.completed},{"peak_live_cumulative",a.peak_live},{"reserved_bursts",a.reserved_bursts},{"outer_queue",memory.backend->queue_depth()}}},{"host_engine_seconds",engine},{"modeled_materialization_address_sha256",(hb?J(hb->addresses.hex()):ab?J(ab->addresses.hex()):gb?J(gb->addresses.hex()):vb?J(nullptr):pb?J(pb->addresses.hex()):sb?J(sb->address_digest.hex()):fb?J(fb->address_digest.hex()):nb?J(nb->address_digest.hex()):ib?J(ib->address_digest.hex()):cb?J(cb->address_digest.hex()):J(rb->address_digest.hex()))}};
  // Existing Builder wall timers only; keep host diagnostics outside the
  // execution object and its exact simulated timing/hash comparison domain.
  J frontend={{"build",nullptr},{"retire",nullptr},{"retire_invariant",nullptr},{"retire_hash",nullptr},
      {"available",false},{"scope","Existing per-call Builder wall seconds; retire includes invariant and hash sub-timers. Excludes constructor/model setup, scheduler/HBFSIM and graph destruction; not additive with host_engine_seconds."}};
  if constexpr(requires {builder.build_seconds;builder.retire_seconds;builder.retire_invariant_seconds;builder.retire_hash_seconds;}) {
      frontend["build"]=builder.build_seconds;frontend["retire"]=builder.retire_seconds;
      frontend["retire_invariant"]=builder.retire_invariant_seconds;frontend["retire_hash"]=builder.retire_hash_seconds;
      frontend["available"]=true;
  }
  result["pipeline"][index]["host_frontend_seconds"]=std::move(frontend);
  if(pb){
   p::need(pb->copy_done==2576*k.executed&&pb->zero_done==16*k.executed&&pb->tensor_work==33554432*k.executed,"P28 copy/tensor census");
   const U read_lines=10240*k.executed,write_lines=64*k.executed;
   p::need(d.at("pre_l1_reads")==read_lines&&d.at("pre_l1_writes")==write_lines&&
      d.at("l1_bypassed_transactions")==read_lines+write_lines&&
      d.at("l1_read_hits")==0&&d.at("l1_read_misses")==0&&
      d.at("l1_write_hits")==0&&d.at("l1_write_misses")==0,
      "P28 read-copy and store bypass counters");
   e["P28_node_audit"]=pb->receipt();
  }
  if(gb){const auto&m=*lease->nextmodel;p::need(gb->copy_done==m.copies*k.executed&&gb->zero_done==m.zero_copies*k.executed&&gb->tensor_work==m.tensors*2048*k.executed,"new GEMM per-CTA work");U read_lines=(k.family=="GEMMGate"?20480:k.family=="GEMMDown"?35840:10240)*k.executed,write_lines=64*k.executed;p::need(d.at("pre_l1_reads")==read_lines&&d.at("pre_l1_writes")==write_lines&&d.at("l1_bypassed_transactions")==read_lines+write_lines&&d.at("l1_read_hits")==0&&d.at("l1_read_misses")==0&&d.at("l1_write_hits")==0&&d.at("l1_write_misses")==0,"new GEMM read-copy and store bypass counters");e["GEMM_node_audit"]=gb->receipt();}
  if(ab){const auto&m=*lease->attentionmodel;p::need(ab->copy_done==m.copies*k.executed&&ab->zero_done==m.zero_copies*k.executed&&ab->tensor_work==m.tensors*2048*k.executed,"Attention copy/shared/tensor work");p::need(d.at("pre_l1_reads")==m.read_lines*k.executed&&d.at("pre_l1_writes")==m.write_lines*k.executed&&d.at("l1_bypassed_transactions")==((m.copy_lines+m.write_lines)*k.executed)&&d.at("l1_write_hits")==0&&d.at("l1_write_misses")==0,"Attention ordinary global, known copy, and store bypass census");e["Attention_node_audit"]=ab->receipt();}
  if(hb)e["Helper_node_audit"]=hb->receipt();
  if(store){auto t=store->telemetry();p::need(t.at("live_nodes")==0&&t.at("live_ctas")==0,"resident graph not reclaimed");e["resident_graph"]=t;}result["pipeline"][index]["execution"]=e;sum_windows+=r.quiescent_end_cycle-r.start_cycle;sum_kernel+=r.kernel_end_cycle-r.start_cycle;sum_drain+=r.drain_cycles;sum_r+=rd;sum_w+=wr;previous=r.quiescent_end_cycle;if(store){dag.cta_graph_store=nullptr;store.reset();}if(gemm)active.graph_leave();
  });if(gemm)active.end_call();p::need(session_l2==&session.l2()&&backend_identity==memory.backend.get()&&frames.receipt().at("encoded_payload_bytes")==frozen_encoded_bytes&&frames.size()==8&&frames.receipt().at("live_decoded_bytes")==0,"fixed Runtime/cache/backend/raw frames");std::cerr<<J({{"progress_completed_calls",index+1},{"total_calls",prepared.size()},{"family",k.family},{"source_launch_key",k.call.at("source_launch_key")},{"continuous_cycle",session.cycle()}}).dump()<<std::endl;
 };for(std::size_t index=0;index<prepared.size();++index){try{const auto hbf_before=memory.backend->service_diagnostics();execute_one(index);result["pipeline"][index]["execution"]["hbf_service"]={{"before",coupling::native_service_statistics(hbf_before)},{"after",coupling::native_service_statistics(memory.backend->service_diagnostics())}};hybrid_full::emit_completed_summary(result["pipeline"][index],index+1);}catch(const std::exception&e){throw std::runtime_error("Execute "+prepared[index]->family+" "+prepared[index]->call.at("source_launch_key").get<std::string>()+": "+e.what());}}
 p::need(sum_windows==session.cycle()&&sum_kernel+sum_drain==sum_windows,"workflow time closure");auto d=delta(stats(session.statistics()),stats(initial));p::need(d.at("dram_fill_bytes").get<U>()==sum_r&&d.at("dram_writeback_bytes").get<U>()==sum_w,"workflow traffic closure");memory.backend->finalize();sg_hbf::require_external_only(session.l2());auto physical=memory.backend->physical_statistics();p::need(physical.read_bytes==sum_r&&physical.write_bytes==sum_w,"final native ledger closure");model.finish();
 const auto& clock=in.at("memory_model").at("clock");long double cycle_ns=clock.at("period_ps_numerator").get<long double>()/clock.at("period_ps_denominator").get<long double>()/1000;for(auto& item:result["pipeline"]){auto& e=item["execution"];double ns=double(e.at("window_cycles").get<U>()*cycle_ns);e["time_ns"]=ns;e["DRAM_GBps"]=double(e.at("DRAM_read_bytes").get<U>()+e.at("DRAM_write_bytes").get<U>())/ns;}
 double ns=double(sum_windows*cycle_ns);result["workflow"]={{"window_cycles",sum_windows},{"kernel_cycles_sum",sum_kernel},{"drain_cycles_sum",sum_drain},{"clock",clock},{"time_ns",ns},{"DRAM_read_bytes",sum_r},{"DRAM_write_bytes",sum_w},{"DRAM_GBps",double(sum_r+sum_w)/ns},{"counter_delta",d},{"kernel_prefixed_node_sha256",workflow_digest.hex()},{"end_is_quiescent",session.is_quiescent()}};result["memory_backend"]={{"identity",memory.device_identity},{"physical_statistics",tiny_hbf_reporting::native_statistics(physical)},{"memory_path",tiny_hbf_reporting::path_statistics(*memory.backend)},{"runtime_instances",1},{"continuous_sessions",1},{"finalize_calls",1}};result["host"]={{"engine_seconds",engine_total},{"total_seconds",std::chrono::duration<double>(Clock::now()-start).count()}};result["P28_model_scope"]={{"known_BYPASS_L1",true},{"CONSTANT_policy_recovered",false},{"hardware_timing_calibrated",false},{"full_compute_memory_overlap_fraction",nullptr},{"tensor_width",cfg.tensor_core_width},{"tensor_latency_cycles",cfg.tensor_core_latency_cycles},{"declared_FMA_per_HMMA",2048},{"issue_cycles_per_HMMA",64}};result["large_frame_typed_constructions"]=active.constructions;result["host_capacity"]["final_child_peak_RSS_bytes"]=prefill_gemm::peak_rss();result["source_pool"]=model.receipt();result["source_pool"]["target_model_calls"]=prepared.size();result["source_pool"]["original_source_addresses_used_for_execution"]=false;result["compressed_frame_cache"]=frames.receipt();result["compressed_frame_cache"]["active_typed_GEMM_or_Attention_models_max"]=1;if(full){result["status"]="MODELED_FULL_1138_CONTINUOUS_EXECUTED";result["scope"]="ALL1138_CANONICAL_FIXED_WORKLOAD_STRUCTURAL_MODEL";result["all_registered_calls_executed"]=true;result["complete_SGLang_phase"]=true;result["full_LLM_state"]=true;result["full_LLM_time_ns"]=ns;result["full_LLM_bandwidth_GBps"]=double(sum_r+sum_w)/ns;result["cache_history_scope"]="one initial-cold canonical1138 modeled sequence; no per-call reset or final dirty flush; rawVA reuse assumption remains";}{
 const auto ds=session.l2().dirty_sector_snapshot();
 result["dirty_sector_evaluation"]={
 {"mode",TILEGEN_DIRTY_SECTOR_MODE},{"store_mask_requests",ds.store_mask_requests},
 {"range_visits",ds.range_visits},{"range_intersections",ds.range_intersections},
 {"unknown_store_rejections",ds.unknown_store_rejections},
 {"store_mask_popcounts",ds.store_mask_popcounts},{"eviction_popcounts",ds.eviction_popcounts},
 {"writeback_run_lengths",ds.writeback_run_lengths},{"eviction_masks",ds.eviction_masks},
 {"eviction_run_counts",ds.eviction_run_counts},
 {"dirty_sector_creations",ds.dirty_sector_creations},{"evicted_dirty_sectors",ds.evicted_dirty_sectors},
 {"resident_dirty_lines",ds.resident_dirty_lines},{"resident_dirty_sectors",ds.resident_dirty_sectors},
 {"pending_dirty_lines",ds.pending_dirty_lines},{"pending_dirty_sectors",ds.pending_dirty_sectors},
 {"outstanding_writeback_bytes",ds.outstanding_writeback_bytes},{"unadmitted_writeback_bytes",ds.unadmitted_writeback_bytes},
 {"dirty_sector_ledger_closed",ds.dirty_sector_ledger_closed},{"writeback_byte_ledger_closed",ds.writeback_byte_ledger_closed},
 {"pending_capacity",session.l2().pending_dram_admission_capacity()},
 {"sizeof_CacheLineState",sizeof(g::L2Cache::CacheLineState)},
 {"sizeof_MSHREntry",sizeof(g::L2Cache::MSHREntry)},
 {"sizeof_Transaction",sizeof(g::L2Cache::Transaction)}};
 }
 tiny_full::require_closed(*shared_l2,memory);
 if(tiny_call_count){result["compressed_frame_cache"]["active_typed_GEMM_or_Attention_models_max_scope"]="original fine Active container only; tiny provider models are separate";result["workflow"]["kernel_prefixed_node_sha256"]=nullptr;result["status"]=full?"MODELED_HYBRID_FULL_1138_CONTINUOUS_EXECUTED":"MODELED_HYBRID_BOUNDED_SUBSEQUENCE_EXECUTED";}
 result["hybrid_context"]={{"tiny_calls",tiny_call_count},{"fine_calls",prepared.size()-tiny_call_count},{"routes",routes},{"q",hybrid_full::options.q},{"silu_q1_explicit",hybrid_full::options.silu_q1},{"same_L2_backend_and_absolute_cycle",true},{"fine_fallback_does_not_imply_all_fine_result",true},{"timing_equivalence_to_all_fine",tiny_call_count==0},{"hardware_timing_qualified",false},{"model_setup_seconds_before_frame_validation",model_setup_seconds},{"bindings",tiny_bindings.receipt()},{"typed_framed_models_maximum_simultaneous_bound",tiny_call_count?5:1},{"host_engine_scope","Sum of heterogeneous route windows: fine run_kernel only; tiny binding+template compilation+runtime+drain+census. Destructors outside per-call timers are included in pre-output CPP and child RAW; do not treat engine sum as pure-core comparison."},{"memory_service_host_seconds",nullptr},{"memory_service_host_seconds_status","not separately instrumented; counters are work counts, not time"}};
 result["memory_backend"]["hbf_service"]=coupling::native_service_statistics(memory.backend->service_diagnostics());
 result["full_trace_saved"]=false;result["retry_host_work"]=g::retry_host::snapshot();return result;
}

} // namespace canonical_full
#include "../../../direct_native.h"
#ifndef TILEGEN_NO_EXECUTABLE_MAIN
int main(int argc,char**argv) {
 const auto process_start=std::chrono::steady_clock::now();
 GTSim::retry_host::prefix_enabled=true;GTSim::retry_host::ready_front_enabled=true;
 try {
  bool validate=false,eager=false,events=true,full=false;
  std::string mode="cosim",trace_path;
  std::uint64_t trace_cap=native_trace::default_max_bytes,phase_ctas=0;
  for(int i=1;i<argc;++i) {
   const std::string a=argv[i];
   if(a=="--help") {
    std::cout<<"TileGen native B1: --mode=cosim|cosim-fast|direct --trace=PATH --max-trace-bytes=N --full-workflow\n"
             <<"cosim preserves native compute dependencies; cosim-fast is the explicitly approximate hybrid profile.\n"
             <<"direct uses the same native addresses with deterministic functional cache order, without timing.\n"
             <<"direct --phase-ctas=N additionally exports an approximate CTA-stage compute profile (1..4096).\n";
    return 0;
   }
   if(a.rfind("--mode=",0)==0)mode=a.substr(7);
  }
  native_program::need(mode=="cosim"||mode=="cosim-fast"||mode=="direct","unknown mode");
  hybrid_full::options.all_fine=mode!="cosim-fast";
  if(mode=="cosim-fast") {
   hybrid_full::options.phase_width=16;hybrid_full::options.memory_epoch=8;
   hbf_drain_cli::policy.drain=sg_hbf::DrainMode::Independent;
  }
  for(int i=1;i<argc;++i) {
   const std::string a=argv[i];
   if(a.rfind("--mode=",0)==0)continue;
   if(a.rfind("--trace=",0)==0)trace_path=a.substr(8);
   else if(a.rfind("--phase-ctas=",0)==0) {
    const auto value=a.substr(13);std::size_t used=0;
    native_program::need(!value.empty()&&value[0]!='-',"positive stage CTA size");
    phase_ctas=std::stoull(value,&used);
    native_program::need(used==value.size()&&phase_ctas>0&&phase_ctas<=4096,"stage CTA size must be 1..4096");
   }
   else if(a.rfind("--max-trace-bytes=",0)==0) {
    const auto value=a.substr(18);std::size_t used=0;
    native_program::need(!value.empty()&&value[0]!='-',"positive trace byte limit");
    trace_cap=std::stoull(value,&used);native_program::need(used==value.size(),"integer trace byte limit");
   }
   else if(a=="--validate-only")validate=true;
   else if(a=="--eager-reference")eager=true;
   else if(a=="--event-off")events=false;
   else if(a=="--full-workflow")full=true;
   else if(hbf_drain_cli::parse(a)){}
   else if(hybrid_full::options.parse(a)){}
   else native_program::need(false,"unknown option");
  }
  native_program::need(mode!="direct"||(!validate&&!eager&&events&&!trace_path.empty()),"direct requires trace path and functional execution");
  native_program::need(!validate||trace_path.empty(),"validate-only cannot save an execution trace");
  native_program::need(!phase_ctas||mode=="direct","stage profile export requires direct mode");
  nlohmann::json result;
  {
   auto transport=compressed_frame::read_control(std::cin);auto control=transport.at("decoded_control");
   compressed_frame::Cache frames(transport.at("frames"));
   for(std::uint64_t i=0;i<transport.at("frames").size();++i)frames.read_one(std::cin);
   frames.finish(std::cin);native_program::need(frames.size()==8,"exact eight compressed frames");
   std::map<std::string,nlohmann::json>expected;
   for(const auto&q:control.at("frames"))native_program::need(expected.emplace(q.at("key"),q).second,"unique decoded declaration");
   native_program::need(expected.size()==8,"eight decoded declarations");
   for(const auto&q:frames.decoded_manifest())native_program::need(expected.at(q.at("key"))==q,"exact original decoded SHA/length");
   for(const std::string k:{"P28QKV","GEMMO","GEMMGate","GEMMDown","AttentionPrefill","AttentionDecode1","AttentionDecode2","Helpers"})
    native_program::need(frames.contains(k),"closed compressed frame key domain");
   const auto context_sha=tiny_sha::sha256(transport.dump());
   std::unique_ptr<native_trace::Writer> writer;
   if(!trace_path.empty())writer=std::make_unique<native_trace::Writer>(trace_path,
      mode=="direct"?native_trace::Mode::FunctionalDirect:native_trace::Mode::NativeCosim,context_sha,trace_cap);
   unified_trace::writer=writer.get();
   result=mode=="direct"?direct_native::run(control,frames,full,*writer,phase_ctas):canonical_full::run(control,frames,validate,eager,events,false,full);
   unified_trace::writer=nullptr;
   if(writer) {
    const auto receipt=writer->finish();
    const auto& ledger=result.at(mode=="direct"?"cache":"workflow");
    native_program::need(receipt.read_bytes==ledger.at("DRAM_read_bytes")&&
      receipt.write_bytes==ledger.at("DRAM_write_bytes"),"exported/cache byte conservation");
    result["trace"]=receipt.to_json();result["full_selected_trace_saved"]=true;result["full_trace_saved"]=full;
   }
   result["transport_control_sha256"]=context_sha;
   result["mode"]=mode;result["batch_size"]=1;result["writeback_request_bytes"]=32;
   result["version"]="tilegen-trace-cosim-20260918-r1";
   result["precision"]={{"native_memory_rules",true},{"compute_dependencies_executed",!validate&&mode!="direct"},
      {"functional_order_matches_cosim",false},{"hybrid_timing_approximation",mode=="cosim-fast"}};
  }
  result["hybrid_pre_output_CPP_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-process_start).count();
  std::cout<<result.dump()<<'\n';return 0;
 }catch(const std::exception&e){std::cerr<<nlohmann::json({{"status","REJECTED"},{"reason",e.what()}}).dump()<<'\n';return 2;}
}
#endif
