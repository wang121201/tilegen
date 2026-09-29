#pragma once
#include "backend.h"
#include "simulator_session.h"
#include "../../tilegen-full-r1/source_root.h"
#include <nlohmann/json.hpp>
namespace coupling {
using J=nlohmann::json;using U=std::uint64_t;namespace g=GTSim;
inline void need(bool x,const char* why){if(!x)throw std::runtime_error(why);}
inline U natural(const J& j){need(j.is_number_integer()&&!j.is_boolean(),"integer profile/map field required");if(!j.is_number_unsigned())need(j.get<std::int64_t>()>=0,"negative profile field");return j.get<U>();}
struct Span {U source,bytes,service;};
class ServiceMapper final:public g::L2DramAddressMapper {
    const g::L2DramAddressMapper& source_;
    std::map<U,Span> spans_;
public:
    ServiceMapper(const J& mapping,const g::L2DramAddressMapper& source,U capacity):source_(source){
        need(mapping.at("schema")=="SG_SOURCE_TO_SERVICE_MAP_V1"&&mapping.at("qualification")=="PACKED_SOURCE_GENERATED_128B_LINES_NOT_PHYSICAL_HARDWARE_ADDRESSES","explicit generated source/service map required");
        const auto& rows=mapping.at("spans");need(rows.is_array()&&!rows.empty()&&rows.size()<=8192,"bounded service map");
        std::vector<std::pair<U,U>> physical;
        for(const auto& r:rows){Span s{natural(r.at("source_base")),natural(r.at("bytes")),natural(r.at("service_base"))};
            need(s.bytes>0&&s.source%128==0&&s.service%128==0&&s.bytes%128==0&&s.bytes<=UINT64_MAX-s.source,"aligned source service extent");
            need(s.service<capacity&&s.bytes<=capacity-s.service,"service mapping exceeds physical capacity");
            need(spans_.emplace(s.source,s).second,"duplicate source span");physical.push_back({s.service,s.service+s.bytes});}
        U end=0;for(const auto& [base,s]:spans_){need(base>=end,"overlapping source spans must be canonicalized as aliases");end=base+s.bytes;}
        std::sort(physical.begin(),physical.end());for(std::size_t i=1;i<physical.size();++i)need(physical[i].first>=physical[i-1].second,"distinct VAs alias physical service incorrectly");
    }
    U map(const g::CacheLineKey& key)const override{
        const U va=source_.map(key);auto it=spans_.upper_bound(va);need(it!=spans_.begin(),"source line has no service mapping");--it;const auto& s=it->second;
        need(va>=s.source&&va-s.source<s.bytes&&128<=s.bytes-(va-s.source),"full native128B line unmapped");return s.service+(va-s.source);
    }
};

inline void match_reference(const J& j,const hbfsim::physical::hbm::HbmConfig& c){
    need(j.at("schema")=="SG_GDDR6_PROFILE_V1"&&j.at("profile_standard")=="GDDR6"&&
         j.at("command_protocol")=="GENERIC_BANKED_SUBSET"&&j.at("JEDEC_controller_validated")==false,
         "explicit unvalidated GDDR6 reference identity required");
    need(j.at("qualification")=="EXPLICIT_UNCALIBRATED_GDDR6_SERVICE_MODEL",
         "native candidate must retain its uncalibrated qualification");
    const auto& g=j.at("geometry");const auto& t=j.at("timing");const auto& r=j.at("refresh");
    need(c.device.stacks==1&&c.device.pseudo_channels_per_channel==1,"GDDR6 reference maps to one internal stack and one pseudo-channel");
#define MATCH_INT(key,field) need(natural(g.at(key))==c.device.field,"reference/native geometry mismatch: " key)
    MATCH_INT("service_channels",channels_per_stack);MATCH_INT("channel_width_bits",channel_width_bits);
    MATCH_INT("burst_length",burst_length);MATCH_INT("data_rate_per_command_clock",data_rate_per_command_clock);
    MATCH_INT("bank_groups",bank_groups_per_pseudo_channel);MATCH_INT("banks_per_group",banks_per_group);
    MATCH_INT("row_bytes",channel_row_size_bytes);MATCH_INT("capacity_bytes",capacity_bytes);
#undef MATCH_INT
    need(g.at("pin_rate_Gbps").get<double>()==c.device.pin_rate_Gbps,"reference/native pin rate mismatch");
    need(t.at("address_mapping_ns").get<double>()==c.controller.address_mapping_ns,"reference/native mapping latency mismatch");
#define MATCH_TIME(field) need(t.at(#field).get<double>()==c.timing.field,"reference/native timing mismatch: " #field)
    MATCH_TIME(tRCDRD_ns);MATCH_TIME(tRCDWR_ns);MATCH_TIME(tCL_ns);MATCH_TIME(tCWL_ns);
    MATCH_TIME(tRP_ns);MATCH_TIME(tRAS_ns);MATCH_TIME(tRC_ns);MATCH_TIME(tWR_ns);MATCH_TIME(tRTP_ns);
    MATCH_TIME(tRRD_S_ns);MATCH_TIME(tRRD_L_ns);MATCH_TIME(tFAW_ns);MATCH_TIME(tWTR_S_ns);MATCH_TIME(tWTR_L_ns);MATCH_TIME(tRTW_ns);
#undef MATCH_TIME
    need(natural(t.at("tCCD_S_cycles"))==c.timing.tCCD_S_cycles&&natural(t.at("tCCD_L_cycles"))==c.timing.tCCD_L_cycles,
         "reference/native tCCD mismatch");
    need(t.at("frfcfs_cap_ns").get<double>()==c.controller.frfcfs_cap_ns,"reference/native scheduler cap mismatch");
    need(r.at("mode")=="disabled"&&!c.controller.refresh_enabled&&!c.controller.same_bank_refresh,
         "reference/native refresh mode mismatch");
    need(r.at("tREFI_ns").get<double>()==c.timing.tREFI_ns&&r.at("tRFC_ns").get<double>()==c.timing.tRFC_ns&&
         r.at("tRREFD_ns").get<double>()==c.timing.tRREFD_ns&&c.timing.tRFCsb_ns==c.timing.tRFC_ns,
         "reference/native inactive refresh timing mismatch");
    need(natural(j.at("queue_depth"))==c.controller.queue_depth,"reference/native queue depth mismatch");
}
inline J native_identity(const hbfsim::physical::hbm::HbmConfig& c,const std::string& path){
    J timing=J::object();
#define TIME(field) timing[#field]=c.timing.field
    TIME(tRCDRD_ns);TIME(tRCDWR_ns);TIME(tCL_ns);TIME(tCWL_ns);TIME(tRP_ns);TIME(tRAS_ns);TIME(tRC_ns);
    TIME(tWR_ns);TIME(tRTP_ns);TIME(tCCD_S_cycles);TIME(tCCD_L_cycles);TIME(tRRD_S_ns);TIME(tRRD_L_ns);
    TIME(tFAW_ns);TIME(tWTR_S_ns);TIME(tWTR_L_ns);TIME(tRTW_ns);TIME(tREFI_ns);TIME(tRFC_ns);TIME(tRFCsb_ns);TIME(tRREFD_ns);
#undef TIME
    return {{"schema","NATIVE_HBFSIM_GDDR6_INTENT_IDENTITY_V1"},
        {"upstream_commit","d7a2ca64614a6d9ce8d7a69beb77ce78b66df1a8"},{"core_modified",true},{"core_modification_scope","bounded independent-PC drain API; original global service method preserved; TraceOff call-site argument guards; modeled command/state updates unchanged"},
        {"intended_memory","GDDR6"},{"intended_protocol","GENERIC_BANKED_SUBSET"},
        {"raw_core_standard",std::string(c.standard)},{"raw_core_type","hbfsim::physical::hbm::HbmDevice"},
        {"raw_core_tier","HBM"},{"native_GDDR6_device_type_implemented",false},
        {"JEDEC_controller_validated",false},{"hardware_timing_calibrated",false},
        {"chip_identity_verified",false},{"physical_GPU_mapping_recovered",false},
        {"native_config_file",path},{"effective_parameter_source","NATIVE_SYSTEM_CONFIG_BUILDER_RESOLVED_CFG"},
        {"historical_GDDR6_JSON_checked_equal_to_corresponding_native_fields",true},
        {"resolved_device",{{"capacity_bytes",c.device.capacity_bytes},{"stacks",c.device.stacks},
            {"channels_per_stack",c.device.channels_per_stack},{"pseudo_channels_per_channel",c.device.pseudo_channels_per_channel},
            {"bank_groups_per_pseudo_channel",c.device.bank_groups_per_pseudo_channel},{"banks_per_group",c.device.banks_per_group},
            {"channel_row_size_bytes",c.device.channel_row_size_bytes},{"channel_width_bits",c.device.channel_width_bits},
            {"burst_length",c.device.burst_length},{"pin_rate_Gbps",c.device.pin_rate_Gbps},
            {"data_rate_per_command_clock",c.device.data_rate_per_command_clock}}},
        {"resolved_timing",timing},
        {"resolved_controller",{{"address_mapping_ns",c.controller.address_mapping_ns},
            {"refresh_enabled",c.controller.refresh_enabled},{"same_bank_refresh",c.controller.same_bank_refresh},
            {"queue_depth",c.controller.queue_depth},{"frfcfs_cap_ns",c.controller.frfcfs_cap_ns},
            {"interleave_bytes",c.effective_interleave_bytes()},
            {"replicate_symmetric_pseudo_channels",c.controller.replicate_symmetric_pseudo_channels}}},
        {"derived",{{"burst_bytes",c.burst_bytes()},{"command_clock_ns",c.command_clock_period_ns()},
            {"command_clock_MHz",c.command_clock_MHz()},
            {"aggregate_peak_DQ_GBps",c.channel_bandwidth_GBps()*c.device.stacks*c.device.channels_per_stack}}},
        {"address_mapping_scheme",std::string(c.address_mapping_scheme())},
        {"address_mapping_qualification","GENERIC_SERVICE_MAPPING_NOT_GPU_PHYSICAL_BANK_CHANNEL_MAP"}};
}
inline J native_statistics(const hbfsim::physical::hbm::HbmStats& s){
    J j=J::object();
#define STAT(field) j[#field]=s.field
    STAT(read_bytes);STAT(write_bytes);STAT(controller_buffer_read_bytes);STAT(controller_buffer_write_bytes);
    STAT(controller_buffer_transfers);STAT(controller_buffer_bus_busy_ns);STAT(row_hits);STAT(row_misses);
    STAT(row_conflicts);STAT(activations);STAT(precharges);STAT(refresh_count);STAT(bus_busy_ns);
    STAT(finish_ns);STAT(pseudo_channels);STAT(active_pseudo_channels);STAT(max_pseudo_channel_accesses);
    STAT(max_queue_occupancy);STAT(max_pseudo_channel_busy_ns);STAT(avg_active_pseudo_channel_busy_ns);
    STAT(replicated_requests);STAT(replicated_bursts);
#undef STAT
    j["first_arrival_ns"]=std::isfinite(s.first_arrival_ns)?J(s.first_arrival_ns):J(nullptr);
    J work=J::object();
#define WORK(field) work[#field]=s.stage_work.field
    WORK(ingress_queue_wait_ns);WORK(scheduler_queue_wait_ns);WORK(address_mapping_ns);WORK(translation_ns);
    WORK(mapping_dram_ns);WORK(write_buffer_dram_ns);WORK(refresh_stall_ns);WORK(precharge_ns);WORK(activation_ns);
    WORK(command_ns);WORK(array_read_ns);WORK(array_program_ns);WORK(array_erase_ns);WORK(media_lane_transfer_ns);
    WORK(page_buffer_ns);WORK(sram_staging_ns);WORK(channel_transfer_ns);WORK(tsv_transfer_ns);WORK(hb_io_transfer_ns);
    WORK(transport_latency_ns);WORK(ecc_queue_wait_ns);WORK(ecc_latency_ns);WORK(maintenance_ns);
#undef WORK
    j["overlapping_stage_work_ns"]=work;return j;
}
inline void path_keys(const J& j,std::initializer_list<const char*> names){
    need(j.is_object()&&j.size()==names.size(),"memory_path fields differ from explicit schema");
    for(const auto* name:names)need(j.contains(name),"required memory_path field missing");
}
inline sg_hbf::PathRate path_rate(const J& j){
    need(j.is_object()&&j.contains("mode"),"explicit path rate mode required");
    const auto mode=j.at("mode").get<std::string>();
    if(mode=="unlimited"){path_keys(j,{"mode"});return {};}
    need(mode=="rational_serial","unknown path service rate mode");path_keys(j,{"mode","numerator","denominator"});
    return {true,natural(j.at("numerator")),natural(j.at("denominator"))};
}
inline sg_hbf::PathConfig path_config(const J& j){
    path_keys(j,{"schema","qualification","request_fixed_ps","response_fixed_ps","request_rate","response_rate","max_outer_live","writeback_ack_policy"});
    need(j.at("schema")=="SG_GPU_MEMORY_PATH_V1"&&j.at("qualification")=="DIAGNOSTIC_UNCALIBRATED",
         "new memory path requires explicit DIAGNOSTIC_UNCALIBRATED identity");
    need(j.at("writeback_ack_policy")=="fixed_response_delay_no_payload","writeback ACK semantics must be explicit");
    sg_hbf::PathConfig p;p.request_fixed_ps=natural(j.at("request_fixed_ps"));p.response_fixed_ps=natural(j.at("response_fixed_ps"));
    p.max_outer_live=natural(j.at("max_outer_live"));p.request_rate=path_rate(j.at("request_rate"));p.response_rate=path_rate(j.at("response_rate"));
    p.validate();return p;
}
inline J path_statistics(const sg_hbf::MemoryPathBackend& b){
    const auto& s=b.path_statistics();J j;
#define PATH(field) j[#field]=s.field
    PATH(outer_accepted);PATH(outer_delivered);PATH(outer_rejected);PATH(physical_admitted);PATH(physical_completed);
    PATH(peak_outer_live);PATH(peak_ingress);PATH(peak_responses);PATH(peak_physical_live);
    PATH(request_service_busy_ps);PATH(response_service_busy_ps);PATH(response_payload_bytes);
    PATH(outer_admission_wait_ps);PATH(request_queue_wait_ps);PATH(request_fixed_work_ps);PATH(request_service_work_ps);PATH(native_admission_wait_ps);
    PATH(physical_core_ready_latency_ps);PATH(physical_observation_lag_ps);PATH(physical_observation_segment_ps);PATH(response_queue_wait_ps);
    PATH(response_fixed_work_ps);PATH(response_service_work_ps);PATH(delivery_poll_lag_ps);PATH(end_to_end_observed_latency_ps);
    PATH(last_native_arrival_ps);PATH(last_physical_core_ready_ps);PATH(last_response_ready_ps);PATH(last_delivery_ps);
    PATH(physical_release_while_response_pending);PATH(zero_path_direct_completions);PATH(step_calls);
    PATH(outer_identity_fnv1a64);PATH(inner_binding_fnv1a64);
#undef PATH
    j["first_issue_ps"]=s.first_issue_ps==UINT64_MAX?J(nullptr):J(s.first_issue_ps);
    j["first_native_arrival_ps"]=s.first_native_arrival_ps==UINT64_MAX?J(nullptr):J(s.first_native_arrival_ps);
    j["final_outer_live"]=b.queue_depth();j["final_physical_live"]=b.physical_queue_depth();
    j["final_ingress"]=b.ingress_queue_depth();j["final_responses"]=b.response_queue_depth();
    j["qualification"]="DIAGNOSTIC_UNCALIBRATED";
    j["physical_core_ready_is_raw_per_parent_device_finish"]=false;
    j["physical_core_ready_definition"]="inner completion cycle poll boundary; raw native finish only in native statistics";
    j["response_start_definition"]="physical completion observed at current poll; no future or retroactive return reservation";
    j["sums_are_overlapping_request_work_not_kernel_critical_path"]=true;
    J segments=J::object();
    for(std::size_t i=0;i<sg_hbf::kDeliveredSegmentNames.size();++i)segments[sg_hbf::kDeliveredSegmentNames[i]]=s.delivered_segments_ps[i];
    j["additive_delivered_segments_ps"]={{"segments",segments},{"sum",s.delivered_segment_sum_ps},
        {"parent_count",s.delivered_parent_segment_closures},{"closed",s.delivered_segment_sum_ps==s.end_to_end_observed_latency_ps&&s.delivered_parent_segment_closures==s.outer_delivered},
        {"definition","sum over delivered parents of observed_delivery_ps minus original_issue_ps; per-parent closure enforced"},
        {"scope","DELIVERED_PARENTS_ONLY; closes at every poll, including while other parents remain live"},
        {"kernel_elapsed_time",false}};
    j["physical_observation_lag_ps_semantics"]="NONADDITIVE_OBSERVATION_DIAGNOSTIC; zero response already includes this lag in delivery_poll_lag";
    j["physical_observation_segment_ps_semantics"]="PHYSICAL_COMPLETED_WORK: zero in direct-return mode; observed-minus-core-ready otherwise; use delivered-only ledger for additive end-to-end";
    j["legacy_work_counter_scope"]="Mixed accepted/physical-completed/delivered prefixes; do not sum legacy fields as end-to-end segments";
    j["request_rate_unit"]="parent/ps";j["response_rate_unit"]="read_payload_byte/ps";
    j["service_rounding"]="ceil per serial service quantum to integer ps";
    j["request_admission_policy"]=b.path_config().direct_request()?"DIRECT_NATIVE_BACKPRESSURE_FOR_ZERO_REQUEST_PATH":"BOUNDED_OUTER_FIFO_THEN_NATIVE_CREDITS";
    j["native_credits_released_at_physical_completion_before_response"]=true;
    j["outer_inner_identity_spaces_separate"]=true;j["trace_saved"]=false;
    return j;
}
inline J native_service_statistics(const sg_hbf::NativeServiceDiagnostics& s){
    const auto& c=s.cadence;J cadence;
#define FIELD(name) cadence[#name]=c.name
    FIELD(quantum);FIELD(step_calls);FIELD(legacy_steps);FIELD(periodic_boundary_visits);
    FIELD(admission_guard_visits);FIELD(service_batches);FIELD(same_cycle_coalesces);FIELD(step_poll_skips);
    FIELD(actual_service_before_calls);FIELD(actual_service_steps);FIELD(completions_visible);
    FIELD(late_visible_completions);FIELD(total_visibility_delay_cycles);FIELD(max_visibility_delay_cycles);
#undef FIELD
    return {{"schema","HBF_DRAIN_NATIVE_DIAGNOSTIC_V1"},{"drain",s.drain==sg_hbf::DrainMode::Global?"global":"independent"},
        {"cadence",cadence},{"independent_api_calls",s.independent_api_calls},
        {"independent_fast_drains",s.independent_fast_drains},{"fallback_global_drains",s.fallback_global_drains},
        {"pc_service_one_calls",s.pc_service_one_calls},
        {"old_service_counter_scope","native_service_before_calls counts actual original method calls; native_service_steps counts successful original global rounds; independent fast-path PC calls are excluded"},
        {"late_visibility_scope","adapter-return cycle minus native physical completion cycle; not L2/Runtime callback time or GPU stall"}};
}
struct Runtime {
    std::unique_ptr<sg_hbf::MemoryPathBackend> backend;
    std::unique_ptr<ServiceMapper> mapper;
    J profile,device_identity;
    bool gddr6_active=false;
    Runtime(const J& input,g::SimulatorConfig& cfg,const g::L2DramAddressMapper& source,sg_hbf::NativeServicePolicy service_policy={}):profile(input.at("memory_model")){
        need(profile.at("schema")=="SG_FRAGMENT_MEMORY_PROFILE_V1"&&profile.at("qualification")=="EXPLICIT_UNCALIBRATED_MODEL_PARAMETERS",
             "explicit fragment memory profile required");
        need(profile.at("core_profile")=="pinned_RTX4000_Ada_footprint_reference","unknown core profile");
        const auto& clock=profile.at("clock");
        sg_hbf::Clock c(natural(clock.at("period_ps_numerator")),natural(clock.at("period_ps_denominator")));
        const double mhz=1e6*static_cast<double>(c.cycle_denominator)/static_cast<double>(c.ps_numerator);
        need(std::abs(mhz-cfg.core_frequency_mhz)<1e-8,"explicit clock differs from pinned core profile");
        const auto mode=profile.at("backend").get<std::string>();
        need(mode=="internal"||mode=="hbf_gddr6","only internal or native hbf_gddr6 is supported; old surrogate is excluded");
        need(profile.at("hbm_timing_scale").get<double>()==1.0,"native cfg cannot be silently timing-rescaled");
        const auto logical_path=profile.at("native_hbfsim_config_file").get<std::string>();
        const auto path=tilegen_source::resolve(logical_path);
        const auto native=sg_hbf::native_config_file(path);
        match_reference(profile.at("gddr6"),native);
        device_identity=native_identity(native,path);
        mapper=std::make_unique<ServiceMapper>(input.at("service_address_map"),source,native.device.capacity_bytes);
        const U latency=natural(profile.at("internal_miss_latency_cycles"));need(latency<=100000,"internal miss latency bound");
        cfg.l2_miss_penalty_cycles=static_cast<int>(latency);cfg.memory_model_semantics.end_to_end_miss_latency_cycles=static_cast<int>(latency);
        gddr6_active=mode=="hbf_gddr6";
        if(gddr6_active){
            need(natural(profile.at("max_live"))==4096&&natural(profile.at("credits_per_pc"))==32,
                 "selected TileGen coupling requires max_live4096 and 32 burst credits per service channel");
            need(profile.contains("memory_path"),"external backend requires explicit memory_path diagnostic configuration");
            need(cfg.memory_model_semantics.l2_miss_latency==g::L2MissLatencySemantics::END_TO_END_FROM_MISS_DECISION,
                 "outer response delivery requires GTSim END_TO_END completion semantics");
            backend=std::make_unique<sg_hbf::MemoryPathBackend>(c,native,path_config(profile.at("memory_path")),4096,32,service_policy);
        }
    }
    J finish(const g::ContinuousL2Session& session)const{
        J out={{"profile",profile},{"scope","independent_cold_direct_norm_full_grid"},
            {"physical_address_mapping_known",false},{"HBM_timing_surrogate",false},{"GDDR_hardware_validated",false},
            {"raw_or_expanded_trace_saved",false},{"internal_DRAM_active",session.l2().uses_internal_dram_backend()},
            {"external_HBM_active",false},{"external_GDDR6_profile_active",gddr6_active},{"external_DRAM_active",bool(backend)},
            {"command_engine_implementation",backend?"LATEST_HBFSIM_NATIVE_ASYNC_HbmDevice":"GTSIM_INTERNAL_FIXED_MISS_LATENCY"},
            {"JEDEC_GDDR6_command_protocol_implemented",false}};
        if(gddr6_active)out["GDDR6_profile_identity"]=device_identity;
        const auto s=session.statistics();
        out["modeled_DRAM_read_bytes"]=s.dram_fill_bytes;
        out["modeled_DRAM_writeback_bytes"]=s.dram_writeback_bytes;
        out["dirty_resident_cache_lines_are_not_implicitly_flushed"]=true;
        if(backend){
            backend->finalize();sg_hbf::require_external_only(session.l2());
            out["memory_path_statistics"]=path_statistics(*backend);
            out["memory_path_qualification"]="DIAGNOSTIC_UNCALIBRATED";
            out["completion_boundary"]="OUTER_RESPONSE_DELIVERY_TO_GTSIM_L2";
            out["internal_604_cycle_reference_active"]=false;
            need(session.l2().pending_dram_admission_count()==0,"native L2 pending admissions did not close");
            const auto& a=backend->admission_statistics();
            out["admission"]={{"accepted",a.accepted},{"completed",a.completed},{"blocked_attempts",a.blocked},
                {"physical_calls",a.physical_calls},{"physical_call_semantics","NATIVE_ENQUEUE_NOT_SYNCHRONOUS_ISSUE"},
                {"peak_live",a.peak_live},{"peak_PC_credits",a.peak_pseudo_channel_credits},
                {"total_admission_wait_ps",a.total_admission_delay_ps},{"last_admission_ps",a.last_admission_ps},
                {"last_completion_ps",a.last_completion_ps},{"L2_peak_pending",session.l2().peak_pending_dram_admissions()},
                {"L2_rejections",session.l2().dram_admission_rejections()},{"final_pending",session.l2().pending_dram_admission_count()},
                {"native_service_before_calls",a.native_service_before_calls},{"native_service_steps",a.native_service_steps},
                {"native_enqueue_execution_checks",a.native_enqueue_execution_checks},
                {"native_enqueue_service_violations",a.native_enqueue_service_violations},
                {"native_completions_posted",a.native_completions_posted},{"future_completions_held",a.future_completions_held},
                {"peak_due_completions",a.peak_due_completions},{"reserved_bursts",a.reserved_bursts},
                {"peak_reserved_bursts",a.peak_reserved_bursts},{"last_service_frontier_ps",a.last_service_frontier_ps},
                {"max_future_completion_lead_ps",a.max_future_completion_lead_ps}};
            out["admission"]["accepted_completed_boundary"]="OUTER_ACCEPT_AND_OUTER_L2_DELIVERY";
            out["admission"]["last_completion_ps_boundary"]="NATIVE_PHYSICAL_FINISH_ROUNDED_UP_TO_PS_NOT_OUTER_DELIVERY";
            out["admission"]["last_outer_delivery_ps"]=backend->path_statistics().last_delivery_ps;
            out["admission_model"]="OUTER_LIFETIME_UNTIL_L2_DELIVERY_NATIVE_BURST_CREDITS_ONLY_UNTIL_PHYSICAL_COMPLETION";
            out["conservative_admission_limit"]="credits may remain held after native child queue release; not measured GPU queue capacity";
            out["physical_statistics_window"]="CUMULATIVE_ONLY_AFTER_BACKEND_AND_L2_ADMISSION_QUIESCENCE";
            out["native_HBFSIM_statistics"]=native_statistics(backend->physical_statistics());
        }
        return out;
    }
};
} // namespace coupling
