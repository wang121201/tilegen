#pragma once
// Native HBFSIM asynchronous command service behind conservative external
// parent/burst credits. Never use issue(), pump(), or speculative draining.
#include "memory.h"
#include "app/system_config.hpp"
#include "physical/hbm/hbm_device.hpp"
#include "prepared/directed_time.h"
#include <algorithm>
#include <map>
#include <numeric>
#include <optional>
#include <stdexcept>

namespace sg_hbf {
using U=std::uint64_t;
namespace g=GTSim;
namespace p=hbfsim::physical;
inline void need(bool v,const char* m){if(!v)throw std::runtime_error(m);}
inline U ratio(U v,U n,U d,bool ceil){
    need(d!=0,"zero clock denominator");
    const __uint128_t product=static_cast<__uint128_t>(v)*n;
    const __uint128_t result=product/d+(ceil&&product%d!=0);
    need(result<=UINT64_MAX,"clock overflow");return static_cast<U>(result);
}
struct Clock {
    U ps_numerator,cycle_denominator;
    Clock(U n,U d):ps_numerator(n),cycle_denominator(d){
        need(n>0&&d>0,"positive explicit rational clock required");
        const auto k=std::gcd(n,d);ps_numerator/=k;cycle_denominator/=k;
    }
    U issue_ps(U c)const{return ratio(c,ps_numerator,cycle_denominator,true);}
    U poll_ps(U c)const{return ratio(c,ps_numerator,cycle_denominator,false);}
    U completion_cycle(U ps)const{return ratio(ps,cycle_denominator,ps_numerator,true);}
};
inline double ps_to_ns(U ps){return directed_time::picoseconds_to_ns_never_early(ps);}
inline U ns_to_ps(double ns){return directed_time::nanoseconds_to_picoseconds_never_early(ns);}
inline double strict_poll_ns(U ps){
    // ps_to_ns is the smallest binary64 not below the exact rational ps/1000.
    // Its predecessor is strictly below that boundary (zero stays zero).
    // service_before is itself strict. No command at/future of the core
    // frontier is allowed to issue, including with binary64 rounding.
    return ps==0?0.0:std::nextafter(ps_to_ns(ps),0.0);
}
inline p::hbm::HbmConfig native_config_file(const std::string& path){
    need(!path.empty(),"explicit native HBFSIM config file required");
    hbfsim::app::SystemConfigBuilder builder;builder.apply_file(path);
    return builder.resolve().hbm;
}
struct AdmissionStatistics {
    U accepted=0,completed=0,blocked=0,physical_calls=0;
    U peak_live=0,peak_pseudo_channel_credits=0;
    U total_admission_delay_ps=0,last_admission_ps=0,last_completion_ps=0;
    U native_service_before_calls=0,native_service_steps=0;
    U native_enqueue_execution_checks=0,native_enqueue_service_violations=0;
    U native_completions_posted=0,future_completions_held=0,peak_due_completions=0;
    U reserved_bursts=0,peak_reserved_bursts=0,last_service_frontier_ps=0;
    U max_future_completion_lead_ps=0;
    U actual_request_shape_fnv1a64=14695981039346656037ULL;
};

enum class DrainMode { Global, Independent };
struct NativeServicePolicy { DrainMode drain=DrainMode::Global; unsigned quantum=1; };
struct ServiceCadenceDiagnostics {
    unsigned quantum=1;
    U step_calls=0,legacy_steps=0,periodic_boundary_visits=0;
    U admission_guard_visits=0,service_batches=0,same_cycle_coalesces=0,step_poll_skips=0;
    U actual_service_before_calls=0,actual_service_steps=0;
    U completions_visible=0,late_visible_completions=0,total_visibility_delay_cycles=0,max_visibility_delay_cycles=0;
};

struct NativeServiceDiagnostics {
    DrainMode drain=DrainMode::Global;
    ServiceCadenceDiagnostics cadence;
    U independent_api_calls=0,independent_fast_drains=0,fallback_global_drains=0,pc_service_one_calls=0;
};

class HbfCreditBackend final:public g::L2DramCompletionBackend {
    struct Live {
        g::L2DramRequest request;
        U admission_ps=0,completion_ps=UINT64_MAX,native_ticket=0;
        std::map<std::size_t,U> credits;
        bool posted=false;
        p::PhysicalCompletion physical;
    };
    Clock clock_;
    p::hbm::HbmDevice device_;
    p::hbm::HbmConfig config_;
    U max_live_,credits_per_pc_;
    std::vector<U> occupied_;
    std::map<U,Live> live_;
    std::map<U,U> native_to_source_;
    std::map<std::pair<U,U>,U> due_;
    g::L2DramRuntimeStatistics stats_;
    AdmissionStatistics admission_;
    bool stepped_=false,failed_=false;
    U last_step_=0,last_issue_=0,last_admission_cycle_=0;
    DrainMode drain_mode_=DrainMode::Global;
    U independent_api_calls_=0,independent_fast_drains_=0,fallback_global_drains_=0,pc_service_one_calls_=0;
    unsigned service_quantum_=1;
    ServiceCadenceDiagnostics cadence_;
    std::optional<U> service_cycle_;
    static void cadence_add(U& target,U value=1) {
        need(value<=UINT64_MAX-target,"service cadence counter overflow");target+=value;
    }
    // No completion take/delivery, no credit release, no L2/Runtime callback.
    // Publish the same-cycle guard only after the full strict pump returns false.
    void service_only(U current_cycle) {
        if(service_cycle_ && *service_cycle_==current_cycle){cadence_add(cadence_.same_cycle_coalesces);return;}
        need(!service_cycle_ || current_cycle>*service_cycle_,"service cadence frontier regressed");
        const U through=clock_.poll_ps(current_cycle);const double boundary=strict_poll_ns(through);
        if(drain_mode_==DrainMode::Global) {
        for(;;) {
            ++admission_.native_service_before_calls;
            if(!device_.service_before(boundary))break;
            ++admission_.native_service_steps;
        }
        } else {
            cadence_add(independent_api_calls_);
            const auto result=device_.drain_independent_before(boundary);
            if(result.used_independent) {
                need(result.legacy_global_rounds==0,"independent result mixed counter units");
                cadence_add(independent_fast_drains_);cadence_add(pc_service_one_calls_,result.pc_service_one_calls);
            } else {
                need(result.pc_service_one_calls==0,"fallback result mixed counter units");
                cadence_add(fallback_global_drains_);
                cadence_add(admission_.native_service_steps,result.legacy_global_rounds);
                cadence_add(admission_.native_service_before_calls,result.legacy_global_rounds);
                cadence_add(admission_.native_service_before_calls); // final original false call
            }
        }
        admission_.last_service_frontier_ps=through;service_cycle_=current_cycle;
        cadence_add(cadence_.service_batches);
    }
    void observe_delivery(const std::vector<g::L2DramCompletion>& out,U cycle) {
        cadence_add(cadence_.step_calls);
        cadence_add(cadence_.completions_visible,out.size());
        for(const auto& completion:out) {
            need(completion.completion_cycle<=cycle,"service cadence early completion");
            const U delay=cycle-completion.completion_cycle;
            if(delay){cadence_add(cadence_.late_visible_completions);cadence_add(cadence_.total_visibility_delay_cycles,delay);}
            cadence_.max_visibility_delay_cycles=std::max(cadence_.max_visibility_delay_cycles,delay);
        }
    }

    static bool is_write(const g::L2DramRequest& r){
        need(r.cause==g::L2DramRequestCause::FILL_READ||r.cause==g::L2DramRequestCause::DIRTY_WRITEBACK,"unknown memory cause");
        return r.cause==g::L2DramRequestCause::DIRTY_WRITEBACK;
    }
    static bool execution_unchanged(const p::hbm::HbmStats& a,const p::hbm::HbmStats& b){
        return a.read_bytes==b.read_bytes&&a.write_bytes==b.write_bytes&&
            a.activations==b.activations&&a.precharges==b.precharges&&
            a.row_hits==b.row_hits&&a.row_misses==b.row_misses&&a.row_conflicts==b.row_conflicts&&
            a.refresh_count==b.refresh_count&&a.bus_busy_ns==b.bus_busy_ns&&a.finish_ns==b.finish_ns;
    }
    std::map<std::size_t,U> route(const g::L2DramRequest& r)const{
        const bool write = is_write(r);
        need(valid_native_request_shape(r), "native read128B/write-sector run shape invalid");
        (void)write;
        need(r.address<config_.device.capacity_bytes&&r.bytes<=config_.device.capacity_bytes-r.address,"physical capacity/tail violation");
        std::map<std::size_t,U> result;
        for(U offset=0;offset<r.bytes;offset+=32){
            const U address=r.address+offset;const auto a=device_.decode(address);
            need(device_.encode(a)==address,"native physical address roundtrip differs");
            const std::size_t pc=(static_cast<std::size_t>(a.stack)*config_.device.channels_per_stack+a.channel)*
                config_.device.pseudo_channels_per_channel+a.pseudo_channel;
            need(pc<occupied_.size(),"decoded service channel out of bounds");++result[pc];
        }
        return result;
    }
public:
    static bool valid_native_request_shape(const g::L2DramRequest& r) {
        if (r.cause == g::L2DramRequestCause::FILL_READ)
            return r.bytes >= 32 && r.bytes <= 128 && r.bytes % 32 == 0 &&
                r.address % 32 == 0 && r.bytes <= 128 - (r.address % 128);
        if (r.cause != g::L2DramRequestCause::DIRTY_WRITEBACK) return false;
        if constexpr (g::kTilegenDirtySectorMode == 2)
            return r.bytes >= 32 && r.bytes <= 128 && r.bytes % 32 == 0 &&
                r.address % 32 == 0 && r.bytes <= 128 - (r.address % 128);
        return r.bytes == 128 && r.address % 128 == 0;
    }
    HbfCreditBackend(Clock clock,p::hbm::HbmConfig config,U max_live=4096,U credits_per_pc=32,unsigned service_quantum=1,DrainMode drain_mode=DrainMode::Global):
        clock_(clock),device_(config),config_(device_.config()),max_live_(max_live),credits_per_pc_(credits_per_pc),
        occupied_(static_cast<std::size_t>(config_.device.stacks)*config_.device.channels_per_stack*
            config_.device.pseudo_channels_per_channel,0){
        need(service_quantum==1||service_quantum==2||service_quantum==4||service_quantum==8,"service cadence quantum 1/2/4/8");
        need(drain_mode==DrainMode::Global||drain_mode==DrainMode::Independent,"unknown HBF drain mode");
        service_quantum_=service_quantum;drain_mode_=drain_mode;
        need(max_live>0&&max_live<=4096,"bounded physical parent capacity");
        need(config_.burst_bytes()==32,"native GDDR6 service profile must produce 32B bursts");
        need(occupied_.size()>0&&occupied_.size()<=4096,"bounded native channel geometry");
        need(credits_per_pc_>=4&&credits_per_pc_<=32&&credits_per_pc_<=config_.controller.queue_depth,
             "burst credits must fit native queue and bounded 32-credit service contract");
        need(!config_.controller.replicate_symmetric_pseudo_channels,"initial native admission audit requires replication disabled");
        need(!config_.controller.refresh_enabled&&!config_.controller.same_bank_refresh,
             "selected native GDDR6 candidate requires explicitly disabled refresh");
    }
    U issue_cycle_to_ps(U cycle)const override{return clock_.issue_ps(cycle);}
    void enqueue(const g::L2DramRequest& r)override{
        need(try_enqueue(r,r.issue_cycle),"legacy enqueue requires retryable caller when saturated");
    }
    bool try_enqueue(const g::L2DramRequest& r,U current_cycle)override{
        need(!failed_,"failed physical backend");
        try {
            need(admission_.accepted<UINT64_MAX&&r.request_id==admission_.accepted&&r.source_sequence==r.request_id,
                 "noncontiguous request identity or duplicate admission");
            need(current_cycle<=INT64_MAX&&r.issue_cycle<=INT64_MAX,"native signed-cycle range");
            need(r.issue_time_ps==clock_.issue_ps(r.issue_cycle),"original issue clock mismatch");
            need(current_cycle>=r.issue_cycle&&(!stepped_||current_cycle>=last_step_),"retry/current cycle regressed");
            need(admission_.accepted==0||r.issue_cycle>=last_issue_,"source issue order regressed");
            const bool write=is_write(r);const auto credits=route(r);
            bool full=live_.size()>=max_live_;
            for(const auto& [pc,n]:credits){
                need(n<=credits_per_pc_,"single parent exceeds per-channel burst-credit capacity");
                full|=occupied_.at(pc)>credits_per_pc_-n;
            }
            // A rejected request changes only the blocked-attempt counter.
            // In particular, no service/step/future completion is consumed.
            if(full){++admission_.blocked;return false;}
            const U admission_ps=clock_.issue_ps(current_cycle);
            need(admission_.accepted==0||admission_ps>=admission_.last_admission_ps,"physical admission time regressed");
            if(service_quantum_>1){cadence_add(cadence_.admission_guard_visits);service_only(current_cycle);}
            const auto before=device_.execution_stats();
            const auto ticket=device_.enqueue(p::PhysicalRequest{
                .id="coupled/"+std::to_string(r.request_id),.tier=p::Tier::HBM,
                .op=write?p::Op::Write:p::Op::Read,.address_space=p::AddressSpace::Physical,
                .trace={p::TraceMode::Off,false},.arrival_ns=ps_to_ns(admission_ps),
                .addr=r.address,.bytes=r.bytes,.stream_id=0,.heatmap_source=p::HeatmapTrafficSource::Direct});
            ++admission_.native_enqueue_execution_checks;
            if(!execution_unchanged(before,device_.execution_stats())){
                ++admission_.native_enqueue_service_violations;
                throw std::runtime_error("native enqueue performed implicit service despite burst credits");
            }
            need(native_to_source_.emplace(ticket,r.request_id).second,"duplicate native ticket");
            need(live_.emplace(r.request_id,Live{r,admission_ps,UINT64_MAX,ticket,credits,false,{}}).second,
                 "duplicate live source request");
            for(const auto& [pc,n]:credits){
                occupied_[pc]+=n;admission_.reserved_bursts+=n;
                admission_.peak_pseudo_channel_credits=std::max(admission_.peak_pseudo_channel_credits,occupied_[pc]);
            }
            ++admission_.accepted;++admission_.physical_calls;last_issue_=r.issue_cycle;
            for(U value:{r.request_id,r.address,U(r.bytes),U(write)})
                for(unsigned byte=0;byte<8;++byte){admission_.actual_request_shape_fnv1a64^=(value>>(8*byte))&255;
                    admission_.actual_request_shape_fnv1a64*=1099511628211ULL;}
            last_admission_cycle_=current_cycle;admission_.last_admission_ps=admission_ps;
            need(admission_ps-r.issue_time_ps<=UINT64_MAX-admission_.total_admission_delay_ps,"admission wait overflow");
            admission_.total_admission_delay_ps+=admission_ps-r.issue_time_ps;
            admission_.peak_live=std::max<U>(admission_.peak_live,live_.size());
            admission_.peak_reserved_bursts=std::max(admission_.peak_reserved_bursts,admission_.reserved_bursts);
            if(write) { ++stats_.writeback_requests;
                need(r.bytes<=UINT64_MAX-stats_.writeback_bytes,"write-byte overflow"); stats_.writeback_bytes+=r.bytes;
            } else { ++stats_.fill_requests;
                need(r.bytes<=UINT64_MAX-stats_.fill_bytes,"read-byte overflow"); stats_.fill_bytes+=r.bytes; }
            stats_.peak_queue_depth=std::max<U>(stats_.peak_queue_depth,live_.size());
            return true;
        }catch(...){failed_=true;throw;}
    }
private:
    std::vector<g::L2DramCompletion> step_legacy(U current_cycle){
        need(!failed_,"failed physical backend");
        try {
            need(current_cycle<=INT64_MAX&&(!stepped_||current_cycle>=last_step_),"physical step regressed or overflowed");
            need(admission_.accepted==0||current_cycle>=last_admission_cycle_,"step precedes an already admitted arrival");
            stepped_=true;last_step_=current_cycle;
            const U through=clock_.poll_ps(current_cycle);const double boundary=strict_poll_ns(through);
            for(;;){
                ++admission_.native_service_before_calls;
                if(!device_.service_before(boundary))break;
                ++admission_.native_service_steps;
            }
            admission_.last_service_frontier_ps=through;
            for(auto& [ticket,physical]:device_.take_completions()){
                auto routed=native_to_source_.find(ticket);need(routed!=native_to_source_.end(),"unknown native completion ticket");
                auto& x=live_.at(routed->second);const auto& r=x.request;
                need(!x.posted&&x.native_ticket==ticket,"duplicate native completion");
                need(physical.id=="coupled/"+std::to_string(r.request_id)&&physical.op==(is_write(r)?p::Op::Write:p::Op::Read),
                     "native completion identity differs");
                need(physical.arrival_ns==ps_to_ns(x.admission_ps)&&std::isfinite(physical.finish_ns)&&
                     physical.finish_ns>=physical.start_ns&&physical.start_ns>=physical.arrival_ns,
                     "native completion timing differs");
                const U finish=ns_to_ps(physical.finish_ns);
                need(finish>=x.admission_ps&&physical.logical_bytes==r.bytes&&physical.physical_bytes==r.bytes,
                     "native completion time/byte conservation failed");
                need(physical.spans.empty()&&physical.note.empty()&&physical.resource_path.empty(),"physical trace retention forbidden");
                x.posted=true;x.completion_ps=finish;x.physical=std::move(physical);
                need(due_.emplace(std::make_pair(finish,r.source_sequence),r.request_id).second,"duplicate completion ordering key");
                native_to_source_.erase(routed);++admission_.native_completions_posted;
                if(finish>through){
                    ++admission_.future_completions_held;
                    admission_.max_future_completion_lead_ps=std::max(admission_.max_future_completion_lead_ps,finish-through);
                }
            }
            admission_.peak_due_completions=std::max<U>(admission_.peak_due_completions,due_.size());
            std::vector<g::L2DramCompletion> out;
            while(!due_.empty()&&due_.begin()->first.first<=through){
                const auto id=due_.begin()->second;const auto& x=live_.at(id);const auto& r=x.request;
                const bool write=is_write(r);const U finish_cycle=clock_.completion_cycle(x.completion_ps);
                need(x.posted&&finish_cycle>=r.issue_cycle&&finish_cycle<=current_cycle,"early/future native completion");
                out.push_back({r.request_id,r.source_sequence,r.issue_cycle,r.issue_time_ps,finish_cycle,r.key,write});
                for(const auto& [pc,n]:x.credits){
                    need(occupied_.at(pc)>=n&&admission_.reserved_bursts>=n,"burst credit underflow");
                    occupied_[pc]-=n;admission_.reserved_bursts-=n;
                }
                admission_.last_completion_ps=x.completion_ps;++admission_.completed;
                if(write) { ++stats_.writeback_completions; stats_.writeback_completed_bytes+=r.bytes; }
                else { ++stats_.fill_completions; stats_.fill_completed_bytes+=r.bytes; }
                live_.erase(id);due_.erase(due_.begin());
            }
            return out;
        }catch(...){failed_=true;throw;}
    }
    std::vector<g::L2DramCompletion> step_cadenced(U current_cycle){
        need(!failed_,"failed physical backend");
        try {
            need(current_cycle<=INT64_MAX&&(!stepped_||current_cycle>=last_step_),"physical step regressed or overflowed");
            need(admission_.accepted==0||current_cycle>=last_admission_cycle_,"step precedes an already admitted arrival");
            stepped_=true;last_step_=current_cycle;
            const U through=clock_.poll_ps(current_cycle);
            if(current_cycle%service_quantum_==0){cadence_add(cadence_.periodic_boundary_visits);service_only(current_cycle);}
            else cadence_add(cadence_.step_poll_skips);
            for(auto& [ticket,physical]:device_.take_completions()){
                auto routed=native_to_source_.find(ticket);need(routed!=native_to_source_.end(),"unknown native completion ticket");
                auto& x=live_.at(routed->second);const auto& r=x.request;
                need(!x.posted&&x.native_ticket==ticket,"duplicate native completion");
                need(physical.id=="coupled/"+std::to_string(r.request_id)&&physical.op==(is_write(r)?p::Op::Write:p::Op::Read),
                     "native completion identity differs");
                need(physical.arrival_ns==ps_to_ns(x.admission_ps)&&std::isfinite(physical.finish_ns)&&
                     physical.finish_ns>=physical.start_ns&&physical.start_ns>=physical.arrival_ns,
                     "native completion timing differs");
                const U finish=ns_to_ps(physical.finish_ns);
                need(finish>=x.admission_ps&&physical.logical_bytes==r.bytes&&physical.physical_bytes==r.bytes,
                     "native completion time/byte conservation failed");
                need(physical.spans.empty()&&physical.note.empty()&&physical.resource_path.empty(),"physical trace retention forbidden");
                x.posted=true;x.completion_ps=finish;x.physical=std::move(physical);
                need(due_.emplace(std::make_pair(finish,r.source_sequence),r.request_id).second,"duplicate completion ordering key");
                native_to_source_.erase(routed);++admission_.native_completions_posted;
                if(finish>through){
                    ++admission_.future_completions_held;
                    admission_.max_future_completion_lead_ps=std::max(admission_.max_future_completion_lead_ps,finish-through);
                }
            }
            admission_.peak_due_completions=std::max<U>(admission_.peak_due_completions,due_.size());
            std::vector<g::L2DramCompletion> out;
            while(!due_.empty()&&due_.begin()->first.first<=through){
                const auto id=due_.begin()->second;const auto& x=live_.at(id);const auto& r=x.request;
                const bool write=is_write(r);const U finish_cycle=clock_.completion_cycle(x.completion_ps);
                need(x.posted&&finish_cycle>=r.issue_cycle&&finish_cycle<=current_cycle,"early/future native completion");
                out.push_back({r.request_id,r.source_sequence,r.issue_cycle,r.issue_time_ps,finish_cycle,r.key,write});
                for(const auto& [pc,n]:x.credits){
                    need(occupied_.at(pc)>=n&&admission_.reserved_bursts>=n,"burst credit underflow");
                    occupied_[pc]-=n;admission_.reserved_bursts-=n;
                }
                admission_.last_completion_ps=x.completion_ps;++admission_.completed;
                if(write) { ++stats_.writeback_completions; stats_.writeback_completed_bytes+=r.bytes; }
                else { ++stats_.fill_completions; stats_.fill_completed_bytes+=r.bytes; }
                live_.erase(id);due_.erase(due_.begin());
            }
            return out;
        }catch(...){failed_=true;throw;}
    }
public:
    std::vector<g::L2DramCompletion> step(U current_cycle)override {
        try {
            auto out=(service_quantum_==1&&drain_mode_==DrainMode::Global)?step_legacy(current_cycle):step_cadenced(current_cycle);
            observe_delivery(out,current_cycle);return out;
        }catch(...){failed_=true;throw;}
    }
    ServiceCadenceDiagnostics cadence_diagnostics()const {
        auto out=cadence_;out.quantum=service_quantum_;
        if(service_quantum_==1&&drain_mode_==DrainMode::Global){out.legacy_steps=out.step_calls;out.service_batches=out.step_calls;}
        out.actual_service_before_calls=admission_.native_service_before_calls;
        out.actual_service_steps=admission_.native_service_steps;return out;
    }
    NativeServiceDiagnostics service_diagnostics()const {
        return {drain_mode_,cadence_diagnostics(),independent_api_calls_,independent_fast_drains_,fallback_global_drains_,pc_service_one_calls_};
    }
    // Unknown until native commands have produced a completion description.
    // UINT64_MAX is not quiescence: queue_depth() still includes that request.
    U next_completion_cycle()const{return due_.empty()?UINT64_MAX:clock_.completion_cycle(due_.begin()->first.first);}
    U next_completion_ps()const{return due_.empty()?UINT64_MAX:due_.begin()->first.first;}
    U max_live()const{return max_live_;}
    std::size_t admission_capacity()const override{return max_live_;}
    U credits_per_pc()const{return credits_per_pc_;}
    const p::hbm::HbmConfig& native_config()const{return config_;}
    const std::vector<U>& occupied_credits()const{return occupied_;}
    const g::L2DramRuntimeStatistics& statistics()const override{return stats_;}
    const AdmissionStatistics& admission_statistics()const{return admission_;}
    std::size_t queue_depth()const override{return live_.size();}
    p::hbm::HbmStats physical_statistics()const{return device_.stats();}
    void finalize()const{
        need(!failed_&&live_.empty()&&due_.empty()&&native_to_source_.empty()&&admission_.accepted==admission_.completed,
             "unclosed native asynchronous backend");
        need(admission_.reserved_bursts==0&&std::all_of(occupied_.begin(),occupied_.end(),[](U n){return n==0;}),
             "unreturned physical burst credits");
        const auto physical=device_.stats();
        need(stats_.fill_bytes==stats_.fill_completed_bytes&&stats_.writeback_bytes==stats_.writeback_completed_bytes,
             "native accepted/completed byte conservation differs");
        need(physical.read_bytes==stats_.fill_bytes&&physical.write_bytes==stats_.writeback_bytes,
             "physical/native actual-byte traffic differs");
        need(admission_.native_enqueue_service_violations==0&&admission_.native_enqueue_execution_checks==admission_.accepted&&
             admission_.native_completions_posted==admission_.accepted,"native admission/service audit did not close");
        need(physical.max_queue_occupancy<=credits_per_pc_&&physical.replicated_requests==0&&physical.refresh_count==0,
             "native queue/replication/refresh contract differs");
    }
};
inline void require_external_only(const g::L2Cache& cache){
    need(!cache.uses_internal_dram_backend(),"external HBFSIM and GTSim internal DRAM are mutually exclusive");
}
} // namespace sg_hbf
