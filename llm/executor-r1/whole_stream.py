"""Execute complete current-process GLOBAL programs and explicit API history.

This is a new executor, not a change to the frozen registry's admission flags.
The source input remains separate from the explicitly modeled cache/SM schedule.
No byte multiplier or omitted launch fallback exists. Diagnostic dirty drains
are opt-in interventions; the default preserves continuous cache residency.
"""
import argparse
import os
import collections
import hashlib
import importlib.util
import json
import math
import resource
import subprocess
import sys
import time
from pathlib import Path

K = Path(os.environ.get('TILEGEN_NATIVE_TREE', str(Path(__file__).resolve().parents[2] / 'kernel-complete-input-r1'))).resolve()
SUPPORT = Path(os.environ.get('TILEGEN_NATIVE_SUPPORT_TREE', str(Path(__file__).resolve().parent.parent))).resolve()
sys.path.insert(0, str(K / 'native-whole-graph-r2'))
from graph_io import Graph, checked, pin


def need(value, message):
    if not value:
        raise ValueError(message)


def parse_dirty_age_accesses(text):
    """Accept an explicit unsigned decimal access count, including disabled=0."""
    if not isinstance(text, str) or not text.isascii() or not text.isdecimal():
        raise argparse.ArgumentTypeError('dirty age must be an unsigned decimal uint64')
    value = int(text)
    if value >= 1 << 64:
        raise argparse.ArgumentTypeError('dirty age exceeds uint64')
    return value


def dirty_age_expectation(value, clock="global"):
    need(clock in ("global", "set"), "expected dirty age clock must be global or set")
    need(clock == "global" or value is not None, "set clock requires an explicit expected dirty age budget")
    return dict(expected_dirty_age_clock=clock, expected_dirty_age_accesses=64000000 if value is None else value,
                expected_dirty_age_source='legacy_default_64000000' if value is None
                else 'explicit_cli_expected_dirty_age_accesses')


def validate_cache_configuration(configuration, expected_dirty_age_accesses, expected_dirty_age_clock="global"):
    need(type(expected_dirty_age_accesses) is int and
         0 <= expected_dirty_age_accesses < 1 << 64, 'expected dirty age must be uint64')
    need(configuration['L2']['EF_hit_numerator'] == 288, 'frozen EF h288 changed')
    need(expected_dirty_age_clock in ('global', 'set'), 'expected dirty age clock must be global or set')
    need(configuration['L2'].get('dirty_age_clock', 'global') == expected_dirty_age_clock, 'runner dirty age clock differs from expected clock')
    actual = configuration['L2']['dirty_age_accesses']
    need(type(actual) is int and actual == expected_dirty_age_accesses,
         'runner dirty age differs from explicit/default expected access count')


def load_module(path, support=False):
    path = Path(path).resolve()
    need(path.is_relative_to(K) or (support and path.is_relative_to(SUPPORT)), 'runtime must be inside declared task source trees')
    spec = importlib.util.spec_from_file_location('current_smoke_registry', path)
    module = importlib.util.module_from_spec(spec)
    previous = list(sys.path)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = previous
    return module


def all_entries(registry):
    result = dict(getattr(registry, 'entries', {}))
    if hasattr(registry, 'base'):
        result.update(all_entries(registry.base))
    result.update(getattr(registry, 'extras', {}))
    return result


def api_effects(node):
    """Preserve actual device endpoints; model transfers through the fixed L2.

    This functional cache route is an explicit simulator assumption, not a claim
    about copy-engine transaction sizes, DMA coherence or actual issue ordering.
    """
    op = node['operation']
    need(op['direction_known'] and op['device_address_extent_qualified'],
         'unknown API direction/device extent')
    need(op['geometry'] == 'linear' and op['height'] == op['depth'] == 1,
         'nonlinear API requires a separate source-qualified implementation')
    size = op['requested_bytes']
    need(type(size) is int and size >= 0 and size == op['width_bytes'], 'API size')
    directions = {
        ('memcpy', 'host_to_device'): [('WRITE', 'destination')],
        ('memcpy', 'device_to_host'): [('READ', 'source')],
        ('memcpy', 'device_to_device'): [('READ', 'source'), ('WRITE', 'destination')],
        ('memset', 'device_fill'): [('WRITE', 'destination')],
    }
    key = op['action'], op['direction']
    need(key in directions, 'unsupported API action/direction')
    result = []
    for operation, endpoint_name in directions[key]:
        endpoint = op[endpoint_name]
        address = endpoint['address_u64']
        need(endpoint['kind'] == 'device' and type(address) is int and address > 0,
             'actual device endpoint required')
        need(all(endpoint.get(v, 0) == 0 for v in ('x_bytes', 'y', 'z')),
             'linear offset endpoint not implemented')
        need(address + size <= 1 << 64, 'device address extent overflow')
        result.append(dict(type='api_range', operation=operation, address=address, bytes=size))
    return result


def source_policy(event):
    # The native opcode is the source of the hint. A chosen model interprets it;
    # this does not infer a new Ada replacement algorithm from its spelling.
    parts = set(event['opcode'].split('.'))
    op = event['operation']
    need(op in ('READ', 'WRITE', 'ATOMIC_RMW', 'GLOBAL_TO_SHARED'), 'unknown global operation')
    need(not (parts & {'LU', 'CV'}), 'unimplemented source cache hint')
    # Explicit functional interpretation: EL preserves its source opcode but uses normal priority.
    # No hardware evict-last behavior or stronger timing/retention claim is made.
    return dict(bypass_l1=op == 'ATOMIC_RMW' or 'BYPASS' in parts,
                l2_priority='evict_first' if 'EF' in parts else 'normal',
                semantic=event.get('role', 'unclassified_source_role'))


CONTROL_ACTIONS = {
    'block_reduction_and_barrier', 'release_fence_and_block_barrier',
    'CTA_last_condition', 'acquire_fence', 'final_block_reduction_and_barrier',
}
MERGE_CONTROL_ACTIONS = {
    'conditional_index_branch', 'cp_async_nonexecuted_prefetch',
    'shared_merge_and_block_barriers', 'skip_next_prefetch_and_remaining_unrolled_iterations',
    'shared_state_reduction', 'skip_null_lse_and_exit', 'exit',
}
MERGE_CODE = 'ee37b075dfd417ab8f86a1bb914a8180541f11ebaad95d5b3d1130de6f4d3b63'


def normalized_event(event, entry=None):
    row = dict(event)
    if 'operation' not in row or 'lane_addresses' not in row:
        need('operation' not in row and 'lane_addresses' not in row,
             'incomplete memory event cannot be treated as control')
        merge_control = row.get('action') in MERGE_CONTROL_ACTIONS
        if merge_control:
            need(entry is not None and entry['family'] == 'qwen_merge'
                 and entry['code_sha256'] == MERGE_CODE,
                 'merge control requires its qualified current source program')
            need(entry['control_witness']['schema'] == 'CURRENT_MERGE_CPU_PLAN_AND_HTOD_WITNESS_V1'
                 and entry['control_witness']['values'] == [0, 2], 'merge control requires current plan witness')
            cta = row.get('cta_linear_id')
            need(type(cta) is int and 0 <= cta < 48, 'merge control CTA extent')
            if row['action'] == 'exit':
                need(cta >= 12 and row.get('pc') == 0x100, 'merge inactive CTA exit')
            else:
                need(cta < 12, 'active merge control CTA')
            if row['action'] == 'conditional_index_branch':
                need(row.get('conditional_not_observed') is False and row.get('indptr') == [0, 2],
                     'unqualified merge control branch')
        need(row.get('kind') == 'control' and (row.get('action') in CONTROL_ACTIONS or merge_control),
             'unimplemented non-memory event; cannot silently discard it')
        need(not row.get('DRAM_flush', False), 'unimplemented source flush')
        return None
    # The frozen merge provider names the GLOBAL projection of this LDGSTS
    # "READ". Preserve its exact addresses/mask, and expose the native direction
    # expected by the common consumer; shared destinations remain out of scope.
    if (entry is not None and entry['family'] == 'qwen_merge'
            and entry['code_sha256'] == MERGE_CODE and row.get('pc') == 0x750):
        if entry.get('binding', {}).get('schema') == 'CURRENT_QWEN_P256_P512_MERGE_BINDING_V1':
            need(row['operation'] == 'GLOBAL_TO_SHARED'
                 and row['width'] == row.get('transfer_width') == 16
                 and row.get('transfer_policy') == 1
                 and row.get('source_read_mask') == row.get('global_effective_mask')
                 and row.get('shared_destination_projection_omitted') is True,
                 'unexpected current merge async-transfer projection')
        else:
            need(row['operation'] == 'READ' and row['width'] == 16,
                 'unexpected frozen merge async-read projection')
            row['source_projection_operation'] = row['operation']
            row['operation'] = 'GLOBAL_TO_SHARED'
            row['source_read_mask'] = row['global_effective_mask']
    row.setdefault('cta_linear_id', row.get('cta'))
    row.setdefault('cta_warp_id', row.get('warp'))
    need(type(row['cta_linear_id']) is int and type(row['cta_warp_id']) is int,
         'source CTA/warp coordinate missing')
    mask = row.get('global_effective_mask')
    if mask is None:
        mask = row['source_read_mask'] if row['operation'] == 'GLOBAL_TO_SHARED' else row['effective_mask']
    need(type(mask) is int and 0 <= mask <= 0xffffffff and mask & ~row['effective_mask'] == 0,
         'global mask must be a subset of execution mask')
    row['global_effective_mask'] = mask
    return row


def bind_static_opcode(event, static_by_pc):
    """Recover hints from this function's actual SASS, never a default opcode."""
    row = dict(event)
    need(row['pc'] in static_by_pc, 'source PC outside current function SASS')
    instruction = static_by_pc[row['pc']]
    opcode = instruction['opcode']
    need(instruction['nvbit_size'] == row['width'], 'source width differs from native SASS')
    if 'opcode' in row:
        need(row['opcode'] == opcode, 'source opcode differs from native SASS')
    else:
        row['opcode'] = opcode
        row['opcode_provenance'] = 'exact_current_function_static_PC'
    operation = row['operation']
    compatible = {
        'READ': opcode.startswith(('LDG.', 'LD.')),
        'WRITE': opcode.startswith(('STG.', 'ST.')),
        'GLOBAL_TO_SHARED': opcode.startswith('LDGSTS.'),
        'ATOMIC_RMW': opcode.startswith(('ATOM', 'RED.')),
    }
    need(compatible.get(operation, False), 'source direction differs from native opcode')
    return row


def preflight(graph, entries):
    value = graph.value
    contract = value['input_contract']
    need((contract['prefill_length'], contract['decode_steps']) in ((32, 2), (64, 2), (128, 2), (256, 2), (512, 2), (128, 4), (128, 8), (128, 16)),
         'only admitted native Prefill/D2 sweep and P128 decode regression cases allowed')
    need(contract['batch_size'] == 1 and contract['dtype'] == 'bfloat16'
         and contract['warmup_runs'] == 1 and contract['sampling_retained']
         and not contract['output_feedback'] and not contract['cuda_graph'], 'inference contract')
    with checked(value['artifacts']['nodes']).open() as stream:
        nodes = list(map(json.loads, stream))
    kernels = {n['id'] for n in nodes if n['kind'] == 'native_kernel'}
    need(kernels == set(entries), 'all and only current kernel nodes require a native source program')
    streams = set()
    api_counts = collections.Counter()
    previous = -1
    for node in nodes:
        need(node['submission_event'] > previous and node['process'] == value['process'],
             'graph submission order/process differs')
        previous = node['submission_event']
        kind = node['kind']
        if kind in ('native_kernel', 'memory_api_submission'):
            streams.add((node['context_handle_u64'], node['stream_u64']))
        if kind == 'native_kernel':
            entry = entries[node['id']]
            need(entry['process'] == value['process'] and entry['code_sha256'] == node['code_sha256'],
                 'registry/kernel process or code differs')
            need(entry['node_id'] == node['id'] and entry['native_launch_id'] == node['native_launch_id']
                 and entry['function_id'] == node['function_id']
                 and entry['graph_argument_sha256'] == node['argument_record_sha256'],
                 'registry entry is not bound to this exact current launch/raw ABI')
            raw = graph.argument(node['id'])
            key = f"epoch-{raw['epoch_id']}-launch-{raw['epoch_launch_ordinal']}"
            need(entry['source_launch_key'] == key, 'entry source launch key differs')
            binding = entry['binding']
            expected_binding = dict(native_launch_id=node['native_launch_id'], function_id=node['function_id'],
                                    phase=raw['phase'], grid=node['grid'], block=node['block'],
                                    source_launch_key=key, code_sha256=node['code_sha256'], process=value['process'])
            for field, expected in expected_binding.items():
                if field in binding:
                    need(binding[field] == expected, 'source binding differs: ' + field)
            need(entry.get('global_program_complete_for_supported_specialization') is True,
                 'source program is not complete for current specialization')
            if entry.get('requires_explicit_execution_schedule'):
                if binding.get('schema') == 'CURRENT_QWEN_P256_P512_QKV_SERIAL_SOURCE_BINDING_V1':
                    # Reuse the same current-ABI/static/initialization and full
                    # dependency-order checks as the emitted native command.
                    sweep = load_module(SUPPORT / 'fast-prefill-sweep-r1/command_builder.py', support=True)
                    sweep.build_command(entry, graph)
                else:
                    down = load_module(K / 'llama-prefill-down-native-program-r1/provider.py')
                    need(entry['code_sha256'] == down.CODE, 'unqualified stateful source program')
                    down.qualify_static(graph.function_static(entry['function_id']))
                    down.validate_initialization(binding, entry['initialization'])
        elif kind == 'memory_api_submission':
            api_effects(node)
            api_counts[node['operation']['direction']] += 1
        elif kind == 'allocation_API_observation':
            need(node['observation']['action'] in ('allocate', 'free'), 'unknown allocation operation')
        elif kind == 'observer_sampling_flush':
            need(node.get('workload_operation') is False, 'opaque application operation cannot be excluded')
        else:
            raise ValueError('unimplemented graph node kind: ' + kind)
    need(len(streams) == 1, 'multiple GPU streams require explicit synchronization model')
    expected = [f'{stage}/{phase}' for stage in ('Warmup', 'Measured')
                for phase in ['Prefill'] + [f'Decode{i}' for i in range(1, contract['decode_steps'] + 1)]]
    need([p['phase'] for p in value['phases']] == expected, 'complete contract-derived phase order')
    timeline = [(n['submission_event'], 'node', n) for n in nodes]
    for phase in value['phases']:
        for edge in ('begin', 'end'):
            observed = graph.record(phase[edge])
            need(observed['epoch_id'] == phase['epoch_id'], 'phase epoch mismatch')
            timeline.append((observed['event_ordinal'], 'phase_' + edge, phase['phase']))
    timeline.sort(key=lambda row: row[0])
    need(len({row[0] for row in timeline}) == len(timeline), 'ambiguous source event ordinal')
    return timeline, dict(kernel_count=len(kernels), memory_API_directions=dict(api_counts),
                         observed_single_stream=list(next(iter(streams))), phases=expected,
                         all_initialization_nodes_retained=True, no_terminal_dirty_flush=True)


def serial_down_schedule():
    """One legal modeled serial schedule; not a measured hardware poll schedule."""
    actions = []
    for part in range(9):
        for n in range(16):
            cta = 16 * part + n
            actions.append(dict(action='body', cta=cta))
            if part:
                actions.extend(dict(action='poll', cta=cta, warp=w) for w in range(8))
            actions.extend(dict(action='output_group', cta=cta, group=g) for g in range(8))
            actions.append(dict(action='publish', cta=cta))
    return actions


class Writer:
    def __init__(self, stream):
        self.stream, self.hash, self.bytes, self.records = stream, hashlib.sha256(), 0, 0
        self.source_hash, self.source_bytes, self.source_records = hashlib.sha256(), 0, 0

    def __call__(self, row):
        raw = (json.dumps(row, separators=(',', ':')) + '\n').encode()
        self.stream.write(raw)
        self.hash.update(raw)
        self.bytes += len(raw)
        self.records += 1
        if not (row['type'] == 'drain' or
                (row['type'] == 'snapshot' and '/diagnostic-drain/' in row['label'])):
            self.source_hash.update(raw)
            self.source_bytes += len(raw)
            self.source_records += 1


def diagnostic_drain(send, label, phase):
    """Separate intervention traffic from the natural source phase endpoint."""
    send(dict(type='snapshot', label=label + '/diagnostic-drain/begin'))
    send(dict(type='drain', label=label, phase=phase))
    send(dict(type='snapshot', label=label + '/diagnostic-drain/end'))


def phase_traffic(label, before_row, after_row):
    """Difference additive counters; retain endpoint residency as a gauge."""
    before, after = before_row['cumulative'], after_row['cumulative']
    keys = ['DRAM_read_bytes', 'DRAM_write_bytes', 'source_read_effect_bytes',
            'source_write_effect_bytes', 'dirty_sector_creations', 'evicted_dirty_sectors',
            'age_writeback_bytes', 'capacity_eviction_writeback_bytes', 'L2_forwarded_access_sequence',
            'drain_writeback_bytes', 'explicit_drain_count', 'masked_writeback_requests',
            'writeback_enabled_byte_coverage', 'L2_sector_read_hits', 'L2_sector_read_misses']
    for cause in ('load_fill', 'read_merge', 'atomic_read', 'capacity_merge', 'age_merge',
                  'drain_merge', 'old_store_RFO', 'old_atomic_RFO'):
        keys += ['DRAM_' + cause + suffix for suffix in ('_requests', '_bytes')]
    delta = {k: after[k] - before[k] for k in keys if k in before and k in after}
    need(all(v >= 0 for v in delta.values()), 'phase cumulative counters regressed')
    need(before['dirty_tail_bytes'] + delta['dirty_sector_creations'] * 32 ==
         delta['DRAM_write_bytes'] + after['dirty_tail_bytes'], 'phase dirty ledger does not close')
    cause_keys = ['DRAM_' + c + '_bytes' for c in ('load_fill', 'read_merge', 'atomic_read',
                  'capacity_merge', 'age_merge', 'drain_merge', 'old_store_RFO', 'old_atomic_RFO')]
    if all(k in delta for k in cause_keys):
        need(sum(delta[k] for k in cause_keys) == delta['DRAM_read_bytes'], 'typed DRAM reads do not close')
    row = dict(phase=label, **delta, dirty_start_bytes=before['dirty_tail_bytes'],
               dirty_end_bytes=after['dirty_tail_bytes'], dirty_ledger_closed=True)
    if 'CPU_minutes_since_run_start' in before_row and 'CPU_minutes_since_run_start' in after_row:
        row['cache_CPU_minutes'] = after_row['CPU_minutes_since_run_start'] - before_row['CPU_minutes_since_run_start']
    if 'dirty_ownership' in before_row and 'dirty_ownership' in after_row:
        b, e = before_row['dirty_ownership'], after_row['dirty_ownership']
        values = ('write_bytes','enabled_write_byte_coverage','masked_writeback_requests','writeback_merge_read_bytes')
        dimensions = ('first_writer_phase','last_writer_phase','trigger_phase','reason',
                      'first_writer_semantic','last_writer_semantic','trigger_semantic')
        def key(r): return tuple(r.get(k) for k in dimensions)
        prior = {key(r):r for r in b['writebacks_cumulative']}
        flows = []
        for item in e['writebacks_cumulative']:
            p = prior.get(key(item), {})
            flow = {k:item.get(k) for k in dimensions}
            flow.update({k:item.get(k,0)-p.get(k,0) for k in values})
            need(all(flow[k] >= 0 for k in values), 'ownership cumulative counters regressed')
            if flow['write_bytes']: flows.append(flow)
        need(sum(r['write_bytes'] for r in flows) == delta['DRAM_write_bytes'], 'phase write ownership does not close')
        row['dirty_ownership'] = dict(scope=e['scope'], writeback_flows=flows,
            resident_start=b['resident_dirty_carry'], resident_end=e['resident_dirty_carry'])
    return row


def gemv_command(entry, graph, provider):
    binding = entry['binding']
    static = graph.function_static(entry['function_id'])
    stages = provider.qualify_static(binding['code_sha256'], static)
    policies = {}
    for role, opcode, operation in [('weights', 'LDG.E.EF.U16', 'READ'),
                                    ('activation', 'LDG.E.U16.STRONG.SM', 'READ'),
                                    ('bias', 'LDG.E.U16', 'READ'), ('output', 'STG.E.U16', 'WRITE')]:
        policies[role] = source_policy(dict(role=role, opcode=opcode, operation=operation))
    return dict(type='native_gemv_program', binding=binding, stages=stages,
                epilog=provider.EPILOG[binding['code_sha256']], policies=policies,
                cta_range=[0, math.prod(binding['grid'])])


def gemm_command(entry, graph, provider, contracts):
    binding = entry['binding']
    role = binding['role']
    provider.qualify_static(graph.function_static(entry['function_id']), role)
    read = 'LDGSTS.E.BYPASS.LTC128B.128.CONSTANT' if role == 'gate_up_proj' else 'LDG.E.LTC128B.128'
    write = 'STG.E.EF.64' if role == 'gate_up_proj' else 'STG.E.64'
    policies = {name: source_policy(dict(role=name, opcode=opcode, operation=operation))
                for name, opcode, operation in [('weights', read, 'GLOBAL_TO_SHARED' if role == 'gate_up_proj' else 'READ'),
                    ('activation', read, 'GLOBAL_TO_SHARED' if role == 'gate_up_proj' else 'READ'),
                    ('bias', 'LDG.E.LTC128B.64', 'READ'), ('output', write, 'WRITE')]}
    return dict(type='qwen_p32_gemm_program', binding=binding, static_contract=contracts[role],
                policies=policies, cta_range=[0, math.prod(binding['grid'])])


def emit(timeline, registry, entries, send, progress, graph=None, fast_gemv=False, fast_gemm=False, fast_prefill=False, fast_down=False, fast_p32_prefill=False, launch_resources=None, fast_prefill_sweep=False, drain_policy='none'):
    need(drain_policy in ('none', 'measured-phase-end', 'run-end'), 'explicit drain policy')
    send(dict(type='run_begin', schema='TILEGEN_SOURCE_CACHE_STREAM_V1', sm_policy='cta_mod_48'))
    counts = collections.Counter()
    function_statics = {}
    schemas = {entry['binding'].get('schema') for entry in entries.values()}
    gemv = load_module(K / 'qwen-gemv-native-program-r1/provider.py') if fast_gemv and 'CURRENT_QWEN_GEMV_SOURCE_BINDING_V1' in schemas else None
    llama_gemv = load_module(K / 'llama-gemv-native-program-r1/provider.py') if fast_gemv and 'CURRENT_LLAMA_GEMV_SOURCE_BINDING_V1' in schemas else None
    last_measured_phase = 'Measured/Decode' + str(graph.value['input_contract']['decode_steps'])
    prefill = load_module(SUPPORT / 'fast-prefill-r2/command_builder.py', support=True) if fast_prefill else None
    prefill_sweep = load_module(SUPPORT / 'fast-prefill-sweep-r1/command_builder.py', support=True) if fast_prefill_sweep else None
    down_fast = load_module(SUPPORT / 'fast-down-r1/command.py', support=True) if fast_down else None
    p32_prefill = load_module(SUPPORT / 'fast-llama-p32-prefill-r1/command_builder.py', support=True) if fast_p32_prefill else None
    gemm = load_module(K / 'p32d2-smoke-r1/prefill-programs/provider.py') if fast_gemm else None
    gemm_contracts = json.loads((K / 'p32d2-smoke-r1/cache-runner-gemm-r1/source-contracts.json').read_text()) if fast_gemm else None
    for ordinal, kind, payload in timeline:
        if kind.startswith('phase_'):
            edge = kind.split('_')[1]
            if payload == 'Measured/Prefill' and edge == 'begin':
                send(dict(type='snapshot', label='Measured/Full/begin'))
            send(dict(type='snapshot', label=payload + '/' + edge))
            if payload == last_measured_phase and edge == 'end':
                send(dict(type='snapshot', label='Measured/Full/end'))
            if drain_policy == 'measured-phase-end' and edge == 'end' and payload.startswith('Measured/'):
                diagnostic_drain(send, payload, payload + '/diagnostic-drain')
                counts['diagnostic_drains'] += 1
            progress(dict(event=kind, phase=payload, source_event_ordinal=ordinal, counts=dict(counts)))
            continue
        node = payload
        node_kind = node['kind']
        phase = node['phase'] or '<phase-external>/' + node['history']['region']
        if node_kind == 'native_kernel':
            entry = entries[node['id']]
            ident = node['native_launch_id']
            begin = dict(type='begin_kernel', id=ident, phase=phase, semantic=entry['family'] + ':' + node['module_scope'],
                         grid=node['grid'], block=node['block'])
            if launch_resources is not None:
                begin['observed_shared_bytes'] = launch_resources[ident]
            send(begin)
            schema = entry['binding'].get('schema')
            selected_gemv = gemv if schema == 'CURRENT_QWEN_GEMV_SOURCE_BINDING_V1' else llama_gemv if schema == 'CURRENT_LLAMA_GEMV_SOURCE_BINDING_V1' else None
            if selected_gemv is not None and entry['code_sha256'] in selected_gemv.RANGES:
                send(gemv_command(entry, graph, selected_gemv))
                counts['native_CPP_GEMV_programs'] += 1
            elif p32_prefill is not None and schema == 'CURRENT_LLAMA_P32_PREFILL_SOURCE_BINDING_V1':
                need(entry['family'] == 'llama_p32_prefill_four_gemm', 'finite P32 Prefill family')
                send(p32_prefill.build_command(entry, graph))
                counts['native_CPP_LLAMA_P32_PREFILL_programs'] += 1
            elif (gemm is not None and entry['family'] == 'p32_prefill_four_gemm'
                    and entry['binding'].get('schema') == 'CURRENT_P32_QWEN_GEMM_SOURCE_BINDING_V1'):
                send(gemm_command(entry, graph, gemm, gemm_contracts))
                counts['native_CPP_GEMM_programs'] += 1
            elif prefill_sweep is not None and prefill_sweep.supports(entry):
                send(prefill_sweep.build_command(entry, graph))
                counts['native_CPP_PREFILL_SWEEP_programs'] += 1
                if entry.get('requires_explicit_execution_schedule'):
                    counts['explicit_serial_splitK_programs'] += 1
                    # QKV: two dependent parts, one modeled poll per four-warps
                    # CTA. EL follows the existing declared normal-priority
                    # approximation; actual hardware polling is not measured.
                    need(schema == 'CURRENT_QWEN_P256_P512_QKV_SERIAL_SOURCE_BINDING_V1', 'qualified sweep stateful family')
                    counts['source_EL_events_modeled_normal_priority'] += 8 * node['grid'][0] * node['grid'][1]
            elif prefill is not None and any(schema == c['schema'] and entry['code_sha256'] == c['code'] for c in prefill.CONTRACTS.values()):
                send(prefill.build_command(entry, graph))
                counts['native_CPP_PREFILL_programs'] += 1
            elif down_fast is not None and entry.get('requires_explicit_execution_schedule'):
                need(entry['code_sha256'] == down_fast.source.CODE, 'unsupported stateful fast program')
                send(down_fast.command(entry['binding'], entry['initialization'], graph.function_static(entry['function_id']), serial_down_schedule(), range(144)))
                counts['native_CPP_SPLITK_programs'] += 1
                counts['explicit_serial_splitK_programs'] += 1
                counts['source_EL_events_modeled_normal_priority'] += 16 * 8 * 8
            else:
                fid = entry['function_id']
                if fid not in function_statics:
                    function_statics[fid] = {r['offset']: r for r in graph.function_static(fid)}
                kwargs = {}
                if entry.get('requires_explicit_execution_schedule'):
                    kwargs['execution_schedule'] = serial_down_schedule()
                    counts['explicit_serial_splitK_programs'] += 1
                if entry['family'] == 'argmax' or entry['binding'].get('schema') == 'CURRENT_ARGMAX_GLOBAL_FAMILY_BINDING_V1':
                    kwargs['cta_completion_order'] = list(range(math.prod(node['grid'])))
                for event in registry.events(node['id'], **kwargs):
                    source_event = event
                    event = normalized_event(event, entry)
                    if event is None:
                        counts['source_control_events'] += 1
                        counts['source_control_action:' + source_event['action']] += 1
                        continue
                    event = bind_static_opcode(event, function_statics[fid])
                    if 'EL' in event['opcode'].split('.'):
                        counts['source_EL_events_modeled_normal_priority'] += 1
                    send(dict(type='memory', event=event, policy=source_policy(event)))
                    counts['source_global_instructions'] += 1
            send(dict(type='end_kernel', id=ident))
            counts['kernels'] += 1
        elif node_kind == 'memory_api_submission':
            ident = node['native_memory_operation_id']
            send(dict(type='begin_api', id=ident, phase=phase, api=node['operation']['cuda_api'],
                      dma_model='L2_COHERENT_FUNCTIONAL_128B_CHUNKS'))
            for effect in api_effects(node):
                send(effect)
            send(dict(type='end_api', id=ident))
            counts['memory_APIs'] += 1
        elif node_kind == 'allocation_API_observation':
            # A CUDA allocation/free does not erase cache tags/dirty sectors.
            # Actual VAs remain the cache identity; semantic owner is per access.
            counts['allocation_metadata_no_cache_flush'] += 1
            send(dict(type='allocation_metadata', node=node))
        elif node_kind == 'observer_sampling_flush':
            counts['excluded_observer_only_operations'] += 1
        else:
            raise ValueError('unknown graph node during execution')
        if counts['kernels'] % 25 == 0:
            progress(dict(event='node_completed', graph_node=node['id'], phase=phase,
                          source_event_ordinal=ordinal, counts=dict(counts)))
    if drain_policy == 'run-end':
        diagnostic_drain(send, 'RunEnd', 'Diagnostic/RunEndDrain')
        counts['diagnostic_drains'] += 1
    send(dict(type='run_end'))
    return dict(counts)


def main():
    parser = argparse.ArgumentParser()
    for name in ('graph', 'registry', 'runtime', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--runner', type=Path)
    parser.add_argument('--runner-range-output', type=Path,
                        help='TileGen-owned post-cache range sink; no direct memory trace is retained')
    parser.add_argument('--phase-scope', choices=('complete', 'measured-only'), default='complete',
                        help='Execute the complete graph or only Measured/Prefill..Decode phases with a cold cache')
    parser.add_argument('--launch-resources', type=Path,
                        help='Graph-SHA-bound per-launch observed shared/L1 carveout; no inference from dynamic shared')
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--fast-gemv', action='store_true')
    parser.add_argument('--fast-gemm', action='store_true')
    parser.add_argument('--fast-prefill', action='store_true')
    parser.add_argument('--fast-prefill-sweep', action='store_true')
    parser.add_argument('--fast-down', action='store_true')
    parser.add_argument('--fast-p32-prefill', action='store_true')
    parser.add_argument('--drain-policy', choices=('none', 'measured-phase-end', 'run-end'), default='none',
                        help='Opt-in diagnostic intervention, not a native CUDA phase flush')
    parser.add_argument('--expected-dirty-age-accesses', type=parse_dirty_age_accesses,
                        help='Admit only this runner dirty-age setting (uint64; 0 disables age); default 64000000. Does not configure the runner.')
    parser.add_argument('--expected-dirty-age-clock', choices=('global', 'set'), default='global',
                        help='Expected clock unit: global forwarded lines or same-group forwarded lines. Does not configure runner.')
    args = parser.parse_args()
    age_expectation = dirty_age_expectation(args.expected_dirty_age_accesses, args.expected_dirty_age_clock)
    need(not args.output.exists(), 'fresh result directory required')
    if args.runner_range_output:
        need(not args.runner_range_output.exists() or args.runner_range_output.is_fifo(),
             'fresh runner range output required (or a FIFO)')
    args.output.mkdir(parents=True)
    start, cpu_start = time.monotonic(), time.process_time()
    source_pins = {name: pin(getattr(args, name)) for name in ('graph', 'registry', 'runtime')}
    source_pins['executor'] = pin(Path(__file__))
    launch_resources = None
    if args.launch_resources:
        source_pins['launch_resources'] = pin(args.launch_resources)
        resources = json.loads(args.launch_resources.read_text())
        need(resources['schema'] == 'OBSERVED_LAUNCH_CARVEOUT_V1', 'resource metadata schema')
        need(resources['graph_sha256'] == source_pins['graph']['sha256'], 'resource graph identity')
        need(resources['evidence_scope'] == 'per_kernel_launch', 'range-level shared metadata is insufficient')
        launch_resources = {}
        for row in resources['kernels']:
            ident, shared = row['native_launch_id'], row['observed_shared_bytes']
            need(type(ident) is int and ident >= 0 and ident not in launch_resources, 'unique launch resource identity')
            need(type(shared) is int and shared in (8192, 16384, 32768, 65536, 102400), 'supported observed shared profiles; 8/16 KiB explicitly extrapolated')
            launch_resources[ident] = shared
    extra_runtime_pins = []
    if args.fast_gemm:
        extra_runtime_pins = [pin(K / 'p32d2-smoke-r1' / relative) for relative in (
            'prefill-programs/provider.py', 'prefill-programs/contracts.json',
            'cache-runner-gemm-r1/source-contracts.json')]
    for enabled, directory in ((args.fast_prefill, 'fast-prefill-r2'), (args.fast_prefill_sweep, 'fast-prefill-sweep-r1'), (args.fast_down, 'fast-down-r1'), (args.fast_p32_prefill, 'fast-llama-p32-prefill-r1')):
        if enabled:
            extra_runtime_pins.extend(pin(p) for p in sorted((SUPPORT / directory).iterdir()) if p.is_file() and p.suffix in ('.py', '.json', '.h'))
    graph = Graph(args.graph)
    registry = load_module(args.runtime, support=True).Registry(args.registry)
    process = None
    state = dict(status='PREFLIGHT_RUNNING', source_pins=source_pins, fast_source_runtime_pins=extra_runtime_pins, cache_policy_changed=True,
                 experimental_cache_environment={k:v for k,v in os.environ.items() if k.startswith('TILEGEN_') and k != 'TILEGEN_NATIVE_TREE'},
                 source_hint_interpretation={'EF': 'fixed_h288', 'EL': 'explicitly_modeled_as_normal_priority'},
                 splitK_schedule='serial_parts_then_tiles_one_poll_per_warp_after_predecessor_publish',
                 compute_stall_cosimulation=False, hardware_warp_schedule_claimed=False,
                 execution_scope=args.phase_scope,
                 DMA_model='L2_COHERENT_FUNCTIONAL_128B_CHUNKS', terminal_dirty_flush=args.drain_policy == 'run-end',
                 drain_policy=args.drain_policy,
                 **age_expectation,
                 measured_cache_history_intervened=args.drain_policy == 'measured-phase-end',
                 NCU_accuracy_accepted=False)
    def write_state():
        (args.output / 'status.json').write_text(json.dumps(state, indent=2) + '\n')
    write_state()
    try:
        entries = all_entries(registry)
        if graph.value['input_contract']['prefill_length'] in (64, 256, 512):
            need(args.fast_prefill_sweep, 'new Prefill source programs require --fast-prefill-sweep')
        if args.fast_gemv:
            schemas = {entry['binding'].get('schema') for entry in entries.values()}
            for schema, directory in [('CURRENT_QWEN_GEMV_SOURCE_BINDING_V1', 'qwen-gemv-native-program-r1'),
                                      ('CURRENT_LLAMA_GEMV_SOURCE_BINDING_V1', 'llama-gemv-native-program-r1')]:
                if schema in schemas:
                    extra_runtime_pins.extend(pin(p) for p in sorted((K/directory).iterdir()) if p.is_file() and p.suffix in ('.py', '.json', '.h'))
        timeline, receipt = preflight(graph, entries)
        if args.phase_scope == 'measured-only':
            selected = [p for p in receipt['phases'] if p.startswith('Measured/')]
            timeline = [(ordinal, kind, payload) for ordinal, kind, payload in timeline
                        if (kind == 'node' and
                            (payload.get('kind') == 'allocation_API_observation' or payload.get('phase') in selected))
                        or (kind.startswith('phase_') and payload in selected)]
            need(any(kind == 'node' for _, kind, _ in timeline), 'measured-only scope selected no kernels')
            receipt = dict(receipt, phases=selected, execution_scope='MEASURED_ONLY_COLD_CACHE',
                           selected_kernel_count=sum(kind == 'node' and payload['kind'] == 'native_kernel'
                                                     for _, kind, payload in timeline),
                           complete_graph_validated=True)
        if launch_resources is not None:
            expected = {n['native_launch_id'] for _, kind, n in timeline if kind == 'node' and n['kind'] == 'native_kernel'}
            if args.phase_scope == 'complete':
                need(set(launch_resources) == expected, 'observed launch resources must cover complete initialization/warmup/measured history exactly')
            else:
                need(expected <= set(launch_resources), 'observed launch resources must cover selected measured kernels')
        state['input_preflight'] = receipt
        state['status'] = 'PASS_COMPLETE_NATIVE_INPUT_PREFLIGHT'
        write_state()
        if args.preflight_only:
            return
        need(args.runner and args.runner.is_file(), 'real cache binary is required')
        state['cache_runner'] = pin(args.runner)
        with (args.output / 'runner.stdout').open('wb') as stdout, (args.output / 'runner.stderr').open('wb') as stderr, (args.output / 'progress.jsonl').open('w') as journal:
            runner_command = [str(args.runner.resolve()), '--input', '-', '--summary',
                str((args.output / 'cache-summary.json').resolve()), '--snapshots',
                str((args.output / 'cache-snapshots.jsonl').resolve())]
            if args.runner_range_output:
                runner_command.extend(['--range-output', str(args.runner_range_output.resolve())])
            process = subprocess.Popen(runner_command, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr)
            state.update(status='RUNNING_COMPLETE_NATIVE_GRAPH_CACHE', runner_pid=process.pid)
            if args.runner_range_output:
                state['postcache_range_output'] = str(args.runner_range_output.resolve())
            write_state()
            send = Writer(process.stdin)
            def progress(row):
                row.update(wall_seconds=time.monotonic() - start, producer_CPU_seconds=time.process_time() - cpu_start)
                journal.write(json.dumps(row, separators=(',', ':')) + '\n')
                journal.flush()
                process.stdin.flush()
            state['executed_counts'] = emit(timeline, registry, entries, send, progress, graph, args.fast_gemv, args.fast_gemm, args.fast_prefill, args.fast_down, args.fast_p32_prefill, launch_resources, args.fast_prefill_sweep, args.drain_policy)
            process.stdin.close()
            need(process.wait() == 0, 'cache runner failed; inspect stderr')
            state['source_stream'] = dict(bytes=send.bytes, records=send.records, sha256=send.hash.hexdigest(), retained_full_trace=False)
            state['source_stream_without_drain_interventions'] = dict(bytes=send.source_bytes, records=send.source_records, sha256=send.source_hash.hexdigest())
        result = json.loads((args.output / 'cache-summary.json').read_text())
        state['actual_cache_configuration'] = result['configuration']
        validate_cache_configuration(result['configuration'], age_expectation['expected_dirty_age_accesses'], age_expectation['expected_dirty_age_clock'])
        need(result['status'] == 'PASS_STREAMED_SOURCE_FUNCTIONAL_CACHE_RUN', 'cache completion status')
        need(result['input_raw_SHA256'] == send.hash.hexdigest() and result['input_bytes'] == send.bytes
             and result['input_lines'] == send.records, 'producer/consumer stream identity differs')
        observations = {}
        for row in map(json.loads, (args.output / 'cache-snapshots.jsonl').open()):
            if row['type'] == 'snapshot':
                need(row['label'] not in observations, 'duplicate phase boundary snapshot')
                observations[row['label']] = row
        phases = []
        for label in receipt['phases'] + ['Measured/Full']:
            phases.append(phase_traffic(label, observations[label + '/begin'], observations[label + '/end']))
        drains = []
        for key in observations:
            if key.endswith('/diagnostic-drain/begin'):
                label = key[:-len('/begin')]
                drains.append(phase_traffic(label, observations[key], observations[label + '/end']))
        (args.output / 'phase-traffic.json').write_text(json.dumps(dict(
            status='PASS_COMPLETE_SMOKE_PHASE_TRAFFIC' if args.phase_scope == 'complete'
                   else 'PASS_MEASURED_SMOKE_PHASE_TRAFFIC', input_contract=graph.value['input_contract'],
            model_only=True, NCU_accuracy_claimed=False, phase_rows=phases, diagnostic_drain_rows=drains,
            drain_policy=args.drain_policy, measured_cache_history_intervened=args.drain_policy == 'measured-phase-end',
            diagnostic_drains_are_native_operations=False,
            source_stream_without_drain_interventions=state['source_stream_without_drain_interventions'],
            actual_cache_configuration=state['actual_cache_configuration'],
            **age_expectation,
            DMA_model=state['DMA_model'], source_stream=state['source_stream'],
            execution_scope=args.phase_scope), indent=2) + '\n')
        state.update(status='PASS_COMPLETE_NATIVE_GRAPH_CACHE_EXECUTION' if args.phase_scope == 'complete'
                     else 'PASS_MEASURED_NATIVE_GRAPH_CACHE_EXECUTION', cache_replay=True,
                     execution_scope=args.phase_scope)
    except BaseException as exc:
        state.update(status='FAIL_NATIVE_GRAPH_EXECUTION', error_type=type(exc).__name__, error=str(exc))
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=30)
        raise
    finally:
        registry.close()
        graph.close()
        for name in ('graph', 'registry', 'runtime'):
            need(pin(getattr(args, name)) == source_pins[name], 'frozen source changed during execution')
        if args.launch_resources:
            need(pin(args.launch_resources) == source_pins['launch_resources'], 'observed launch resource metadata changed')
        need(pin(Path(__file__)) == source_pins['executor'], 'executor changed during execution')
        for item in extra_runtime_pins:
            checked(item)
        if 'cache_runner' in state:
            need(pin(args.runner) == state['cache_runner'], 'cache binary changed during execution')
        state['producer_CPU_minutes'] = (time.process_time() - cpu_start) / 60
        state['child_CPU_minutes'] = (resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime) / 60
        state['wall_minutes'] = (time.monotonic() - start) / 60
        write_state()
        print(json.dumps({k: state[k] for k in ('status', 'producer_CPU_minutes', 'child_CPU_minutes', 'wall_minutes')}))


if __name__ == '__main__':
    main()
