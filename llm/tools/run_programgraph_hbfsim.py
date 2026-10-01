#!/usr/bin/env python3
"""Run the canonical ProgramGraph -> FunctionalCache -> HBFSim service path.

The range stream is connected through a FIFO owned by this TileGen adapter;
no post-cache trace file is created or retained.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading


def main() -> None:
    ap = argparse.ArgumentParser()
    for name in ('graph', 'registry', 'runtime', 'runner', 'service', 'config', 'output'):
        ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--service-map', '--service_map', dest='service_map', type=Path, required=True)
    ap.add_argument('--whole-stream', type=Path,
                    default=Path(__file__).resolve().parent.parent / 'executor-r1' / 'whole_stream.py')
    ap.add_argument('--launch-resources', type=Path)
    ap.add_argument('--phase-scope', choices=('complete', 'measured-only'), default='measured-only')
    ap.add_argument('--drain-policy', choices=('none', 'measured-phase-end', 'run-end'), default='none')
    ap.add_argument('--fast-gemv', action='store_true')
    ap.add_argument('--fast-gemm', action='store_true')
    ap.add_argument('--fast-prefill', action='store_true')
    ap.add_argument('--fast-prefill-sweep', action='store_true')
    ap.add_argument('--fast-down', action='store_true')
    ap.add_argument('--fast-p32-prefill', action='store_true')
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit('fresh output directory required')
    args.output.mkdir(parents=True)
    fifo = args.output / 'postcache.ranges.fifo'
    os.mkfifo(fifo)
    program_out = args.output / 'programgraph'
    service_stderr = (args.output / 'hbfsim-service.stderr').open('wb')
    service = subprocess.Popen([str(args.service), '--config', str(args.config),
                                '--service-map', str(args.service_map)],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=service_stderr)
    pump_error = []

    def pump() -> None:
        try:
            with fifo.open('rb') as source:
                shutil.copyfileobj(source, service.stdin)
            service.stdin.close()
        except BaseException as exc:  # report in result, never hide a broken pipe
            pump_error.append(repr(exc))
            try:
                service.stdin.close()
            except Exception:
                pass

    pump_thread = threading.Thread(target=pump, name='tilegen-range-pump')
    pump_thread.start()
    env = os.environ.copy()
    cmd = [sys.executable, '-B', str(args.whole_stream), '--graph', str(args.graph),
           '--registry', str(args.registry), '--runtime', str(args.runtime),
           '--runner', str(args.runner), '--output', str(program_out),
           '--phase-scope', args.phase_scope, '--drain-policy', args.drain_policy,
           '--runner-range-output', str(fifo)]
    if args.launch_resources:
        cmd += ['--launch-resources', str(args.launch_resources)]
    for enabled, flag in ((args.fast_gemv, '--fast-gemv'), (args.fast_gemm, '--fast-gemm'),
                          (args.fast_prefill, '--fast-prefill'),
                          (args.fast_prefill_sweep, '--fast-prefill-sweep'),
                          (args.fast_down, '--fast-down'),
                          (args.fast_p32_prefill, '--fast-p32-prefill')):
        if enabled:
            cmd.append(flag)
    with (args.output / 'programgraph.stdout').open('wb') as stdout, \
            (args.output / 'programgraph.stderr').open('wb') as stderr:
        producer = subprocess.run(cmd, stdout=stdout, stderr=stderr)
    if producer.returncode:
        # Unblock the FIFO reader when the producer rejected its inputs before
        # opening the sink; the service must still be reaped and reported.
        with fifo.open('wb'):
            pass
    pump_thread.join(timeout=30)
    if pump_thread.is_alive():
        service.kill()
        raise SystemExit('range pump did not terminate')
    # communicate() otherwise attempts to flush the already-closed pump pipe.
    service.stdin = None
    service_stdout, _ = service.communicate(timeout=60)
    service_stderr.close()
    lines = [line for line in service_stdout.decode().splitlines() if line.strip()]
    result = json.loads(lines[-1]) if lines else {'status': 'FAIL_HBFSIM_SERVICE_NO_RESULT'}
    result.update({'producer_returncode': producer.returncode,
                   'service_returncode': service.returncode,
                   'programgraph_output': str(program_out),
                   'fifo_streamed': True,
                   'postcache_trace_retained': False,
                   'pump_errors': pump_error})
    (args.output / 'hbfsim-result.json').write_text(json.dumps(result, indent=2) + '\n')
    try:
        fifo.unlink()
    except FileNotFoundError:
        pass
    print(json.dumps(result))
    if producer.returncode or service.returncode or result.get('status') != 'PASS_STREAMING_POSTCACHE_HBFSIM':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
