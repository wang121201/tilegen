#!/usr/bin/env python3
"""Build a TileGen-owned logical-address map for the HBFSim service.

The ProgramGraph contains allocation observations, not a complete memory trace.
This map packs those observed allocation spans into the service address space;
it is not a claim about physical GPU VA translation.
"""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--graph', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    graph = json.loads(args.graph.read_text())
    nodes_path = Path(graph['artifacts']['nodes']['path'])
    spans = []
    for line in nodes_path.open():
        node = json.loads(line)
        if node.get('kind') != 'allocation_API_observation':
            continue
        obs = node.get('observation', {})
        if obs.get('action') != 'allocate':
            continue
        base, size = int(obs['base_u64']), int(obs['bytes'])
        if size:
            spans.append((base, base + size))
    if not spans:
        raise SystemExit('no allocation observations')
    merged = []
    for base, end in sorted(spans):
        base = (base // 128) * 128
        end = ((end + 127) // 128) * 128
        if merged and base <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((base, end))
    packed = []
    service = 0
    for base, end in merged:
        size = end - base
        packed.append({'source_base': base, 'bytes': size, 'service_base': service})
        service += size
    result = {
        'schema': 'PROGRAMGRAPH_SERVICE_MAP_V1',
        'mapping': 'ALLOCATED_SOURCE_SPANS_PACKED_FOR_TILEGEN_HBFSIM_SERVICE',
        'source_graph': {'path': str(args.graph.resolve()), 'sha256': digest(args.graph)},
        'source_nodes': {'path': str(nodes_path), 'sha256': digest(nodes_path)},
        'spans': packed,
        'capacity_bytes': service,
        'physical_address_claimed': False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit('fresh output required')
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': 'PASS_PROGRAMGRAPH_SERVICE_MAP', 'spans': len(packed),
                      'capacity_bytes': service, 'output': str(args.output)}))


if __name__ == '__main__':
    main()
