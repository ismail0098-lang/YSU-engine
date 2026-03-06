#!/usr/bin/env python3
"""
compare_architectures.py — Cross-architecture SASS comparison tool.

Compares two result directories from disassemble_all.ps1 runs on different GPUs.
Produces a comprehensive Markdown report detailing:
  1. ISA differences (instructions present on one GPU but not the other)
  2. Instruction frequency shifts
  3. Encoding format differences (word size, opcode field layout)
  4. Register pressure differences per kernel
  5. Latency/throughput comparison tables (if benchmark CSVs are present)

Usage:
    python compare_architectures.py <dir_gpu_A> <dir_gpu_B> [--output COMPARE.md]

Example:
    python compare_architectures.py results/Ada_RTX4070TiS_20260306_190541 \
                                     results/Pascal_GTX1050Ti_20260310_143022
"""

import argparse
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


def parse_sass_file(path):
    """Parse a .sass file from cuobjdump. Returns list of (addr, mnemonic, full_line)."""
    instructions = []
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            m = re.match(r'\s*/\*([0-9a-fA-F]+)\*/\s+(\S+)', line)
            if m:
                addr = m.group(1)
                mnemonic = m.group(2)
                instructions.append((addr, mnemonic, line.rstrip()))
    return instructions


def get_instruction_census(instructions):
    """Count occurrences of each mnemonic (base opcode, e.g. FADD not FADD.FTZ)."""
    base_counter = Counter()
    full_counter = Counter()
    for _, mnem, _ in instructions:
        full_counter[mnem] += 1
        base = mnem.split('.')[0]
        base_counter[base] += 1
    return base_counter, full_counter


def parse_encoding_file(path):
    """Parse ENCODING_ANALYSIS.md or .raw files for encoding data."""
    encodings = {}
    if not os.path.exists(path):
        return encodings
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    # Try to parse from raw disassembly
    for line in content.split('\n'):
        m = re.match(r'\s*/\*([0-9a-fA-F]+)\*/\s+(\S+).*?;\s*/\*\s*([0-9a-fA-Fx]+)\s*\*/', line)
        if m:
            mnem = m.group(2).split('.')[0]
            enc = m.group(3)
            if mnem not in encodings:
                encodings[mnem] = []
            encodings[mnem].append(enc)
    return encodings


def parse_latency_output(path):
    """Parse latency benchmark stdout text. Returns dict of instruction -> cycles."""
    results = {}
    if not os.path.exists(path):
        return results
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            m = re.match(r'(\S+)\s+(\d+\.\d+)\s*$', line)
            if m:
                results[m.group(1)] = float(m.group(2))
    return results


def parse_throughput_output(path):
    """Parse throughput benchmark stdout text."""
    results = {}
    if not os.path.exists(path):
        return results
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            m = re.match(r'(\S+)\s+.*?(\d+\.\d+)\s+ops/clk/SM', line)
            if m:
                results[m.group(1)] = float(m.group(2))
    return results


def load_gpu_data(result_dir):
    """Load all data from a result directory."""
    data = {
        'dir': result_dir,
        'name': os.path.basename(result_dir),
        'probes': {},
        'all_instructions': [],
        'base_census': Counter(),
        'full_census': Counter(),
    }

    sass_files = sorted(Path(result_dir).glob('*.sass'))
    for sf in sass_files:
        instrs = parse_sass_file(sf)
        probe_name = sf.stem
        base_c, full_c = get_instruction_census(instrs)
        data['probes'][probe_name] = {
            'instructions': instrs,
            'base_census': base_c,
            'full_census': full_c,
            'count': len(instrs),
        }
        data['all_instructions'].extend(instrs)
        data['base_census'] += base_c
        data['full_census'] += full_c

    # Try to load benchmark results
    for name in ['latency_results.txt', 'latency_output.txt', 'latency.txt']:
        p = os.path.join(result_dir, name)
        lat = parse_latency_output(p)
        if lat:
            data['latency'] = lat
            break

    for name in ['throughput_results.txt', 'throughput_output.txt', 'throughput.txt']:
        p = os.path.join(result_dir, name)
        tp = parse_throughput_output(p)
        if tp:
            data['throughput'] = tp
            break

    return data


def generate_comparison(gpu_a, gpu_b, output_path):
    """Generate the full comparison report."""
    lines = []
    L = lines.append

    name_a = gpu_a['name']
    name_b = gpu_b['name']

    L(f"# SASS Architecture Comparison")
    L(f"")
    L(f"**GPU A:** {name_a}")
    L(f"**GPU B:** {name_b}")
    L(f"")
    L(f"---")
    L(f"")

    # ── Section 1: Instruction Set Differences ──
    L(f"## 1. Instruction Set Differences")
    L(f"")

    all_base_a = set(gpu_a['base_census'].keys())
    all_base_b = set(gpu_b['base_census'].keys())
    only_a = sorted(all_base_a - all_base_b)
    only_b = sorted(all_base_b - all_base_a)
    common = sorted(all_base_a & all_base_b)

    L(f"| Category | Count |")
    L(f"|---|---|")
    L(f"| Instructions on **both** GPUs | {len(common)} |")
    L(f"| Only on **{name_a}** | {len(only_a)} |")
    L(f"| Only on **{name_b}** | {len(only_b)} |")
    L(f"")

    if only_a:
        L(f"### Instructions ONLY on {name_a}")
        L(f"")
        L(f"These instructions were **added** between the two architectures (or were compiled differently):")
        L(f"")
        L(f"| Instruction | Count |")
        L(f"|---|---|")
        for op in only_a:
            L(f"| `{op}` | {gpu_a['base_census'][op]} |")
        L(f"")

    if only_b:
        L(f"### Instructions ONLY on {name_b}")
        L(f"")
        L(f"| Instruction | Count |")
        L(f"|---|---|")
        for op in only_b:
            L(f"| `{op}` | {gpu_b['base_census'][op]} |")
        L(f"")

    # ── Section 2: Instruction Frequency Comparison ──
    L(f"## 2. Instruction Frequency Comparison (Common Instructions)")
    L(f"")
    L(f"| Instruction | {name_a} | {name_b} | Delta | Notes |")
    L(f"|---|---|---|---|---|")

    for op in sorted(common, key=lambda x: gpu_a['base_census'].get(x,0)+gpu_b['base_census'].get(x,0), reverse=True):
        ca = gpu_a['base_census'][op]
        cb = gpu_b['base_census'][op]
        delta = cb - ca
        sign = "+" if delta > 0 else ""
        note = ""
        if ca > 0 and cb > 0:
            ratio = cb / ca
            if ratio > 2.0:
                note = f"{ratio:.1f}x more on B"
            elif ratio < 0.5:
                note = f"{1/ratio:.1f}x more on A"
        L(f"| `{op}` | {ca} | {cb} | {sign}{delta} | {note} |")
    L(f"")

    # ── Section 3: Per-Probe Comparison ──
    L(f"## 3. Per-Probe Instruction Count")
    L(f"")
    all_probes = sorted(set(list(gpu_a['probes'].keys()) + list(gpu_b['probes'].keys())))
    L(f"| Probe | {name_a} | {name_b} | Delta |")
    L(f"|---|---|---|---|")
    for p in all_probes:
        ca = gpu_a['probes'].get(p, {}).get('count', 0)
        cb = gpu_b['probes'].get(p, {}).get('count', 0)
        if ca == 0:
            L(f"| {p} | — | {cb} | N/A (only B) |")
        elif cb == 0:
            L(f"| {p} | {ca} | — | N/A (only A) |")
        else:
            delta = cb - ca
            sign = "+" if delta > 0 else ""
            L(f"| {p} | {ca} | {cb} | {sign}{delta} |")
    L(f"")

    total_a = len(gpu_a['all_instructions'])
    total_b = len(gpu_b['all_instructions'])
    L(f"**Totals:** {name_a} = {total_a} instructions, {name_b} = {total_b} instructions")
    L(f"")

    # ── Section 4: Instruction Modifier Differences ──
    L(f"## 4. Instruction Modifier / Variant Differences")
    L(f"")
    L(f"Full mnemonic variants (including modifiers like .FTZ, .STRONG, etc.) that differ:")
    L(f"")

    all_full_a = set(gpu_a['full_census'].keys())
    all_full_b = set(gpu_b['full_census'].keys())
    only_full_a = sorted(all_full_a - all_full_b)
    only_full_b = sorted(all_full_b - all_full_a)

    if only_full_a:
        L(f"### Variants only on {name_a}")
        L(f"")
        for v in only_full_a[:50]:  # cap at 50
            L(f"- `{v}` ({gpu_a['full_census'][v]}x)")
        L(f"")

    if only_full_b:
        L(f"### Variants only on {name_b}")
        L(f"")
        for v in only_full_b[:50]:
            L(f"- `{v}` ({gpu_b['full_census'][v]}x)")
        L(f"")

    # ── Section 5: Latency Comparison ──
    lat_a = gpu_a.get('latency', {})
    lat_b = gpu_b.get('latency', {})

    if lat_a or lat_b:
        L(f"## 5. Instruction Latency Comparison (cycles)")
        L(f"")
        L(f"| Instruction | {name_a} | {name_b} | Ratio (B/A) | Winner |")
        L(f"|---|---|---|---|---|")

        all_lat = sorted(set(list(lat_a.keys()) + list(lat_b.keys())))
        for instr in all_lat:
            va = lat_a.get(instr)
            vb = lat_b.get(instr)
            sa = f"{va:.2f}" if va else "—"
            sb = f"{vb:.2f}" if vb else "—"
            if va and vb:
                ratio = vb / va
                winner = name_a if va < vb else name_b if vb < va else "Tie"
                L(f"| {instr} | {sa} | {sb} | {ratio:.2f}x | {winner} |")
            else:
                L(f"| {instr} | {sa} | {sb} | — | — |")
        L(f"")

    # ── Section 6: Throughput Comparison ──
    tp_a = gpu_a.get('throughput', {})
    tp_b = gpu_b.get('throughput', {})

    if tp_a or tp_b:
        L(f"## 6. Instruction Throughput Comparison (ops/clk/SM)")
        L(f"")
        L(f"| Instruction | {name_a} | {name_b} | Ratio (A/B) | Winner |")
        L(f"|---|---|---|---|---|")

        all_tp = sorted(set(list(tp_a.keys()) + list(tp_b.keys())))
        for instr in all_tp:
            va = tp_a.get(instr)
            vb = tp_b.get(instr)
            sa = f"{va:.1f}" if va else "—"
            sb = f"{vb:.1f}" if vb else "—"
            if va and vb:
                ratio = va / vb if vb > 0 else float('inf')
                winner = name_a if va > vb else name_b if vb > va else "Tie"
                L(f"| {instr} | {sa} | {sb} | {ratio:.2f}x | {winner} |")
            else:
                L(f"| {instr} | {sa} | {sb} | — | — |")
        L(f"")

    # ── Section 7: Key Architectural Observations ──
    L(f"## 7. Key Architectural Observations")
    L(f"")
    L(f"### Encoding format")
    L(f"")

    # Check encoding word sizes from raw files
    raw_a = sorted(Path(gpu_a['dir']).glob('*.raw'))
    raw_b = sorted(Path(gpu_b['dir']).glob('*.raw'))

    L(f"- **{name_a}**: {len(raw_a)} raw dump files available")
    L(f"- **{name_b}**: {len(raw_b)} raw dump files available")
    L(f"")

    # Instruction word size detection
    for label, raws in [(name_a, raw_a), (name_b, raw_b)]:
        if raws:
            sample = raws[0]
            with open(sample, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            hex_words = re.findall(r'0x[0-9a-fA-F]+', content)
            if hex_words:
                max_len = max(len(h) - 2 for h in hex_words[:20])  # minus "0x"
                bits = max_len * 4
                L(f"- **{label}** encoding word: ~{bits}-bit (max {max_len} hex digits observed)")
    L(f"")

    # ISA generation differences
    L(f"### ISA generation differences")
    L(f"")

    # Detect known arch-specific instructions
    pascal_only = {'XMAD', 'DBAR', 'DEPBAR', 'TEXDEPBAR', 'NOP', 'XMAD', 'VMNMX', 'VMAD'}
    ada_only = {'IADD3', 'LOP3', 'HMMA', 'IMMA', 'LDGSTS', 'WARPSYNC', 'NANOSLEEP'}

    found_pascal = sorted(all_base_a & pascal_only) if all_base_a else []
    found_ada_a = sorted(all_base_a & ada_only) if all_base_a else []
    found_ada_b = sorted(all_base_b & ada_only) if all_base_b else []

    L(f"Notable instruction class observations:")
    L(f"")

    if only_a:
        L(f"- **{name_a}** has {len(only_a)} unique instructions: {', '.join(f'`{x}`' for x in only_a[:15])}")
    if only_b:
        L(f"- **{name_b}** has {len(only_b)} unique instructions: {', '.join(f'`{x}`' for x in only_b[:15])}")

    # Detect IADD3 vs IADD / XMAD changes
    if 'IADD3' in all_base_a and 'IADD3' not in all_base_b:
        L(f"- **IADD3** (3-input add) present on {name_a} but NOT {name_b} — the older arch may use **IADD** or **XMAD** instead")
    if 'XMAD' in all_base_b and 'XMAD' not in all_base_a:
        L(f"- **XMAD** (extended multiply-add) present on {name_b} but NOT {name_a} — replaced by **IMAD** + **IADD3** on newer arch")
    if 'LOP3' in all_base_a and 'LOP3' not in all_base_b:
        L(f"- **LOP3** (3-input logic) present on {name_a} but NOT {name_b} — older arch uses **LOP**/**LOP32I** instead")
    if 'HMMA' in all_base_a and 'HMMA' not in all_base_b:
        L(f"- **HMMA** (tensor core) present on {name_a} but NOT {name_b} — tensor cores were introduced in Volta (SM 7.0)")
    L(f"")

    # ── Section 8: What This Means ──
    L(f"## 8. Summary for Paper")
    L(f"")
    L(f"- Total unique base opcodes: **{name_a}** = {len(all_base_a)}, **{name_b}** = {len(all_base_b)}")
    L(f"- Common opcodes: {len(common)}")
    L(f"- ISA evolution: {len(only_a)} instructions added to A, {len(only_b)} instructions unique to B")
    L(f"- Total instructions compiled: **{name_a}** = {total_a}, **{name_b}** = {total_b}")

    if lat_a and lat_b:
        common_lat = set(lat_a.keys()) & set(lat_b.keys())
        if common_lat:
            avg_ratio = sum(lat_b[k]/lat_a[k] for k in common_lat) / len(common_lat)
            L(f"- Average latency ratio ({name_b}/{name_a}): {avg_ratio:.2f}x")

    L(f"")

    # Write output
    report = '\n'.join(lines) + '\n'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Comparison report written to: {output_path}")
    print(f"  {len(lines)} lines")


def main():
    parser = argparse.ArgumentParser(description='Compare SASS disassembly from two GPU architectures')
    parser.add_argument('dir_a', help='Result directory for GPU A')
    parser.add_argument('dir_b', help='Result directory for GPU B')
    parser.add_argument('--output', '-o', default='COMPARISON.md', help='Output file path')
    args = parser.parse_args()

    if not os.path.isdir(args.dir_a):
        print(f"Error: {args.dir_a} is not a directory", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(args.dir_b):
        print(f"Error: {args.dir_b} is not a directory", file=sys.stderr)
        sys.exit(1)

    print(f"Loading GPU A: {args.dir_a}")
    gpu_a = load_gpu_data(args.dir_a)
    print(f"  {len(gpu_a['all_instructions'])} instructions from {len(gpu_a['probes'])} probes")

    print(f"Loading GPU B: {args.dir_b}")
    gpu_b = load_gpu_data(args.dir_b)
    print(f"  {len(gpu_b['all_instructions'])} instructions from {len(gpu_b['probes'])} probes")

    generate_comparison(gpu_a, gpu_b, args.output)


if __name__ == '__main__':
    main()
