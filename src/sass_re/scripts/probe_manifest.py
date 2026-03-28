#!/usr/bin/env python3
"""
Probe manifest and runner generator for the SASS reverse-engineering toolkit.

This keeps the recursive probe inventory in one place and provides enough
metadata for compile/disassembly sweeps and Nsight Compute runners.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys
from dataclasses import dataclass
from typing import Iterable


ROOT = pathlib.Path(__file__).resolve().parent.parent
PROBE_ROOT = ROOT / "probes"
RUNNER_ROOT = ROOT / "runners"

SKIP_PROBES = {
    "probe_optix_host_pipeline.cu": "host-side OptiX pipeline setup, not a device probe",
    "probe_optix_real_pipeline.cu": "host-side OptiX real-pipeline probe, profiled via custom runner",
    "probe_optix_callable_pipeline.cu": "host-side OptiX callable-pipeline probe, profiled via custom runner",
    "probe_ofa_pipeline.cu": "host-side Optical Flow Accelerator probe, profiled via custom runner",
    "probe_nvenc_nvdec_pipeline.cu": "host-side NVENC/NVDEC probe, profiled via custom runner",
    "probe_cudnn_conv_mining.cu": "host-side cuDNN convolution + library-mining probe, profiled via custom runner",
}

TEXTURE_PROBES = {
    "probe_texture_surface.cu",
    "texture_surface/probe_tmu_behavior.cu",
}

CUSTOM_RUNNER_KINDS = {
    "mbarrier/probe_mbarrier_core.cu": "mbarrier",
    "data_movement/probe_cp_async_zfill.cu": "cp_async_zfill",
    "probe_barrier_arrive_wait.cu": "barrier_arrive_wait",
    "probe_barrier_coop_groups_sync.cu": "barrier_coop_groups",
    "probe_cooperative_launch.cu": "cooperative_launch",
    "probe_tiling_hierarchical.cu": "tiling_hierarchical",
    "barrier_sync2/probe_depbar_explicit.cu": "depbar_explicit",
    "optix/probe_optix_real_pipeline.cu": "optix_pipeline",
    "optix/probe_optix_callable_pipeline.cu": "optix_callable_pipeline",
    "optical_flow/probe_ofa_pipeline.cu": "optical_flow",
    "video_codec/probe_nvenc_nvdec_pipeline.cu": "video_codec",
    "cudnn/probe_cudnn_conv_mining.cu": "cudnn",
}

KERNEL_RE = re.compile(
    r'(?:extern\s+"C"\s+)?__global__\s+void'
    r'(?:\s+__launch_bounds__\((?P<launch_bounds>[^)]*)\))?'
    r"\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*"
    r"\((?P<args>[^)]*)\)",
    re.S,
)

BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.S)
LINE_COMMENT_RE = re.compile(r"//[^\n]*")


@dataclass
class KernelArg:
    name: str
    decl: str
    type_name: str
    kind: str


@dataclass
class Kernel:
    name: str
    launch_bounds: str
    args: list[KernelArg]


@dataclass
class Probe:
    probe_id: str
    relative_path: str
    basename: str
    compile_enabled: bool
    runner_kind: str
    supports_generic_runner: bool
    skip_reason: str
    kernels: list[Kernel]


def split_args(arg_blob: str) -> list[str]:
    arg_blob = " ".join(arg_blob.split())
    if not arg_blob or arg_blob == "void":
        return []
    parts: list[str] = []
    cur: list[str] = []
    depth = 0
    for ch in arg_blob:
        if ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
            continue
        if ch in "(<[":
            depth += 1
        elif ch in ")>]":
            depth = max(depth - 1, 0)
        cur.append(ch)
    if cur:
        parts.append("".join(cur).strip())
    return [part for part in parts if part]


def strip_comments(text: str) -> str:
    text = BLOCK_COMMENT_RE.sub(" ", text)
    text = LINE_COMMENT_RE.sub("", text)
    return text


def parse_arg(arg_decl: str) -> KernelArg:
    decl = " ".join(arg_decl.split())
    match = re.search(r"(?P<type>.+?)\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)$", decl)
    if not match:
        return KernelArg(name="arg", decl=decl, type_name=decl, kind="unsupported")
    type_name = match.group("type").strip()
    name = match.group("name")
    kind = classify_arg(type_name)
    return KernelArg(name=name, decl=decl, type_name=type_name, kind=kind)


def normalize_scalar(type_name: str) -> str:
    base = re.sub(r"\b(const|volatile|restrict|__restrict__)\b", "", type_name)
    base = base.replace("*", " ")
    base = " ".join(base.split())
    return base


def classify_arg(type_name: str) -> str:
    stripped = " ".join(type_name.split())
    base = normalize_scalar(stripped)
    if "cudaTextureObject_t" in stripped:
        return "texture_object"
    if "cudaSurfaceObject_t" in stripped:
        return "surface_object"
    if "*" in stripped:
        return "pointer"
    scalar_map = {
        "bool": "scalar_bool",
        "char": "scalar_int",
        "signed char": "scalar_int",
        "unsigned char": "scalar_uint",
        "int8_t": "scalar_int",
        "uint8_t": "scalar_uint",
        "short": "scalar_int",
        "unsigned short": "scalar_uint",
        "int16_t": "scalar_int",
        "uint16_t": "scalar_uint",
        "int": "scalar_int",
        "unsigned": "scalar_uint",
        "unsigned int": "scalar_uint",
        "int32_t": "scalar_int",
        "uint32_t": "scalar_uint",
        "long": "scalar_int",
        "unsigned long": "scalar_uint",
        "long long": "scalar_long_long",
        "unsigned long long": "scalar_unsigned_long_long",
        "int64_t": "scalar_long_long",
        "uint64_t": "scalar_unsigned_long_long",
        "size_t": "scalar_unsigned_long_long",
        "float": "scalar_float",
        "double": "scalar_double",
        "cudaTextureObject_t": "texture_object",
        "cudaSurfaceObject_t": "surface_object",
    }
    return scalar_map.get(base, "unsupported")


def parse_launch_bounds(bounds: str) -> int | None:
    if not bounds:
        return None
    head = bounds.split(",", 1)[0].strip()
    if head.isdigit():
        return int(head)
    return None


def parse_kernels(text: str) -> list[Kernel]:
    kernels: list[Kernel] = []
    scrubbed = strip_comments(text)
    for match in KERNEL_RE.finditer(scrubbed):
        args = [parse_arg(arg) for arg in split_args(match.group("args"))]
        kernels.append(
            Kernel(
                name=match.group("name"),
                launch_bounds=(match.group("launch_bounds") or "").strip(),
                args=args,
            )
        )
    return kernels


def build_probe(path: pathlib.Path) -> Probe:
    relative_path = path.relative_to(PROBE_ROOT).as_posix()
    basename = path.name
    text = path.read_text(encoding="utf-8", errors="ignore")
    kernels = parse_kernels(text)
    short_hash = hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:8]
    probe_id = f"{path.with_suffix('').relative_to(PROBE_ROOT).as_posix().replace('/', '__')}__{short_hash}"
    compile_enabled = basename not in SKIP_PROBES
    skip_reason = SKIP_PROBES.get(basename, "")
    runner_kind = "plain"
    if not kernels and re.search(r"\bint\s+main\s*\(", text):
        runner_kind = "skip"
        skip_reason = "standalone host-style probe with no __global__ kernels"
    elif any(
        kernel.name.startswith(("__raygen__", "__closesthit__", "__miss__", "__intersection__", "__anyhit__"))
        for kernel in kernels
    ):
        runner_kind = "skip"
        skip_reason = "OptiX program entrypoints require a dedicated OptiX runner"
    if relative_path in CUSTOM_RUNNER_KINDS:
        runner_kind = CUSTOM_RUNNER_KINDS[relative_path]
    elif relative_path in TEXTURE_PROBES or basename in TEXTURE_PROBES:
        runner_kind = "texture_surface"
    elif not compile_enabled:
        runner_kind = "skip"
    elif any(arg.kind in {"texture_object", "surface_object"} for kernel in kernels for arg in kernel.args):
        runner_kind = "texture_surface"
    supported_kinds = {
        "pointer",
        "scalar_bool",
        "scalar_int",
        "scalar_uint",
        "scalar_long_long",
        "scalar_unsigned_long_long",
        "scalar_float",
        "scalar_double",
    }
    supports_generic_runner = runner_kind == "plain" and all(
        arg.kind in supported_kinds for kernel in kernels for arg in kernel.args
    )
    return Probe(
        probe_id=probe_id,
        relative_path=relative_path,
        basename=basename,
        compile_enabled=compile_enabled,
        runner_kind=runner_kind,
        supports_generic_runner=supports_generic_runner,
        skip_reason=skip_reason,
        kernels=kernels,
    )


def iter_probes() -> Iterable[Probe]:
    for path in sorted(PROBE_ROOT.rglob("probe_*.cu")):
        yield build_probe(path)


def probe_to_dict(probe: Probe) -> dict:
    return {
        "probe_id": probe.probe_id,
        "relative_path": probe.relative_path,
        "basename": probe.basename,
        "compile_enabled": probe.compile_enabled,
        "runner_kind": probe.runner_kind,
        "supports_generic_runner": probe.supports_generic_runner,
        "skip_reason": probe.skip_reason,
        "kernels": [
            {
                "name": kernel.name,
                "launch_bounds": kernel.launch_bounds,
                "args": [
                    {
                        "name": arg.name,
                        "decl": arg.decl,
                        "type_name": arg.type_name,
                        "kind": arg.kind,
                    }
                    for arg in kernel.args
                ],
            }
            for kernel in probe.kernels
        ],
    }


def print_tsv(probes: Iterable[Probe], header: bool) -> None:
    if header:
        print(
            "\t".join(
                [
                    "probe_id",
                    "relative_path",
                    "basename",
                    "compile_enabled",
                    "runner_kind",
                    "supports_generic_runner",
                    "kernel_names",
                    "skip_reason",
                ]
            )
        )
    for probe in probes:
        print(
            "\t".join(
                [
                    probe.probe_id,
                    probe.relative_path,
                    probe.basename,
                    "1" if probe.compile_enabled else "0",
                    probe.runner_kind,
                    "1" if probe.supports_generic_runner else "0",
                    ",".join(kernel.name for kernel in probe.kernels),
                    probe.skip_reason,
                ]
            )
        )


def cpp_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def default_scalar(arg: KernelArg) -> str:
    lname = arg.name.lower()
    if arg.kind == "scalar_bool":
        return "true"
    if arg.kind == "scalar_float":
        if lname == "lo":
            return "0.25f"
        if lname == "hi":
            return "0.75f"
        if lname in {"u", "v", "w"} or lname.startswith(("u", "v", "w")):
            return "0.5f"
        if "thresh" in lname:
            return "0.25f"
        return "1.0f"
    if arg.kind == "scalar_double":
        return "1.0"
    if lname in {"nx", "ny", "nz"}:
        return "16"
    if lname in {"w", "h"}:
        return "32"
    if lname in {"m", "n", "k"}:
        return "128"
    if lname in {"mod", "n_bins"}:
        return "64"
    if "iterations" in lname or lname == "stages":
        return "8"
    if lname == "ksize":
        return "7"
    if lname == "stride":
        return "16"
    if lname == "log_n":
        return "8"
    if "tile" in lname:
        return "2"
    if "count" in lname:
        return "4"
    if "threshold" in lname:
        return "0"
    return "64"


def block_dim(kernel: Kernel) -> int:
    value = parse_launch_bounds(kernel.launch_bounds)
    if value is None:
        return 32
    return min(value, 256)


def make_runner_source(probe: Probe) -> str:
    include_path = cpp_escape((PROBE_ROOT / probe.relative_path).as_posix())
    lines: list[str] = []
    lines.extend(
        [
            "#include <cuda_runtime.h>",
            "#include <stdint.h>",
            "#include <stdio.h>",
            "#include <stdlib.h>",
            "",
            "#define SASS_RE_EMBEDDED_RUNNER 1",
            f'#include "{include_path}"',
            "",
            "static int check_cuda(cudaError_t err, const char *file, int line) {",
            "    if (err != cudaSuccess) {",
            '        fprintf(stderr, "CUDA error: %s (%d) at %s:%d\\n",',
            "                cudaGetErrorString(err), (int)err, file, line);",
            "        return 1;",
            "    }",
            "    return 0;",
            "}",
            "#define CHECK_CUDA(expr) do { if (check_cuda((expr), __FILE__, __LINE__)) return 1; } while (0)",
            "",
            "static void fill_float(float *data, size_t count) {",
            "    for (size_t i = 0; i < count; ++i) data[i] = ((int)(i % 257u) - 128) * 0.125f;",
            "}",
            "static void fill_double(double *data, size_t count) {",
            "    for (size_t i = 0; i < count; ++i) data[i] = ((int)(i % 257u) - 128) * 0.125;",
            "}",
            "static void fill_int(int *data, size_t count, int mod, int positive_only) {",
            "    for (size_t i = 0; i < count; ++i) {",
            "        int v = (int)(i % (size_t)(mod > 0 ? mod : 1));",
            "        data[i] = positive_only ? v : (v - mod / 2);",
            "    }",
            "}",
            "static void fill_uint(unsigned int *data, size_t count, unsigned int mod) {",
            "    for (size_t i = 0; i < count; ++i) data[i] = (unsigned int)(i % (size_t)(mod ? mod : 1u));",
            "}",
            "static void fill_short(short *data, size_t count, int mod) {",
            "    for (size_t i = 0; i < count; ++i) data[i] = (short)((int)(i % (size_t)(mod > 0 ? mod : 1)) - mod / 2);",
            "}",
            "static void fill_ushort(unsigned short *data, size_t count, unsigned int mod) {",
            "    for (size_t i = 0; i < count; ++i) data[i] = (unsigned short)(i % (size_t)(mod ? mod : 1u));",
            "}",
            "static void fill_ll(long long *data, size_t count, long long mod) {",
            "    for (size_t i = 0; i < count; ++i) data[i] = (long long)(i % (size_t)(mod > 0 ? mod : 1)) - mod / 2;",
            "}",
            "static void fill_ull(unsigned long long *data, size_t count, unsigned long long mod) {",
            "    for (size_t i = 0; i < count; ++i) data[i] = (unsigned long long)(i % (size_t)(mod ? mod : 1ull));",
            "}",
            "static void fill_s8(signed char *data, size_t count) {",
            "    for (size_t i = 0; i < count; ++i) data[i] = (signed char)((int)(i % 63u) - 31);",
            "}",
            "static void fill_u8(unsigned char *data, size_t count, unsigned int mod) {",
            "    for (size_t i = 0; i < count; ++i) data[i] = (unsigned char)(i % (size_t)(mod ? mod : 1u));",
            "}",
            "",
            "int main(void) {",
            "    const size_t kElemCount = 4096;",
            "",
        ]
    )
    ptr_index = 0
    cleanup_lines: list[str] = []
    for kernel in probe.kernels:
        lines.append(f"    /* kernel: {kernel.name} */")
        arg_names: list[str] = []
        for arg_index, arg in enumerate(kernel.args):
            var_name = f"{kernel.name}_arg_{arg_index}"
            arg_names.append(var_name)
            if arg.kind == "pointer":
                lname = arg.name.lower()
                base = normalize_scalar(arg.type_name)
                is_output = any(tok in lname for tok in ("out", "dst", "result", "sink", "flag", "lock", "count", "hist"))
                is_index = any(tok in lname for tok in ("idx", "index", "indices", "gather", "scatter", "permute", "rev"))
                lines.append(f"    void *{var_name} = NULL;")
                if base in {"float"}:
                    bytes_expr = "sizeof(float) * kElemCount"
                    lines.append(f"    CHECK_CUDA(cudaMalloc(&{var_name}, {bytes_expr}));")
                    if is_output:
                        lines.append(f"    CHECK_CUDA(cudaMemset({var_name}, 0, {bytes_expr}));")
                    else:
                        host_name = f"{var_name}_host"
                        lines.append(f"    float *{host_name} = (float*)malloc({bytes_expr});")
                        lines.append(f"    if (!{host_name}) {{ fprintf(stderr, \"malloc failed\\n\"); return 1; }}")
                        lines.append(f"    fill_float({host_name}, kElemCount);")
                        lines.append(f"    CHECK_CUDA(cudaMemcpy({var_name}, {host_name}, {bytes_expr}, cudaMemcpyHostToDevice));")
                        cleanup_lines.append(f"    free({host_name});")
                elif base in {"double"}:
                    bytes_expr = "sizeof(double) * kElemCount"
                    lines.append(f"    CHECK_CUDA(cudaMalloc(&{var_name}, {bytes_expr}));")
                    if is_output:
                        lines.append(f"    CHECK_CUDA(cudaMemset({var_name}, 0, {bytes_expr}));")
                    else:
                        host_name = f"{var_name}_host"
                        lines.append(f"    double *{host_name} = (double*)malloc({bytes_expr});")
                        lines.append(f"    if (!{host_name}) {{ fprintf(stderr, \"malloc failed\\n\"); return 1; }}")
                        lines.append(f"    fill_double({host_name}, kElemCount);")
                        lines.append(f"    CHECK_CUDA(cudaMemcpy({var_name}, {host_name}, {bytes_expr}, cudaMemcpyHostToDevice));")
                        cleanup_lines.append(f"    free({host_name});")
                elif base in {"int", "char", "signed char", "short", "long", "int8_t", "int16_t", "int32_t"}:
                    ctype = "int"
                    bytes_expr = "sizeof(int) * kElemCount"
                    fill = "fill_int"
                    if base in {"char", "signed char", "int8_t"}:
                        ctype = "signed char"
                        bytes_expr = "sizeof(signed char) * kElemCount"
                        fill = "fill_s8"
                    elif base in {"short", "int16_t"}:
                        ctype = "short"
                        bytes_expr = "sizeof(short) * kElemCount"
                        fill = "fill_short"
                    host_name = f"{var_name}_host"
                    lines.append(f"    CHECK_CUDA(cudaMalloc(&{var_name}, {bytes_expr}));")
                    if is_output:
                        lines.append(f"    CHECK_CUDA(cudaMemset({var_name}, 0, {bytes_expr}));")
                    else:
                        lines.append(f"    {ctype} *{host_name} = ({ctype}*)malloc({bytes_expr});")
                        lines.append(f"    if (!{host_name}) {{ fprintf(stderr, \"malloc failed\\n\"); return 1; }}")
                        if fill == "fill_int":
                            lines.append(f"    {fill}({host_name}, kElemCount, { '256' if is_index else '64' }, 1);")
                        elif fill == "fill_short":
                            lines.append(f"    {fill}({host_name}, kElemCount, { '256' if is_index else '64' });")
                        else:
                            lines.append(f"    {fill}({host_name}, kElemCount);")
                        lines.append(f"    CHECK_CUDA(cudaMemcpy({var_name}, {host_name}, {bytes_expr}, cudaMemcpyHostToDevice));")
                        cleanup_lines.append(f"    free({host_name});")
                elif base in {"unsigned char", "unsigned short", "unsigned", "unsigned int", "unsigned long", "uint8_t", "uint16_t", "uint32_t"}:
                    ctype = "unsigned int"
                    bytes_expr = "sizeof(unsigned int) * kElemCount"
                    fill = "fill_uint"
                    if base in {"unsigned char", "uint8_t"}:
                        ctype = "unsigned char"
                        bytes_expr = "sizeof(unsigned char) * kElemCount"
                        fill = "fill_u8"
                    elif base in {"unsigned short", "uint16_t"}:
                        ctype = "unsigned short"
                        bytes_expr = "sizeof(unsigned short) * kElemCount"
                        fill = "fill_ushort"
                    host_name = f"{var_name}_host"
                    lines.append(f"    CHECK_CUDA(cudaMalloc(&{var_name}, {bytes_expr}));")
                    if is_output:
                        lines.append(f"    CHECK_CUDA(cudaMemset({var_name}, 0, {bytes_expr}));")
                    else:
                        lines.append(f"    {ctype} *{host_name} = ({ctype}*)malloc({bytes_expr});")
                        lines.append(f"    if (!{host_name}) {{ fprintf(stderr, \"malloc failed\\n\"); return 1; }}")
                        lines.append(f"    {fill}({host_name}, kElemCount, { '256u' if is_index else '64u' });")
                        lines.append(f"    CHECK_CUDA(cudaMemcpy({var_name}, {host_name}, {bytes_expr}, cudaMemcpyHostToDevice));")
                        cleanup_lines.append(f"    free({host_name});")
                elif base in {"long long", "int64_t"}:
                    bytes_expr = "sizeof(long long) * kElemCount"
                    host_name = f"{var_name}_host"
                    lines.append(f"    CHECK_CUDA(cudaMalloc(&{var_name}, {bytes_expr}));")
                    if is_output:
                        lines.append(f"    CHECK_CUDA(cudaMemset({var_name}, 0, {bytes_expr}));")
                    else:
                        lines.append(f"    long long *{host_name} = (long long*)malloc({bytes_expr});")
                        lines.append(f"    if (!{host_name}) {{ fprintf(stderr, \"malloc failed\\n\"); return 1; }}")
                        lines.append(f"    fill_ll({host_name}, kElemCount, { '256' if is_index else '64' });")
                        lines.append(f"    CHECK_CUDA(cudaMemcpy({var_name}, {host_name}, {bytes_expr}, cudaMemcpyHostToDevice));")
                        cleanup_lines.append(f"    free({host_name});")
                elif base in {"unsigned long long", "size_t", "uint64_t"}:
                    bytes_expr = "sizeof(unsigned long long) * kElemCount"
                    host_name = f"{var_name}_host"
                    lines.append(f"    CHECK_CUDA(cudaMalloc(&{var_name}, {bytes_expr}));")
                    if is_output:
                        lines.append(f"    CHECK_CUDA(cudaMemset({var_name}, 0, {bytes_expr}));")
                    else:
                        lines.append(f"    unsigned long long *{host_name} = (unsigned long long*)malloc({bytes_expr});")
                        lines.append(f"    if (!{host_name}) {{ fprintf(stderr, \"malloc failed\\n\"); return 1; }}")
                        lines.append(f"    fill_ull({host_name}, kElemCount, { '256ull' if is_index else '64ull' });")
                        lines.append(f"    CHECK_CUDA(cudaMemcpy({var_name}, {host_name}, {bytes_expr}, cudaMemcpyHostToDevice));")
                        cleanup_lines.append(f"    free({host_name});")
                else:
                    lines.append(f"    CHECK_CUDA(cudaMalloc(&{var_name}, 1u << 20));")
                    lines.append(f"    CHECK_CUDA(cudaMemset({var_name}, 0, 1u << 20));")
                ptr_index += 1
            else:
                lines.append(f"    {arg.type_name} {var_name} = {default_scalar(arg)};")
        if arg_names:
            arg_array = ", ".join(f"(void*)&{name}" for name in arg_names)
            lines.append(f"    void *{kernel.name}_args[] = {{ {arg_array} }};")
        else:
            lines.append(f"    void **{kernel.name}_args = NULL;")
        bx = block_dim(kernel)
        lines.append(
            f"    CHECK_CUDA(cudaLaunchKernel((const void*){kernel.name}, dim3(1, 1, 1), dim3({bx}, 1, 1), "
            f"{kernel.name}_args, 0, 0));"
        )
        lines.append("    CHECK_CUDA(cudaDeviceSynchronize());")
        lines.append("")
    lines.extend(
        [
            *cleanup_lines,
        ]
    )
    for kernel in probe.kernels:
        for arg_index, arg in enumerate(kernel.args):
            if arg.kind == "pointer":
                var_name = f"{kernel.name}_arg_{arg_index}"
                lines.append(f"    cudaFree({var_name});")
    lines.extend(
        [
            '    printf("probe runner complete\\n");',
            "    return 0;",
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def find_probe(probe_name: str) -> Probe:
    needle = probe_name.strip()
    for probe in iter_probes():
        if needle in {probe.relative_path, probe.basename, probe.probe_id}:
            return probe
    raise SystemExit(f"probe not found: {probe_name}")


def cmd_emit(args: argparse.Namespace) -> int:
    probes = list(iter_probes())
    if args.format == "json":
        payload = [probe_to_dict(probe) for probe in probes]
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print_tsv(probes, header=args.header)
    return 0


def cmd_generate_runner(args: argparse.Namespace) -> int:
    probe = find_probe(args.probe)
    if probe.runner_kind != "plain":
        raise SystemExit(f"probe {probe.relative_path} requires runner kind {probe.runner_kind}")
    if not probe.supports_generic_runner:
        raise SystemExit(f"probe {probe.relative_path} has unsupported generic runner args")
    out_path = pathlib.Path(args.output)
    out_path.write_text(make_runner_source(probe), encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SASS RE recursive probe manifest")
    sub = parser.add_subparsers(dest="cmd", required=True)

    emit = sub.add_parser("emit", help="emit the recursive manifest")
    emit.add_argument("--format", choices=["json", "tsv"], default="tsv")
    emit.add_argument("--header", action="store_true")
    emit.set_defaults(func=cmd_emit)

    runner = sub.add_parser("generate-runner", help="generate a generic probe runner")
    runner.add_argument("--probe", required=True, help="relative path, basename, or probe_id")
    runner.add_argument("--output", required=True, help="output .cu path")
    runner.set_defaults(func=cmd_generate_runner)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
