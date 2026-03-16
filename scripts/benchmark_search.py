#!/usr/bin/env python3
"""
Codebook Search Benchmark.

Generates z random codewords and measures codebook search time.
Reports total time, per-query average, and throughput.

Usage examples:
  python scripts/benchmark_search.py --n 20 --u 128 --z 1000 --distance hamming
  python scripts/benchmark_search.py --n 24 --u 127 --z 100 --distance levenshtein
  python scripts/benchmark_search.py --n 24 --u 127 --z 100 --distance levenshtein --batch-size 50
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Project root (one level up from scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

import src.paths as paths
from src.codebooks import (
    BCHCodewordGenerator,
    CodebookBase,
    CodebookHamming,
    CodebookLevenshtein,
    CodebookLevenshteinTriton,
    LinearCodeGenerator,
    RNGCodewordGenerator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark codebook search time with random codewords"
    )
    # Codebook parameters
    parser.add_argument("--n", type=int, required=True, help="Codebook size parameter (2^n entries)")
    parser.add_argument("--u", type=int, required=True, help="Code length in bits")
    parser.add_argument("--distance", type=str, required=True, choices=["hamming", "levenshtein", "levenshtein-triton"], help="Distance metric")
    parser.add_argument("--codebook-generation", type=str, default="linear", choices=["rng", "bch", "linear"], help="Codeword generation algorithm (default: linear)")
    parser.add_argument("--secret-key", type=str, default="25111915", help="Codebook secret key (default: 25111915)")
    parser.add_argument("--bch-m", type=int, default=None, help="BCH field degree m (n0=2^m-1). If None, auto-select optimal")

    # Benchmark parameters
    parser.add_argument("--z", type=int, required=True, help="Number of random query codewords")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for search (default: all z at once)")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup queries before timing (default: 1)")
    parser.add_argument("--faiss-threads", type=int, default=1, help="Number of OpenMP threads for Faiss (default: 1)")

    # Output
    parser.add_argument("--save-dir", type=str, default="output", help="Directory to save benchmark results (default: output)")

    return parser.parse_args()


def _run_cmd(cmd: str) -> str:
    """Run a shell command and return stripped stdout, or '' on failure."""
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        return r.stdout.strip()
    except Exception:
        return ""


def collect_system_info() -> dict[str, str]:
    """Collect CPU, GPU, and RAM specs."""
    info: dict[str, str] = {}

    # --- CPU ---
    if platform.system() == "Windows":
        raw = _run_cmd("wmic cpu get Name /value")
        # "Name=Intel(R) Core(TM) i7-..."
        for line in raw.splitlines():
            if line.startswith("Name="):
                info["cpu_model"] = line.split("=", 1)[1].strip()
                break

        raw = _run_cmd("wmic cpu get NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed /value")
        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("MaxClockSpeed="):
                mhz = line.split("=", 1)[1]
                info["cpu_max_clock_mhz"] = mhz
            elif line.startswith("NumberOfCores="):
                info["cpu_cores"] = line.split("=", 1)[1]
            elif line.startswith("NumberOfLogicalProcessors="):
                info["cpu_threads"] = line.split("=", 1)[1]
    else:
        # Linux / macOS
        raw = _run_cmd("lscpu")
        for line in raw.splitlines():
            if line.startswith("Model name:"):
                info["cpu_model"] = line.split(":", 1)[1].strip()
            elif line.startswith("CPU(s):"):
                info["cpu_threads"] = line.split(":", 1)[1].strip()
            elif line.startswith("Core(s) per socket:"):
                cores_per = line.split(":", 1)[1].strip()
            elif line.startswith("Socket(s):"):
                sockets = line.split(":", 1)[1].strip()
            elif line.startswith("CPU max MHz:"):
                info["cpu_max_clock_mhz"] = line.split(":", 1)[1].strip().split(".")[0]
        if "cpu_cores" not in info:
            try:
                info["cpu_cores"] = str(int(cores_per) * int(sockets))
            except Exception:
                pass

    # --- RAM ---
    if platform.system() == "Windows":
        # Total size
        raw = _run_cmd("wmic computersystem get TotalPhysicalMemory /value")
        for line in raw.splitlines():
            if line.startswith("TotalPhysicalMemory="):
                total_bytes = int(line.split("=", 1)[1])
                info["ram_total_gb"] = f"{total_bytes / (1024**3):.1f}"

        # Speed (per DIMM — take first)
        raw = _run_cmd("wmic memorychip get Speed /value")
        for line in raw.splitlines():
            if line.startswith("Speed="):
                info["ram_speed_mhz"] = line.split("=", 1)[1].strip()
                break

        # Type
        raw = _run_cmd("wmic memorychip get SMBIOSMemoryType /value")
        type_map = {"20": "DDR", "21": "DDR2", "24": "DDR3", "26": "DDR4", "34": "DDR5"}
        for line in raw.splitlines():
            if line.startswith("SMBIOSMemoryType="):
                code = line.split("=", 1)[1].strip()
                info["ram_type"] = type_map.get(code, f"type-{code}")
                break
    else:
        raw = _run_cmd("free -b")
        for line in raw.splitlines():
            if line.startswith("Mem:"):
                total_bytes = int(line.split()[1])
                info["ram_total_gb"] = f"{total_bytes / (1024**3):.1f}"
                break
        # Speed (needs dmidecode / root — best effort)
        raw = _run_cmd("sudo dmidecode -t memory 2>/dev/null | grep -m1 'Speed:'")
        if raw:
            info["ram_speed_mhz"] = raw.split(":")[-1].strip().replace(" MHz", "")

    # --- GPU ---
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu_model"] = props.name
            info["gpu_vram_gb"] = f"{props.total_memory / (1024**3):.1f}"
            info["gpu_sm_count"] = str(props.multi_processor_count)
            info["cuda_version"] = torch.version.cuda or "N/A"
    except Exception:
        pass

    return info


def format_system_info(info: dict[str, str]) -> list[str]:
    """Format system info dict into lines for the results file."""
    lines = ["System Info:"]

    cpu = info.get("cpu_model", "unknown")
    cores = info.get("cpu_cores", "?")
    threads = info.get("cpu_threads", "?")
    clock = info.get("cpu_max_clock_mhz")
    cpu_line = f"  CPU: {cpu} ({cores}C/{threads}T"
    if clock:
        cpu_line += f", {clock} MHz"
    cpu_line += ")"
    lines.append(cpu_line)

    ram_gb = info.get("ram_total_gb", "?")
    ram_type = info.get("ram_type", "")
    ram_speed = info.get("ram_speed_mhz", "")
    ram_line = f"  RAM: {ram_gb} GB"
    if ram_type or ram_speed:
        ram_line += f" {ram_type}"
        if ram_speed:
            ram_line += f"-{ram_speed}"
    lines.append(ram_line)

    gpu = info.get("gpu_model")
    if gpu:
        vram = info.get("gpu_vram_gb", "?")
        sm = info.get("gpu_sm_count", "?")
        cuda = info.get("cuda_version", "?")
        lines.append(f"  GPU: {gpu} ({vram} GB VRAM, {sm} SMs, CUDA {cuda})")
    else:
        lines.append("  GPU: N/A")

    return lines


def generate_random_codeword(sample_idx: int, seed: int, u: int) -> np.ndarray:
    """Generate a deterministic random codeword from sample index and seed."""
    hash_input = f"{sample_idx + seed}"
    hash_val = int(hashlib.sha256(hash_input.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(hash_val)
    return rng.integers(0, 2, size=u, dtype=np.uint8)


def benchmark_hamming(
    codebook: CodebookHamming,
    queries: np.ndarray,
    u: int,
    batch_size: int | None,
) -> dict:
    """Benchmark Hamming (Faiss) search."""
    z = queries.shape[0]
    packed_queries = CodebookBase._pack_bits_2d(queries, u)

    if batch_size is None:
        batch_size = z

    # Batch search
    total_queries = 0
    t_start = time.perf_counter()

    for i in range(0, z, batch_size):
        batch = packed_queries[i : i + batch_size]
        distances, indices = codebook.index.search(batch, 1)
        total_queries += len(batch)

    t_elapsed = time.perf_counter() - t_start

    return {
        "total_time_s": t_elapsed,
        "total_queries": total_queries,
        "per_query_ms": (t_elapsed / total_queries) * 1000,
        "throughput_qps": total_queries / t_elapsed,
        "batch_size": batch_size,
    }


def benchmark_levenshtein(
    codebook: CodebookLevenshtein,
    queries: np.ndarray,
    batch_size: int | None,
) -> dict:
    """Benchmark Levenshtein (Numba) search."""
    z = queries.shape[0]

    if batch_size is None:
        batch_size = z

    query_list = [queries[i] for i in range(z)]

    total_queries = 0
    t_start = time.perf_counter()

    for i in range(0, z, batch_size):
        batch = query_list[i : i + batch_size]
        codebook.find_closest_match_batch(batch, distances_only=True)
        total_queries += len(batch)

    t_elapsed = time.perf_counter() - t_start

    return {
        "total_time_s": t_elapsed,
        "total_queries": total_queries,
        "per_query_ms": (t_elapsed / total_queries) * 1000,
        "throughput_qps": total_queries / t_elapsed,
        "batch_size": batch_size,
    }


def write_results(args: argparse.Namespace, results: dict, sys_info: dict[str, str], save_path: str) -> None:
    """Write benchmark results to text file."""
    lines = []
    lines.append("=" * 60)
    lines.append("Codebook Search Benchmark")
    lines.append("=" * 60)
    lines.append("")
    lines.extend(format_system_info(sys_info))
    lines.append("")
    lines.append("Codebook Parameters:")
    lines.append(f"  n (codebook size): {args.n} (2^{args.n} = {2**args.n:,} entries)")
    lines.append(f"  u (codeword length): {args.u} bits")
    lines.append(f"  distance metric: {args.distance}")
    lines.append(f"  codebook generation: {args.codebook_generation}")
    lines.append(f"  secret key: {args.secret_key}")
    if args.codebook_generation == "bch":
        lines.append(f"  bch_m: {args.bch_m}")
    lines.append("")
    lines.append("Benchmark Parameters:")
    lines.append(f"  z (queries): {args.z}")
    lines.append(f"  seed: {args.seed}")
    lines.append(f"  batch size: {results['batch_size']}")
    lines.append(f"  warmup queries: {args.warmup}")
    if args.distance == "hamming":
        lines.append(f"  faiss threads: {args.faiss_threads}")
    lines.append("")
    lines.append("-" * 60)
    lines.append("Results:")
    lines.append("-" * 60)
    lines.append(f"  Total search time: {results['total_time_s']:.4f} s")
    lines.append(f"  Total queries: {results['total_queries']}")
    lines.append(f"  Per-query average: {results['per_query_ms']:.4f} ms")
    lines.append(f"  Throughput: {results['throughput_qps']:.2f} queries/s")
    lines.append("")

    text = "\n".join(lines)
    print(text)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Results saved to: {save_path}")


def main() -> None:
    args = parse_args()
    paths.ensure_directories()

    print("Collecting system info...")
    sys_info = collect_system_info()

    # Build codebook
    if args.codebook_generation == "rng":
        generator_cls = RNGCodewordGenerator
        generator_kwargs: dict = {}
    elif args.codebook_generation == "linear":
        generator_cls = LinearCodeGenerator
        generator_kwargs = {}
    else:
        generator_cls = BCHCodewordGenerator
        generator_kwargs = {"m": args.bch_m}

    base_codebook = CodebookBase(
        n=args.n,
        u=args.u,
        secret_key=args.secret_key,
        generator_cls=generator_cls,
        generator_kwargs=generator_kwargs,
    )

    if args.distance == "hamming":
        import faiss
        faiss.omp_set_num_threads(args.faiss_threads)
        codebook = CodebookHamming(base_codebook)
    elif args.distance == "levenshtein-triton":
        codebook = CodebookLevenshteinTriton(base_codebook)
    else:
        codebook = CodebookLevenshtein(base_codebook)

    # Generate random codewords
    print(f"Generating {args.z} random codewords (seed={args.seed})...")
    queries = np.stack([
        generate_random_codeword(s, args.seed, args.u) for s in range(args.z)
    ])

    # Warmup (JIT compilation for Numba, cache warming for Faiss)
    if args.warmup > 0:
        print(f"Running {args.warmup} warmup queries...")
        warmup_queries = queries[:args.warmup]
        if args.distance == "hamming":
            packed = CodebookBase._pack_bits_2d(warmup_queries, args.u)
            codebook.index.search(packed, 1)
        else:
            codebook.find_closest_match_batch(
                [warmup_queries[i] for i in range(len(warmup_queries))],
                distances_only=True,
            )

    # Benchmark
    print(f"Benchmarking {args.z} queries (batch_size={args.batch_size or args.z})...")
    if args.distance == "hamming":
        results = benchmark_hamming(codebook, queries, args.u, args.batch_size)
    else:
        results = benchmark_levenshtein(codebook, queries, args.batch_size)

    # Output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(
        args.save_dir,
        f"benchmark_n{args.n}_u{args.u}_z{args.z}"
        f"_dist-{args.distance}_c-{args.codebook_generation}_{timestamp}",
    )
    save_path = os.path.join(save_dir, "results.txt")
    write_results(args, results, sys_info, save_path)


if __name__ == "__main__":
    main()
