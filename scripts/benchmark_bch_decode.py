#!/usr/bin/env python3
"""
BCH Decode Benchmark.

Encodes z random watermarks with BCHCodec, optionally injects bit errors,
then measures decode time. Reports total time, per-decode average, and throughput.

Designed for direct comparison with benchmark_search.py results from the
codebook-based (master) branch.

Usage examples:
  python scripts/benchmark_bch_decode.py --n 16 --u 127 --z 1000
  python scripts/benchmark_bch_decode.py --n 20 --u 127 --z 1000 --error-rate 0.05
  python scripts/benchmark_bch_decode.py --n 24 --u 255 --z 500 --bch-m 8
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

from src.bch_codec import BCHCodec
from src.bch_params import format_bch_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark BCH decode time with random watermarks"
    )
    # BCH parameters
    parser.add_argument("--n", type=int, required=True, help="Watermark message length in bits (2^n unique watermarks)")
    parser.add_argument("--u", type=int, required=True, help="Codeword length in bits")
    parser.add_argument("--bch-m", type=int, default=None, help="BCH field degree m (n0=2^m-1). If None, auto-select optimal")

    # Benchmark parameters
    parser.add_argument("--z", type=int, required=True, help="Number of random watermarks to encode/decode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--error-rate", type=float, default=0.0, help="Fraction of bits to flip in each codeword before decoding (default: 0.0)")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup decodes before timing (default: 5)")

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
        for line in raw.splitlines():
            if line.startswith("Name="):
                info["cpu_model"] = line.split("=", 1)[1].strip()
                break

        raw = _run_cmd("wmic cpu get NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed /value")
        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("MaxClockSpeed="):
                info["cpu_max_clock_mhz"] = line.split("=", 1)[1]
            elif line.startswith("NumberOfCores="):
                info["cpu_cores"] = line.split("=", 1)[1]
            elif line.startswith("NumberOfLogicalProcessors="):
                info["cpu_threads"] = line.split("=", 1)[1]
    else:
        raw = _run_cmd("lscpu")
        cores_per = "?"
        sockets = "1"
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
        raw = _run_cmd("wmic computersystem get TotalPhysicalMemory /value")
        for line in raw.splitlines():
            if line.startswith("TotalPhysicalMemory="):
                total_bytes = int(line.split("=", 1)[1])
                info["ram_total_gb"] = f"{total_bytes / (1024**3):.1f}"

        raw = _run_cmd("wmic memorychip get Speed /value")
        for line in raw.splitlines():
            if line.startswith("Speed="):
                info["ram_speed_mhz"] = line.split("=", 1)[1].strip()
                break

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
        raw = _run_cmd("sudo dmidecode -t memory 2>/dev/null | grep -m1 'Speed:'")
        if raw:
            info["ram_speed_mhz"] = raw.split(":")[-1].strip().replace(" MHz", "")

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

    return lines


def generate_random_watermark(sample_idx: int, seed: int, n: int) -> int:
    """Generate a deterministic random watermark integer from sample index and seed."""
    hash_input = f"{sample_idx + seed}"
    hash_val = int(hashlib.sha256(hash_input.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(hash_val)
    return int(rng.integers(0, 2**n))


def inject_bit_errors(codeword: np.ndarray, error_rate: float, rng: np.random.Generator) -> np.ndarray:
    """Flip a fraction of bits in the codeword to simulate channel errors."""
    if error_rate <= 0.0:
        return codeword
    noisy = codeword.copy()
    n_errors = max(1, int(len(codeword) * error_rate))
    flip_positions = rng.choice(len(codeword), size=n_errors, replace=False)
    noisy[flip_positions] ^= 1
    return noisy


def benchmark_bch_decode(
    codec: BCHCodec,
    codewords: list[np.ndarray],
) -> dict:
    """Benchmark BCH decode on a list of (possibly noisy) codewords."""
    z = len(codewords)
    successful = 0
    decode_failures = 0

    t_start = time.perf_counter()

    for cw in codewords:
        _msg, bit_errors = codec.decode(cw)
        if bit_errors >= codec.native_codeword_length:
            decode_failures += 1
        else:
            successful += 1

    t_elapsed = time.perf_counter() - t_start

    return {
        "total_time_s": t_elapsed,
        "total_queries": z,
        "per_query_ms": (t_elapsed / z) * 1000,
        "throughput_qps": z / t_elapsed,
        "successful": successful,
        "decode_failures": decode_failures,
    }


def benchmark_bch_encode(
    codec: BCHCodec,
    watermarks: list[int],
) -> dict:
    """Benchmark BCH encode on a list of watermark integers."""
    z = len(watermarks)

    t_start = time.perf_counter()

    for wm in watermarks:
        codec.encode_int(wm)

    t_elapsed = time.perf_counter() - t_start

    return {
        "total_time_s": t_elapsed,
        "total_queries": z,
        "per_query_ms": (t_elapsed / z) * 1000,
        "throughput_qps": z / t_elapsed,
    }


def write_results(
    args: argparse.Namespace,
    codec: BCHCodec,
    encode_results: dict,
    decode_results: dict,
    sys_info: dict[str, str],
    save_path: str,
) -> None:
    """Write benchmark results to text file."""
    lines = []
    lines.append("=" * 60)
    lines.append("BCH Decode Benchmark")
    lines.append("=" * 60)
    lines.append("")
    lines.extend(format_system_info(sys_info))
    lines.append("")
    lines.append("BCH Parameters:")
    lines.append(f"  {format_bch_summary(codec.params, args.n)}")
    lines.append(f"  n (watermark bits): {args.n} (2^{args.n} = {2**args.n:,} unique watermarks)")
    lines.append(f"  u (codeword length): {args.u} bits")
    lines.append(f"  t (error correction): {codec.error_correction_capability} bits")
    lines.append(f"  n0 (native BCH length): {codec.native_codeword_length}")
    if codec.params.padding > 0:
        lines.append(f"  padding: {codec.params.padding} bits")
    if codec.params.shortening > 0:
        lines.append(f"  shortening: {codec.params.shortening} bits")
    lines.append("")
    lines.append("Benchmark Parameters:")
    lines.append(f"  z (queries): {args.z}")
    lines.append(f"  seed: {args.seed}")
    lines.append(f"  error rate: {args.error_rate}")
    lines.append(f"  warmup decodes: {args.warmup}")
    lines.append("")
    lines.append("-" * 60)
    lines.append("Encode Results:")
    lines.append("-" * 60)
    lines.append(f"  Total encode time: {encode_results['total_time_s']:.4f} s")
    lines.append(f"  Per-encode average: {encode_results['per_query_ms']:.4f} ms")
    lines.append(f"  Throughput: {encode_results['throughput_qps']:.2f} encodes/s")
    lines.append("")
    lines.append("-" * 60)
    lines.append("Decode Results:")
    lines.append("-" * 60)
    lines.append(f"  Total decode time: {decode_results['total_time_s']:.4f} s")
    lines.append(f"  Total queries: {decode_results['total_queries']}")
    lines.append(f"  Per-decode average: {decode_results['per_query_ms']:.4f} ms")
    lines.append(f"  Throughput: {decode_results['throughput_qps']:.2f} decodes/s")
    lines.append(f"  Successful decodes: {decode_results['successful']}")
    lines.append(f"  Decode failures: {decode_results['decode_failures']}")
    lines.append("")

    text = "\n".join(lines)
    print(text)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Results saved to: {save_path}")


def main() -> None:
    args = parse_args()

    print("Collecting system info...")
    sys_info = collect_system_info()

    # Create BCH codec (galois.BCH constructor takes ~10s for large m)
    print(f"Creating BCH codec (n={args.n}, u={args.u}, bch_m={args.bch_m})...")
    t_init_start = time.perf_counter()
    codec = BCHCodec(n=args.n, u=args.u, bch_m=args.bch_m, verbose=True)
    t_init = time.perf_counter() - t_init_start
    print(f"BCH codec initialized in {t_init:.2f} s\n")

    # Generate random watermarks
    print(f"Generating {args.z} random watermarks (seed={args.seed})...")
    watermarks = [generate_random_watermark(i, args.seed, args.n) for i in range(args.z)]

    # Encode all watermarks
    print(f"Encoding {args.z} watermarks...")
    codewords = [codec.encode_int(wm) for wm in watermarks]

    # Inject bit errors if requested
    if args.error_rate > 0:
        n_flips = max(1, int(args.u * args.error_rate))
        print(f"Injecting ~{n_flips} bit errors per codeword (error_rate={args.error_rate})...")
        error_rng = np.random.default_rng(args.seed + 1000)
        codewords = [inject_bit_errors(cw, args.error_rate, error_rng) for cw in codewords]

    # Warmup (JIT compilation for galois internals, cache warming)
    if args.warmup > 0:
        print(f"Running {args.warmup} warmup decodes...")
        for cw in codewords[:args.warmup]:
            codec.decode(cw)

    # Benchmark encode
    print(f"Benchmarking {args.z} encodes...")
    encode_results = benchmark_bch_encode(codec, watermarks)

    # Benchmark decode
    print(f"Benchmarking {args.z} decodes...")
    decode_results = benchmark_bch_decode(codec, codewords)

    # Output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    m_tag = f"m{codec.params.m}"
    err_tag = f"_err{args.error_rate}" if args.error_rate > 0 else ""
    save_dir = os.path.join(
        args.save_dir,
        f"benchmark_bch_n{args.n}_u{args.u}_z{args.z}_{m_tag}{err_tag}_{timestamp}",
    )
    save_path = os.path.join(save_dir, "results.txt")
    write_results(args, codec, encode_results, decode_results, sys_info, save_path)


if __name__ == "__main__":
    main()
