import os
import sys
import subprocess
import logging
CURRENT_FILE_PATH = os.path.abspath(__file__)
WORK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_FILE_PATH)))
# print(WORK_DIR)
sys.path.extend([f"{WORK_DIR}/sparse_warp_spec/tests", f"{WORK_DIR}/third_party/megablocks", f"{WORK_DIR}/hare"])

from preprocess import parse_nsys_avg_ns, setup_logging

profile_prefix = "nsys nvprof --cpu-profiling off --profile-from-start=off --force-overwrite -o /tmp/test"
clean_cache_cmd = "rm -rf /tmp/nvidia/nsight_systems" 

def run_command(cmd):
    logger = logging.getLogger("file")
    logger.info(f"command: {cmd}")
    result = subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
    if result.returncode != 0:
        logger.error(f"Error: {result.stdout}\nfailed with code {result.returncode}")
        logger.error(f"Error: {cmd}\nfailed with code {result.returncode}")
        logger.error(f"stderr: {result.stderr}")
        return None
    return result

def test_Samoyeds(m, n, k, sparsity=0.9):
    assert sparsity in [0.5, 0.75, 0.9], "sparsity must be in [0.5, 0.75, 0.9]"
    if sparsity == 0.5:
        N, M = 1, 1
    elif sparsity == 0.75:
        N, M = 1, 2
    else:
        N, M = 1, 5
    cmd = f"cd {WORK_DIR}/third_party/Samoyeds/Samoyeds-Kernel"
    cmd += f" && {profile_prefix} ./build/./benchmark/./horizontal_ssmm_benchmark -m {m} -k {k} -n {n} -N {N} -M {M} --vector_length 128 --seed 42 -t"
    result = run_command(cmd)
    if result is None:
        return 0
    # import pdb; pdb.set_trace()
    avg_ns = parse_nsys_avg_ns(result.stdout, "_HorizontalSsmmKernel")
    return avg_ns

def test_CLASP(m, n, k, sparsity=0.9):
    assert sparsity in [0.5, 0.75, 0.9], "sparsity must be in [0.5, 0.75, 0.9]"
    density = 1 - sparsity
    cmd = f"cd {WORK_DIR}/third_party/venom/build"
    cmd += f" && {profile_prefix} ./src/benchmark_spmm --sparsity-type cvs --spmm CLASP --gemm cuBlas --precision half --block-size 8 --m {m} --k {k} --n {n} --d {density} --check"
    result = run_command(cmd)
    if result is None:
        return 0
    # import pdb; pdb.set_trace()
    avg_ns = parse_nsys_avg_ns(result.stdout, "spmm_amp::wmmaSpmmKernel")
    return avg_ns

def test_Sputnik(m, n, k, sparsity=0.9):
    assert sparsity in [0.5, 0.75, 0.9], "sparsity must be in [0.5, 0.75, 0.9]"
    density = 1 - sparsity
    cmd = f"cd {WORK_DIR}/third_party/venom/build"
    cmd += f" && {profile_prefix} ./src/benchmark_spmm --sparsity-type csr --spmm sputnik --gemm cuBlas --precision half --acc_t fp16 --m {m} --k {k} --n {n} --d {density}"
    result = run_command(cmd)
    if result is None:
        return 0
    
    if result.returncode != 0:
        print(f"Error: {cmd} failed with code {result.returncode}")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        return 0
    avg_ns = parse_nsys_avg_ns(result.stdout, "sputnik")
    return avg_ns

def test_Spatha(m, n, k, sparsity=0.9):
    assert sparsity in [0.5, 0.75, 0.9], "sparsity must be in [0.5, 0.75, 0.9]"
    if sparsity == 0.5:
        N, M = 2, 4
    elif sparsity == 0.75:
        N, M = 2, 7
    else:
        N, M = 2, 20
    cmd = f"cd {WORK_DIR}/third_party/venom/build"
    cmd += f" && {profile_prefix} ./src/benchmark_spmm --sparsity-type n-to-m --spmm spatha --gemm cuBlas \
        --precision half --meta-block-size 32 --block-size 4 \
        --nn_row {N} --mm_row {M} --m {m} --k {k} --n {n} --d 0.5 --bm 128 --bn 64 --bk 32 --wm 32 --wn 64 --wk 32 --mm 16 --mn 8 --mk 32 --nstage 2 --random --check"
    result = run_command(cmd)
    if result is None:
        return 0
    # import pdb; pdb.set_trace()
    avg_ns = parse_nsys_avg_ns(result.stdout, "spatha")
    return avg_ns

def test_cuBLAS(m, n, k, sparsity=0.9):
    cmd = f"cd {WORK_DIR}/third_party/venom/build"
    cmd += f" && {profile_prefix} ./src/benchmark_spmm --sparsity-type cvs --spmm CLASP --gemm cuBlas --precision half --block-size 8 --m {m} --k {k} --n {n} --d 0.25 --check"
    result = run_command(cmd)
    if result is None:
        return 0
    # import pdb; pdb.set_trace()
    avg_ns = parse_nsys_avg_ns(result.stdout, "cutlass")
    return avg_ns

def test_SSD(m, n, k, sparsity=0.9, num_expert=8, bh=128, bw=16):
    assert sparsity in [0.5, 0.75, 0.9], "sparsity must be in [0.5, 0.75, 0.9]"
    if sparsity == 0.5:
        N, M = 1, 1
    elif sparsity == 0.75:
        N, M = 1, 2
    else:
        N, M = 1, 5
    cmd = f"cd {WORK_DIR}/sparse_warp_spec/tests"
    cmd += f" && {profile_prefix} python test_kernel.py --m {m} --n {n} --k {k} --bn {N} --bm {M} --ssd --num_expert {num_expert} --bh {bh} --bw {bw}"
    result = run_command(cmd)
    if result is None:
        return 0
    # import pdb; pdb.set_trace()
    avg_ns = parse_nsys_avg_ns(result.stdout, "fastssd")
    return avg_ns

def test_DSS(m, n, k, sparsity=0.9, num_expert=8, bh=128, bw=16):
    assert sparsity in [0.5, 0.75, 0.9], "sparsity must be in [0.5, 0.75, 0.9]"
    if sparsity == 0.5:
        N, M = 1, 1
    elif sparsity == 0.75:
        N, M = 1, 2
    else:
        N, M = 1, 5
    cmd = f"cd {WORK_DIR}/sparse_warp_spec/tests"
    cmd += f" && {profile_prefix} python test_kernel.py --m {m} --n {n} --k {k} --bn {N} --bm {M} --dss --num_expert {num_expert} --bh {bh} --bw {bw}"
    result = run_command(cmd)
    if result is None:
        return 0
    # import pdb; pdb.set_trace()
    avg_ns = parse_nsys_avg_ns(result.stdout, "fastdss")
    return avg_ns

if __name__ == "__main__":
    res = test_Samoyeds(1024, 1024, 1024)
    print(res)
    res = test_CLASP(1024, 1024, 1024)
    print(res)
    res = test_Sputnik(1024, 1024, 1024)
    print(res)
    res = test_Spatha(1024, 1024, 1024)
    print(res)
    res = test_cuBLAS(1024, 1024, 1024)
    print(res)
    res = test_SSD(1024, 1024, 1024)
    print(res)
    res = test_DSS(1024, 1024, 1024)
    print(res)