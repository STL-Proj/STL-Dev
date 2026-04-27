# -----------------------------
# Imports
# -----------------------------
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime

import google_benchmark as benchmark
import numpy as np
import torch

# -----------------------------
# Setup
# -----------------------------
# Add parent directory to sys.path to import STL package modules
PARENT_DIR = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
sys.path.append(PARENT_DIR)
print("Parent directory added to sys.path:", ".../" + os.path.basename(PARENT_DIR))

# Path to test data
DATA_TEST_PATH = PARENT_DIR + "/data" + "/test"
print(
    "Dataset directory used:",
    ".../"
    + os.path.basename(PARENT_DIR)
    + DATA_TEST_PATH.split(os.path.basename(PARENT_DIR))[-1],
)

from STL_main.STL_2D_FFT_Torch import STL_2D_FFT_Torch

# Import STL modules
from STL_main.STL_2D_Kernel_Torch import STL_2D_Kernel_Torch
from STL_main.Synthesis import synthesize_from_maps
from STL_main.torch_backend import _DEFAULT_DEVICE

device = _DEFAULT_DEVICE
assert (
    device.type == "cuda"
), "GPU is required for this benchmark for memory testing. Please ensure CUDA is properly set up."


# Run benchmarks in silent mode to avoid cluttering benchmark output
@contextmanager
def silent():
    """Suppress stdout and stderr temporarily."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = open(os.devnull, "w")
    sys.stderr = sys.stdout
    try:
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


# Setup benchmark output directory and file naming
benchmark_name = os.path.splitext(os.path.basename(__file__))[0]
base_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs", benchmark_name
)
os.makedirs(base_dir, exist_ok=True)

# Setup benchmark output directory and file naming
datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
sys.argv += [
    f"--benchmark_out={os.path.join(base_dir, f'run_{datetime_str}.csv')}",
    "--benchmark_out_format=csv",
    "--benchmark_min_warmup_time=0.1",
    "--benchmark_time_unit=s",
    "--benchmark_counters_tabular=true",
    "--benchmark_repetitions=4",
]

# Load target image
im = np.load(DATA_TEST_PATH + "/" + "Turb_6.npy")[:, None, :, :]  # [Nb, Nc, Nx, Ny]
im_target = im[0, 0, :, :]  # Mono channel image

# Instantiate data classes
data_target_kernel = STL_2D_Kernel_Torch(im_target, pbc=True)
data_target_fft = STL_2D_FFT_Torch(im_target, pbc=True)


# -----------------------------
# Benchmark registration
# -----------------------------
def make_benchmark(
    name, data_target, computation_backend=None, has_fewer_convolutions=False
):
    @benchmark.register(name=name)
    @benchmark.option.use_manual_time()
    @benchmark.option.arg(1)
    @benchmark.option.arg(5)
    @benchmark.option.arg(10)
    @benchmark.option.arg(15)
    @benchmark.option.arg(20)
    @benchmark.option.complexity(benchmark.oN)
    def _bench(state):
        nbatch = state.range(0)
        total_memory = 0

        while state:
            start_time = time.perf_counter()
            start_memory = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()

            with silent():
                synthesize_from_maps(
                    data_target,
                    pbc_running=True,
                    nbatch=nbatch,
                    computation_backend=computation_backend,
                    has_fewer_convolutions=has_fewer_convolutions,
                )

            end_time = time.perf_counter()
            end_memory = torch.cuda.memory_allocated()
            peak_mem = torch.cuda.max_memory_allocated()

            state.set_iteration_time(end_time - start_time)
            total_memory += peak_mem - start_memory

        state.counters["Time_per_batch"] = benchmark.Counter(
            nbatch, benchmark.Counter.kIsRate | benchmark.Counter.kInvert
        )

        state.counters["Memory"] = total_memory / state.iterations
        state.complexity_n = nbatch

    return _bench


make_benchmark("Kernel", data_target_kernel, False)
make_benchmark("Kernel_Sihao", data_target_kernel, True)
make_benchmark("FFT", data_target_fft, False)
make_benchmark("FFT_Sihao", data_target_fft, True)

if __name__ == "__main__":
    console_file = os.path.join(base_dir, f"run_{datetime_str}.txt")

    f = open(console_file, "w")

    # Save original stdout/stderr file descriptors
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    old_stdout_fd = os.dup(stdout_fd)
    old_stderr_fd = os.dup(stderr_fd)

    # Redirect stdout/stderr to the file
    os.dup2(f.fileno(), stdout_fd)
    os.dup2(f.fileno(), stderr_fd)

    try:
        benchmark.main()
    finally:
        # Restore original stdout/stderr
        os.dup2(old_stdout_fd, stdout_fd)
        os.dup2(old_stderr_fd, stderr_fd)
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)
        f.close()
