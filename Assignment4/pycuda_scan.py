import numpy as np
import math
import time
import matplotlib.pyplot as plt
from functools import wraps
from time import time
import contextlib
from adjustText import adjust_text

#import relevant pycuda modules

# decorator to compute execution time
def time_it(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        return result, (te - ts) * 1e3
    return wrap

# to compute cuda execution time
@contextlib.contextmanager
def cuda_sync(stats):
    start = cuda.Event()
    end = cuda.Event()
    start.record()
    yield
    end.record()
    end.synchronize()
    stats['elapsed'] += start.time_till(end)
    
MAX_THREAD_PER_BLOCK = cudatools.get_default_device().get_attribute(cuda.device_attribute.MAX_THREADS_PER_BLOCK)
MAX_BLOCKS = cudatools.get_default_device().get_attribute(cuda.device_attribute.MAX_GRID_DIM_X)

def get_block_size(N):
    # can change this function to return dynamic output
    # currently set as constant
    return MAX_THREAD_PER_BLOCK

def get_grid_dim(N, block_size):
    # return the grid dim based on input size
    # check if it is possible to schedule the kernel
    blocks = int(math.ceil(N / block_size))
    if blocks > MAX_BLOCKS:
        raise Exception("cannot schedule more than {} blocks".format(MAX_BLOCKS))
    return blocks

def get_dim_args(N, double_sized_block = False):
    block_size = get_block_size(N)
    # define a 1D block - of blocsize length
    block_dim = (block_size, 1, 1)
    # if double_sized_block, double the size
    if double_sized_block:
        block_size = block_size * 2
    # get griddim corresponding to blocksize
    grid_dim = (get_grid_dim(N, block_size), 1, 1)
    return {"block": block_dim, "grid": grid_dim}

class PrefixSum:
    def __init__(self):
        self.sample_testcases = map(
            lambda t: (np.array(t[0]).astype(np.int64), np.array(t[1]).astype(np.int64)), 
            [
                #([1,1,1,1,1], [1,2,3,4,5]), - complete this with custom examples
    
            ])
        self.mod = self.load_module()
        self.work_inefficient_scan = self.mod.get_function("work_inefficient_scan")
        self.work_efficient_scan = self.mod.get_function("work_efficient_scan")
        self.add_block_sums = self.mod.get_function("add_block_sums")

    def load_module(self):
        kernel = """
            __global__ void work_inefficient_scan(int64_t* src, int64_t* dst, int64_t* aux, int N, int B) {
                // prefix sum scan within each block
            
            }

            __global__ void add_block_sums(int64_t* dst, int64_t* aux, int N, int B) {
                // We are adding aux[i] to each of values in a block of dst
                // We only want to add the first M - 1 blocks
            }
            
            __global__ void work_efficient_scan(int64_t* src, int64_t* dst, int64_t* aux, int N, int B) {
                // Shared memory contains 2 * block_size == B
            }
        """
        return SourceModule(kernel)
    
    @time_it
    def prefix_sum_python(self, a):
        # complete function
        return b

    @time_it
    def prefix_sum_gpu_naive(self, src_h):
        # testing naive implementation
        return self.gpu_scan(self.work_inefficient_scan, src_h)

    @time_it
    def prefix_sum_gpu_work_efficient(self, src_h):
        # testing efficient implementation
        return self.gpu_scan(self.work_efficient_scan, src_h, double_sized_block = True)
    
    def gpu_scan(self, scanfunc, src_h, double_sized_block = False):
        stats = {'elapsed':0}
        src_d = gpuarray.to_gpu(src_h)

        def hierarchical_scan(src_d):
            N = len(src_d)
            main_scan_dim_args = get_dim_args(N, double_sized_block)
            num_blocks = main_scan_dim_args['grid'][0]
            block_size = main_scan_dim_args['block'][0]
            if double_sized_block:
                block_size *= 2

            dst_d = gpuarray.zeros_like(src_d)
            aux_d = gpuarray.zeros((num_blocks,), dtype=src_d.dtype)
            
            with cuda_sync(stats):
                scanfunc(src_d, dst_d, aux_d, np.int32(N), np.int32(block_size),
                     shared = block_size * src_d.dtype.itemsize, 
                     **main_scan_dim_args)
            
            # base case -> scan can fit within one thread block
            if num_blocks == 1:
                return dst_d
            
            scanned_aux_d = hierarchical_scan(aux_d)

            M = len(scanned_aux_d)
            aux_add_dim_args = get_dim_args(M)

            # add block sums to dst_d which contains blocked partial prefix sums atm
            with cuda_sync(stats):
                self.add_block_sums(
                    dst_d, scanned_aux_d, np.int32(N), np.int32(block_size), 
                    **aux_add_dim_args)
            
            return dst_d
        
        dst_d = hierarchical_scan(src_d)
        return dst_d.get(), stats['elapsed']

    def test_prefix_sum_python(self):
        for test, truth in self.sample_testcases:
            out, _ = self.prefix_sum_python(test)
            assert(np.allclose(out, truth, atol=1e-5))
    
    def test_prefix_sum_gpu_naive(self):
        for test, truth in self.sample_testcases:
            out, _ = self.prefix_sum_gpu_naive(test)
            assert(np.allclose(out, truth, atol=1e-5))

    def test_prefix_sum_gpu_work_efficient(self):
        for test, truth in self.sample_testcases:
            out, _ = self.prefix_sum_gpu_work_efficient(test)
            assert(np.allclose(out, truth, atol=1e-5))

def plot_stats(title, sizes, pystats, cudastats):
    #plot function
    assert len(py_t) == len(cudastats[0][1])
    x = list(range(1, len(pystats)+1))
    plt.figure(figsize=(16, 8))
    plt.title(title)
    plt.xlabel("Array sizes")
    plt.ylabel("Execution time (ms)")
    
    plt.plot(x, pystats, linestyle='--', marker='o', label="Execution time using Python")
    
    texts = []
    for idx, time in zip(x, pystats):
        texts.append(plt.text(idx, time, "Python: {}ms".format(round(time, 2))))

    n = len(cudastats)
    for i, (label, t) in enumerate(cudastats):
        plt.plot(x, t, linestyle='--', marker='o', label="Execution time using {}".format(label))
        for idx, time in zip(x, t):
            texts.append(plt.text(idx, time, "{}: {}ms".format(label, round(time, 2))))
    
    plt.yscale("log")
    plt.xticks(x, sizes)
    plt.legend()
    adjust_text(texts, arrowprops=dict(arrowstyle="->", color='black', lw=0.5))
    plt.savefig(title + '.png')
    plt.show()

def get_avg_stats(func, A, iters):
    return np.mean([func(A)[1] for _ in range(iters)])

def get_avg_raw_stats(func, A, iters):
    return np.mean([func(A)[0][1] for _ in range(iters)])

if __name__ == "__main__":
    # collect stats
    iters = 10
    #SIZES = [128, 2048, ....]

    prefix_sum = PrefixSum()
    print("Verifying correctness with small testcases...")
    prefix_sum.test_prefix_sum_python()
    prefix_sum.test_prefix_sum_gpu_naive()
    prefix_sum.test_prefix_sum_gpu_work_efficient()

    pystats = []
    cudastats = {}
    cudarawstats = {}
    for N in SIZES:
        pyfunc = prefix_sum.prefix_sum_python
        cudafuncs = [prefix_sum.prefix_sum_gpu_naive, prefix_sum.prefix_sum_gpu_work_efficient]
        if not cudastats:
            cudastats = {n: [] for n in map(lambda f: f.__name__, cudafuncs)}
        if not cudarawstats:
            cudarawstats = {n: [] for n in map(lambda f: f.__name__, cudafuncs)}
        print("Computing prefix sum for array with size {}...".format(N))
        A = np.random.randint(0, 100, size=N).astype(np.int64)
        truth, _ = pyfunc(A)
        print("Verifying correctness using python...")
        for cudafunc in cudafuncs:
            out, _ = cudafunc(A)
            out, _ = out
            try:
                assert(np.allclose(truth, out, atol=1e-5))
            except:
                print(out)
                print("==================")
                print(truth)
                raise
        print("Getting runtime stats...")
        # get stats
        pystats.append(get_avg_stats(pyfunc, A, iters))
        for func in cudafuncs:

            cudastats[func.__name__].append(get_avg_stats(func, A, iters))
            cudarawstats[func.__name__].append(get_avg_raw_stats(func, A, iters))
            



    # plot stats
    plot_stats("PyCUDA - Entire runtime stats", SIZES, pystats, list(cudastats.items()))
    plot_stats("PyCUDA - CUDA only runtime stats", SIZES, pystats, list(cudarawstats.items()))
