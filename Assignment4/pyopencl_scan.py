import numpy as np
import time
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from functools import wraps
from time import time
import contextlib
from adjustText import adjust_text

# import relevant pyopencl functions - use cl.array
MAX_THREAD_PER_BLOCK = min(cl.device_info.MAX_WORK_GROUP_SIZE, 1024)

def get_block_size(N):
    # return MAX_THREADS_PER_BLOCK by default
    return MAX_THREAD_PER_BLOCK
    
def get_grid_dim(N, block_size):
    blocks = int(math.ceil(N / block_size))
    return blocks

def get_dim_args(N, double_sized_block = False):
    block_size = get_block_size(N)
    block_dim = (block_size, 1, 1)
    if double_sized_block:
        block_size = block_size * 2
    grid_dim = (get_grid_dim(N, block_size), 1, 1)
    return {"block": block_dim, "grid": grid_dim}


#decorator to compute time of execution
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
def cl_sync(stats):
    start = time()
    yield
    end = time()
    stats['elapsed'] = (end - start) * 1e3

class PrefixSum:
    def __init__(self):
        self.sample_testcases = map(
            lambda t: (np.array(t[0]).astype(np.int64), np.array(t[1]).astype(np.int64)), 
            [


            ])
        
        # Get platform and device property
        NAME = 'NVIDIA CUDA'
        platforms = cl.get_platforms()
        devs = None
        for platform in platforms:
            if platform.name == NAME:
                devs = platform.get_devices()       

        # Create Context:
        self.ctx = cl.Context(devs)

        # Setup Command Queue:
        self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

        self.mod = self.load_module()
        self.work_inefficient_scan = self.mod.work_inefficient_scan
        self.work_efficient_scan = self.mod.work_efficient_scan
        self.add_block_sums = self.mod.add_block_sums

    def load_module(self):
        kernel = """
            __kernel void work_inefficient_scan(__global long* src, __global long* dst, __global long* aux, __local long* shm, const int N, const int B) {
               
            }

            __kernel void add_block_sums(__global long* dst, __global long* aux, const int N, const int B) {
               
            }
            
            __kernel void work_efficient_scan(__global long* src, __global long* dst, __global long* aux, __local long* shm, const int N, const int B) {
               
            }
        """
        return cl.Program(self.ctx, kernel).build()
    
    @time_it
    def prefix_sum_python(self, a):
        # complete python function
       pass
       
    @time_it
    def prefix_sum_gpu_naive(self, src_h):
        # complete naive method
        pass

    @time_it
    def prefix_sum_gpu_work_efficient(self, src_h):
        # complete work efficient method
        pass
    
    def gpu_scan(self, scanfunc, src_h, double_sized_block = False):
        # initiate stats
        stats = {'elapsed': 0}
        # initiate src_device
        src_d = cl_array.to_device(self.queue, src_h)

        def hierarchical_scan(src_d):
            # complete this function

    def test_prefix_sum_python(self):
        # complete test
        pass
    
    def test_prefix_sum_gpu_naive(self):
        # complete test
        pass

    def test_prefix_sum_gpu_work_efficient(self):
        # complete test
        pass



def plot_stats(title, sizes, pystats, cudastats):
    # plot function 
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
    
    # replicate testing method from pycuda code
    # plot stats

