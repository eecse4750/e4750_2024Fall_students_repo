# EECS E4750: Heterogeneous Computing for Signal and Data Processing (Fall 2024)

## Assignment-3: 2D kernels, Matrices, Memory Locality optimizations using Shared Memory, Constant Memory, using Modular Programming

Deadline: Wednesday - 23 Oct 2024

Total points: 100

### References to help you with assignment
* [PyCuda Examples](https://github.com/inducer/pycuda/tree/main/examples)
* [NVidia Blog](https://developer.nvidia.com/blog/tag/cuda/)
* [Video about Convolution](https://www.youtube.com/watch?v=A6qR7UhLvng)
* [Preprocessor Directives and Macros](https://www.informit.com/articles/article.aspx?p=1732873&seqNum=13)
* Other references from previous assignments may also be relevant

### Introduction
For this assignment you will implement 2D convolution using GPU. The kernel code will employ varying levels of memory locality optimizations based on compile time defines used. With neither `Shared_mem_optimized` nor `Constant_mem_optimized` compile time defines used, the code should pursue a simple(r) approach which doesn't use shared memory or constant memory. Later you will incorporate shared memory (`Shared_mem_optimized` defined), then constant memory along with shared memory (`Shared_mem_optimized` and `Constant_mem_optimized` defined) in the kernel code using conditional compile.

$NOTE$: The operation being performed is technically correlation and not convolution. The kernel is not "flipped", and directlty moved across the input matrix to generate the output. This is a common misnomer especially in the ML community. Because in most cases it doesn't matter whether a kernel is flipped, and as it is more efficient to apply a kernel as it is, correlation is performed rather than convolution. However, it is still called "convolution".

The algorithm (for a $5\times 5$ filter) can be summarized as:
```math
Y = X * K

```
$X$ is the input matrix

$K$ is the filer (typically called kernel, but we refrain from using that term as we deal with other kernels!!!)

$Y$ is the output

```
Y(i,j) = sum (m = 0 to 4) {
	 sum (n = 0 to 4) { 
	 	X[m][n] * K[i+m-2][j+n-2] 
	 } 
}
```
where 0 <= i < K.height and 0 <= j < K.width

For this assignment you can assume that the elements that are "outside" the matrix K, are treated as if they had value zero. You can assume the kernel size ($5 \times 5$) for this assignment but write your code to work for general odd dimnesion kernel sizes.

Refer to [this link](https://en.wikipedia.org/wiki/Convolution) for more details on convolution.

### Programming Part (70 Points)

All the timing, and plots should be taken from running the code in the Cloud Machine. DO NOT produce analysis on personal machines.

Your submission should contain 5 files.

1. Report : E4750.2024Fall.(uni).assignment3.report.PDF : In PDF format containing information presented at [link](https://github.com/eecse4750/e4750_2024Fall_students_repo/wiki/Homework-Reports)
2. modules.py : completed kernels and modules
3. test.py : completed script to test methods on input vector sizes
4. profile.py : completed script to use for profiling with ncu
5. (uni)_ncu_report.ncu-rep : NCU report generated using profile.py


### Problem set up

Follow the templates to create methods and the kernel code to perform the following


(20 Points) 

1. Write a Kernel function to perform the above convolution operation without using shared memory or constant memory and name it `conv_gpu`. Define a python method `conv_naive` to call this kernel without incorporating the modular compile time defines (calling kernel stored in `self.module_naive_gpu` in the template code)

(20 Points) 

2. Extend this `conv_gpu` function to incorporate shared memory optimization when `Shared_mem_optimized` is defined. Define a python method `conv_shared` to call this kernel incorporating `Shared_mem_optimized` define (calling kernel stored in `self.module_shared_mem_optimized` in the template code).

(10 Points) 

3. Further extend this `conv_gpu` function to incorporate shared memory optimization and constant memory optimization when `Shared_mem_optimized` and `Constant_mem_optimized` are defined. Define a python method `conv_constant_shared` to call this kernel incorporating `Shared_mem_optimized` and `Constant_mem_optimized` defines (calling kernel stored in `self.module_const_mem_optimized` in the template code).

(10 Points) 

4. Complete the conv_scipy method to compute convolution using scipy.signal.convolve2d. Ensure that the output of convolution performed using the kernels is checked using this - the input filter to the convolution kernels needs to be flipped as conv_scipy computed actual convolution.

(10 Points) 

5. Record the time taken to execute convolution including memory transfer operations (do not include kernel only execution times) for the following matrix dimensions: 16 x 16, 64 x 64, 256 x 256, 1024 x 1024, 4096 x 4096. Run each case multiple times (50 iterations) and record the average of the time.

### Theory Problems(30 points) 
(5 points) 
1. Compare the recorded times against the serial implementation for all the above cases (the three methods). Which approach is faster and why?

(15 points) 

2. Profile the three methods with Nsight Compute using profile.py and analyse what you see in the report UI. Investigate the speedup proposed and reason behind. In the "GPU Speed Of Light Throughput" section of Details" tab, look at the graphs produced:
    1. Floating Point Operations Roofline
    2. GPU Throughput
  
To help with profiling, a guide will be update on the student directory soon!

(5 points) 

3. Can this approach be scaled for very big kernels? Why?

(5 points) 

4. Assuming M > N:

    ```
    Code 1: for(int i=0; i<M; i++) for(int j=0; j<N;j++) val = A[i][j];

    Code 2: for(int j=0; j<N; j++) for(int i=0; i<M;i++) val = A[i][j];
    ```

Will the above two codes give the same performance? Why/Why not?

