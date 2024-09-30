# EECS E4750: Heterogeneous Computing for Signal and Data Processing (Fall 2024)

## Assignment-2: Modular Code, Compile Time Arguments and Divergence Analysis

Due date: October 9 2024

Total points: 100

### Introduction

This assignment introduces writing modular code using built in math libraries, generating device functions, and using print statements. It is divided into a programming and theory section.

### Relevant Documentation

1. [Preprocessor Directives and Macros](https://www.informit.com/articles/article.aspx?p=1732873&seqNum=13)
2. [Printing in C](https://cplusplus.com/reference/cstdio/printf/)
3. [Python Raw Strings](https://www.pythontutorial.net/python-basics/python-raw-strings/)
4. [Taylor Series](https://people.math.sc.edu/girardi/m142/handouts/10sTaylorPolySeries.pdf)
5. [Taylor Series Video, if interested](https://www.youtube.com/watch?v=3d6DsjIBzJ4)
6. [CUDA Math Library](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE)

For PyOpenCL:
1. [OpenCL Runtime: Platforms, Devices & Contexts](https://documen.tician.de/pyopencl/runtime_platform.html)
2. [pyopencl.array](https://documen.tician.de/pyopencl/array.html#the-array-class)

For PyCUDA:
1. [Documentation Root](https://documen.tician.de/pycuda/index.html)
2. [Memory tools](https://documen.tician.de/pycuda/util.html#memory-pools)
3. [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)

Additional Readings (if interested):
1. [Floating Point Accuracy](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html)
2. [Kahan Summation](https://en.wikipedia.org/wiki/Kahan_summation_algorithm)

### Additional References
Consult the [git wiki page](https://github.com/eecse4750/e4750_2024Fall_students_repo/wiki) for relevant tutorials.

1. Synchronization:
    1. There are two ways to synchronize threads across blocks in PyCuda:
        1. Using pycuda.driver.Context.synchronize()
        2. Using CUDA Events. Usually using CUDA Events is a better way to synchronize, for details you can go through: https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/ , its a short interesting read.
            1. For using CUDA events you will get an instance of cuda event using cuda.event(), example: event = cuda.event().
            2. You will also time cuda events by recording particular time instances using event.record().
            3. You will synchronize all threads in an event using event.synchronize().
            4. Note: you might need to synchronize event.record() too.
            5. You will use cuda events time_till to record execution time. 
   
    
    2. To Synchronize PyOpenCL kernels you have to use kernel.wait() functionality. PyOpenCL kernels are by default executed as an event afaik.
2. For examples related to PyOpenCL please refer to https://github.com/HandsOnOpenCL/Exercises-Solutions. However there is no such comprehensive list for PyCuda.

3. Note: Sometimes synchronization is not required because of the order of operations you keep, consider the following example:
4. Consider a case in which a kernel call is followed by an enqueue_copy call/ memcpy call from device to host,  in this case you can leave wait()/event.synchronize() out because the kernel call and the copy function call are enqueued in the proper sequence. Also generally copy from device to host call is blocking on host unless you explicitly use async copy in which case you will have to synchronize.
5. In case you get error like: "device not ready" most likely the error is with synchronization.

6. You can also use time.time() to record execution time since the difference is usually very small.


## Programming Problem (80 points)

All the timing, and plots should be taken from running the code in the Cloud Machine. DONOT produce analysis on personal machines.

Your submission should contain 4 files.


1. Report    : E4750.2024Fall.(uni).assignment2.report.PDF   : In PDF format containing information presented at [Homework-Reports.md](https://github.com/eecse4750/e4750_2024Fall_students_repo/wiki/Homework-Reports) , the plots, print and profiling results, and the answers for theory questions.

2. CUDA/OpenCL modules : cudacl_modules.py : In .py format containing all completed methods and kernel codes.

3. PyCUDA Script: pycuda_script.py : In .py format containing the completed script along with comments.

4. PyOpenCL Script : pyopencl_script.py : In .py format containing the completed script along with comments.

Replace (uni) with your uni ID. An example report would be titled E4750.2024Fall.zk2172.assignment2.report.PDF 

### Problem set up

You are given a part of the template code in the main function to iterate between the 1D vectors (Float32 Datatype) having values:

$10^{-3}*(1,2,3,...,N)$ 

with **N** taking different values for different CPU/GPU computation scenarios. The different scenarios to iterate through have been written in the form of a nested for loop with the CPU methods part completed. You are expected to follow the template and complete the code for performing GPU computation.

The programming section contains two tasks.
1. Writing Kernel Code (For PyOpenCL and PyCUDA)
2. Running the Code and Analysing (including Plotting)

You are given a CPU method (**CPU_Sine**) to compare your results to. Please check the end of this README file for the template. It is recommended to refer to the pre filled parts of CUDA Template to supplement with the question context, and implement a similar structure for OpenCL.

#### Task - 1 : Kernel Code (50 points)

You will be using python string concatenation to compile multiple kernel, as shown in the template code. The task is to complete the modular parts of kernel as described

1. *(10 CUDA + 10 OpenCL points = 20 Points)* Create a python string `kernel_main_wrapper` and define a kernel function `main_function` inside it. This main function should

    1. take in an input vector, and
        1. for even index of the input vector, use CUDA/OpenCL built in math functions to compute sine of input argument.
        2. for odd index of the input vector, use a user defined device function `sine_taylor` (described in next sub task) for computing sine of input argument. 
        
    2. Implement print statements of the form (*"Hello from index <array_index>"*) encapsulated inside compile_time arguments as mentioned in the CUDA Template (One example print statement has already been implemented in the template code). The OpenCL Kernel should have the same print statements in the same locations.

    Out of the 10 points each (for CUDA and OpenCL case), the split up is 5 points for correctly computing in `kernel_main_wrapper`, and 5 points for priniting each thread index.
    
2. *(15 CUDA + 15 OpenCL points = 30 Points)* Create a python string `kernel_device` and define a kernel function `sine_taylor` (accepting a float input and returning a float datatype) that computes the sine of input using taylor series approximation upto Q terms, where Q is given as a compile time argument TAYLOR_COEFFS. (Upto the term with $x^(2Q -1) \over (2Q -1)!$.


    Out of the 15 points (for both CUDA and OpenCL case), the split up is 5 Points for variable declaration, 5 points for computation of individual terms, and 5 points for getting final sum of all terms.

#### Task - 2: Analysis (30 points)

This task involves using appropriate compiled kernel to perform different operations in both PyCUDA and PyOpenCL. Complete the GPU methods using explicit memory allocation (using `pycuda.driver.mem_alloc()` in `sine_device_mem_gpu` and `pyopencl.array.to_device` in `deviceSine`). Do not forget to retrieve the result from device memory using the appropriate functions. You will use the variable named `printing_properties` to choose the appropriate kernel to run (look at CUDA Template, end of getSourceModule method and inside the sine_device_mem_gpu method for reference) in both PyOpenCL and PyCUDA. Each sub division carries 5 points (If only one of PyCUDA or PyOpenCL is performed, only 3 out of 5 marks will be awarded in each case).

1. *(5 Points)* For array Sizes $(N = 10,10^2)$ use the kernel compiled in self.module_with_print_nosync (or equivalent in OpenCL) to make the sinusoid computations in GPU, and observe the print messages. Do you see any pattern? Describe the pattern. Why do you think this is so?
2. *(5 Points)* For array Sizes $(N = 10,10^2)$ use the kernel compiled in self.module_with_print_with_sync to make the sinusoid computations in GPU, and observe the print messages. Do you see any pattern? Is it the same as the previous case? Why do you think this is so?
3. *(5 Points)* For array Sizes $(N = 10,10^2,10^3...10^4)$ use the kernel compiled in self.module_no_print to make the sinusoid computations in GPU and time the execution including memory copy. Compare with CPU results (using CPU function in template code). (Use 50 iterations in the main code and take the average). You may use numpy's isclose function for comparing the results - tweak the tol parameter to make observations.
4. *(5 Points)* Change the sine_taylor function to compute for 5 taylor series terms by modifying the `kernel_device` function (Change to #define TAYLOR_COEFFS 5) For array Sizes $(N = 10,10^2,10^3...10^6)$ use the kernel compiled in self.module_no_print to make the sinusoid computations in GPU and time the execution including memory copy. Compare with CPU computation results (using CPU function in template code). (Use 50 iterations in the main code and take the average)
5. *(5 CUDA + 5 OpenCL Points = 10 Points)* Plot timing results from GPU with 10000 Taylor Series Terms, GPU with 5 Taylor Series terms and CPU for array Sizes computed in question 3 and 4 (You can take the time to be 0 for array sizes not computed in Question 3 and plot).

NOTE: You may want to use convert the times to a log scale while plotting for better visibility of effects.

## Theory Problems (20 points)

1. *(5 points)* Cuda provides a "syncthreads" method, explain what is it and where is it used? Give an example for its application? Consider the following kernel code for doubling each vector, will syncthreads be helpful here?

```
__global__ void doublify(float *c_d, const float *a_d, const int len) {
        int t_id =  blockIdx.x * blockDim.x + threadIdx.x;
        c_d[t_id] = a_d[t_id]*2;
        __syncthreads();
}

```

2. *(5 points)* Briefly explain the difference between private memory, local memory & global memory. What happens when you use too much private memory?

3. *(5 points)* Explain the elements of an NVIDIA GPU.
    1. What is a streaming multiprocessor
    2. What are cores?
    3. What are processing blocks?
    4. (2 points) Query the GPU you are using on GCP and state the number of SMs in it.



4. *(5 points)* Explain how threads are assigned to an SM when a kernel is launched. Answer the following:
    
    1. What resource is a thread block assigned to?
    2. What do threads within the same block share?
    3. Explain synchronization within a block.
    4. What is a warp?
    5. How are warps executed?

