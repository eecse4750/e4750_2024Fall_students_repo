# EECS E4750: Heterogeneous Computing for Signal and Data Processing (Fall 2024)

## Assignment-1: Introduction to memory management and profiling in PyCUDA and PyOpenCL.

Due Date: See in Courseworks

Total points: 100

### Goal

The goal of the assignment is to compare and contrast the different method(s) of host-to-device memory allocation for simple elementwise operations on vector(s) and introduction to profiling. The assignment is divided into a programming section, and a theory section. The programming section contains tasks for (Py)CUDA, and tasks for (Py)OpenCL. 

### Relevant Documentation

For PyOpenCL:
1. [OpenCL Runtime: Platforms, Devices & Contexts](https://documen.tician.de/pyopencl/runtime_platform.html)
2. [pyopencl.array](https://documen.tician.de/pyopencl/array.html#the-array-class)
3. [pyopencl.Buffer](https://documen.tician.de/pyopencl/runtime_memory.html#buffer)

For PyCUDA:
1. [PyCUDA Documentation](https://documen.tician.de/pycuda/index.html)
2. [Memory tools](https://documen.tician.de/pycuda/util.html#memory-pools)
3. [gpuarrays](https://documen.tician.de/pycuda/array.html)

### Hints


This [git wiki page](https://github.com/eecse4750/e4750_2024Fall_students_repo/wiki) has relevant tutorials to help you get started.

These concepts will help you write code that executes the expected outputs:
1. Synchronization:

As you read more on this, think about why it is an important part of a code for a heterogeneous system and also specifically how you will use it for this assignment.

    1. There are two ways to synchronize threads across blocks in PyCuda:
        1. Using pycuda.driver.Context.synchronize()
        2. Using CUDA Events. Usually using CUDA Events is a better way to synchronize, for details you can go through: [https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/] , its a short interesting read.
            1. For using CUDA events you will get an instance of cuda event using cuda.event(), example: event = cuda.event().
            2. You will also time cuda events by recording particular time instances using event.record().
            3. You will synchronize all threads in an event using event.synchronize().
            4. For example you can refer to a brief part of assignment solution I showed during my recitation :). There is a minor issue with that code though but the idea is correct.
            5. Note: you need to synchronize event.record() too.
            6. You will use cuda events time_till to record execution time. 
   
    
    2. To Synchronize PyOpenCL kernels you have to use kernel.wait() functionality. PyOpenCL kernels are by default executed as an event afaik.
    
2. For examples related to PyOpenCL please refer to https://github.com/HandsOnOpenCL/Exercises-Solutions. 



## Programming Problems - (80 points)

Your submission should contain five files. Modify the provided templates and rename accordingly.

1. Report    : E4750.2024Fall.(uni).assignment1.report.pdf   : In PDF format containing information presented at (Homework-Reports.md) , the plots, print and profiling results, and the answers for theory questions.
   
2. PyCUDA solution   : E4750.2024Fall.(uni).assignment1.PyCUDA.py  : In .py format containing the methods, along with comments.

3. CudaModule solution   : E4750.2024Fall.(uni).assignment1.CudaModule.py  : In .py format containing the methods and kernel codes, along with comments.

4. PyOpenCL solution : E4750.2024Fall.(uni).assignment1.PyOpenCL.py  : In .py format containing the methods, along with comments.

4. clModule solution : E4750.2024Fall.(uni).assignment1.clModule.py  : In .py format containing the methods and kernel codes, along with comments.

Replace (uni) with your uni ID. An example report would be titled E4750.2024Fall.zk2172.assignment1.report.pdf.

Detailed instructions about each section are included with the code templates and problem setup.


## Problem Setup

You will perform parallel vector addition in this assignment. The goal is to help you get comfortable with PyCUDA and PyOpenCL while solving an introductory problem whose execution on a GPU is straightforward. You will be introduced to essential aspects of computing with CUDA and OpenCL through their Python wrappers (the concepts used in the wrappers translate to the respective functions in the native programs). In addition to this, you will profile the programs to evaluate the effectiveness of introducing parallelism. You will profile using two methods—one using timestamps and the other using NVIDIA's Nsight application.

The template given in the main function iterates through a 1D vector - $[1,2,..,N]$, with $N$ taking values of $(10,10^2,10^3...10^8)$ for different CPU/GPU implementations of vector addition. Note that the vectors used are numpy arrays with the float32 datatype. (What is the reason for using float32 type ?)

The CPU methods are provided directly and you are expected to complete the GPU methods. 
### CPU Methods
There are two version of the CPU code
1. `CPU_numpy_Add` : Numpy addition of vectors
2. `CPU_Loop_Add` : Addition of a scalar to a vector using a for loop

### GPU Methods
The GPU methods differ slightly for pycuda and pyopencl
1. Each includes a direct utilization of kernels with

### Tasks

#### Kernels

`This section is worth 15 points`

You will write kernels for both PyCUDA and PyOpenCL solutions. There are four kernels that you have to complete in total; two each for PyCUDA and PyOpenCL - `Add_two_vectors_GPU` and `Add_to_each_element_GPU`.

One kernel has been completed for your reference. The other three are to be finished by you and each carries 5 points.

1. `Add_two_vectors_GPU`: A vector matching the input length containing the same scalar for each index $b$ is to be added to the input.
2. `Add_to_each_element_GPU`: A scalar $b$ is to be added to the input vector.

Note: The scalar $b$ is of type float32 (Think about why this is)

In the following section, the `is_b_a_vector` will be used to choose which kernel is used for computation.

#### PyCUDA code 

`This section is worth 40 points`

In each scenario, the `SourceModule` functionality will be used to compile the written kernel, that will be used to perform the parallel computation on the device.


There are four methods to be completed 
1. *(10 points)* `add_device_mem_gpu`:
   
    Both (`Add_two_vectors_GPU`, `Add_to_each_element_GPU`) operations will be performed using explicit device memory allocations using `pycuda.driver.mem_alloc()`. Note that you need to retieve the computed result from the device to the host using the appropriate API. Record the following:

   i. Execution time including memory transfer (HostToDevice copy + Kernel + DeviceToHost copy).
   
   ii. Execution time excluding memory transfer (Only Kernel). 
2. *(10 points)* `add_host_mem_gpu`:

   Both operations will be performed without explicit device memory allocations (Which method of pycuda.driver will do this?).
   Record the following:

   i. Execution time including memory transfer.

    Why not execution time without memory transfer?
   
3. *(4 points)* `add_gpuarray_kernel`:

    Complete this method using the `gpuarray` functionality instead of `mem_alloc` while performing computation **with** the kernel written by you. For the  `Add_to_each_element_GPU` operation, can you directly use the numpy scalar value, or do you have to convert it to an array? Record the following:
   
   i. Execution time including memory transfer (HostToDevice copy + Kernel + DeviceToHost copy).
   
   ii. Execution time excluding memory transfer (Only Kernel). 


4. *(4 points)* `add_gpuarray_no_kernel`:

    Complete this method using the `gpuarray` functionality instead of `mem_alloc` while performing computation **without** the kernel written by you (Use numpy like syntax - add vectors sent to gpu directly). Record the following:
   
   i. Execution time including memory transfer (HostToDevice copy + Kernel + DeviceToHost copy).
   
   ii. Execution time excluding memory transfer (Only Kernel). 

Compare the results with `CPU_numpy_Add`. 

In addition to these methods, also complete the profiling section of the code.


5. Call each of the eight `GPU` cases and two `CPU` cases one by one for each vector length $(10,10^2,..,10^6)$. Run loops for each operation, for each vector size, for each iteration, for each method. The following order of precedence is to be followed in the loop.

```python
for operation in operations: #(add vector/ add scalar)
    for size in vector sizes: # (10^1,10^2,...,10^N)
        for iter in iterations: #(set as 50)
            for method in methods: #(GPU/CPU)
                # compute and save execution time
       #compute average time
```
6. Now exclude the slower of the two CPU cases and extend the same analysis for vector lengths $(10,10^2,..,10^8)$

7. *(4 points)* Plot the average execution times, including memory transfer for GPU operations, against vector size (display power of 10 in xticks ranging from $1$ to $8$).

8. *(4 points)* Plot the average execution times, excluding memory transfer for GPU operations, against vector size (display power of 10 in xticks ranging from $1$ to $8$).

9. *(4 points)* Profile `add_device_mem_gpu` for both operations - `Add_to_each_element_GPU` and `Add_two_vectors_GPU` kernels. Explain the differences seen. For profiling use NVIDIA Nsight Compute CLI. Also visualize generated reports with Nsight Compute UI. 

#### PyOpenCL code

`This section is worth 25 points`

There are two methods to be completed.

1. *(10 points)* `deviceAdd`:

Complete the method to perform both operations (`Add_two_vectors_GPU`, `Add_to_each_element_GPU`) using `pyopencl.array` to load inputs to device memory. Record execution time including memory transfers.

2. *(10 points)* `bufferAdd`:

Complete the method to perform both operations using `pyopencl.Buffer` to load inputs to device memory. Record execution time including memory transfers.

In addition to these, complete the profiling section of the code

3. Call each of the four `GPU` cases and two `CPU` cases one by one for each vector length $(10,10^2,..,10^6)$. Run loops for each operation, for each vector size, for each iteration (set max iterations as $50$), for each method. The following order of precedence is to be followed in the loop.

```python
for operation in operations: #(add vector/ add scalar)
    for size in vector sizes: # (10^1,10^2,...,10^N)
        for iter in iterations: #(set as 50)
            for method in methods: #(GPU/CPU)
                # compute and save execution time
       #compute average time
```

4.  Now exclude the slower of the two CPU cases and extend the same analysis for vector lengths $(10,10^2,..,10^8)$. There are now five cases - four GPU + one CPU.

5. *(5 points)* Plot the average execution times for the four GPU cases and the faster CPU case against vector size (display power of 10 in xticks ranging from $1$ to $8$).




The code templates are provided in the files in the repository. The files `PyCUDA.py` and `PyOpenCL.py` include a template code for profiling the kernels with different operations and methods. The `CudaModule.py` and `clModule.py` files contain templates for the class definitions of the pycuda and pyopencl vector addition modules.

## Theory - (20 points)

You may use any resource for this independent study.

1. *(3 points)* Explain the concepts of threads, tasks, and processes. (For both CPUs and GPUs). 
2. *(2 points)* Which CPU method is faster and why?
3. *(4 points)* Explain how parallel approaches in PyCUDA and PyOpenCL compare against CPU methods.
4. *(4 points)* Explain which method in PyOpenCL proved to be faster and why. You may refer to the documentation.
5. *(4 points)* Explain which method in PyCUDA proved to be faster and why. You may refer to the documentation.
6. *(3 points)* Between PyCUDA and PyOpenCL, which do you prefer? State your reasons. 


 
