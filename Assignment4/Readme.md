# EECS E4750: Heterogeneous Computing for Signal and Data Processing (Fall 2024)

## Assignment-4: Prefix Scan + CNNs (what is this?)

Total points: 100
Deadline: November 11, 2024 - 11:59PM 

## References to help you with assignment
### Prefix Scan
* [Kogge-Stone adder](https://en.wikipedia.org/wiki/Kogge%E2%80%93Stone_adder)
* [Brent-Kung adder](https://en.wikipedia.org/wiki/Brent%E2%80%93Kung_adder)
### CNN
* [CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network)

### General References
* [PyCuda Examples](https://github.com/inducer/pycuda/tree/main/examples)
* [PyOpenCL Examples](https://github.com/inducer/pyopencl/tree/main/examples)
* [Nvidia Blog](https://developer.nvidia.com/blog/tag/cuda/)
* [L2 norm](https://www.digitalocean.com/community/tutorials/norm-of-vector-python)

* **Textbook:** Hwu, W. W., Kirk, D. B., El Hajj, I. (2022). Programming Massively Parallel Processors: A Hands-on Approach.
  
Other versions of the textbook are useful as well!

## Prefix Scan
---
For this assignment you will be working on the inclusive parallel scan on a 1D list. 

Mathematically, an inclusive scan operation takes a binary associative operator $\bigoplus$ and an input array of n elements $[x_0, x_1, . . ., x_{n-1}]$, and returns the following output array: $[x_0, (x_0\bigoplus x_1),...,(x_0\bigoplus x_1,. . .\bigoplus x_{n-1})]$

The operator we will use is  the addition $"+"$ operator. There are many uses for all-prefix-sums, including, but not limited to sorting, lexical analysis, string comparison, polynomial evaluation, stream compaction, and building histograms and data structures (graphs, trees, etc.) in parallel. 

Your kernel should be able to handle input lists of arbitrary length. To simplify the task, the input list will be at most of length $134215680$ (which can be split up as $2048 \times 65535$) elements. This means that the computation can be performed using only one kernel launch. 

The boundary condition can be handled by filling “identity value (0 for sum)” into the shared memory of the last block when the length is not a multiple of the thread block size. 


Example: The prefix sum operation on the array $\text{[3 1 7 0 4 1 6 3]}$, would return $\text{[3 4 11 11 15 16 22 25]}$.

>NOTE: You may refer to the **class presentations** on prefix scan to implement the following methods. You can also refer to the **textbook** to read more about the specifics of the algorithm in great detail!

### Naive Implementation

This is a straightforward algorithm to iteratively reduce the input vector and find the resulting vector. There are issues with the naive implementation that will lead to waste of compute resources. To consider these, we perform optimizations and make the kernel work efficient.

### Work Efficient Implementation

One important consideration in analyzing parallel algorithms is work efficiency. The work efficiency of an algorithm refers to the extent to which the work that is performed by the algorithm is close to the minimum amount of work needed for the computation.





## Convolutional Neural Network (CNN) 
---
This is a class on parallel programming, so why care about neural networks?

$${\color{magenta}{\text{BECAUSE APPLICATIONS OF GPU PROGRAMMING IN THIS AREA ARE IMMENSE}}}$$
$${\color{magenta}{\text{Data processing requirements are growing rapildly for AI workloads, and GPUs help bring down the processing time by a LOT}}}$$

<!-- There is an increasing demand for efficient computing because the workloads in AI keep increasing exponentially.  -->




>You do not need any experience in Deep Learning/ Machine Learning to do this part of the Homework.
>We will treat it as a system whose performance we are attempting to optimize - HOW?
>By looking at its constituents
>- Convolutional Layer (performs convolution on input)
>- Activation Layer (applies a function to input)
>- Linear Layer (Matmul + Vector Addition - GEMM!) 

We worked on all these problems in the course until this point!
The goal of this section is to introduce you to performing sequential operations on data using multiple kernels in the context of CNNs and introduce some aspects of its architecutre.





>NOTE📝: We are going to perform a single sequence of simple operations that constitute a CNN.


These are the operations that we will follow:

1) Convolution of input image 
- We pass a single image - three channels into the network, and convolve it with a single filter.
- This operation involves the following steps:
    - Convolution is performed channel wise and the result of each channel is summed
    - The output of the convolution between an image and kernel will be a single matrix, because we are convolving with only a single kernel.  

$$Y = X\circledast K$$


>input shape = (28, 28, 3)
>
>filter shape = (3, 3, 3)
>
>output shape = (28, 28, 1)
>
>How to do this?
>- perform channel wise "same" conv2d
>- add each channel to get output




2) Activation - **relu**
- The output is passed through a relu activation function - $max(0,x)$ for an input $x$
- [relu](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))

$$ Y = ReLU(X) $$
$$Y(i,j) = max(X(i,j), 0) $$


>input shape = (28, 28, 1)
>
>output shape = (28, 28, 1)

  
3) Flatten
- The result is now flattened into a 1D vector
$$ X \text{: append columns to a single row}$$

>input shape = (28, 28, 1)
>
>output shape = (28*28, 1)


4) Fully Connected Layer
- The vector is now passed through a fully connected layer

$$ Y = W\times X + b $$


>W is a matrix and b is a vector
>
>input shape = (28*28, 1)
>
>output shape = (10, 1)


An that's it. We have a CNN ready to be used!
(P.S It is a terrible design, but fun to build for the first time.)

---
### Submission
You need to include four files with your submission:

1) Report: E4750.2024Fall.(uni).assignment4.report.pdf : Self contained report on the results and analysis for the different tasks - guidelines at [link](https://github.com/eecse4750/e4750_2024Fall_students_repo/wiki/Homework-Reports)
2) cnn.py : completed kernels + methods
3) pycuda_scan.py : completed kernels + methods
4) pyopencl_scan.py : completed kernels + methods

---
### Programming Part (70 points)
---
#### Prefix scan (60 points)

**NOTE**📝

>All the following tasks have to be implemented in BOTH PyCUDA and PyOpenCL - points split equally


##### Task 1.1 (Naive Python function) (10 points)
1) Implement a sequential python function for the inclusive prefix scan on a $1D$ list. (6 points)
2) Write two test cases - each is a pair of an input list and the corresponding prefix scan output. Length of each test case must be five. (4 points)

##### Task 1.2 (Parallel Functions in PyCUDA and PyOpencl) (50 points)


1) Implement a naive parallel scan algorithm (20 points)
2) Implement a work efficient parallel scan algorithm (20 points)
3) Write test cases to verify your output with the python sequential algorithm. The lengths to be included in the test case are: (2 points)

>(a) 128
>
>(b) 2048
>
>(c) 128*2048 = 262144
>
>(d) 2048*2048 = 4194304
>
>(e) 2048*65535 = 134215680 

4) For each input test case record the time of execution(total & gpu - methods provided) for the three functions - python, naive, work-efficient. Provide a graph of execution time and compare performance of the algorithms. Also compare the space and time complexities. (8 points)
>You are already given methods to compute execution time with. Please follow the pycuda script given and implement the same in the pyopencl code.
>Plotting function is also provided to you.
>Ensure that you include some analysis of the plot in the report.

#### CNN (10 points)

**NOTE**📝

This section only requires you to use PyCUDA.
There is not serious analysis involved here. The goal is to introduce writing a sequence of kernel operations to create a simple "CNN" system. At the end, you will only show that your module works - pass an input, get an output.

##### Task 2.1
1) Complete kernels provided to you. (5 points)

2) Complete python methods to execute the kernels.
The forward method must include all operations sequentially performed in some input. (3 points)

3) Finally, test the network on a simple input. (1point)

4) Include insights gained through this building process, and any challenges faced. Suggest how to make it better at the kernel level and at the system level. (1 point)

### Theory Problems (30 points) 

1. For the work efficient scan kernel based on reduction trees and inverse reduction trees, assume that we have 2048 elements (each block has BLOCK_SIZE=1024 threads) in each section and warp size is 32, how many warps in each block will have control divergence during the reduction tree phase iteration where stride is 16? For your convenience, the relevant code fragment from the kernel is given below: (5 points)

```
for (unsigned int stride = 1; stride <= BLOCK_SIZE; stride = stride*2) {
    int index = (threadIdx.x+1)*stride*2 - 1;
    if(index < 2*BLOCK_SIZE) {XY[index] += XY[index-stride];}
    __syncthreads();
}
```

(A) 0
(B) 1
(C) 16
(D) 32

2. Consider that NVIDIA GPUs execute warps of 32 parallel threads using SIMT. What's the difference between SIMD and SIMT? What is the worst choice as the number of threads per block to chose in this case among the following and why? (5 points)

(A) 1  
(B) 16  
(C) 32  
(D) 64  

3. What is a bank conflict? Give an example for bank conflict. (5 points) 

4. For the following basic reduction kernel code fragment, if the block size is 1024 and warp size is 32, how many warps in a block will have divergence during the iteration where stride is equal to 1? (5 points)

```
unsigned int t = threadIdx.x;
Unsigned unsigned int start = 2*blockIdx.x*blockDim.x;
partialSum[t] = input[start + t];
partialSum[blockDim.x+t] = input[start+ blockDim.x+t];
for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2)
{
    __syncthreads();
    if (t % stride == 0) {partialSum[2*t]+= partialSum[2*t+stride];}
}
```

(A) 0  
(B) 1  
(C) 16  
(D) 32  


5. Consider the following code for finding the sum of all elements in a vector. The following code doesn't always work correctly explain why? Also suggest how to fix this? (Hint: use atomics.) (5 points) 
```
__global__ void vectSum(int* d_vect,size_t size, int* result){
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    while(tid < size){
        *result+=d_vect[tid];
        tid+=blockDim.x * gridDim.x;
    }
}
```

6. Consider the work-efficient parallel sum-scan algorithm you implemented. If we would like to compute the L2 norm (square root of sum of squares of all elements) of the vector along with the sum-scan results, using the same kernel, what would be the optimal way to implement it? (Assume the time to compute square root in the end is negligible, focus on getting sum of squares of all elements) (5 points)
