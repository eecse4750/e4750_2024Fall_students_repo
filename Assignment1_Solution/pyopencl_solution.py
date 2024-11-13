"""
E4750 2024 Fall
This is the solution for clModule of Assignment 1
"""

import pyopencl as cl
import pyopencl.tools
import pyopencl.array as cl_array
import time
import numpy as np
class clModule:
    def __init__(self):
        """
        **Do not modify this code**
        Attributes for instance of clModule
        Includes OpenCL context, command queue, kernel code.
        """

        # Get platform and device property
        NAME = 'NVIDIA CUDA'
        platforms = cl.get_platforms()
        
        # Returns a list of platform instances and stores it in a string vector called platforms.
        # Basically gets all components on pc that supports and creates a pyopencl.platforms() instance with name platforms.
        # This platforms is a string vector, with many elements, each element being an instance of GPU, or CPU or any other supported opencl platform.
        # Each of these elements obtained using get_platforms() themselves have attributes (defined already on the device like gpu driver binding to PC)
        # These attributes specifies if it is of type CPU, GPU (mentioned in here as device), etc.

        devs = None
        # Initialize devs to None, basically we are creating a null list.
        # Then we go through each element of this platforms vector. Each such element has a method get_devices() defined on it.
        # This will populate the available processors (like number of available GPU threads etc)
        for platform in platforms:
            if platform.name == NAME:
                devs = platform.get_devices()

        # Create Context:
        # A context is an abstraction for parallel computation. pyopencl.context() method operates on input device and generates an instance of context (here we name it ctx)
        # All variables and operations will be bound to a context as an input argument for opencl functions. This way we can choose explicitly which device we want the code to run on through openCL.
        # Here devs contains devices (GPU threads) and hence the context self.ctx holds information of, and operates on GPU threads.
        self.ctx = cl.Context(devs)

        # Setup Command Queue:
        # A command queue is used to explicitly specify queues within a context. Context by itself has methods pass information from host memory to device memory and vice versa.
        # But a queue can be used to have fine grained control on which part of the data should be accessed in which sequence or order, or acts as a control on the data flow.
        # Here pyopencl.CommandQueue takes in input context and sets some properties (used for enabling debugging options etc), creates a commandqueue bound to this context and stores it to self.queue
        self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
        
        # kernel - will not be provided for future assignments!
        # The arguments (output:c and inputs:a,b) stored in global memory are passed with __global type. The other argument n, containing the number of elements is additionally passed
        # with a qualifier const to allow the compiler to optimize for it (it is the same value that is to be passed to each thread)
        # get_global_id will get the Global Item ID (equivalent to thread ID in cuda) for the current instance.
        # The if condition before the actual computation is for bounds checking to ensure threads donot operate on invalid indexes.
        kernel_code = """

            __kernel void Add_two_vectors_GPU(__global float* c, __global float* a, __global float* b, const unsigned int n)
            {
                unsigned int i = get_global_id(0);
                if (i < n) {
                    c[i] = a[i] + b[i];
                }
            }

            __kernel void Add_to_each_element_GPU(__global float* c, __global float* a, __global float* b,  const unsigned int n)
            {
                unsigned int i = get_global_id(0);
                if(i < n){
                    c[i] = a[i] + b[0];
                }
            }
        """ 
        
        # Build kernel code
        # The context (which holds the GPU on which the code should run in) and the kernel code (stored as a string, allowing for metaprogramming if required) are passed onto cl.Program.
        # pyopencl.Program(context,kernelcode).build is similar to sourceModule in Cuda and it returns the kernel function (equivalent of compiled code) that we can pass inputs to.
        # This is stored in self.prg the same way in cuda it is stored in func.
        self.prg = cl.Program(self.ctx, kernel_code).build()

    def deviceAdd(self, a, b, length, is_b_a_vector):
        """
        Function to perform vector addition using the cl.array class
        Arguments:
            a             :   1st Vector
            b             :   number or vector of equal numbers with same length as a
            length        :   length of vector a
            is_b_a_vector :   Boolean Describing if b is a vector or a number
        Returns:
            c       :   vector sum of arguments a and b
            time_   :   execution time for pocl function 
        """
        # [TODO: Students should write code for the entire method for both cases of is_b_a_vector]
        # device memory allocation

        l = np.uint32(length)

        time_start = time.time()
        
        a_gpu = cl_array.to_device(self.queue, a)
        b_gpu = cl_array.to_device(self.queue, b)
        c_gpu = cl_array.empty_like(a_gpu)

        
        
        # execute operation.
        if (is_b_a_vector == False):
            # Use `Add_to_each_element_GPU` Kernel
            func = self.prg.Add_to_each_element_GPU(self.queue, a.shape, None, c_gpu.data, a_gpu.data, b_gpu.data, l)
        else:
            # Use `Add_two_vectors_GPU` Kernel.
            func = self.prg.Add_two_vectors_GPU(self.queue, a.shape, None, c_gpu.data, a_gpu.data, b_gpu.data, l)
        
        # wait for execution to complete.
        func.wait()
        # Copy output from GPU to CPU [Use .get() method]
        c = c_gpu.get(self.queue)
        # Record execution time.
        time_end = time.time()

        time_exec = time_end - time_start
        # return a tuple of output of addition and time taken to execute the operation.
        return c, time_exec

    def bufferAdd(self, a, b, length, is_b_a_vector):
        """
        Function to perform vector addition using the cl.Buffer class
        Returns:
            c               :    vector sum of arguments a and b
            end - start     :    execution time for pocl function 
        """
        # [TODO: Students should write code for the entire method for both cases of is_b_a_vector]
        # Create three buffers (plans for areas of memory on the device)

        l = np.uint32(length)

        time_start = time.time()
        
        a_gpu = cl.Buffer(self.ctx, cl.mem_flags.COPY_HOST_PTR, size=a.nbytes, hostbuf=a)
        b_gpu = cl.Buffer(self.ctx, cl.mem_flags.COPY_HOST_PTR, size=b.nbytes, hostbuf=b)
        c_gpu = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, size=a.nbytes)
        c = np.empty_like(a)

        
        # execute operation.
        if (is_b_a_vector == True):
            # Use `Add_two_vectors_GPU` Kernel.
            func = self.prg.Add_two_vectors_GPU(self.queue, a.shape, None, c_gpu, a_gpu, b_gpu, l)
        else:
            # Use `Add_to_each_element_GPU` Kernel
            func = self.prg.Add_to_each_element_GPU(self.queue, a.shape, None, c_gpu, a_gpu, b_gpu, l)
        
        # Wait for execution to complete.
        func.wait()
        # Copy output from GPU to CPU [Use enqueue_copy]
        cl.enqueue_copy(self.queue, c, c_gpu)
        # Record execution time.
        time_end = time.time()
        # return a tuple of output of addition and time taken to execute the operation.
        time_exec = time_end - time_start
        return c, time_exec

    def CPU_numpy_Add(self, a, b, length, is_b_a_vector):
        """
        Function to perform vector addition on host(CPU) using numpy add method
        Arguments:
            a             :   1st Vector
            b             :   number or vector of equal numbers with same length as a
            length        :   length of vector a
            is_b_a_vector :   Boolean Describing if b is a vector or a number
        """
        start = time.time()
        c = a + b
        end = time.time()

        return c, end - start

    def CPU_Loop_Add(self, a, b, length, is_b_a_vector):
        """
        Function to perform vector addition on host(CPU) using numpy add method
        Arguments:
            a             :   1st Vector
            b             :   number or vector of equal numbers with same length as a
            length        :   length of vector a
            is_b_a_vector :   Boolean Describing if b is a vector or a number
        """

        start = time.time()
        c = np.empty_like(a)
        for index in np.arange(0,length):
            if (is_b_a_vector == True):
                c[index] = a[index] + b[index]
            else:
                c[index] = a[index] + b
        end = time.time()

        return c, end - start