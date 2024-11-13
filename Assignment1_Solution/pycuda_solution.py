"""
E4750 2024 Fall
This is the solution for CudaModule of Assignment 1
"""

class CudaModule:
    def __init__(self):
        """
        Attributes for instance of deviceAdd module
        Includes kernel code and input variables.
        """
        # Compile the kernel code when an instance
        # of this class is made. This way it only
        # needs to be done once for the functions
        # you will call from this class.
         self.threads_per_block_x = 1024 # Our current input is 1D so it makes sense to assume a 1D thread block. (other two dimensions are 1)
        # self.threads_per_block_y = 1
        # self.threads_per_block_z = 1
        self.mod = self.getSourceModule()

    def getSourceModule(self):
        """
        Compiles Kernel in Source Module to be used by functions across the class.
        """
        # define your kernel below.
         kernelwrapper = """
        __global__ void Add_to_each_element_GPU(float *a, float *b, float *c, int n)
        {
            int idx = blockDim.x * blockIdx.x + threadIdx.x;
            if (idx < n) c[idx] = a[idx] + b[0];
        }

        __global__ void Add_two_vectors_GPU(float *a, float *b, float *c, int n)
        {
            int idx = blockDim.x * blockIdx.x + threadIdx.x;
            if (idx < n) c[idx] = a[idx] + b[idx];
        }
        """
        return SourceModule(kernelwrapper)

    
    def add_device_mem_gpu(self, a, b, length, is_b_a_vector):
        """
        Function to perform on-device parallel vector addition
        by explicitly allocating device memory for host variables.
        Arguments:
            a                               :   numpy array of size: length
            b                               :   numpy array of size: length
            length                          :   length of numpy arrays 
        Returns
            c                               :   addition result
            time                            :   execution time
        """
         # [TODO: Students should write code for the entire method for both cases of is_b_a_vector]
        nbytes = a.nbytes
        l = np.int32(a.size)
        c = np.empty_like(a)
        
        # Event objects to mark the start and end points
        start_memcpy = cuda.Event()
        start_kernel = cuda.Event()
        end_kernel = cuda.Event()
        end_memcpy = cuda.Event()
        
        # Device memory allocation for input and output arrays
        a_gpu = cuda.mem_alloc(nbytes)
        b_gpu = cuda.mem_alloc(nbytes)
        c_gpu = cuda.mem_alloc(nbytes)

        # record start time - total time including memcpy includes copying from host to device, performing computations, and copying from device back to host
        start_memcpy.record()
        # Copy data from host to device
        cuda.memcpy_htod(a_gpu, a)
        cuda.memcpy_htod(b_gpu, b)

        # NOTE: The following sections (mod, func, block/grid dims) ideally need to be included before recording start - we are currently ignoring the overhead of the cpu definitions.
        mod = self.getSourceModule()
        # Call the kernel function from the compiled module
        if (is_b_a_vector == True):
            # Use `Add_two_vectors_GPU` Kernel.
            func = mod.get_function("Add_two_vectors_GPU")
        else:
            # Use `Add_to_each_element_GPU` Kernel
            func = mod.get_function("Add_to_each_element_GPU")

        # Get grid and block dim
        blockdims = (self.threads_per_block_x, 1, 1)
        griddims = (int(np.ceil(length/self.threads_per_block_x)),1, 1)
        
        # Record execution time and call the kernel loaded to the device
        start_kernel.record()
        event = func(a_gpu, b_gpu, c_gpu, l, block=blockdims, grid=griddims)
        end_kernel.record()
        # Wait for the event to complete
        # event.synchronize() - redundant but can perform
        # Copy result from device to the host
        cuda.memcpy_dtoh(c, c_gpu)
        end_memcpy.record()
        cuda.Context.synchronize() # synchronize the context to ensure that the following time difference calculations are accurate
        # return a tuple of output of addition and time taken to execute the operation.

        time_total = start_memcpy.time_till(end_memcpy)
        time_kernel = start_kernel.time_till(end_kernel)

        return (c, time_total, time_kernel)
    
    def add_host_mem_gpu(self, a, b, length, is_b_a_vector):
        """
        Function to perform on-device parallel vector addition
        without explicitly allocating device memory for 
        host variables.
        Arguments:
            a                               :   numpy array of size: length
            b                               :   numpy array of size: length
            length                          :   length of numpy arrays 
        Returns
            c                               :   addition result
            time                            :   execution time
        """
        # [TODO: Students should write code for the entire method for both cases of is_b_a_vector]
        nbytes = a.nbytes
        l = np.int32(a.size)
        c = np.empty_like(a)

        # Event objects to mark the start and end points
        start_memcpy = cuda.Event()
        end_memcpy = cuda.Event()
        
        mod = self.getSourceModule()
        
        # Get grid and block dim
        blockdims = (self.threads_per_block_x, 1, 1)
        griddims = (int(np.ceil((length)/blockdims[0])), 1)

        # Call the kernel function from the compiled module
        if (is_b_a_vector == True):
            # Use `Add_two_vectors_GPU` Kernel.
            func = mod.get_function("Add_two_vectors_GPU")
        else:
            # Use `Add_to_each_element_GPU` Kernel
            func = mod.get_function("Add_to_each_element_GPU")
            
        # Record execution time and call the kernel loaded to the device
        """
        We are recording total execution time here
        Because we are using the cuda.In/cuda.Out methods to read and write data to/from gpu,
        performing record operations will include mem copy times.
        cuda.In() and cuda.Out() methods are managed by the host
        """
        
        start_memcpy.record()
        func(cuda.In(a), cuda.In(b), cuda.Out(c), l, block=blockdims, grid=griddims)
        end_memcpy.record()
        # Wait for the event to complete
        cuda.Context.synchronize()
        return c, start_memcpy.time_till(end_memcpy)
        # return a tuple of output of addition and time taken to execute the operation.


    def add_gpuarray_no_kernel(self, a, b, length, is_b_a_vector):
        """
        Function to perform on-device parallel vector addition
        without explicitly allocating device memory for 
        host variables (use gpuarray.to_gpu instead) and WITHOUT calling the kernel. The operation
        is defined using numpy-like syntax. 
        Arguments:
            a                               :   numpy array of size: length
            b                               :   numpy array of size: length
            length                          :   length of numpy arrays 
        Returns
            c                               :   addition result
            time                            :   execution time
        """
       # [TODO: Students should write code for the entire method. Sufficient to be able to do for is_b_a_vector == True case alone. Bonus points if is_b_a_vector == False case is solved by passing a single number to GPUarray and performing the addition]
        if(not is_b_a_vector):
            b = np.array(b)
        nbytes = a.nbytes
        l = np.int32(a.size)
        c = np.empty_like(a)

        # Event objects to mark start and end points
        start_memcpy = cuda.Event()
        start_kernel = cuda.Event()
        end_kernel = cuda.Event()
        end_memcpy = cuda.Event()
        
        
        # Allocate device memory using gpuarray class 
        start_memcpy.record()
        
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
     
        # Record execution time and execute operation with numpy syntax
        start_kernel.record()
        c_gpu = a_gpu + b_gpu
        end_kernel.record()
        # Wait for the event to complete

        # Fetch result from device to host
        c = c_gpu.get()
        end_memcpy.record()

        cuda.Context.synchronize()               
        # return a tuple of output of addition and time taken to execute the operation.
        
        total_time = start_memcpy.time_till(end_memcpy)
        kernel_time = start_kernel.time_till(end_kernel)

        return c, total_time, kernel_time
        
    def add_gpuarray(self, a, b, length, is_b_a_vector):
        """
        Function to perform on-device parallel vector addition
        without explicitly allocating device memory for 
        host variables (use gpuarray.to_gpu instead). In this scenario make sure that 
        you call the kernel function.
        Arguments:
            a                               :   numpy array of size: length
            b                               :   numpy array of size: length
            length                          :   length of numpy arrays 
        Returns
            c                               :   addition result
            time                            :   execution time
        """
        # [TODO: Students should write code for the entire method. Sufficient to be able to do for is_b_a_vector == True case alone. Bonus points if is_b_a_vector == False case is solved by passing a single number to GPUarray and performing the addition]
        if(not is_b_a_vector):
            b = np.array(b)
        nbytes = a.nbytes
        l = np.int32(a.size)
        c = np.empty_like(a)
        mod = self.getSourceModule()
        # Create cuda events to mark the start and end of array.
        start_memcpy = cuda.Event()
        start_kernel = cuda.Event()
        end_kernel = cuda.Event()
        end_memcpy = cuda.Event()
        
        # Get function defined in class defination
        if (is_b_a_vector == True):
            # Use `Add_two_vectors_GPU` Kernel.
            func = mod.get_function("Add_two_vectors_GPU")
        else:
            # Use `Add_to_each_element_GPU` Kernel
            func = mod.get_function("Add_to_each_element_GPU")

        # Get grid and block dim
        blockdims = (self.threads_per_block_x, 1, 1)
        griddims = (int(np.ceil((length)/blockdims[0])), 1)

        # Allocate device memory for a, b, output of addition using gpuarray class     
        start_memcpy.record()
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c_gpu = gpuarray.to_gpu(c)
        # NOTE - a better declaration would be to declare c_gpu before start_memcpy.record() as we do not care about the contents
        # For this assignment, this is ignored

        
        # Record execution time and execute operation
        start_kernel.record()
        func(a_gpu, b_gpu, c_gpu, l, block=blockdims, grid=griddims)
        end_kernel.record()
        # Wait for the event to complete
        
        # Fetch result from device to host
        c = c_gpu.get()
        end_memcpy.record()
        cuda.Context.synchronize()
            
        # return a tuple of output of addition and time taken to execute the operation.
        total_time = start_memcpy.time_till(end_memcpy)
        kernel_time = start_kernel.time_till(end_kernel)

        return c, total_time, kernel_time