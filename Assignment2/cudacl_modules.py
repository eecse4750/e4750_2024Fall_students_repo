#import required modules

class CudaModule:
    def __init__(self):
        """
        Attributes for instance of CudaModule
        Includes kernel code and input variables.
        """
        self.threads_per_block_x = 1024 # Students can modify this number.
        self.threads_per_block_y = 1
        self.threads_per_block_z = 1
        self.threads_total = self.threads_per_block_x * self.threads_per_block_y * self.threads_per_block_z

        self.getSourceModule()

    def getSourceModule(self):
        """
        Compiles Kernel in Source Module to be used by functions across the class.
        """
        # define your kernel below.
        kernel_printer_end = """
        #define PRINT_ENABLE_AFTER_COMPUTATION
        """

        kernel_printer = """
        #define PRINT_ENABLE_DEBUG
        """

        kernel_main_wrapper = r"""

        __global__ void main_function(float *input_value, float *computed_value, int n)
        {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            if((idx%2) == 0){
                [TODO]: STUDENTS SHOULD WRITE CODE TO USE CUDA MATH FUNCTION TO COMPUTE SINE OF INPUT VALUE
                #ifdef PRINT_ENABLE_DEBUG
                if(idx<n)
                {
                    printf("Hello from index %d \n", idx);
                }
                #endif
            }
            else{
                [TODO]: STUDENTS SHOULD WRITE CODE TO CALL THE DEVICE FUNCTION sine_taylor TO COMPUTE SINE OF INPUT VALUE
                #ifdef PRINT_ENABLE_DEBUG
                if(idx<n)
                {
                    [TODO]: STUDENTS SHOULD WRITE CODE TO PRINT THE INDEX OF THE ARRAY BEING COMPUTED
                }
                #endif
            }

            #ifdef PRINT_ENABLE_AFTER_COMPUTATION
            if(idx<n)
            {
                [TODO]: STUDENTS SHOULD WRITE CODE TO PRINT THE INDEX OF THE ARRAY BEING COMPUTED
            }
            #endif     
        }
        """

        kernel_device = """
        #define TAYLOR_COEFFS 10000

        __device__ float sine_taylor(float in)
        {
            [TODO]: STUDENTS SHOULD WRITE CODE FOR COMPUTING TAYLOR SERIES APPROXIMATION FOR SINE OF INPUT, WITH TAYLOR_COEFFS TERMS.
        }
        """

        # Compile kernel code and store it in self.module_*

        self.module_no_print = SourceModule(kernel_device + kernel_main_wrapper)
        self.module_with_print_nosync = SourceModule(kernel_printer + kernel_device + kernel_main_wrapper)
        self.module_with_print_with_sync = SourceModule(kernel_printer_end + kernel_device + kernel_main_wrapper)

        # SourceModule is the Cuda.Compiler and the kernelwrapper text is given as input to SourceModule. This compiler takes in C code as text inside triple quotes (ie a string) and compiles it to CUDA code.
        # When we call this getSourceModule method for an object of this class, it will return the compiled kernelwrapper function, which will now take inputs along with block_specifications and grid_specifications.
    
    def sine_device_mem_gpu(self, a, length, printing_properties):
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
        # [TODO: Students should write code for the entire method for all cases of printing_properties]

        # Event objects to mark the start and end points

        # Device memory allocation for input and output arrays

        # Copy data from host to device

        # Call the kernel function from the compiled module
        if(printing_properties == 'No Print'):
            mod = self.module_no_print.get_function("main_function")
        elif(printing_properties == 'Print'):
            mod = self.module_with_print_nosync.get_function("main_function")
        else:
            mod = self.module_with_print_with_sync.get_function("main_function")

        # Get grid and block dim
        
        # Record execution time and call the kernel loaded to the device

        # Wait for the event to complete

        # Copy result from device to the host

        # return a tuple of output of sine computation and time taken to execute the operation.
        pass

 
    def CPU_Sine(self, a, length, printing_properties):
        """
        Function to perform vector addition on host(CPU) using numpy add method
        Arguments:
            a       :   1st Vector
            b       :   number or vector of equal numbers with same length as a
            length  :   length of vector a
        """
        start = time.time()
        c = np.sin(a)
        end = time.time()

        return c, end - start


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
        
        # define your kernel below.
        kernel_printer_end = """
        #define PRINT_ENABLE_AFTER_COMPUTATION
        """

        kernel_printer = """
        #define PRINT_ENABLE_DEBUG
        """

        kernel_main_wrapper = r"""

        __kernel void main_function(float *input_value, float *computed_value, int n)
        {
            TODO: STUDENTS TO WRITE KERNEL CODE MATCHING FUNCTIONALITY (INLCUDING PRINT AND COMPILE TIME CONDITIONS) OF THE CUDA KERNEL.
        }
        """

        kernel_device = """
        #define TAYLOR_COEFFS 10000

        float sine_taylor(float in)
        {
            [TODO]: STUDENTS SHOULD WRITE CODE FOR COMPUTING TAYLOR SERIES APPROXIMATION FOR SINE OF INPUT, WITH TAYLOR_COEFFS TERMS.
        }
        """

        # Compile kernel code and store it in self.module_*

        self.module_no_print = cl.Program(self.ctx, kernel_device + kernel_main_wrapper).build()
        self.module_with_print_nosync = cl.Program(self.ctx, kernel_printer + kernel_device + kernel_main_wrapper).build()
        self.module_with_print_with_sync = cl.Program(self.ctx, kernel_printer_end + kernel_device + kernel_main_wrapper).build()
        
        # Build kernel code
        # The context (which holds the GPU on which the code should run in) and the kernel code (stored as a string, allowing for metaprogramming if required) are passed onto cl.Program.
        # pyopencl.Program(context,kernelcode).build is similar to sourceModule in Cuda and it returns the kernel function (equivalent of compiled code) that we can pass inputs to.
        # This is stored in self.prg the same way in cuda it is stored in func.
        self.prg = cl.Program(self.ctx, kernel_code).build()

    def deviceSine(self, a, b, length, is_b_a_vector):
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
        # [TODO: Students should write code for the entire method]
        # device memory allocation

        # execute operation.
        if(printing_properties == 'No Print'):
            #[TODO: Students to get appropriate compiled kernel]
        elif(printing_properties == 'Print'):
            #[TODO: Students to get appropriate compiled kernel]
        else:
            #[TODO: Students to get appropriate compiled kernel]

        # wait for execution to complete.

        # Copy output from GPU to CPU [Use .get() method]

        # Record execution time.

        # return a tuple of output of addition and time taken to execute the operation.
        pass

    def CPU_Sine(self, a, length, printing_properties):
        """
        Function to perform vector addition on host(CPU) using numpy add method
        Arguments:
            a       :   1st Vector
            b       :   number or vector of equal numbers with same length as a
            length  :   length of vector a
        """
        start = time.time()
        c = np.sin(a)
        end = time.time()

        return c, end - start
