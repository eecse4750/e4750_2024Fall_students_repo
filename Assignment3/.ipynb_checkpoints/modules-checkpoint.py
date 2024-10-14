# import required modules


# module that performs convolution2D
class convModule():
    def __init__(self):

        # can you have a size > 32?
        self.threadsX = 32
        self.threadsY = 32
        self.threadsZ = 1
        self.blockDims = (self.threadsX, self.threadsY, self.threadsZ)
        self.grid_dimension_function = lambda a, b, c, d, e, f : (int(np.ceil(a/(b+1-c))),int(np.ceil(d/(e+1-f))))
        self.getSourceModule()

    def getSourceModule(self):

        kernel_enable_shared_mem_optimizations = """
        #define Shared_mem_optimized
        """
        kernel_enable_constant_mem_optimizations = """
        #define Constant_mem_optimized
        """

        kernelwrapper = r"""

		[TODO: Students to write entire kernel code. An example of using the ifdef and ifndef is shown below. The example can be modified if necessary]

        #ifndef Constant_mem_optimized
        __kernel void conv_gpu(__global float* a, __global float* b, __global float* c, const unsigned int in_matrix_num_rows, const unsigned int in_matrix_num_cols, const unsigned int in_mask_num_rows, const unsigned int in_mask_num_cols)
        #endif
        #ifdef Constant_mem_optimized
        __kernel void conv_gpu(__global float* a, __constant float* mask, __global float* c, const unsigned int in_matrix_num_rows, const unsigned int in_matrix_num_cols, const unsigned int in_mask_num_rows, const unsigned int in_mask_num_cols)
        #endif
        {
            [TODO: Perform required tasks, likely some variable declaration, and index calculation, maybe more]

            #ifdef Shared_mem_optimized

			[TODO: Perform some part of Shared memory optimization routine, maybe more]

            #endif

			[TODO: Perform required tasks, mostly relating to the computation part. More #ifdef and #ifndef can be added as necessary]
        }

"""
        self.module_naive_gpu = SourceModule(kernelwrapper)
        self.module_shared_mem_optimized = SourceModule(kernel_enable_shared_mem_optimizations + kernelwrapper)
        self.module_const_mem_optimized = SourceModule(kernel_enable_shared_mem_optimizations + kernel_enable_constant_mem_optimizations + kernelwrapper)
        




    def conv_naive(self, X, K, Xshape, Kshape):

        # allocate host memory to store output of convolution
       
        
		# get function from module
        
		# declare events to perform time stamping
        
		# initiate dimensions
        
        
        # allocate memory in gpu
        
        # copy data

		# invoke kernel
        
		# copy data from gpu to host
        
		# record end of execution - synchronize??
        
        #compute time of eexcution
        
		# return values
        

    def conv_shared(self, X, K, Xshape, Kshape):

		# same as conv_naive

    def conv_constant_shared(self, X, K, Xshape, Kshape):

		# same as conv_naive
		
    
    def conv_scipy(self, X, K):

		# function to compute convolution between input X and filter K using inbuilt scipy function

		# record time of execution
        
		# perform scipy conv
       
		# compute time of execution
      
		# return results
    


class tester():
    def __init__(self, X_sizes=2**np.arange(4, 14, 2), K_size=5, iterations=10):
        self.X_sizes = X_sizes
        self.K_size = 5
        self.iterations = iterations
        self.cm = convModule()
        self.avgtime_cpu = []
        self.avgtime_gpu = []

    # generate random matrices to test convolution on
    def getRandMatrices(self, Xsize):
        Xshape = (Xsize, Xsize)
        Ksize = self.K_size
        Kshape = (Ksize, Ksize)
        X = np.random.random(Xshape).astype(np.float32)
        K = np.random.random(Kshape).astype(np.float32)\
        # generate flipped matrix to input to the kernels
        # Kflipped = np.flip( fill in the details ).astype(np.float32)
        return X, K, Kflipped

    # print avg execution times stored
    def printExecutionTimes(self):
        if(len(self.avgtime_cpu)==0 or len(self.avgtime_gpu)==0):
            raise Exception(f"No time computed...please run tester on sample data!")
        else:
            cpu_times = (self.avgtime_cpu)
            cpu_times = [f"{time:.3f}" for time in cpu_times]
            gpu_times = (self.avgtime_gpu)
            gpu_times = [f"{time:.3f}" for time in gpu_times]
            print(f"The average execution times for vector sizes {self.X_sizes} are:\n")
            print(f"CPU times:\n{cpu_times}")
            print(f"GPU times:\n{gpu_times}")

    def clearCache(self):
        self.avgtime_cpu = []
        self.avgtime_gpu = []
            
    def testMethod(self, method='naive', prints=False):
        # use prints to enable/disable print()
        if(method=='naive'):
            test_func = self.cm.conv_naive
        elif(method=='shared'):
            test_func = self.cm.conv_shared
        elif(method=='constant_shared'):
            test_func = self.cm.conv_constant_shared
        else:
            raise Exception(f"Method \"{method}\" is not recognized. Allowed methods - \"naive\", \"shared\", \"constant_shared\" ")
        print("\n"+"*"*30)
        print(f"Testing {method}")
        print("*"*30+"\n")
        for size in self.X_sizes:
            if(prints):
                print(f'Testing {method} for X shape = {size}*{size}\n')
            times_cpu = []
            times_gpu = []
            for i in range(self.iterations):
                # get X, K, Kflipped
                
                # call functions based on method
                # store Y_gpu, Y_cpu, time_gpu, time_cpu
                
                # compute valid - np.allclose(Y_cpu, Y_gpu)
                if(not valid):
                    # compute diff mean and throw error
                    diff_mean = np.mean(Y_cpu-Y_gpu)
                    print(f'\tHuge error encountered between CPU and GPU computations!\n\tAvg diff = {diff_mean}\n')
                    raise Exception('Mismatch!! Error in computation.\n')
                


				# append times_cpu
                
				# append times_gpu
                



			# compute_average stats per method - toggle print using "prints"
            
            if(prints):
                print(f'\tavg cpu time = {avg_time_cpu}\n')
            
			# append results for vector size to self.avgtime_cpu, self.avgtime_gpu
         
            if(prints):
                print(f'\tavg gpu time = {avg_time_gpu}\n')

    
            
    		
        
        

