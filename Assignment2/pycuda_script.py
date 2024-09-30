# import required modules

# List all main methods
all_main_methods = ['CPU Sine', 'Sine_device_mem_gpu']
# List the two operations
all_operations = ['No Print', 'Print', 'Sync then Print']
# List the size of vectors
vector_sizes = 10**np.arange(1,3)
# List iteration indexes
iteration_indexes = np.arange(1,50)
# Select the list of valid operations for profiling
valid_operations = all_operations

# Create an instance of the cuda module
graphicscomputer = CudaModule()

for current_operation in valid_operations:
    #TODO: STUDENTS CAN USE THIS SPACE TO WRITE NECESSARY TIMING ARRAYS, PERSONAL DEBUGGING PRINT STATEMENTS, ETC        
    for vector_size in vector_sizes:
        #TODO: STUDENTS CAN USE THIS SPACE TO WRITE NECESSARY TIMING ARRAYS, PERSONAL DEBUGGING PRINT STATEMENTS, ETC

        #THE FOLLOWING VARIABLE SHOULD NOT BE CHANGED
        a_array_np = 0.001*np.arange(1,vector_size+1).astype(np.float32) #Generates an Array of Numbers 0.001, 0.002, ... 

        for iteration in iteration_indexes:
            #TODO: STUDENTS CAN USE THIS SPACE TO WRITE NECESSARY TIMING ARRAYS, PERSONAL DEBUGGING PRINT STATEMENTS, ETC

            for current_method in all_main_methods:
                if(current_method == 'CPU Sine'):
                    #TODO: STUDENTS TO GET OUTPUT TIME AND COMPUTATION FROM CPU_Sine
                else:
                    if(current_method == 'Sine_device_mem_gpu'):
                        #TODO: STUDENTS TO GET OUTPUT TIME AND COMPUTATION FROM sine_device_mem_gpu

                    #TODO: STUDENTS TO COMPARE RESULTS USING ISCLOSE FUNCTION
    #TODO: STUDENTS CAN USE THIS SPACE TO WRITE NECESSARY TIMING ARRAYS, PERSONAL DEBUGGING PRINT STATEMENTS, ETC
    #TODO: Include scirpt used to plot results - comment in great detail and ensure the difference between the plotting code and the rest can be noticed