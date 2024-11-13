def getSourceModule(self):
        '''
        Here we have each thread computing each output point (so each thread takes in adjacent point and does elementwise multiplication and sums).
        We have defined 32 x 32 threads per block and mask size as 5 x 5. So Each block in effect will compute (32 + 1 -5) x (32 + 1 -5) ie 28 x 28 outputs,
        so we need a total of (inputmatrix_numrows/28) x (inputmatrix_numcolumns/28) blocks. Blocks and threads are indexed from 0 so it makes the code easier.
        In the conv_gpu_naive code we will be accessing all the elements from the global memory (where all the elements of the whole input matrix are stored)
        so it is accessible to all blocks and grids there. When we need to do shared memory (splitting and storing the input matrix to shared memory), we will
        need to copy things to shared and also need to add additional borders necessary for convolution in each block (we will need to have overlaps between blocks).
        Finally we also try to copy the whole mask into constant memory (to be cached to all cores). Since in C, we cannot generate M[MASK_SIZE][MASK_SIZE]
        array without initializing MASK_SIZE (ie the MASK_SIZE cannot be passed as a variable from our host code), we explicitly define MASK_SIZE as 5 here.
        and then create the array in the GPU code. This will store it into the constant memory in the GPU die from where it will be cached into our GPU cores.
        '''
        '''
        KERNEL CODE FOR CONV_GPU_NAIVE NOTE:
        indexing starts from 0 so the last element is (matrixdimension - 1).
        For the (i,j)th element of the output matrix it will convolve index (0,0) of the mask with ((i-n/2),(j-m/2))th element of the input matrix and the rest
        follows, only the (1+n/2,1+m/2)th element will convolve with (i,j)th element of the input matrix. (note:: n and m are assumed odd here. If they are even,
        then it gets more complicated). It is this mapping that we do by making row_count_ini and column_count_ini functions in our conv_gpu_naive function.
        We initialize ctemp to 0 and then keep adding all these individual sums in a for loop! (parallelization is each thread working on individual output).

        we will do tiling, where we create a tile of memory inside the block for the threads inside the block. It is not necessary for the first kernel but we do it
        anyway to make the whole code more symmetric and uniform (else we will need to generate another helper function for grid dimension for the first kernel alone).
        Our output tile will be smaller than the input tile, with the relation output_tile_size = input_tile_size + 1 - mask_size.
        We input 32x32 elements from memory by using the 32 x 32 threads and output a 28 x 28 collection (32 + 5 - 1 = 28). Notice that the input
        to two blocks should have overlaps by this design! We need to code it in a way to ensure that we include all those into our code!.
        tile_rows and tile_cols are used to represent the size of the output from this one CUDA block.
        Here our tx and ty will vary since we are not using the full size of the blocks
        NOTE:: The cores will be separated and some cores inside a block will be made idle (by the if condition) inside the block.
        The int tx and int ty values use blockIdx.x * tile_cols (instead of blockIdx.x * blockDim.x) and blockIdx.y * tile_rows (instead of blockIdx.y * blockDim.y)
        This is done to map these cores to the proper output memory index (the memory index in global memory). There are gaps in the actual core number or id computed
        using blockDim as some of the cores are made idle and dont output any value for our code. This is a trade off taken to ensure the memory copy
        uses all cores well in our case. The opposite can also be done, and we might be able to make an assessment comparing both for the GPU used. (not part of exercise)
        The final order count is also done with respect to the global memory arrangement and mapping tx and ty values to that. (tx and ty are a function of block dimensions,
        thread sizes, mask sizes and our current positions!). The final multiplication is to be done with "in_matrix_num_cols" for the "row_count_present" since it is
        the way it is arranged in global memory. Global memory holds data structure of a in that way. We need to access that particular element of a.

        NOTE: The if conditions are introduced earlier since I feel that that can save from making the other CPUs accessing and competing for memory.
        The precedence is also chosen in such a way, first it should be within the 28 x 28 from the 32 x 32 so <tile_rows and <tile cols (we use the first
        28 threads x 28 threads in the block to do useful work and relax the remaining (32x32)-(28x28) threads) is introduced first. Then we also introduce
        the condition where the block contains only partial useful threads since the input matrix size is not a whole multiple.

        NOTE:Also notice that row_count_ini will encompass the value of the elements of the whole 32 x 32 that will be needed! (in case of the value being not
        in the input matrix, we will get a negative row_count_ini or row_count_ini>=matrix_row size and we block these cases using if else statements). The ini
        is different from the current element (actually separated by mask/2 distance from the current element).

        cuda uses row major order. Also note that the threadidx.x corresponds to the x axis component of the threadidx which means it corresponds to column number.
        while threadidx.y corresponds to row number. Notice that this is quite confusing and needs to be taken care of. tx = column number, ty = row number.
        Since its row major it traverses entire row (ie all columns x in a row) and then moves to next row. So we have the expression as (row_id*num_cols + col_id)
        [row_count_present*in_matrix_num_cols + column_count_present], where row_count_present depends on row_count_ini which depends on ty (not tx),
        and column_count_present depends on column_count_ini which depends on tx (not ty). In the if loop, i represents the row id of mask. The mask also follows
        row major order so it will be row_id*num_columns + col_id which is (i*in_mask_num_cols + j).
        '''

        kernel_enable_shared_mem_optimizations = """
        #define Shared_mem_optimized
        """

        kernel_enable_constant_mem_optimizations = """
        #define Constant_mem_optimized
        """
        
        kernelwrapper = r"""
        #include<stdio.h>
        #define MASK_SIZE 5
        #define TILE_SIZE
        __constant__ float M[MASK_SIZE][MASK_SIZE];

        #ifndef Constant_mem_optimized
        __global__ void conv_gpu(float *a,float *b, float *c, int in_matrix_num_rows, int in_matrix_num_cols, int in_mask_num_rows, int in_mask_num_cols)
        #endif
        #ifdef Constant_mem_optimized
        __global__ void conv_gpu(float *a, float *c, int in_matrix_num_rows, int in_matrix_num_cols, int in_mask_num_rows, int in_mask_num_cols)
        #endif
        {
            int tile_rows = blockDim.y - in_mask_num_rows + 1;
            int tile_cols = blockDim.x - in_mask_num_cols + 1;

            int ty = blockIdx.y * tile_rows + threadIdx.y;
            int tx = blockIdx.x * tile_cols + threadIdx.x;
            int n = in_mask_num_rows/2;
            int m = in_mask_num_cols/2;
            float temp1;
            float temp2;

            int row_count_ini = ty - n;
            int column_count_ini = tx - m;

            #ifdef Shared_mem_optimized

            __shared__ float Matrix_block[32][32];
            if((row_count_ini>=0)&&(row_count_ini<in_matrix_num_rows)&&(column_count_ini>=0)&&(column_count_ini<in_matrix_num_cols)){
                Matrix_block[threadIdx.y][threadIdx.x] = a[(row_count_ini*in_matrix_num_cols)+column_count_ini];
            }
            else{
                Matrix_block[threadIdx.y][threadIdx.x] = 0.0;
            }

            __syncthreads();

            #endif

            if ((threadIdx.y)<tile_rows &&(threadIdx.x)<tile_cols){
                float ctemp = 0.0;
                if(ty<in_matrix_num_rows && tx<in_matrix_num_cols){
                    int row_count_present;
                    int column_count_present;
                    for(int i=0; i<in_mask_num_rows;i++){
                        row_count_present = row_count_ini + i;
                        if((row_count_present>=0)&&(row_count_present<in_matrix_num_rows)){
                            for(int j=0; j<in_mask_num_cols;j++){
                                column_count_present = column_count_ini + j;
                                if((column_count_present>=0)&&(column_count_present<in_matrix_num_cols)){
                                    #ifndef Shared_mem_optimized
                                    temp1 = a[(row_count_present*in_matrix_num_cols)+column_count_present];
                                    #endif
                                    #ifdef Shared_mem_optimized
                                    temp1 = (Matrix_block[(threadIdx.y)+i][(threadIdx.x)+j]);
                                    #endif
                                    #ifdef Constant_mem_optimized
                                    temp2 = M[i][j];
                                    #endif
                                    #ifndef Constant_mem_optimized
                                    temp2 = b[(i*in_mask_num_cols)+j];
                                    #endif
                                    ctemp += temp1 * temp2;
                                }
                            }
                        }
                    }
                c[ty*in_matrix_num_cols+tx] = ctemp;
                }
            }
        }
        """

        self.module_naive_gpu = SourceModule(kernelwrapper)
        self.module_shared_mem_optimized = SourceModule(kernel_enable_shared_mem_optimizations + kernelwrapper)
        self.module_const_mem_optimized = SourceModule(kernel_enable_shared_mem_optimizations + kernel_enable_constant_mem_optimizations + kernelwrapper)