"""
Build a simple CNN using pycuda - have fun!
"""
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import gpuarray 

    
INPUT_SHAPE = (28, 28, 3)
FILTER_SHAPE = (3, 3, 3)
NUM_CLASSES = 10


class cudamod:
    def __init__(self):
        self.mod = self.load_module()

    def load_module(self):
        kernel = """
        __global__ void conv2d(float* X, float* K, float* Y, int xwidth, int xsize, int ksize) {
        // computes convolution
        // you may used any - free to use naive from assignment 3!
        }
        
        __global__ void relu(float* X, float* Y, int xwidth, int xheight) {
        // maps input matrix to its relu
        }

        // can include a kernel for flattening input
        
        __global__ void fc(float* X, float* W, float* b, float* Y, int xsize, int ysize) {
        // complete the fully connected kernel
        // maps a vector to a vector through matmul
        }
        """
        return SourceModule(kernel)


cm = cudamod()

class CudaCNN:
    def __init__(self, input_shape=INPUT_SHAPE, filter_shape = FILTER_SHAPE, num_classes=10, cm=cm):
        """
        CNN module implemented with CUDA
        This module only supports inference
        NO training possible
        Args:
            input_shape : shape of input
            num_classes : number of classes 
        """
        self.mod = cm.mod
        self.num_classes = num_classes
        self.conv_func =  self.mod.get_function("conv2d")
        self.relu_func = self.mod.get_function("relu")
        self.fc_func = self.mod.get_function("fc")
        self.init_params()
        
    def init_params(self):
        """
        Initiate network parameters
        Set filter value : K - convolution
        Set FC layer weights : W - fc layer
        Set FC layer bias : b - fc layer
        """
        #self.K
        #self.W
        #self.b
        
        
    def conv2d(self, input_gpu, weights_gpu, out_shape): # you are free to play with the arguments
        """
        out = in(*)k
        Input is seperated into three channels
        Each channel is separately convolved with corresponding filter channel
        The three outputs are then fused together
        This is returned 
        """
        
        return output_gpu
        
        
    def relu(self, input_gpu):
        """
        out = relu(in)
        This layer computes the elementwise relu of the input 
        """
        
        return output_gpu

        
    def flatten(self, input_gpu):
        """
        This layer flattens the input image
        you may either use numpy or create another kernel!
        """
        
        return output_gpu
        
    def fc(self, input_gpu):
        """
        This layer transforms the input vector into another dimension
        out = W*in + b
        """

        return output_gpu

    def forward(self, input_cpu):
        # data transfers to gpu
        # sequentially implement each operation
        """
        1) conv2d
        2) relu
        3) flatten
        4) fully connected
        """
        # get result form gpu
        return output_cpu

    

if __name__ == "__main__":
    """
    In this task, send a sample input of INPUT_SHAPE through the network
    That's all you need to do. Make it work!
    """
    
