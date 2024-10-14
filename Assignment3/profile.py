import numpy as np

from modules import convModule
from modules import tester

# choose a size of 1024
X_sizes=2**np.array([10])
# do not change kernel size
K_size = 5
# ensure iterations are set to 1
iterations = 1
# initiate the tester module
conv_tester = tester(X_sizes=X_sizes, K_size=K_size ,iterations=iterations)


# call conv_tester for each method - iterations = 1 (ensures only one copy of kernel is reflected in the profiler report)


#Once script is finished, run it using ncu - use:
# ncu --target-processes all -o <output_name> --set full python profiler.py
# visualize the report using the UI

# Notes about the report
# The report will have three kernels indexed by 0, 1, 2 - order of execution
# they will correspond to naive, shared, constant_shared
# note down which corresponds to what kernel
# compare performance

# Investigate the speedup proposed and reason behind each 


# In the "GPU Speed Of Light Throughput" section of Details" tab, look at the graphs produced:
#    1) Floating Point Operations Roofline
#    2) GPU Throughput

