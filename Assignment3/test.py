import numpy as np

from modules import convModule
from modules import tester


X_sizes=2**np.arange(4, 14, 2)
K_size = 5
iterations = 50
s
conv_tester = tester(X_sizes=X_sizes, K_size=K_size ,iterations=iterations)


# call tester
conv_tester.testMethod('naive')
# print execution times
conv_tester.printExecutionTimes()
# clear execution time data stored
conv_tester.clearCache()


# repeat for shared and constant_shared

