import numpy as np
import time
n=10000
mat=np.random.randn(n,n)
for i in range(10):
    t1=time.time()
    y=mat@mat
    t2=time.time()
    nop=n**3*2;gflop=nop/(t2-t1)/1e9
    print('multiplied in ',t2-t1,' for ',gflop,' gflops')
