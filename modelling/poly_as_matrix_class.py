import numpy as np
from matplotlib import pyplot as plt
plt.ion()

x=np.linspace(-2,2,1001)
#m=np.asarray([0,1,0,1],dtype='float')
m=np.asarray([0,0,1,0],dtype='float')
A=np.empty([len(x),4])
for i in range(4):
    A[:,i]=x**i
y=A@m
plt.clf()
plt.plot(x,y)
plt.show()

