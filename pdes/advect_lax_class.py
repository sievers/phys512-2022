import numpy as np
from matplotlib import pyplot as plt
plt.ion()

N=300
x=np.zeros(N)
x[N//3:(2*N)//3]=1.0
x_org=x.copy()

a=0.5
b=0.3

for i in range(int(N/a)):
    xx=np.zeros(N+2) #let's make some guard cells
    xx[1:-1]=x
    xx[0]=x[-1]
    xx[-1]=x[0]

    x_cur=b*(xx[2:]+xx[:-2])/2+(1-b)*xx[1:-1]
    grad=(xx[2:]-xx[:-2])/2

    x=x_cur-a*grad
    plt.clf()
    plt.plot(x)
    plt.show()
    plt.pause(0.001)
