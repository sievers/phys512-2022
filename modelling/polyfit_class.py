import numpy as np
from matplotlib import pyplot as plt
plt.ion()

x=np.linspace(-2,2,1001)
#m=np.asarray([0,1,0,1],dtype='float')
m=np.asarray([0,0,1,1],dtype='float')
A=np.empty([len(x),4])
for i in range(4):
    A[:,i]=x**i
y_true=A@m
y=y_true+np.random.randn(len(x))

lhs=A.T@A
rhs=A.T@y
mfit=np.linalg.inv(lhs)@rhs
pred=A@mfit

plt.clf()
plt.plot(x,y,'.')
plt.plot(x,pred)
plt.show()

