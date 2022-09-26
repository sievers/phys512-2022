import numpy as np
from matplotlib import pyplot as plt
plt.ion()

x=np.linspace(-2,2,1001)
#m=np.asarray([0,1,0,1],dtype='float')
ord=25
m=np.zeros(ord)
m[2]=1
m[3]=1
#m=np.asarray([0,0,1,1],dtype='float') above does the same, but optionally extra zeros at the end

A=np.empty([len(x),ord])
for i in range(ord):
    A[:,i]=x**i
y_true=A@m
y=y_true+np.random.randn(len(x))

lhs=A.T@A
rhs=A.T@y
e,v=np.linalg.eigh(lhs)
print('ATA condition is ',e.min()/e.max())
mfit=np.linalg.pinv(lhs,rcond=1e-12)@rhs
pred=A@mfit

print('squared residual is ',np.sum((pred-y)**2))

plt.clf()
plt.plot(x,y,'.')
plt.plot(x,pred)
plt.show()

