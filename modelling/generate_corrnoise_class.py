import numpy as np
from matplotlib import pyplot as plt
plt.ion()

n=500

N=np.empty([n,n])

width=0.1
diag=0.001
for i in range(n):
    for j in range(n):
        dx=i-j
        N[i,j]=1/(1+(dx/width)**2)
    N[i,i]=N[i,i]+diag
L=np.linalg.cholesky(N)

d=L@np.random.randn(n)


x=np.linspace(-5,5,n)
sig=0.2
y=np.exp(-0.5*x**2/sig**2)

d=d+3*y #here is the raw data we would get.  we never look at the noise again
plt.clf()
plt.plot(x,d)
plt.show()
y_org=y.copy()
plt.plot(x,y_org)


#y=np.roll(y,2*width)
y=np.roll(y,50)
plt.plot(x,y)

lhs=y@y
rhs=y@d
amp=rhs/lhs
#print('best-fit noise-free amplitude is ',amp)
Ninv=np.linalg.inv(N)
lhs_good=y@Ninv@y
rhs_good=y@Ninv@d
amp_good=rhs_good/lhs_good
err_good=np.sqrt(1/lhs_good)
print('best-fit amplitude with noise is ',amp_good,' +/- ',err_good)
