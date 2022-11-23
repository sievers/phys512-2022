import numpy as np
from matplotlib import pyplot as plt
plt.ion()


n=100
mask=np.zeros([n,n],dtype='bool')
x=np.linspace(-1,1,n)
xsqr=np.outer(x**2,np.ones(n))
rsqr=xsqr+xsqr.T
R=0.1
mask[rsqr<R**2]=True
mask[:,0]=True
mask[0,:]=True
mask[-1,:]=True
mask[:,-1]=True
bc=np.zeros([n,n])
bc[rsqr<R**2]=1.0


V=np.zeros([n,n])+0.5
V[mask]=bc[mask]
niter=20*n
for i in range(niter):
    neighbors=np.roll(V,1,0)+np.roll(V,-1,0)+np.roll(V,1,1)+np.roll(V,-1,1)
    neighbors=neighbors/4
    rho=V-neighbors
    V=neighbors
    V[mask]=bc[mask]
    if i%10==0:
        plt.clf()
        plt.imshow(rho,vmin=-1e-2,vmax=1e-2)
        plt.pause(0.001)
        plt.show()
