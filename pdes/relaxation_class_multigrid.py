import numpy as np
from matplotlib import pyplot as plt
plt.ion()

def average_neighbors(V):
    tot=np.roll(V,1,0)+np.roll(V,-1,0)+np.roll(V,1,1)+np.roll(V,-1,1)
    return tot/4

def deres(map):
    tmp=np.maximum(map[::2,::2],map[1::2,::2])
    tmp=np.maximum(tmp,map[::2,1::2])
    tmp=np.maximum(tmp,map[1::2,1::2])
    return tmp

def upres(map):
    n=map.shape[0]
    tmp=np.zeros([2*n,2*n])
    tmp[::2,::2]=map
    tmp[1::2,::2]=map
    tmp[::2,1::2]=map
    tmp[1::2,1::2]=map
    return tmp

def laplace(bc,mask,niter,V=None):
    if V is None:
        V=0*bc
    V[mask]=bc[mask]
    for i in range(niter):
        V=average_neighbors(V)
        V[mask]=bc[mask]
    return V

n=1024
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

npass=5
bc_vec=[None]*(npass+1)
mask_vec=[None]*(npass+1)
bc_vec[0]=bc
mask_vec[0]=mask
V_vec=[None]*(npass+1)
rho_vec=[None]*(npass+1)

for i in range(npass):
    bc_vec[i+1]=deres(bc_vec[i])
    mask_vec[i+1]=deres(mask_vec[i])


V_vec[-1]=laplace(bc_vec[-1],mask_vec[-1],1500)
rho_vec[-1]=V_vec[-1]-average_neighbors(V_vec[-1])
for i in range(npass-1,-1,-1):
    V0=upres(V_vec[i+1])
    V_vec[i]=laplace(bc_vec[i],mask_vec[i],100,V0)
    rho_vec[i]=V_vec[i]-average_neighbors(V_vec[i])
    print(i)
assert(1==0)

V=np.zeros([n,n])+0.5
V[mask]=bc[mask]
niter=20*n
for i in range(niter):
    neighbors=np.roll(V,1,0)+np.roll(V,-1,0)+np.roll(V,1,1)+np.roll(V,-1,1)
    neighbors=neighbors/4
    rho=V-neighbors
    V=neighbors
    V[mask]=bc[mask]
    if i%10==1:
        plt.clf()
        #plt.imshow(rho,vmin=-1e-2,vmax=1e-2)
        plt.imshow(V)
        plt.pause(0.001)
        plt.show()
