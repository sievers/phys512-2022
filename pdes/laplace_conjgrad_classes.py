import numpy as np
from matplotlib import pyplot as plt
plt.ion()


def apply_stencil(A,do_A=True):
    tot=np.roll(A,1,axis=0)+np.roll(A,-1,axis=0)+np.roll(A,1,axis=1)+np.roll(A,-1,axis=1)
    if do_A:
        return A-tot/4
    else:
        return -tot/4

class Grid:
    def __init__(self,mask,bc):
        self.mask=mask
        self.bc=bc
    def make_rhs(self):
        out=0*bc
        tmp=apply_stencil(self.bc,False)
        out[self.mask==False]=-tmp[self.mask==False]
        return out
    def __matmul__(self,x):
        tmp=x.copy()
        tmp[mask]=0
        tmp=apply_stencil(tmp)
        tmp[mask]=0 #we don't want to worry about points on the boundary
        return tmp

def conjgrad(A,b,x=None,niter=20):
    if x is None:
        x=0*b
    r=b-A@x
    p=r.copy()
    rtr=np.sum(r**2)
    for i in range(niter):
        Ap=A@p
        pAp=np.sum(p*Ap) #why can't we use @ here?
        alpha=rtr/pAp
        x=x+alpha*p
        r=r-alpha*Ap
        rtr_new=np.sum(r**2)
        beta=rtr_new/rtr
        p=r+beta*p
        rtr=rtr_new
        print('current r^2 is ',rtr_new)
    return x
    

def Ax(V,mask):
    #Vuse=np.zeros([V.shape[0]+2,V.shape[1]+2])
    #Vuse[1:-1,1:-1]=V
    Vuse=V.copy()
    Vuse[mask]=0
    ans=apply_stencil(Vuse)
    ans[mask]=0
    #ans=(Vuse[1:-1,:-2]+Vuse[1:-1,2:]+Vuse[2:,1:-1]+Vuse[:-2,1:-1])/4.0
    #ans=ans-V[1:-1,1:-1]

    return ans

def pad(A):
    AA=np.zeros([A.shape[0]+2,A.shape[1]+2])
    AA[1:-1,1:-1]=A
    return AA
n=100

V=np.zeros([n,n])
bc=0*V

mask=np.zeros([n,n],dtype='bool')
mask[:,0]=True
mask[:,-1]=True
mask[0,:]=True
mask[-1,:]=True
#mask[n//2,n//4:(3*n)//4]=True
mask[n//4:n//2,n//4:n//2]=True
bc[n//4:n//2,n//4:n//2]=1.0
mask[n//4:n//2,n//2:(3*n)//4]=True
bc[n//4:n//2,n//2:(3*n)//4]=-1.0

A=Grid(mask,bc)
b=A.make_rhs()
V=conjgrad(A,b,niter=4*n)
Vfull=V.copy()
Vfull[A.mask]=A.bc[A.mask]
rho=apply_stencil(Vfull)
rho_edge=np.sum(np.abs(rho[A.mask]))
rho_int=np.sum(np.abs(rho[A.mask==False]))
print('interior vs. edge charge ratio is ',rho_int/rho_edge)
