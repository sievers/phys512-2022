import numpy as np
from matplotlib import pyplot as plt

def average_neighbors(mat):
    out=0*mat
    out=out+np.roll(mat,1,0)
    out=out+np.roll(mat,-1,0)
    out=out+np.roll(mat,1,1)
    out=out+np.roll(mat,-1,1)
    return out/4

class Grid:
    def __init__(self,bc,mask):
        self.bc=bc
        self.mask=mask
    def make_rhs(self):
        rhs=average_neighbors(self.bc)
        rhs[self.mask]=0 #we need to zero out the RHS on the boundary
        return rhs
    def __matmul__(self,x):
        x[self.mask]=0
        ave=average_neighbors(x)
        ave[self.mask]=0
        return x-ave

def conjgrad(A,b,x=None,niter=100,plot=False):
    if x is None:
        x=0*b
    r=b-A@x
    p=r.copy()
    rtr=np.sum(r**2)
    for i in range(niter):
        Ap=A@p
        #pAp=p@Ap #wrong if p,Ap aren't already vectors!
        pAp=np.sum(p*Ap)
        alpha=rtr/pAp
        x=x+alpha*p
        r=r-alpha*Ap
        rtr_new=np.sum(r**2)
        beta=rtr_new/rtr
        p=r+beta*p
        print('my current residual squared is ',rtr_new)
        rtr=rtr_new
        if plot:
            plt.clf()
            plt.imshow(x)
            plt.show()
            plt.pause(0.001)
    return x

n=200
mask=np.zeros([n,n],dtype='bool')
bc=np.zeros([n,n])
x=np.linspace(-1,1,n)
xsqr=np.outer(x**2,np.ones(n))
rsqr=xsqr+4*xsqr.T
mask[:,0]=True
mask[0,:]=True
mask[-1,:]=True
mask[:,-1]=True

if False:
    R=0.1
    mask[rsqr<R**2]=True
    bc[rsqr<R**2]=1.0
    mask[n//2,n//3]=True
else:
    bc[2*n//5,n//4:(3*n)//4]=1.0
    bc[3*n//5,n//4:(3*n)//4]=-1.0
    mask[2*n//5,n//4:(3*n)//4]=True
    mask[3*n//5,n//4:(3*n)//4]=True


A=Grid(bc,mask)
b=A.make_rhs()

x=conjgrad(A,b,niter=3*n)
V=x.copy()
V[A.mask]=A.bc[A.mask]
Ex=V-np.roll(V,-1,0)
Ey=V-np.roll(V,-1,1)

rho=V-average_neighbors(V)
