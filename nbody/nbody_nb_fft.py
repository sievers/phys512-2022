import numpy as np
import numba as nb
import time
from matplotlib import pyplot as plt
from scipy import fft


@nb.njit
def inbound_array(xy,n):
    nflt=float(n)
    for i in nb.prange(xy.shape[0]):
        while xy[i,0]<-0.5:
            xy[i,0]+=nflt
        while xy[i,0]>=nflt-0.5:
            xy[i,0]-=nflt

        while xy[i,1]<-0.5:
            xy[i,1]+=nflt
        while xy[i,1]>=nflt-0.5:
            xy[i,1]-=nflt

def inbound_array_np(xy,n):
    xy[xy<-0.5]=xy[xy<-0.5]+n
    xy[xy>=n-0.5]=xy[xy>=n-0.5]-n

def mask_array_np(xy,m,n):
    for i in range(xy.shape[1]):
        m[xy[:,i]<-0.5]=0
        m[xy[:,i]>=n-0.5]=0
        
    
@nb.njit(parallel=True)
def get_grad(xy,pot,grad):
    #n=xy.shape[0]
    n=pot.shape[0]
    for i in nb.prange(xy.shape[0]):
        if xy[i,0]<0:
            ix0=n-1
            ix1=0
            fx=xy[i,0]+1
        else:
            ix0=int(xy[i,0])
            ix1=ix0+1
            fx=xy[i,0]-ix0
            if ix1==n:
                ix1=0
        if xy[i,1]<0:
            iy0=n-1
            iy1=0
            fy=xy[i,1]+1
        else:
            iy0=int(xy[i,1])
            iy1=iy0+1
            fy=xy[i,1]-iy0
            if iy1==n:
                iy1=0
        #potential is f00(1-fx)(1-fy)+f01(1-fx)(fy)+f10(fx)(1-fy)+f11(fx)(fy)
        #grad_x is -f00(1-fy)-f01(fy)+f10(1-fy)+f11(fy)
        #grad_y is -f00(1-fx)+f01(1-fx)-f10(fx)+f11(fx)
        #grad[i,0]=pot[ix0,iy0]*(fy-1)-pot[ix0,iy1]*fy+pot[ix1,iy0]*(1-fy)+pot[ix1,iy1]*ft
        grad[i,0]=(pot[ix1,iy1]-pot[ix0,iy1])*fy+(pot[ix1,iy0]-pot[ix0,iy0])*(1-fy)
        grad[i,1]=(pot[ix1,iy1]-pot[ix1,iy0])*fx+(pot[ix0,iy1]-pot[ix0,iy0])*(1-fx)

        #fy0=pot[ix0,iy0]*(1-fy)+pot[ix0,iy1]*fy
        #fy1=pot[ix1,iy0]*(1-fy)*pot[ix1,iy1]*fy
        #grad[i,0]=fy0*(1-fx)+fy1*fx

        #fx0=pot[ix0,iy0]*(1-fx)+pot[ix1,iy0]*fx
        #fx1=pot[ix0,iy1]*(1-fx)*pot[ix1,iy1]*fx
        #grad[i,1]=fx0*(1-fy)+fx1*fy
        
#@nb.njit
def hist2d(xy,mat):
    nx=xy.shape[0]
    for i in range(nx):
        ix=int(xy[i,0]+0.5)
        iy=int(xy[i,1]+0.5)
        mat[ix,iy]=mat[ix,iy]+1

@nb.njit
def hist2d_wmass(xy,mat,m):
    nx=xy.shape[0]
    for i in range(nx):
        ix=int(xy[i,0]+0.5)
        iy=int(xy[i,1]+0.5)
        if m[i]>0: #we can set m=0 to flag particles
            mat[ix,iy]=mat[ix,iy]+m[i]
    
def get_kernel(n,r0):
    x=np.fft.fftfreq(n)*n
    rsqr=np.outer(np.ones(n),x**2)
    rsqr=rsqr+rsqr.T
    rsqr[rsqr<r0**2]=r0**2
    kernel=rsqr**-0.5
    return kernel


class particles:
    def __init__(self,npart=10000,n=1000,soft=1,periodic=True):
        self.x=np.empty([npart,2])
        self.f=np.empty([npart,2])
        self.v=np.empty([npart,2])
        self.grad=np.empty([npart,2])
        self.m=np.empty(npart)
        self.kernel=[]
        self.kernelft=[]
        self.npart=npart
        self.ngrid=n
        if periodic:
            self.rho=np.empty([self.ngrid,self.ngrid])
            self.pot=np.empty([self.ngrid,self.ngrid])
        else:
            self.rho=np.empty([2*self.ngrid,2*self.ngrid])
            self.pot=np.empty([2*self.ngrid,2*self.ngrid])

        self.soft=soft
        self.periodic=periodic
    def ics_poisson(self):
        self.x[:]=np.random.rand(self.npart,2)*self.ngrid-0.5
        self.m[:]=1
        self.v[:]=0
    def ics_gauss(self):
        self.x[:]=np.random.randn(self.npart,2)*(self.ngrid/12)+self.ngrid/2
        self.m[:]=1
        self.v[:]=0
    def ics_2gauss(self):
        self.x[:]=np.random.randn(self.npart,2)*(self.ngrid/12)+self.ngrid/2
        self.x[:self.npart//2,0]=self.x[:self.npart//2,0]-self.ngrid/5
        self.x[self.npart//2:,0]=self.x[self.npart//2:,0]+self.ngrid/5
        self.m[:]=1
        self.v[:]=0
        self.v[:self.npart//2,1]=25
        self.v[self.npart//2:,1]=-25

    def ics_powlaw(self,ind=-2,amp=0.01):
        vec=np.fft.fftfreq(self.ngrid)*self.ngrid
        rsqr=np.outer(np.ones(self.ngrid),vec**2)
        rsqr=rsqr+rsqr.T
        rsqr[0,0]=1
        ampmat=rsqr**(ind/2)
        ampmat[0,0]=0
        crud=np.random.randn(self.ngrid,self.ngrid)
        crudft=fft.fft2(crud)
        crudft=crudft*ampmat
        crud=np.real(fft.ifft2(crudft))
        crud=crud/np.std(crud)
        m=1+crud*amp
        vv=np.arange(self.ngrid)-0.1
        x,y=np.meshgrid(vv,vv)
        x=np.ravel(x)
        y=np.ravel(y)
        xy=np.vstack([x,y]).T
        self.npart=xy.shape[0]
        self.x=xy
        self.m=np.ravel(m)
        self.v=np.zeros([self.npart,2])
        self.f=np.empty([self.npart,2])
        self.grad=np.empty([self.npart,2])

    def get_kernel(self):
        if self.periodic:
            self.kernel=get_kernel(self.ngrid,self.soft)
        else:
            self.kernel=get_kernel(2*self.ngrid,self.soft)
        self.kernelft=fft.rfft2(self.kernel)
    def get_rho(self):
        if self.periodic:
            inbound_array_np(self.x,self.ngrid)
        else:
            mask_array_np(self.x,self.m,self.ngrid)
        self.rho[:]=0
        hist2d_wmass(self.x,self.rho,self.m)
    def get_pot(self):
        t1=time.time()
        self.get_rho()
        #print('got density: ',time.time()-t1)
        rhoft=fft.rfft2(self.rho)
        #print('got ft 1: ',time.time()-t1)
        n=self.ngrid
        if not(self.periodic):
            n=n*2
        #self.pot=fft.irfft2(rhoft*self.kernelft,[self.ngrid,self.ngrid])
        self.pot=fft.irfft2(rhoft*self.kernelft,[n,n])
        #print('got ft 2: ',time.time()-t1)
    def get_forces(self):
        get_grad(self.x,self.pot,self.grad)
        self.f[:]=self.grad
    def take_step(self,dt=1):
        self.x[:]=self.x[:]+dt*self.v
        self.get_pot()
        self.get_forces()
        self.v[:]=self.v[:]+self.f*dt



parts=particles(npart=10000000,soft=2,periodic=False)

#parts.ics_gauss()
parts.ics_poisson()
#parts.ics_powlaw(ind=-2)
#parts.ics_2gauss()
parts.get_kernel()

plt.ion()
#parts.x=parts.x/2
xy=parts.x.copy()
parts.get_pot()

rho=parts.rho.copy()
pot=parts.pot.copy()
osamp=3


fig = plt.figure()
ax = fig.add_subplot(111)
crap=ax.imshow(parts.rho[:parts.ngrid,:parts.ngrid]**0.5)

for i in range(1500):
    t1=time.time()
    for j in range(osamp):
        parts.take_step(dt=0.02)
    t2=time.time()
    kin=np.sum(parts.v**2)
    pot=np.sum(parts.rho*parts.pot)
    print(t2-t1,kin,pot,kin-0.5*pot)
    #assert(1==0)
    #plt.clf()
    #plt.imshow(parts.rho**0.5)#,vmin=0.9,vmax=1.1)
    #plt.colorbar()


    crap.set_data(parts.rho[:parts.ngrid,:parts.ngrid]**0.5)
    plt.pause(0.001)
