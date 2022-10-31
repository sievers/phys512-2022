import numpy as np
from matplotlib import pyplot as plt

#F=r/(r^2+eps)^(3/2)
#u=r^2+eps, du=2rdr
#=> F=du/2u^3/2, Pot=2u^-1/2=-(r^2+eps)^-1/2
def get_forces(xy,f,soft=0.01,pot=False):
    n=xy.shape[0]
    tot=0
    for i in range(n):
        dx=xy[:,0]-xy[i,0]
        dy=xy[:,1]-xy[i,1]
        rsqr=dx**2+dy**2+soft**2
        rrt=rsqr**-0.5
        r3=rrt**3
        f[i,0]=np.sum(dx*r3)
        f[i,1]=np.sum(dy*r3)
        if pot:
            tot=tot+np.sum(rrt)
    if pot:
        return tot
class particles:
    def __init__(self,n,soft=0.01):
        self.n=n
        self.xy=np.random.randn(n,2)
        self.v=np.zeros([n,2])
        self.f=np.zeros([n,2])
        self.soft=soft
        
    def get_forces(self):
        pot=get_forces(self.xy,self.f,self.soft,pot=True)
        return pot
    def update(self,dt):
        self.xy=self.xy+self.v*dt
        kin1=np.sum(self.v**2)
        pot=self.get_forces()
        self.v=self.v+self.f*dt
        kin2=np.sum(self.v**2)
        kin=0.5*(kin1+kin2)
        print('energies are ',pot/self.n**2,kin/self.n**2,(kin-pot)/self.n**2)
        return kin,pot

parts=particles(4000)
dt=3e-4
plt.ion()
nstep=2000
kk=np.zeros(nstep)
pp=np.zeros(nstep)
for i in range(nstep):
    plt.clf()
    plt.plot(parts.xy[:,0],parts.xy[:,1],'.')
    plt.pause(0.001)
    kk[i],pp[i]=parts.update(dt)

