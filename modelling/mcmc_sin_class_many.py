import numpy as np
from matplotlib import pyplot as plt
plt.ion()

def sinfun(p,x):
    omega=p[0]
    phi=p[1]
    amp=p[2]
    y=amp*np.sin(omega*x+phi)
    dydomega=amp*np.cos(omega*x+phi)*x
    dydphi=amp*np.cos(omega*x+phi)
    dydamp=np.sin(omega*x+phi)
    A=np.empty([len(x),len(p)])
    A[:,0]=dydomega
    A[:,1]=dydphi
    A[:,2]=dydamp

    return y,A

def chisq_fun(p,x,y):
    pred,grad=sinfun(p,x)
    chisq=np.sum((pred-y)**2)
    return chisq

x=np.linspace(-np.pi,np.pi,1001)
phi=np.pi/4
amp=1.5
omega=2.0
y_true=np.sin(omega*x+phi)*amp
y=y_true+np.random.randn(len(x))

plt.clf()
plt.plot(x,y,'.')
plt.show()

p=np.asarray([0.9*omega,phi+0.1,1.1*amp])

pred,grad=sinfun(p,x)
plt.plot(x,pred)

for iter in range(10):
    pred,A=sinfun(p,x)
    r=y-pred
    #pretend noise is 1
    lhs=A.T@A
    rhs=A.T@r
    dp=np.linalg.inv(lhs)@rhs
    p=p+dp
    #print('parameter shift is ',dp)

print('best fit params are ',p)
mod,grad=sinfun(p,x)
plt.plot(x,mod)

nstep=20000
#these are our parameter errors assuming gaussianity
errs=np.sqrt(np.diag(np.linalg.inv(lhs)))

nchain=4
chains=[None]*nchain
for iter in range(nchain):
    chain=np.zeros([nstep,len(p)])
    chain[0,:]=p+3*np.random.randn(len(p))*errs
    chisq=chisq_fun(chain[0,:],x,y)
    for i in range(1,nstep):
        #take trial step
        pp=chain[i-1,:]+1.0*np.random.randn(len(errs))*errs
        chisq_new=chisq_fun(pp,x,y) #evaluate likelihood at new position
        accept=np.exp(-0.5*(chisq_new-chisq))
        if accept>np.random.rand(1):
            chain[i,:]=pp
        else:
            chain[i,:]=chain[i-1,:]
    chains[iter]=chain

#let's do a gelman-rubin test
means=np.zeros([nchain,chain.shape[1]])
scats=np.zeros([nchain,chain.shape[1]])
for i in range(nchain):
    means[i,:]=np.mean(chains[i],axis=0)
    scats[i,:]=np.std(chains[i],axis=0)
#scatter of means
mean_scat=np.std(means,axis=0)
gelman_rubin=mean_scat/np.mean(scats,axis=0)
print('gelman_rubin scatters are ',gelman_rubin)
