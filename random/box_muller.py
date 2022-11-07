import numpy as np

N=100000
x=2*np.random.rand(N)-1
y=2*np.random.rand(N)-1
rsqr=(x**2+y**2)
ind=rsqr<1
x=x[ind]
y=y[ind]
rsqr=rsqr[ind]
rr=np.sqrt(-2*np.log(rsqr)/rsqr)
xx=x*rr
yy=y*rr
xx=np.hstack([xx,yy])
nn=len(xx)
bins=np.linspace(-4,4,401)
a,b=np.histogram(xx,bins)
a=a/(bins[1]-bins[0])/nn
bb=(bins[1:]+bins[:-1])/2
plt.clf()
plt.plot(bb,a)
plt.plot(bb,np.exp(-0.5*bb**2)/np.sqrt(2*np.pi))
plt.show()
