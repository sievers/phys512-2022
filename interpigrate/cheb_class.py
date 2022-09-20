import numpy as np
from matplotlib import pyplot as plt
plt.ion()

x=np.linspace(-1,1,1001)
ord=5
chebs=np.empty([len(x),ord+1])

chebs[:,0]=1
chebs[:,1]=x
for i in range(1,ord):
    chebs[:,i+1]=2*x*chebs[:,i]-chebs[:,i-1]


n=20
x=np.linspace(-1,1,n)
mat=np.polynomial.chebyshev.chebvander(x,n-1)
fun=np.exp
y=fun(x)
fitp=np.linalg.inv(mat)@y
fitp_use=fitp.copy()
nuse=6
fitp_use[nuse:]=0 #zero out all coefficients pass the ones we want
pred=mat@fitp_use
plt.clf()
plt.plot(x,y-pred)
plt.show()

xx=np.linspace(-1,1,nuse)
lmat=np.polynomial.legendre.legvander(xx,nuse-1)
lfitp=np.linalg.inv(lmat)@(fun(xx))
lmat2=np.polynomial.legendre.legvander(x,nuse-1)
lpred=lmat2@lfitp
plt.plot(x,y-lpred)
