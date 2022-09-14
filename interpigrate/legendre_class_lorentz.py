import numpy as np


def lorentz(x):
    return 1.0/(1+x**2)

order=4

npt_targ=9
ninterval=int((npt_targ-1)/order)
print('going to use ',ninterval,' intervals')

#get the coefficients/weights
x=np.linspace(-1,1,order+1)
mat=np.polynomial.legendre.legvander(x,order)
minv=np.linalg.inv(mat)
coeffs=minv[0,:]


npt_use=ninterval*order+1

fun=lorentz

x=np.linspace(-1,1,npt_use)
y=fun(x)
dx=np.median(np.diff(x))
tot=0
for i in range(ninterval):
    i1=i*order  #for 4th order, we want to start at (0,4,8,...) etc.
    i2=(i+1)*order+1
    pts=y[i1:i2]
    tot=tot+np.sum(coeffs*pts)

ans=tot*dx*order
#truth=np.cos(0)-np.cos(1)
truth=np.arctan(1)-np.arctan(-1)
print('my integral is ',ans,truth,ans-truth)
    
               
