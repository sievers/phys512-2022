import numpy as np
from matplotlib import pyplot as plt

n=1000
x=np.random.randn(n,n)
x=x+x.T
e,v=np.linalg.eigh(x)
rcond=10
my_eig=1+np.random.rand(n)*(rcond-1)

A=v@np.diag(my_eig)@v.T  #this gives us a positive-definite matrix 

b=np.random.randn(n)
x_true=np.linalg.inv(A)@b

x=0*b
r=b-A@x
p=r.copy()
niter=50
rtr=np.sum(r**2)
for i in range(niter):
    Ap=A@p
    pAp=p@Ap
    alpha=rtr/pAp
    x=x+alpha*p
    r=r-alpha*Ap
    rtr_new=np.sum(r**2)
    beta=rtr_new/rtr
    p=r+beta*p
    print('my current residual squared is ',rtr_new)
    rtr=rtr_new
