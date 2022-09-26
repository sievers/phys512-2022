import numpy as np

n=10001
ord=7
x=np.linspace(-1,1,n)
A=np.polynomial.chebyshev.chebvander(x,ord)
lhs=A.T@A
e,v=np.linalg.eigh(lhs)
print('rcond is ',e.max()/np.abs(e).min())

y=np.exp(x)
fitp=np.linalg.inv(lhs)@(A.T@y)

nuse=8
pred=A[:,:nuse]@fitp[:nuse]
plt.clf();plt.plot(x,y-pred);plt.show()
