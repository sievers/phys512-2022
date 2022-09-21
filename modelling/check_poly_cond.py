import numpy as np
x=np.linspace(-1,1,2000) #make the x vector
ord=100

A=np.zeros([len(x),ord])
A[:,0]=1.0
for i in range(1,ord):
    A[:,i]=x*A[:,i-1]

T=np.zeros([len(x),ord])
T[:,0]=1
T[:,1]=x
for i in range(2,ord):
    T[:,i]=2*x*T[:,i-1]-T[:,i-2]



ata=np.dot(A.transpose(),A)
eigs,vecs=np.linalg.eig(ata)

u,s,v=np.linalg.svd(A,0)

print('condition number is ' + repr(np.log10(eigs.max()/np.abs(eigs).min())))
print('SVD condition number is ' + repr(np.log10(s.max()/np.abs(s).min())))


ttt=np.dot(T.transpose(),T)
eigs,vecs=np.linalg.eig(ttt)
print("Cheb condition is " + repr(np.max(eigs)/np.min(np.abs(eigs)))
