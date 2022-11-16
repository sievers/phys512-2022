import numpy as np

#let's start by making a positive-definite matrix of size n
n=100
x=np.random.randn(n,n)
A=x.T@x

#for better convergence, let's put a floor on the eigenvalues
#of A.  We could also add something to the diagonal.
e,v=np.linalg.eigh(A)
thresh=1
e[e<thresh]=thresh
print('cleaned condition number is ',e.max()/e.min())
A=v@np.diag(np.sqrt(e))@v.T

#generate the right-hand side
b=np.random.randn(n)


#do the initial setup for conjugate gradient, starting with 0 for the guess
x=0*b
niter=35
r=b-A@x
p=r.copy()
r_old=r
#iterate to solve the matrix equation
for i in range(niter):
    r=b-A@x    
    Ar=A@r
    alpha=r@r/(r@Ar)
    x=x+alpha*r
    mydot=r_old@r
    r_old=r
    print('r squared is ',r@r,mydot)

#print out the scatter of the residual.  Note that Ax-b should be sitting in
#r so this ought to be the same (modulo the scaling by the scatter
#of b), but roundoff error could make things different, so
#starting fresh is sometimes a good sanity check.
print('fractional scatter of answer is ',np.std(A@x-b)/np.std(b))
