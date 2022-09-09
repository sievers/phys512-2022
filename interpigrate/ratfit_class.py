import numpy as np

fun=np.exp

n=3 #numerator order
m=3 #denominator order

x=np.linspace(-1,1,n+m+1)
y=fun(x)

pcols=[x**k for k in range(n+1)]
pmat=np.vstack(pcols)

qcols=[-x**k*y for k in range(1,m+1)]
qmat=np.vstack(qcols)
mat=np.hstack([pmat.T,qmat.T])
coeffs=np.linalg.inv(mat)@y

#now that we have coefficients, we can see how well we did
xfine=np.linspace(-1,1,1001)
#it turns out np.polyval expects the coefficients in reverse order, so
#we have to flip them around.
num=np.polyval(np.flipud(coeffs[:n+1]),xfine)
#note that we need an extra x multiply when evaluating the denominator
#because of how we defined the problem
denom=1+xfine*np.polyval(np.flipud(coeffs[n+1:]),xfine)
pred=num/denom  #our final model is the ratio of the numerator and denominator

plt.clf()
plt.plot(x,y,'*')
plt.plot(xfine,yfine)
plt.plot(xfine,fun(xfine))
plt.show()

print('ratfun err is ',np.std(pred-fun(xfine)))
pp=np.polyfit(x,y,len(x)-1)
poly_pred=np.polyval(pp,xfine)
print('poly err is ',np.std(poly_pred-fun(xfine)))
