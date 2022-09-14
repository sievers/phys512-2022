import numpy as np
from matplotlib import pyplot as plt
plt.ion()


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

p=0
for i in range(n+1):
    p=p+coeffs[i]*xfine**i
qq=1
for i in range(m):
    qq=qq+coeffs[n+1+i]*xfine**(i+1)

y_pred=p/qq
y_true=fun(xfine)
plt.clf()
plt.plot(xfine,y_pred-y_true)
plt.show()

#use numpy to fit same-order polynomial
poly_coeffs=np.polyfit(x,y,len(y)-1)
y_poly=np.polyval(poly_coeffs,xfine)
plt.plot(xfine,y_poly-y_true)
plt.legend(['ratfit resid','poly resid'])
assert(1==0)


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
