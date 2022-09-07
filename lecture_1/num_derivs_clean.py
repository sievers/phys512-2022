import numpy as np
from matplotlib import pyplot as plt

#dx=np.linspace(0,1,1001) #we could do this, but we want a wider log-range of dx
logdx=np.linspace(-15,-1,1001)
dx=10**logdx

fun=np.exp
x0=1

y0=fun(x0)
y1=fun(x0+dx)
d1=(y1-y0)/dx #calculate the 1-sided, first-order derivative
ym=fun(x0-dx)
d2=(y1-ym)/(2*dx) #calculate the 2-sided derivative.
plt.ion() #so we don't have to click away plots!
plt.clf()
#make a log plot of our errors in the derivatives
plt.loglog(dx,np.abs(d1-np.exp(x0)))
plt.plot(dx,np.abs(d2-np.exp(x0)))
plt.show()
plt.savefig('deriv_errors.png')

