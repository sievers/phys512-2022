import numpy as np
from scipy import interpolate as interp
from matplotlib import pyplot as plt

x=np.linspace(-1,1,11)
y=np.exp(x)
y[-2:]=y[-3]


spl=interp.splrep(x,y) #generate the spline using scipy
xfine=np.linspace(-1,1,1001)
yfine=interp.splev(xfine,spl) #evaluate the spline on fine x
ytrue=np.exp(xfine)

plt.ion()
plt.clf()
plt.plot(x,y,'*')
plt.plot(xfine,yfine)
plt.show()
print('error is ',np.std(ytrue-yfine))

