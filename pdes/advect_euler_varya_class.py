import numpy as np
from matplotlib import pyplot as plt
plt.ion()

n=100
rho=np.zeros(n)
rho[n//3:(2*n)//3]=1.0
rho_org=rho.copy()
plt.clf()
plt.plot(rho)
plt.show()
#update a time step
a=1.01 #vdt in grid units
periodic=True
for i in range(int(n/a)):
    rho_new=np.empty(n)
    rho_new[1:]=rho[1:]-(rho[1:]-rho[:-1])*a
    if periodic:
        rho_new[0]=rho[0]-(rho[0]-rho[-1])*a
    else:
        rho_new[0]=0
    plt.clf()
    plt.plot(rho_new)
    plt.pause(0.001)
    rho=rho_new
