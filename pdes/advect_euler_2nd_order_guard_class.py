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
a=0.1 #vdt in grid units
periodic=True
for i in range(int(n/a)):
    

    rho_new=np.empty(n+2)
    rho_new[1:-1]=rho
    rho_new[0]=rho[-1]
    rho_new[-1]=rho[0]

    grad=(rho_new[2:]-rho_new[:-2])/2
    rho_new[1:-1]=rho_new[1:-1]-a*grad
    rho=rho_new[1:-1]

    plt.clf()
    plt.plot(rho_new)
    plt.pause(0.001)
    rho=rho_new
