import numpy as np
from matplotlib import pyplot as plt
plt.ion()

def expfun(x,y):
    return y
def sinfun(x,y):
    return np.sin(y)

def rk2_step_other(fun,x,y,h):
    k1=h*fun(x,y)
    k2=h*fun(x+h/2,y+k1/2)
    return k2

def rk2_step(fun,x,y,h):
    k1=h*fun(x,y)
    k2=h*fun(x+h,y+k1)
    return (k1+k2)/2

def rk2_integrate(fun,x0,x1,y0,n):
    x=np.linspace(x0,x1,n)
    h=x[1]-x[0]
    yvec=0*x
    yvec[0]=y0
    for i in range(len(x)-1):
        yvec[i+1]=yvec[i]+rk2_step_other(fun,x[i],yvec[i],h)
    return yvec,x

y,x=rk2_integrate(expfun,0,1,1,21)

y2,x2=rk2_integrate(sinfun,0,1,2/np.arctan(1),21)
yy=2/np.arctan(np.exp(-x2))
