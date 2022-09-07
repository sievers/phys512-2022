import numpy as np
from matplotlib import pyplot as plt

xi=np.linspace(-2,2,11)
yi=np.exp(xi)

x=np.linspace(xi[0],xi[-1],1001)
y_true=np.exp(x)

y_interp=0*y_true
for i in range(len(x)):
    #argmin gives the index of the minimum point
    #abs takes the absolute value
    #so myind will contain the index of the xi closest
    #to my x point
    myind=np.argmin(np.abs(x[i]-xi))
    #paste in the value of yi at the closes xi
    y_interp[i]=yi[myind]

myfun=np.interp(xi,yi,'linear')
