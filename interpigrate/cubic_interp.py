import numpy as np

xi=np.linspace(-2,2,11)
yi=np.exp(xi)
yi[-2:]=yi[-3]

x=np.linspace(xi[1],xi[-2],1001)
y_true=np.exp(x)
y_interp=np.zeros(len(x))
for i in range(len(x)):    
    ind=np.max(np.where(x[i]>=xi)[0])
    x_use=xi[ind-1:ind+3]
    y_use=yi[ind-1:ind+3]
    pars=np.polyfit(x_use,y_use,3)
    pred=np.polyval(pars,x[i])
    y_interp[i]=pred

print('my rms error is ',np.std(y_interp-y_true))

