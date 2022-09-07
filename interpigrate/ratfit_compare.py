import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate

def lorentz(x):
    return 1/(1+x**2)

def rat_eval(p,q,x):
    top=0
    for i in range(len(p)):
        top=top+p[i]*x**i
    bot=1
    for i in range(len(q)):
        bot=bot+q[i]*x**(i+1)
    return top/bot

def rat_fit(x,y,n,m):
    assert(len(x)==n+m-1)
    assert(len(y)==len(x))
    mat=np.zeros([n+m-1,n+m-1])
    for i in range(n):
        mat[:,i]=x**i
    for i in range(1,m):
        mat[:,i-1+n]=-y*x**i
    print(mat)
    pars=np.dot(np.linalg.pinv(mat),y)
    p=pars[:n]
    q=pars[n:]
    return p,q


#1*p0 + x*p1 +x**2+p2+... -q1*x - q2*x**2... = y

#fun=np.cos;x0=-np.pi/2;x1=np.pi/2;
#fun=lorentz;x0=-1;x1=1
fun=np.sin;x0=0;x1=np.pi;
n=4
m=4

x=np.linspace(x0,x1,n+m-1)
y=fun(x)
p,q=rat_fit(x,y,n,m)
to_check=rat_eval(p,q,x)
print('error at inerpolated points is ',np.std(to_check-y))
xx=np.linspace(x[0],x[-1],1001)
y_true=fun(xx)
pred=rat_eval(p,q,xx)
plt.clf();plt.plot(x,y,'*')
plt.plot(xx,y_true)
plt.plot(xx,pred)

fitp=np.polyfit(x,y,n+m-1)
pred_poly=np.polyval(fitp,xx)

myfun=interpolate.interp1d(x,y,'cubic')
pred_spline=myfun(xx)

print('rat err ',np.std(pred-y_true))
print('poly err ',np.std(pred_poly-y_true))
print('spline err ',np.std(pred_spline-y_true))

plt.ion()
plt.clf();plt.plot(x,y,'*');plt.plot(xx,y_true);plt.plot(xx,pred)

assert(1==0)

xx=np.linspace(-2,2,1001)
yy=np.exp(-0.5*xx**2)
yy_interp=rat_eval(p,q,xx)
plt.ion()
plt.clf()
plt.plot(xx,yy_interp-yy)


#we can use numpy's polynomial fitter to see how that does
pp=np.polyfit(x,y,n+m) #use same number of terms
yy_poly=np.polyval(pp,xx)
plt.plot(xx,yy_poly-yy)
plt.legend(['Ratfun err','Poly err'])
plt.savefig('ratfit_vs_poly.png')
