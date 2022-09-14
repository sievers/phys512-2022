import numpy as np

def heaviside(x):
    return 1.0*(x>0)

def offset_gauss(x):
    return 1+10*np.exp(-0.5*x**2/(0.1)**2)


def integrate(fun,a,b,tol):
    print('calling function from ',a,b)
    x=np.linspace(a,b,5)
    dx=x[1]-x[0]
    y=fun(x)
    #do the 3-point integral
    i1=(y[0]+4*y[2]+y[4])/3*(2*dx)
    i2=(y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/3*dx
    myerr=np.abs(i1-i2)
    if myerr<tol:
        return i2
    else:
        mid=(a+b)/2
        int1=integrate(fun,a,mid,tol/2)
        int2=integrate(fun,mid,b,tol/2)
        return int1+int2


ans=integrate(offset_gauss,-4,6,1e-6)
ans2=integrate(offset_gauss,-4,0,1e-6)+integrate(offset_gauss,0,6,1e-6)
print('answer was ',ans,ans2,ans-(10+np.sqrt(2*np.pi)))
#print('answer was ',ans,ans-(np.exp(1)-np.exp(-1)))
