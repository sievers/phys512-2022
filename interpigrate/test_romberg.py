import numpy as np
from scipy import integrate

a=-1
b=1
for k in range(1,10):
    n=1+2**k
    dx=(b-a)/(n-1.0)
    x=np.linspace(a,b,n)
    y=np.exp(x)
    pred=np.exp(b)-np.exp(a)
    f=dx*integrate.romb(y)
    print('for k=' + repr(k) + ' and ' + repr(n) + ' function calls, error is ' + repr(np.abs(f-pred)))
f=integrate.romberg(np.exp,a,b,show=True)

