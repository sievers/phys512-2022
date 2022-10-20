import numpy as np
import time

def myfft(f):
    N=len(f)
    if N==1: #dft of length-1 vector is itself
        return f
    
    M=N//2
    fe=f[::2]
    fo=f[1::2]
    even_ft=myfft(fe)
    odd_ft=myfft(fo)
    k=np.arange(M)
    twid=np.exp(-2J*np.pi*k/N)
    ft1=even_ft+twid*odd_ft
    ft2=even_ft-twid*odd_ft
    return np.hstack([ft1,ft2])

f=np.random.randn(8192*16)
t1=time.time()
myft=myfft(f)
t2=time.time()
print('my FFT took ',t2-t1)
print(np.std(myft-np.fft.fft(f)))
