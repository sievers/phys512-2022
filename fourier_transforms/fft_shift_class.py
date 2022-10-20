import numpy as np
from matplotlib import pyplot as plt

plt.ion()
N=1000
#we will shift a sawtooth, 'cause that's what the class asked for
x=np.linspace(-5,5,N)
y=x-np.floor(x)
plt.clf()
plt.plot(x,y)
plt.show()

dx=50
yft=np.fft.fft(y)
k=np.arange(len(yft))
ramp=np.exp(2*np.pi*1J*k*dx/N)
yft_shift=yft*ramp
y_shift=np.fft.ifft(yft_shift)
plt.plot(x,y_shift)
