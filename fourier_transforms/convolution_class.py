import numpy as np
from matplotlib import pyplot as plt


#let's make a sawtooth with a convolution!

N=1000
a=100

#a set of spikes/picket fence
f=np.zeros(N)
f[::a]=1
f[int(2.5*a)]=2

g=np.zeros(N)
b=int(1.5*a)  #the width of our sawteeth
g[:b]=np.linspace(1,0,b)

plt.ion()
plt.clf()
plt.plot(f)
plt.plot(g)
plt.show()

F=np.fft.rfft(f)
G=np.fft.rfft(g)
H=F*G
h=np.fft.irfft(H)
plt.plot(h)
