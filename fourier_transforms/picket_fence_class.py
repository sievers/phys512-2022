import numpy as n
from matplotlib import pyplot as plt

plt.ion()

N=150
a=5
y=np.zeros(N)
y[::a]=1
yft=np.fft.fft(y)
plt.clf()
plt.plot(y,'.')
plt.show()
