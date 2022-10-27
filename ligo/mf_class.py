import numpy as np
from matplotlib import pyplot as plt
plt.ion()

nd=1000
y_true=1/np.random.rand(nd)**1.5
y_true=y_true/y_true.max()*3

tau=np.pi
x=np.arange(nd)
kernel=np.exp(-x/tau)

kernel_ft=np.fft.rfft(kernel)
yft=np.fft.rfft(y_true)
y_conv=np.fft.irfft(yft*kernel_ft)

sig=0.05
y=y_conv+np.random.randn(nd)*sig
yft=np.fft.rfft(y)
y_deconv=np.fft.irfft(yft/kernel_ft) #this is the deconvolution
lhs=kernel.T@kernel/sig**2
rhs=np.fft.irfft(np.conj(kernel_ft)*yft)/sig**2
mf=rhs/lhs
print('median matched filter noise: ',np.median(np.abs(mf)))
print('median deconvolution noise: ',np.median(np.abs(y_deconv)))
print('matched filter expected noise: ',1/np.sqrt(lhs))



plt.clf()
plt.plot(y_true,'k')
plt.plot(y_deconv,'r:')
plt.plot(mf,'b:')
plt.show()
plt.legend(['True signal','Deconvolution','MF'])
