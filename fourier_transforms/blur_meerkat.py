import numpy as np
from matplotlib import pyplot as plt

def make_chessboard(ncell,npix):
    n=ncell*npix
    mat=np.zeros([n,n])
    for i in range(ncell):
        for j in range(ncell):
            isblack=(i-j)%2==0
            if isblack:
                mat[i*npix:(i+1)*npix,j*npix:(j+1)*npix]=1
    return mat

def get_smooth_kernel(shape,npx):
    nx=shape[0]
    ny=shape[1]
    xvec=np.arange(nx)
    yvec=np.arange(ny)
    xvec[nx//2:]=xvec[nx//2:]-nx
    yvec[ny//2:]=yvec[ny//2:]-ny
    xexp=np.exp(-0.5*xvec**2/npx**2)
    yexp=np.exp(-0.5*yvec**2/npx**2)
    kernel=np.outer(xexp,yexp)
    return kernel/kernel.sum()

plt.ion()
myim=plt.imread('meerkat_small.png')
#myim=plt.imread('30973.jpg')

red=myim[:,:,0]
#red=make_chessboard(16,16)

kernel=get_smooth_kernel(red.shape,1.5)
redft=np.fft.rfft2(red)
kernelft=np.fft.rfft2(kernel)
red_smooth_ft=redft*kernelft
red_smooth=np.fft.irfft2(red_smooth_ft)

red_smooth_ft2=np.fft.rfft2(red_smooth)
red_ft2=red_smooth_ft2/kernelft
red_ft2[np.isfinite(red_ft2)==False]=0
red_back=np.fft.irfft2(red_ft2)
#red_back=np.fft.irfft2(np.fft.
