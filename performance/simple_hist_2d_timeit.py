import numpy as np
import time
import timeit
import numba as nb


def hist_2d(xy,grid):
    ixy=np.asarray(np.round(xy),dtype='int')
    n=xy.shape[0]
    for i in range(n):
        grid[ixy[i,0],ixy[i,1]]=grid[ixy[i,0],ixy[i,1]]+1

@nb.njit
def hist_2d_nb(xy,grid):
    n=xy.shape[0]
    for i in range(n):
        myx=int(round(xy[i,0]))
        myy=int(round(xy[i,1]))
        grid[myx,myy]=grid[myx,myy]+1



npix=1000
npt=200000
ndim=2
xy=np.random.rand(npt,ndim)*(npix-1)  #-1 is just to make sure we don't go off the edge
grid=np.zeros([npix,npix])


#ipix=np.asarray(np.round(xy),dtype='int')
t1=time.time()
hist_2d(xy,grid)
#for i in range(npt):
#    grid[ipix[i,0],ipix[i,1]]=grid[ipix[i,0],ipix[i,1]]+1
t2=time.time()
print('time per particle to project was ' + repr((t2-t1)/npt))
niter=10
#we can add imports to timeit functions
out=timeit.timeit('hist_2d(xy,grid)',
                  'from __main__ import hist_2d,xy,grid',number=niter)
print('time per particle from timeit is ',out/niter/npt)
hist_2d_nb(xy,grid)
out=timeit.timeit('hist_2d_nb(xy,grid)',
                  'from __main__ import hist_2d_nb,xy,grid',number=niter)
print('numba time per particle from timeit is ',out/niter/npt)

#we can also just import everything in the global namespace
out=timeit.timeit('hist_2d_nb(xy,grid)',globals=globals(),number=niter)
print('globals time per particle from timeit is ',out/niter/npt)
