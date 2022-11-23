import numpy as np
import numba as nb
import time
import timeit


def partition_range(nthread,imax,imin=0):
    lims=np.asarray(np.linspace(imin,imax,nthread+1),dtype='int')
    return lims

@nb.njit(parallel=True)
def parhist(inds,nbin,nthread):
    tmp=np.zeros((nthread,nbin))
    n=len(inds)
    #lims=np.asarray(np.linspace(0,n,nthread+1),dtype='int')

    for id in nb.prange(nthread):
        i1=int(n/nthread*id)
        i2=int(n/nthread*(id+1))
        #for i in np.arange(lims[id],lims[id+1]):
        for i in np.arange(i1,i2):
            tmp[id,inds[i]]+=1
    return np.sum(tmp,axis=0)


n=100000000
nbin=1000
x=np.asarray(np.random.rand(n)*nbin,dtype='int')

myhist=parhist(x,nbin,8)
niter=10
for i in range(niter):
    t1=time.time()
    myhist=parhist(x,nbin,8)
    t2=time.time()
    print(t2-t1)
