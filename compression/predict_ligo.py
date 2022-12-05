import numpy as np
from matplotlib import pyplot as plt
import h5py
import glob
plt.ion()
import os

def read_template(filename):
    dataFile=h5py.File(filename,'r')
    template=dataFile['template']
    th=template[0]
    tl=template[1]
    return th,tl
def read_file(filename):
    dataFile=h5py.File(filename,'r')
    dqInfo = dataFile['quality']['simple']
    qmask=dqInfo['DQmask'][...]

    meta=dataFile['meta']
    #gpsStart=meta['GPSstart'].value
    gpsStart=meta['GPSstart'][()]
    #print meta.keys()
    #utc=meta['UTCstart'].value
    utc=meta['UTCstart'][()]
    #duration=meta['Duration'].value
    duration=meta['Duration'][()]
    #strain=dataFile['strain']['Strain'].value
    strain=dataFile['strain']['Strain'][()]
    dt=(1.0*duration)/len(strain)

    dataFile.close()
    return strain,dt,utc



#fnames=glob.glob("[HL]-*.hdf5")
#fname=fnames[0]
fname='../ligo/H-H1_LOSC_4_V2-1126259446-32.hdf5'
print('reading file ',fname)
strain,dt,utc=read_file(fname)

mynoise=np.std(np.diff(strain))
nbit_noise=5
scale_fac=mynoise/(2**nbit_noise)
istrain=np.asarray(strain/scale_fac,dtype='int')
n=10
nvec=np.zeros(n)
nvec[0]=np.mean(istrain**2)
for i in range(1,n):
    nvec[i]=np.mean(istrain[i:]*istrain[:-i])

mat=np.zeros([n,n])
for i in range(n):
    for j in range(n):
        mat[i,j]=nvec[np.abs(i-j)]
Ninv=np.linalg.inv(mat) #get the LP coefficients
coeffs=-Ninv[-1,:-1]/Ninv[-1,-1] 
vec=0*strain
vec[-(n-1):]=coeffs
pred=np.fft.irfft(np.fft.rfft(istrain)*np.fft.rfft(vec))

