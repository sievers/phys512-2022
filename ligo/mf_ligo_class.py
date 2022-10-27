import numpy as np
from matplotlib import pyplot as plt
import h5py
import glob
plt.ion()


def smooth_vector(vec,sig):
    n=len(vec)
    x=np.arange(n)
    x[n//2:]=x[n//2:]-n
    kernel=np.exp(-0.5*x**2/sig**2) #make a Gaussian kernel
    kernel=kernel/kernel.sum()
    vecft=np.fft.rfft(vec)
    kernelft=np.fft.rfft(kernel)
    vec_smooth=np.fft.irfft(vecft*kernelft) #convolve the data with the kernel
    return vec_smooth

def read_template(filename):
    dataFile=h5py.File(filename,'r')
    template=dataFile['template']
    tp=template[0]
    tx=template[1]
    return tp,tx
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
fname='H-H1_LOSC_4_V2-1126259446-32.hdf5'
print('reading file ',fname)
strain,dt,utc=read_file(fname)

#th,tl=read_template('GW150914_4_template.hdf5')
template_name='GW150914_4_template.hdf5'
tp,tx=read_template(template_name)

x=np.linspace(-np.pi/2,np.pi/2,len(strain))
win=np.cos(x)

noise_ft=np.fft.fft(win*strain)
noise_smooth=smooth_vector(np.abs(noise_ft)**2,10)
noise_smooth=noise_smooth[:len(noise_ft)//2+1] #will give us same length




tobs=dt*len(strain)
dnu=1/tobs
nu=np.arange(len(noise_smooth))*dnu
nu[0]=0.5*nu[1]

Ninv=1/noise_smooth
Ninv[nu>1500]=0
Ninv[nu<20]=0

template_ft=np.fft.rfft(tp*win)
template_filt=template_ft*Ninv
data_ft=np.fft.rfft(strain*win)
rhs=np.fft.irfft(data_ft*np.conj(template_filt))




#plt.clf();plt.loglog(nu,np.abs(noise_ft));plt.show()




#plt.xlim([20000,30000])
#plt.xlim([21600,21800]);plt.ylim([-2e-19,2e-19])
