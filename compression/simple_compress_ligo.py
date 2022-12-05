import numpy as np
from matplotlib import pyplot as plt
import h5py
import glob
from scipy.linalg import toeplitz
plt.ion()

def get_ptable(vec):
    imin=vec.min()
    n=np.max(vec-imin)+1
    ptable=np.zeros(n,dtype='int')
    for i in range(len(vec)):
        ptable[vec[i]-imin]=ptable[vec[i]-imin]+1
    return ptable

def get_s(ptable):
    ii=ptable[ptable>0]
    ii=ii/ii.sum()
    return np.sum(-ii*np.log2(ii))


def write_stream(fname,fac,stream):
    f=open(fname,'w')

    size=np.asarray(stream.itemsize,dtype='uint8')
    #print('dtype is ',size.dtype)
    size.tofile(f)

    fac=np.asarray(fac)
    fac.tofile(f)
    stream.tofile(f)
    f.close()
def read_stream(fname):
    f=open(fname,'rb')
    size=np.fromfile(f,'uint8',1)[0]
    dtype='int'+repr(8*size)
    #print('reading ',dtype)
    fac=np.fromfile(f,np.float,1)
    stream=np.fromfile(f,dtype)
    f.close()
    return fac,stream

def restore_lp(stream,coeffs):
    n=len(stream)
    ans=np.zeros(n)
    m=len(coeffs)
    ans[:m]=stream[:m]
    for i in range(m,n):
        pred=np.round(np.sum(ans[i-m:i]*coeffs))
        ans[i]=stream[i]+pred
    return ans

def write_lp(fname,coeffs,stream,scale):
    f=open(fname,'w')
    size=np.asarray(stream.itemsize,dtype='uint8')
    #print('dtype is ',size.dtype)
    size.tofile(f)
    scale=np.asarray(scale)
    scale.tofile(f)

    ncoeff=len(coeffs)
    ncoeff=np.asarray(ncoeff,dtype='uint16')
    #print('ncoeff written is ',ncoeff)
    ncoeff.tofile(f)
    coeffs.tofile(f)
    stream.tofile(f)
    f.close()

def read_lp(fname,restore=False):
    f=open(fname,'rb')
    size=np.fromfile(f,'uint8',1)[0]
    dtype='int'+repr(8*size)
    #print('reading ',dtype)
    scale=np.fromfile(f,count=1)
    ncoeff=np.fromfile(f,dtype='uint16',count=1)[0]
    #print('ncoeff read is ',ncoeff)
    coeffs=np.fromfile(f,count=ncoeff)
    stream=np.fromfile(f,dtype=dtype)
    f.close()
    if restore:
        stream=restore_lp(stream,coeffs)
        return stream*scale
    else:
        return stream,coeffs,scale


def get_corr(vec):
    n=len(vec)
    vv=np.zeros(2*n)
    vv[:n]=vec
    vvft=np.fft.rfft(vv)
    corr_unnorm=np.fft.irfft(vvft*np.conj(vvft),2*n)
    vv[:n]=1
    vvft=np.fft.rfft(vv)
    mynorm=np.fft.irfft(vvft*np.conj(vvft),2*n)
    ans=corr_unnorm[:n]/mynorm[:n]
    return ans
def evaluate_lp_model_slow(vec,coeffs):
    n=len(vec)
    m=len(coeffs)
    ans=np.zeros(n,dtype='int')
    ans[:m]=vec[:m]
    for i in range(m,n):
        ans[i]=np.round(np.sum(vec[i-m:i]*coeffs))
    return ans

def evaluate_lp_model(vec,coeffs):
    n=len(vec)
    tmp=np.zeros(2*n)
    tmp2=np.zeros(2*n)
    tmp[:n]=vec
    m=len(coeffs)
    tmp2[-m:]=coeffs
    tmpft=np.fft.rfft(tmp)
    tmp2ft=np.fft.rfft(tmp2)
    pred=np.fft.irfft(tmpft*tmp2ft,2*n)
    pred[:m]=0 #we don't have a model for the first m points
    return np.asarray(np.round(pred[:n]),dtype='int')


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
strain_quant=np.asarray(np.round(strain/scale_fac),dtype='int')
print(np.std(strain_quant*scale_fac-strain))
fname='ligo_digitized.dat'
write_stream(fname,scale_fac,strain_quant)
ptable=get_ptable(strain_quant)
print('entropy of raw stream is ',get_s(ptable))
#fac2,strain2=read_stream(fname)

strain_diff=np.diff(strain_quant)
strain_diff=np.hstack([strain_quant[0],strain_diff])
ptable2=get_ptable(strain_diff)
squant=get_s(ptable2)
print('entropy of diff stream is ',squant)
print('ideal file size is ',squant*len(strain_diff)/8)
ss=np.cumsum(strain_diff)
fname='ligo_diff.dat'
write_stream(fname,scale_fac,strain_diff)


mycorr=get_corr(strain_quant)
nsamp=12
N=toeplitz(mycorr[:nsamp])
N=N+100*N[0,0] #this tells us to not think we know about the mean
Ninv=np.linalg.inv(N)
lhs=Ninv[-1,-1]
rhs=Ninv[-1,:-1]
coeffs=-rhs/lhs
pred=evaluate_lp_model(strain_quant,coeffs)
pred2=evaluate_lp_model_slow(strain_quant,coeffs)
delt=strain_quant-pred2
delt[:len(coeffs)]=strain_quant[:len(coeffs)]
p_lp=get_ptable(delt)
s_lp=get_s(p_lp)
print('LP model entropy is ',s_lp,' with ideal size ',s_lp*len(strain)/8)
stream_back=restore_lp(delt,coeffs)
write_lp('ligo_lp.dat',coeffs,delt,scale_fac)

#aa,bb,cc=read_lp('ligo_lp.dat')
mystrain=read_lp('ligo_lp.dat',True)

#plt.clf();plt.plot(strain[100:200]);plt.plot(strain[100:200],'*');plt.plot(pred[100:200]*scale_fac,'o');plt.show()
#plt.savefig('ligo_lp_zoon.png')
