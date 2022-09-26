import numpy as np
from openpyxl import load_workbook
from matplotlib import pyplot as plt


crud=load_workbook('A02_GAIN_MIN20_373.xlsx')
sheet=crud['All']

nu=[sheet['A'+repr(i+1)].value for i in range(4096)]
nu=np.asarray(nu)
gain=[sheet['B'+repr(i+1)].value for i in range(4096)]
gain=np.asarray(gain)
nu=nu/1e6 #convert frequency from Hz to MHz

nu_min=15

ii=nu>nu_min
nu=nu[ii]
gain=gain[ii]

ord=5
ndata=len(nu)
A=np.empty([ndata,ord+1])
for i in range(ord+1):
    A[:,i]=nu**i
lhs=A.T@A
rhs=A.T@gain
m=np.linalg.inv(lhs)@rhs
pred=A@m
plt.clf()
plt.plot(nu,gain)
plt.plot(nu,pred)
plt.show()

N=np.mean((gain-pred)**2)
par_errs=np.sqrt(N*np.diag(np.linalg.inv(lhs)))
tmp=A@np.linalg.inv(lhs)@A.T
mod_err=np.sqrt(N*np.diag(tmp))


assert(1==0)




order=5
A=np.polynomial.polynomial.polyvander(nu,order)

lhs=A.T@A
rhs=A.T@gain
pp=np.linalg.inv(lhs)@rhs
pred_poly=A@pp

print('RMS error after fit is ' + repr(np.sqrt(np.mean( (pred_poly-gain)**2))))

plt.ion()
plt.clf()
plt.plot(nu,gain)
plt.plot(nu,pred_poly)
plt.show()
N=np.std(gain-pred_poly)**2




