import numpy as np

x=np.linspace(-1,1,1001)
ord=25
m=np.zeros(ord)
m[2]=1
m[3]=1
#m=np.asarray([0,0,1,1],dtype='float') above does the same, but optionally extra zeros at the end

A=np.empty([len(x),ord])
for i in range(ord):
    A[:,i]=x**i
y_true=A@m
y=y_true+np.random.randn(len(x))


#x=np.linspace(-1,1,1001)
ords=[5,10,20,30,50,100]
for ord in ords:
    A=np.polynomial.legendre.legvander(x,ord)
    lhs=A.T@A
    e,v=np.linalg.eigh(lhs)
    rcond=e.max()/np.abs(e).min()
    print('on order ',ord,' rcond is ',rcond)

ord=1000
A=np.polynomial.legendre.legvander(x,ord)
lhs=A.T@A
rhs=A.T@y
fitp=np.linalg.inv(lhs)@rhs
pred=A@fitp
plt.clf()
plt.plot(x,y,'.')
plt.plot(x,pred)
plt.show()
