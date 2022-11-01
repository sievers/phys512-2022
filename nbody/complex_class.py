import numpy as np

class Complex:
    def __init__(self,r,i):
        self.r=r
        self.i=i
    def copy(self):
        return Complex(self.r,self.i)
    def abs(self):
        return np.sqrt(self.r**2+self.i**2)
    def add(self,a):
        #b=Complex(self.r,self.i)
        b=a.copy()
        if isinstance(a,Complex):
            b.r=self.r+a.r
            b.i=self.i+a.i
        else:
            b.r=self.r+a
        return b
    def double(self):
        self.r=self.r*2
        self.i=self.i*2
    def __add__(self,a):
        return self.add(a)
    def __radd__(self,a):
        return self.add(a)
    def __repr__(self):
        if self.i<0:
            return repr(self.r)+' - '+repr(-self.i)+'J'
        else:
            return repr(self.r)+' + '+repr(self.i)+'J'
a=Complex(1,2)

