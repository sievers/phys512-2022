import numpy as np
class Complex:
    def __init__(self,r,i):
        self.r=r
        self.i=i
    def copy(self):
        return Complex(self.r,self.i)
    def abs(self):
        return np.sqrt(self.r**2+self.i**2)
    def __add__(self,x):
        ans=Complex(self.r,self.i)
        if isinstance(x,Complex):
            ans.r+=x.r
            ans.i+=x.i
        else:
            try:
                ans.r+=x
            except:
                print('unsupported input type ',type(x),' in add')
                return None
        return ans
    def __radd__(self,x):
        return self.__add__(x)
    def __mul__(self,x):
        if isinstance(x,Complex):
            return Complex(self.r*x.r-self.i*x.i,self.r*x.i+self.i*x.r)
        else:
            return Complex(self.r*x,self.i*x)
    def __rmul__(self,x):
        return self.__mul__(x)
    def __repr__(self):
        if self.i<0:
            return repr(self.r)+' -'+repr(-self.i)+'i'
        else:
            return repr(self.r)+' +'+repr(self.i)+'i'
    def conj(self):
        return Complex(self.r,-self.i)
