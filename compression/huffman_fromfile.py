import numpy as np
import copy

#class for nodes in a tree
class node:
    def __init__(self,name=None,p=0):
        self.name=name
        self.left=None
        self.right=None
        self.prob=p
    def print(self,cur='',mydict=None):
        if (self.left is None)&(self.right is None):
            print(self.name,' ',cur)
            if not(mydict is None):
                mydict[self.name]=cur
        else:
            self.left.print(cur+'0',mydict=mydict)
            self.right.print(cur+'1',mydict=mydict)    
    def to_dict(self,cur='',mydict=None):
        if mydict is None:
            mydict={}
            self.to_dict(mydict=mydict)
            return mydict
        if (self.left is None)&(self.right is None):
            #print(self.name,' ',cur)
            if not(mydict is None):
                mydict[self.name]=cur
        else:
            self.left.to_dict(cur+'0',mydict=mydict)
            self.right.to_dict(cur+'1',mydict=mydict)    
    def get_length(self,depth=0):
        if (self.left is None)&(self.right is None):
            #s=-self.prob*np.log2(self.prob)
            return self.prob*depth
        else:
            s=0
            s=s+self.left.get_length(depth+1)
            s=s+self.right.get_length(depth+1)
        return s
    def decode_token(self,message):
        if not(self.name is None):
            return self.name,message
        else:
            if message[0]=='0':
                return self.left.decode_token(message[1:])
            else:
                return self.right.decode_token(message[1:])
def merge_nodes(probs,nodes):
    ind=np.argmin(probs)
    node1=nodes[ind]
    p1=probs[ind]
    del(probs[ind])
    del(nodes[ind])

    ind=np.argmin(probs)
    node2=nodes[ind]
    p2=probs[ind]
    del(probs[ind])
    del(nodes[ind])
    mynode=node()
    mynode.left=node1
    mynode.right=node2
    myprob=p1+p2
    probs.append(myprob)
    nodes.append(mynode)
    return probs,nodes

def build_tree(names,probs):
    names=copy.copy(names)
    probs=copy.copy(probs)
    n=len(names)
    ptot=np.sum(probs)
    nodes=[None]*len(names)
    for i in range(n):
        nodes[i]=node(names[i],probs[i]/ptot)
    for i in range(n-1):
        probs,nodes=merge_nodes(probs,nodes)
    return nodes[0]

def decode_message(encoded,tree):
    message=''
    #while len(encoded)>0:
    while 1:
        tmp,encoded=tree.decode_token(encoded)
        if tmp=='eom':
            return message
        else:
            message=message+tmp
    return None
def encode_message(message,mydict):
    encoded=''
    for i in range(len(message)):
        encoded=encoded+mydict[message[i]]
    return encoded

def make_ptable(dat):
    mydict={}
    for i in range(len(dat)):
        if dat[i] in mydict.keys():
            mydict[dat[i]]=mydict[dat[i]]+1
        else:
            mydict[dat[i]]=1
    mydict['eom']=1
    names=[key for key in mydict.keys()]
    probs=[mydict[key] for key in mydict.keys()]
    return names,probs

def encoded_to_arr(encoded):
    tmp=np.empty(1,dtype='int')
    nbit=tmp.nbytes*8
    remainder=len(encoded)%nbit
    if remainder>0:
        encoded=encoded+'0'*(nbit-remainder)
    nword=len(encoded)//nbit
    myarr=np.empty(nword,dtype='uint')
    for i in range(nword):
        i1=i*nbit
        i2=(i+1)*nbit
        tmp=encoded[i1:i2]
        try:
            #myarr[i]=int(encoded[i1:i2],2)            
            myarr[i]=int(tmp,2)
        except:
            print('failed had len ',len(tmp),' with message ',tmp)
    return myarr

file=open('huffman_fromfile.py')
dat=file.read()
file.close()

names,probs=make_ptable(dat)
tree=build_tree(names,probs)

print('average bits per character: ',tree.get_length())
pp=np.asarray(probs,dtype='float')
pp=pp/pp.sum()
print('ideal would have been ',-np.sum(pp*np.log2(pp)))

mydict=tree.to_dict()
encoded=encode_message(dat,mydict)+mydict['eom']
print('we compressed to ',len(encoded)/8,' bytes')

to_check=decode_message(encoded,tree)
myarr=encoded_to_arr(encoded)
f=open('myself.compressed','w')
myarr.tofile(f)
