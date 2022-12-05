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
    while len(encoded)>0:
        tmp,encoded=tree.decode_token(encoded)
        message=message+tmp
    return message
def encode_message(message,mydict):
    encoded=''
    for i in range(len(message)):
        encoded=encoded+mydict[message[i]]
    return encoded


#names=['a','b','c']
#probs=[0.4,0.25,0.35]
#names=['a','c','e','d','b']
#probs=[7,6,5,2,1]
names=['a','b','c','d','e','f','g']
probs=[1,1,2,4,8,8,8]
tree=build_tree(names,probs)
mydict={}
tree.print(mydict=mydict)


print('average bits per character: ',tree.get_length())
pp=np.asarray(probs,dtype='float')
pp=pp/pp.sum()
print('ideal would have been ',-np.sum(pp*np.log2(pp)))

message='cede'
encoded=encode_message(message,mydict)
#encoded=''
#for i in range(len(message)):
#    encoded=encoded+mydict[message[i]]
print(message,' when encoded is ',encoded)
to_check=decode_message(encoded,tree)
print('when we decode it we get ',to_check)
