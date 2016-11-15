# -*- coding: utf-8 -*-
import numpy as np
'''
#print(np.random.randn(12,5))
size =[5,8,7]
#print(size[:-1])
#print(size[1:])
weight = [np.random.rand(y,x) for x,y in zip(size[:-1],size[1:])]
#print(weight)
#print(np.zeros(（10,1）))
a=[1,2,3]
a=np.reshape(a,(3,1))
print(a)
b=[[1,1,1],
   [2,2,2],
   [3,3,3],
   [4,4,4]]
for x,y in zip(a,b):
    print(x)
    print(y)
'''
delta_nabla_b = [[1,1,1],[1,1,1]]
nabla_b=[[2,2,2],[2,2,2]]
for x,y in zip(delta_nabla_b,nabla_b):
    print(x,y)
print("---------")
s = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
print(s)