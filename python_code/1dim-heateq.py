#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""1-dim heat-equation"""
from numpy import *
from matplotlib.pyplot import *

# konstanter
L = 0.5
n = 1000
a = L/n
h = 1e-3
D = 4.25e-6 # värmeledningsförmåga
c = h*D/a**2  

# begynnelsevillkor
t1,t2 = 0,50 # tid
T_inside, T_between, T_outside = 50,20,0 # temperatur

T = empty(n+1,float) # temperatur
# definierar om T mha begynnelsevillkor
T[0] = T_inside    # boundary-temp
T[1:n] = T_between # inside-temp
T[n] = T_outside   # boundary-temp

while t1 < t2: # itererar över tiden 
    T[1:n] = T[1:n] + c*( T[2:n+1] - 2*T[1:n] + T[0:n-1] )  # itererar över positionen mellan materialet
    t1 += h # öka tiden med en liten steg 

width = linspace(0,L,n+1)
plot(width, T)
xlabel('bredd'), ylabel('temperatur')
show()

