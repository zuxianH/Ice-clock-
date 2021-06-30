#!/usr/bin/env python
# coding: utf-8

# In[4]:


from numpy import * 
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import *
from IPython import *
from skimage import *
from skimage import color,io
from numba import njit
from tqdm import tqdm
from numpy.fft import *
from numba import jit
from scipy.optimize import curve_fit
from jupyterthemes import jtplot
from scipy.integrate import *
jtplot.style(fscale=1.5, gridlines='--',theme='monokai',ticks=True, grid=False, figsize=(14,14))


# In[132]:


data3 = loadtxt('ice_3.txt') # experimentell data amplitud:tid
t3 = data3[200:400,0] # tiden, har klippt bort början och slutet 
y3 = data3[200:400,1] # amplitud

n3 = len(y3) # antal tidsindelningar
dt3 = (t3[-1] - t3[0])/len(t3) # genomsnitta tidsintervall 
freq3 = 1/(n3*dt3)*arange(n3)  # frekvensen för sväningen
# fourier trasform
# y3, amplitud:tid -> y3hat, energi:frekvens
y3hat = fft(y3)  
E3 = y3hat * conj(y3hat) / n3 # energin för isen  

# filter
ind = E3 < 1000 # filterar bort onödigt höga energinivåer
E3 = E3 * ind 
y3hat = y3hat * ind # filterar bort onödigt höga frekvenser
# invers fourier transform 
# y3hat, energi:frekvens -> y3, amplitud:tid
y3 = ifft(y3hat) 

# plottgrejer
fig,ax = subplots(2,1)

sca(ax[0])
plot(t3,y3.real)
ylabel('relativ höjd'),xlabel('tid')

sca(ax[1])
plot(freq3,E3.real)
ylabel('energi'),xlabel('frekvens')
ylim(0,500)

