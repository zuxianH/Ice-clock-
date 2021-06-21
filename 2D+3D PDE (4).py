#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import *
from IPython import *
from tqdm import tqdm
from scipy.integrate import * 
from time import time
from scipy.interpolate import *
from numpy import * 
from matplotlib.pyplot import *
from numba import njit
from jupyterthemes.jtplot import *
from scipy.optimize import curve_fit
style(theme='monokai',fscale=1,figsize=(16,14),grid=False,ticks=True)


# 2D heat equation (with shape)

# In[2]:


@njit # skapar isen
def ice2d(L,R,T0,S):
    for x in range(L):
        for y in range(L):
            if (x-L//2)**2 + (y-L//2)**2 <= R**2:
                T0[x,y] = T_ice
                S[x,y] = T_oil
    return T0,S

@njit # centrala algoritmen
def solve2d(heatmap,S,radius,stefan,f=0): # beräknar temperaturförändringen
    ice,S = heatmap[0],S[0]
    t = 0
    for t in range(1,total_time): # löser heat eq för varje sekund
        for x in range(L-1):
            for y in range(L-1):
                # beräknar det nya temperaturen baserad på omgivningens temperatur
                ice[x,y] += c*(ice[x+1,y]+ice[x-1,y]+ice[x,y+1]+ice[x,y-1]-4*ice[x,y])
                if ice[x,y] > 0 and stefan: 
                    ice[x,y] = T_oil # ommedelbart höja isens temperatur
                S[x,y] = ice[x,y]    # definierar ytans temperatur
        if t % snapshot == 0:        # lagrar 1 frame vart "snapshot" enhet
            heatmap[f] = ice         # uppdaterar temperaturen
            radius[f] = count_nonzero(S[:L//2,L//2] - ice[x,L//2])
            f += 1
        if ice[L//2,L//2] > 0: # stoppvillkor
            break
        
    return heatmap,radius[:count_nonzero(radius)]/factor,t

# skapar isen
@njit
def ice3d(L,R,T0,S):
    for x in range(L):
        for y in range(L):
            for z in range(L):
                if (x-L//2)**2 + (y-L//2)**2 + (z-L//2)**2 <= R**2:
                    T0[x,y,z] = T_ice
                    S[x,y,z] = T_oil
    return T0,S

@njit # centrala algoritmen
def solve3d(heatmap,S,radius,stefan,f=0): # beräknar temperaturförändringen
    ice,S = heatmap[0],S[0]
    t = 0
    for t in range(1,total_time): # löser heat eq för varje sekund
        for x in range(L-1):
            for y in range(L-1):
                for z in range(L-1):
                    # beräknar det nya temperaturen baserad på omgivningens temperatur
                    ice[x,y,z] += c*(ice[x+1,y,z]+ice[x-1,y,z]+                                     ice[x,y+1,z]+ice[x,y-1,z]+                                     ice[x,y,z+1]+ice[x,y,z-1]+                                     -6*ice[x,y,z])
                    
                    if ice[x,y,z] > 0 and stefan: 
                        ice[x,y,z] = T_oil # ommedelbart höja isens temperatur
                    S[x,y,z] = ice[x,y,z]    # definierar ytans temperatur
            
        if t % snapshot == 0:        # lagrar 1 frame vart "snapshot" enhet
            heatmap[f] = ice         # uppdaterar temperaturen
            radius[f] = count_nonzero(S[:L//2,L//2,L//2] - ice[x,L//2,L//2])
            f += 1
        if ice[L//2,L//2,L//2] > 0: # stoppvillkor
            break
    return heatmap,radius[:count_nonzero(radius)]/factor,t

# konstanter [cm^2/w]
a_water = 0.143e-2
a_oil = 0.15e-2
a_ice = 1.02e-2
ds,dt = 1,1 # ds=1pixel, dt=1s
c = a_ice*dt/ds**2


# In[9]:


"""2D heat equation"""
# verklighet:Python = 1cm:59element
factor = 59
R = round(factor*1) # radien hos isen: cm
L = round(2*(R+6))  # lådans bredd: cm
T_ice = -5
T_oil = 10
# skapar lådan 
T0 = zeros((L,L)) + T_oil
T0 = flip(T0,axis=0)
S = zeros((L,L))     # isens yta
T0,S = ice2d(L,R,T0,S) # temperatur och geometri 

snapshot = 50 # hur ofta varje frames lagras (normaliserad vid 1)   
frames = 34 # frames(R,snapshot)  # totala antal frames
total_time = frames*snapshot # totala tiden som simuleras [s]
# print(f'frames: {frames}')

# skapar arrayer med frames
heat_frames = empty((frames,L,L)) 
shape_frames = heat_frames.copy()
heat_frames[0],shape_frames[0] = T0,S
radius = zeros(frames) # radien 

"""lösningen till heat equation (i frames)"""
heat_frames,radius,final_time = solve2d(heat_frames,shape_frames,radius,stefan=True)
print(f'smälttid: {final_time}s, {round(final_time/60)}min, {round(final_time/3600,1)}h')
print(f'rekommenderat frames: {ceil(final_time/snapshot)}')

# plott grejer
t = linspace(0,final_time,len(radius))
plot(t,radius,label=f'data',c='w',ls='dotted',lw=3)

# kurvanpassning
def f(t,a,b,c):
    return a + b*t + c*t**2
s,cov = curve_fit(f,t,radius)
plot(t,f(t,s[0],s[1],s[2]),
     label=f'anpassad',lw=2,c='r')
for i in range(3):
    print(s[i])
print(f'stdev: {sqrt(diagonal(cov))}')

xlabel(f'tid [s]'),ylabel('radie [cm]'),legend(loc='upper center')
title('2D')
show()


# In[10]:


"""3D heat equation"""

# verklighet:Python = 1cm:59element
factor = 59
R = round(factor*1) # radien hos isen
L = round(2*R+6) # lådans bredd 
T_ice = -5
T_oil = 10

# skapar lådan 
T0 = zeros((L,L,L)) + T_oil
T0 = flip(T0,axis=0)
S = zeros((L,L,L))     # isens yta
T0,S = ice3d(L,R,T0,S) # temperatur och geometri 

snapshot = 50 # hur ofta varje frames lagras (normaliserad vid 1)   
frames = 20 # frames(R,snapshot)  # totala antal frames
total_time = frames*snapshot # totala tiden som simuleras [s]

# skapar arrayer med frames
heat_frames = empty((frames,L,L,L)) 
shape_frames = heat_frames.copy()
heat_frames[0],shape_frames[0] = T0,S
radius = zeros(frames) # radien 

"""lösningen till heat equation (i frames)"""
heat_frames,radius,final_time = solve3d(heat_frames,shape_frames,radius,stefan=True)
print(f'smälttid: {final_time}s, {round(final_time/60)}min, {round(final_time/3600,1)}h')
print(f'rekommenderat frames: {ceil(final_time/snapshot)}')

t = linspace(0,final_time,len(radius))
plot(t,radius,label=f'data',c='w',ls='dotted',lw=3)

# kurvanpassning
def f(t,a,b,c):
    return a + b*t + c*t**2
s,cov = curve_fit(f,t,radius)
plot(t,f(t,s[0],s[1],s[2]),
     label=f'anpassad',lw=2,c='r')
for i in range(3):
    print(s[i])
print(f'stdev: {sqrt(diagonal(cov))}')

xlabel(f'tid [s]'),ylabel('radie [cm]'),legend(loc='upper center')
title('3D')
show()


# In[ ]:


"""animering"""
fig, ax = subplots(figsize=(14,14))
cmap = get_cmap('inferno') # inferno

def animate(t):
    ax.contourf(heat_frames[t],1,cmap=cmap,vmin=T_ice,vmax=T_oil)
    if t % (frames//100) == 0 and t != 0: # progressbar
        print(round(100*t/frames),end='% ')
    return fig,

anim = FuncAnimation(fig,animate,frames=frames,interval=200)
video = anim.to_html5_video() 
html = display.HTML(video) 
display.display(html) 
close() 

