#!/usr/bin/env python
# coding: utf-8

# In[45]:


import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import *
from IPython import *
from scipy.integrate import * 
from time import time
from scipy.interpolate import *


# 2D heat equation (with shape)

# In[43]:


from numpy import * 
from matplotlib.pyplot import *
from numba import njit
from jupyterthemes.jtplot import *
from scipy.optimize import curve_fit
style(theme='monokai',fscale=1,figsize=(16,14))

@njit # skapar isen
def ice(L,R,T0,S):
    for x in range(L):
        for y in range(L):
            if (x-L//2)**2 + (y-L//2)**2 <= R**2:
                T0[x,y] = T_ice
                S[x,y] = T_oil
    return T0,S

@njit # centrala algoritmen
def solve(heatmap,S,radius,stefan,f=0): # beräknar temperaturförändringen
    ice,S = heatmap[0],S[0]
    for t in range(1,total_time): # löser heat eq för varje sekund
        for x in range(L-1):
            for y in range(L-1):
                ice[x,y] += c*(ice[x+1,y]+ice[x-1,y]+ice[x,y+1]+ice[x,y-1]-4*ice[x,y])
                if ice[x,y] > 0 and stefan: 
                    ice[x,y] = T_oil # ommedelbart höja isens temperatur
                S[x,y] = ice[x,y]    # definierar ytans temperatur
        if t % snapshot == 0: # lagrar 1 frame vart "snapshot" enhet
            heatmap[f] = ice  # uppdaterar temperaturen
            radius[f] = count_nonzero(S[:L//2,L//2] - ice[x,L//2])
            f += 1
    return heatmap,radius[:count_nonzero(radius)]/factor

# konstanter [cm^2/w]
a_water = 0.143e-2
a_oil = 0.15e-2
a_ice = 1.02e-2
ds,dt = 1,1 # ds=1pixel, dt=1s
c = a_ice*dt/ds**2
T_ice = -10
T_oil = 10

# verklighet:Python = 1cm:59element
factor = 59
R = round(factor*2) # radien hos isen: cm
L = round(2*(R+6)) # lådans bredd: cm

# skapar lådan 
T0 = zeros((L,L)) + T_oil
T0 = flip(T0,axis=0)
S = zeros((L,L))     # isens yta
T0,S = ice(L,R,T0,S) # temperatur och geometri 

frames = 500 # totala antal frames 
snapshot = 30 # hur ofta varje frames lagras (normaliserad vid 1)    
total_time = frames*snapshot # totala tiden som simuleras 

# skapar arrayer med frames
heat_frames = empty((frames,L,L)) 
shape_frames = heat_frames.copy()
heat_frames[0],shape_frames[0] = T0,S
radius = zeros(frames) # radien 

"""lösningen till heat eq (i frames)"""
heat_frames,radius = solve(heat_frames,shape_frames,radius,stefan=True)

# plott grejer
t = linspace(0,len(radius),len(radius))
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

xlabel(f'tid*snapshot [s]'),ylabel('radie [cm]')
legend(loc='upper center')
show()


# In[ ]:


"""animering"""
t1 = time()
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
t2 = time()
print(t2-t1)


# In[ ]:


"""3D heat equation"""

# konstanter
a_water = 0.143e-2
a_oil = 0.15e-2
a_ice = 1.02e-2
T_ice = -10 
T_oil = 20 

# verklighet:Python = 1cm:59element
R = round(factor*1) # radien hos isen
L = round(factor*3) # lådans bredd 

# skapar lådan 
T0 = zeros((L,L,L)) + T_oil
T0 = flip(T0,axis=0)

# skapar isen
@njit
def ice_ball(L,R,T0):
    for x in range(L):
        for y in range(L):
            for z in range(L):
                if (x-L//2)**2 + (y-L//2)**2 + (z-L//2)**2 <= R**2:
                    T0[x,y,z] = T_ice
    return T0
ice_ball(L,R,T0)
# time = 2000  # totala tiden som simuleras 
# frames = 100 # totala antal frames 
# snapshot = time//frames # anger hur ofta en frame är lagrad
# heat_frames = zeros((frames,L,L,L)) # "frames" antal frames stackade på varandra
# heat_frames[0] = init_heat # första bilden är init_heat

# dx = 1 # ett steg i rummet 
# dt = 1   # ett steg i tiden 
# @njit
# def solve_heat(heatmap):
#     T = heatmap[0] # begynnelse temperatur
#     f = 0          # nuvarande frame
#     for t in range(1,time): # itererar över tiden
#         for x in range(L-1):
#             for y in range(L-1):
#                 for z in range(L-1):
                
#                     if T[x,y,z] < 0: # om temperaturen är under 0
#                         a = a_ice  
#                     else:
#                         a = a_oil
                
#                     # beräknar det nya temperaturen i rummet 
#                     T[x][y][z] = (1-6*a*dt/dx**2) * T[x][y][z] + \
#                     a*dt/dx**2 * \
#                     (T[x+1][y][z] + T[x-1][y][z] + \
#                      T[x][y+1][z] + T[x][y-1][z] + \
#                      T[x][y][z+1] + T[x][y][z-1])
                
#         if t % snapshot == 0: # lagrar 1 frame för vart snapshot enhet
#             heatmap[f] = T
#             f += 1
            
#     return heatmap

# heat_frames = solve_heat(heat_frames) # vart "snapshot" frames är lagrad       

