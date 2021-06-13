#!/usr/bin/env python
# coding: utf-8

# In[6]:


from numpy import * 
from matplotlib.pyplot import *
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import *
from matplotlib import *
from skimage import *
from skimage import color,io
from IPython import *
import numba
from tqdm import tqdm
from numba import jit
from jupyterthemes import jtplot
from scipy.integrate import *
jtplot.style(theme='monokai')
jtplot.style(fscale=1, gridlines='--')
jtplot.style(ticks=True, grid=False, figsize=(14,14))


# In[ ]:


"""3D heat equation"""

# konstanter
a_water = 0.143e-6
a_oil = 0.15e-6 
a_ice = 1.02e-6
T_ice = -10 
T_oil = 20 

R = 10  # radien hos isen
L = 50 # lådans bredd 

# skapar lådan 
init_heat = zeros((L,L,L)) + T_oil
init_heat = flip(init_heat,axis=0)

# skapar isen
for x in range(L):
    for y in range(L):
        for z in range(L):
            if (x-L//2)**2 + (y-L//2)**2 + (z-L//2)**2 <= R**2:
                init_heat[x,y,z] = T_ice

time = 2000  # totala tiden som simuleras 
frames = 100 # totala antal frames 
snapshot = time//frames # anger hur ofta en frame är lagrad
heat_frames = zeros((frames,L,L,L)) # "frames" antal frames stackade på varandra
heat_frames[0] = init_heat # första bilden är init_heat

dx = 1/L # ett steg i rummet 
dt = 1   # ett steg i tiden 

def solve_heat(heatmap):
    T = heatmap[0] # begynnelse temperatur
    f = 0          # nuvarande frame
    for t in tqdm(range(1,time)): # itererar över tiden
        for x in range(L-1):
            for y in range(L-1):
                for z in range(L-1):
                
                    if T[x,y,z] < 0: # om temperaturen är under 0
                        a = a_ice  
                    else:
                        a = a_oil
                
                    # beräknar det nya temperaturen i rummet 
                    T[x][y][z] = (1-6*a*dt/dx**2) * T[x][y][z] +                     a*dt/dx**2 *                     (T[x+1][y][z] + T[x-1][y][z] +                      T[x][y+1][z] + T[x][y-1][z] +                      T[x][y][z+1] + T[x][y][z-1])
                
        if t % snapshot == 0: # lagrar 1 frame för vart snapshot enhet
            heatmap[f] = T
            f += 1
            
    return heatmap

heat_frames = solve_heat(heat_frames) # vart 10e frames är lagrad       


# In[62]:


"""2D heat equation"""

# konstanter
a_water = 0.143e-6
a_oil = 0.15e-6 
a_ice = 1.02e-6
T_ice = -10 
T_oil = 20 

R = 30  # radien hos isen
L = 100 # lådans bredd 

# skapar lådan 
init_heat = zeros((L,L)) + T_oil
init_heat = flip(init_heat,axis=0)

# skapar isen
for x in range(L):
    for y in range(L):
        if (x-L//2)**2 + (y-L//2)**2 <= R**2:
            init_heat[x,y] = T_ice

time = 2000  # totala tiden som simuleras 
frames = 100 # totala antal frames 
tick = time//frames
heat_frames = zeros((frames, L, L)) # "frames" antal frames stackade på varandra
heat_frames[0] = init_heat # första bilden är init_heat

dx = 1/L # ett steg i rummet 
dt = 1   # ett steg i tiden 


def solve_heat(heatmap):
    T = heatmap[0] # begynnelse temperatur
    f = 0          # nuvarande frame
    for t in tqdm(range(1,time)): # itererar över tiden
        for x in range(L-1):
            for y in range(L-1):
                
                if T[x,y] < 0: # om temperaturen är under 0
                    a = a_ice  
                else:
                    a = a_oil
                
                # beräknar det nya temperaturen i rummet 
                T[x][y] = T[x][y] + a*dt/dx**2 *                 (T[x+1][y] + T[x-1][y] + T[x][y+1] + T[x][y-1]                  - 4 * T[x][y])
                
        if t % tick == 0: # lagrar 1 frame vart "tick" tidsenhet
            heatmap[f] = T
            f += 1
            
    return heatmap

heat_frames = solve_heat(heat_frames) # vart 10e frames är lagrad       


# In[59]:


"""animering"""

fig, ax = subplots(figsize=(14,14))
def animate(t):
    ax.contourf(heat_frames[t], cmap=cmap, vmin=T_ice, vmax=T_oil)
    if t % 10 == 0 and t > 0: # progressbar
        print(round(100*(t+0)/frames),end='% ')
    return fig,
  
anim = FuncAnimation(fig, animate,frames=frames, interval=100)
video = anim.to_html5_video() 
html = display.HTML(video) 
display.display(html) 
close() 

