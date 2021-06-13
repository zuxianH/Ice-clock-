#!/usr/bin/env python
# coding: utf-8

# In[2]:


from numpy import * 
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
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


# In[5]:


"""2D heat equation (with shape)"""

# cm^2/w
a_water = 0.143e-2
a_oil = 0.15e-2
a_ice = 1.02e-2

# celcius
T_ice = -10 
T_oil = 20 

# verklighet:Python = 1cm:59element
factor = 59
R = round(factor*2) # radien hos isen: cm
L = round(factor*4.1) # lådans bredd: cm

# skapar lådan 
init_heat = zeros((L,L)) + T_oil
init_heat = flip(init_heat,axis=0)
heat_bool = zeros((L,L))
heat_bool = heat_bool != 0
 
# skapar isen
for x in range(L):
    for y in range(L):
        if (x-L//2)**2 + (y-L//2)**2 <= R**2:
            init_heat[x,y] = T_ice
            heat_bool[x,y] = True

time = 20000  # totala tiden som simuleras 
frames = 2000 # totala antal frames 
snapshot = 10

# skapar arrayer med frames
heat_frames = zeros((frames,L,L)) 
shape_frames = zeros((frames,L,L))
heat_frames[0] = init_heat 
shape_frames[0] = heat_bool
radius = []

ds = 1  # ett steg i rutan rummet 
dt = 1  # ett steg i tiden (s)

def solve_heat(heatmap,heat_bool):
    T,shape = heatmap[0],heat_bool[0] # begynnelse temperatur
    f = 0                             # start-frame
    # itererar över tiden
    for t in tqdm(range(time)):     
        for x in range(L-1):
            for y in range(L-1):
                if T[x,y] < 0: # om temperaturen är under 0
                    a = a_ice
                if T[x,y] >= 0:
                    a = a_oil
                if abs(T[x,y]) < 0.01: # det är is
                    shape[x,y] = True 
                if T[x,y] >= 0.01: # det är inte is 
                    shape[x,y] = False 
                # beräknar det nya temperaturen i rummet 
                T[x,y] += a*dt/ds**2 * (T[x+1,y] + T[x-1,y] + T[x,y+1] + T[x,y-1] - 4 * T[x,y])
        
        # sparar frames
        if t % snapshot == 0: # lagrar 1 frame vart "snapshot" enhet
            heatmap[f] = T
            heat_bool[f] = shape
            radius.append(count_nonzero(shape[:,L//2] == 1))
            f += 1
            
    return heatmap, heat_bool, array(radius)

# frames för värme och isens geometri 
heat_frames,shape_frames,radius = solve_heat(heat_frames,shape_frames)


# In[ ]:


timespan = arange(0,time,snapshot)
plot(timespan,radius)
xlabel('tid'),ylabel('radie')
show()


# In[ ]:


"""animering"""
fig, ax = subplots(figsize=(14,14))
cmap = get_cmap('Greys') # inferno

def animate(t):
#     ax.contourf(heat_frames[t], cmap=cmap, vmin=T_ice, vmax=T_oil)
#     ax.contourf(shape_frames[t],1,cmap=cmap)
    if t % snapshot == 0 and t != 0: # progressbar
        print(round(100*t/frames),end='% ')
    return fig,
  
anim = FuncAnimation(fig,animate,frames=frames,interval=100)
video = anim.to_html5_video() 
html = display.HTML(video) 
display.display(html) 
close() 


# In[ ]:


"""3D heat equation"""

# konstanter
a_water = 0.143e-6
a_oil = 0.15e-6 
a_ice = 1.02e-6
T_ice = -10 
T_oil = 20 

# verklighet:Python = 1cm:59element
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

dx = 1 # ett steg i rummet 
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

heat_frames = solve_heat(heat_frames) # vart "snapshot" frames är lagrad       

