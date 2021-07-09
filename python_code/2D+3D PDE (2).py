#!/usr/bin/env python
# coding: utf-8

# In[89]:


from matplotlib.pyplot import *
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import *
from IPython import *
from numba import *
from jupyterthemes import jtplot
from scipy.integrate import *
from scipy.optimize import * 
from scipy.interpolate import *
jtplot.style(theme='monokai')
jtplot.style(fscale=1, gridlines='--')
jtplot.style(ticks=True, grid=False, figsize=(14,14))


# 2D heat equation (with shape)

# In[92]:


from numpy import * 
# cm^2/w
a_water = 0.143e-2
a_oil = 0.15e-2
a_ice = 1.02e-2
# celcius
T_ice = -10 
T_oil = 20 
# verklighet:Python = 1cm:59element
factor = 59
R = factor*0.5 # radien hos isen: cm
L = round(factor*2.1) # lådans bredd: cm
# skapar lådan 
init_heat = zeros((L,L)) + T_oil
init_heat = flip(init_heat,axis=0)
surface = zeros((L,L))

# skapar isen
for x in range(L):
    for y in range(L):
        if (x-L//2)**2 + (y-L//2)**2 <= R**2:
            init_heat[x,y] = T_ice
            surface[x,y] = T_oil
ds,dt = 1,1

frames = 90 # totala antal frames 
snapshot = 10
time = snapshot*frames # totala tiden som simuleras 

# skapar arrayer med frames
heat_frames = empty((frames,L,L)) 
shape_frames = empty((frames,L,L))
heat_frames[0] = init_heat 
shape_frames[0] = surface
radius = empty(frames)

@njit
def solve(heatmap,surface,radius,constant): # beräknar temperaturförändringen
    T = heatmap[0]
    S = surface[0]
    f = 0
    for t in range(1,time):
        for x in range(L-1):
            for y in range(L-1):
                T[x,y] += a_ice*dt/ds**2 * (T[x+1,y]+T[x-1,y]+T[x,y+1]+T[x,y-1]-4*T[x,y])
                if T[x,y] > 0 and constant:
                    T[x,y] = T_oil 
                S[x,y] = T[x,y]
        if t % snapshot == 0: # lagrar 1 frame vart "snapshot" enhet
            heatmap[f] = T
            radius[f] = count_nonzero( S[:,L//2] - T[x,y] ) 
            f += 1 
    return heatmap,radius/factor  

heat_frames,radius = solve(heat_frames,shape_frames,radius,constant=True)


# In[103]:


# plott grejer
def f(t,a,b,c):
    return a + b*t + c*t**2 

t = linspace(0,time,len(radius))
s,cov = curve_fit(f,t,radius)

plot(t,radius,label=f'data',c='w',ls='dotted',lw=3)
plot(t,f(t,s[0],s[1],s[2]),label=f'anpassad',lw=2)

xlabel('tid [s]'),ylabel('diameter [cm]')
# xlim(0,time),ylim(0,2.1*R/factor)
for i in range(3):
    print(s[i])
sd = sqrt(diag(cov))
print(sd)
plt.legend(loc='upper center')
plt.show()


# In[13]:


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


# In[57]:


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
@njit
def solve_heat(heatmap):
    T = heatmap[0] # begynnelse temperatur
    f = 0          # nuvarande frame
    for t in range(1,time): # itererar över tiden
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

