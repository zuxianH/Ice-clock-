#!/usr/bin/env python
# coding: utf-8

# In[11]:


from numpy import * 
from matplotlib.pyplot import *
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import *
from matplotlib import *
from skimage import *
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


width = 49
length = 50
xv, yv = meshgrid(width, length)
img = color.rgb2gray(io.imread('IceBall.png'))
img = flip(img, axis=0)
Ice_bool = img<0.9

# konstanter
a_water = 0.143e-6
a_Ice = 1.02e-6
T_i = -10 + 273.15
T_o = 20 + 273.15


init_heat = zeros((width,length)) + T_o
init_heat[Ice_bool] = T_i

times = 36000
times_snapshot = 3600
heat_frames = zeros((times_snapshot, width, length))
heat_frames[0] = init_heat

x = 0.5
dx = 0.5/100
dt = 1


def solve_heat(heatmap, ice):
    cs = heatmap[0].copy() # current state
    cf = 0 # current frame
    for t in tqdm(range(1,times)):
        ns = cs.copy() # new state
        for x in range(1, width-1):
            for y in range(1, length-1):
                if ice[x][y]: # om det är is
                    a = a_Ice 
                    ns[x][y] = cs[x][y] + a*dt/dx**2 * (cs[x+1][y] + cs[x-1][y] +                                                    cs[x][y+1] + cs[x][y-1] -                                                    4*cs[x][y])
        cs = ns.copy()
        if t%10==0:
            cf += 1
            heatmap[cf] = cs
            
    return heatmap

heat_frames = solve_heat(heat_frames, Ice_bool)

heat_frames -= 273.15
cmap = get_cmap('inferno')
a = contourf(heat_frames[1000], 100, cmap=cmap, vmin=T_i-273.15, vmax=T_o-273.15)
colorbar(a)

# fig, ax = subplots(figsize=(14,14))
# def animate(i):
#     ax.contourf(heat_frames[10*i], 100, cmap=cmap, vmin=-10, vmax=20)
#     if i % 36 == 0: 
#         print(f'{10*(i%36)}% klar')
#     return fig,


# anim = FuncAnimation(fig, animate,frames=359, interval=50)
# video = anim.to_html5_video() # skapar en video 
# html = display.HTML(video) # gör den möjlig för visualisering
# display.display(html) # spela videon genom att kalla på den
# close() # eliminerar onödiga figurer


# In[ ]:




