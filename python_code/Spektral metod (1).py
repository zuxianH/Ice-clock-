#!/usr/bin/env python
# coding: utf-8

# In[27]:


from numpy import * 
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import *
from skimage import *
from skimage import color,io
from IPython import *
from numba import njit
from tqdm import tqdm
from numpy.fft import *
from numba import jit
from scipy.optimize import curve_fit
from jupyterthemes import jtplot
from scipy.integrate import *
jtplot.style(theme='monokai')
jtplot.style(fscale=1.5, gridlines='--')
jtplot.style(ticks=True, grid=False, figsize=(14,14))


# In[224]:


"""3-dim heat-equation (spektral metoden)"""

def solve(time,u0hat,heat_frames,radius,central,f=0):
    for t in tqdm(range(time)):
        # analytiskt lösning till ODE-system
        uhat = exp(-3*a_ice*k**2*t)*u0hat # 3*k kommer ifrån 3D
        u = irfft(uhat)   # invers fourier transform 
        if t % snapshot == 0: # lagrar ett frame vart "snapshot" sekund
            u = clip(u,T_ice,0)
            diameter = count_nonzero(u)
            radius[f] = diameter/(2*cuts) # normaliserar med antal indelningar
            heat_frames[f] = u  # temperaturen för hela isen
            central[f] = min(u) # centrala (lägsta) temperaturen
            f += 1
        if min(u) > 0: 
            break
    return heat_frames,radius,central,t # temperaturen lagrad i en array

# konstanter
a_water = 0.143e-2
a_oil = 0.15e-2
a_ice = 1.02e-2
T_ice = -10
T_oil = 16

# egenskaper hos rummet
# välj lämpliga förhållande mellan L och R
L = int((T_oil - T_ice))  # lådans bredd
cuts = 100 # indelningar - upplösning
N = cuts*L # antal diskreta punkter
s = linspace(-L,L,N) # lådan
ds = s[1]-s[0]  
R = 3.7 # radie [cm]

# begynnelsevillkor
u = full(N,T_oil)
u[N//2-round(R*cuts):N//2+round(R*cuts)] = T_ice 

k = 2*pi*rfftfreq(N,ds) # frekvenskoefficienter i fourier rum
uhat = rfft(u)          # fourier transfrom
u0hat = uhat.copy()     # begynnelsevillkor i fourier rum   

frames = 593 # minst 100 frames för animering
snapshot = 2
time = snapshot*frames # tiden som simuleras [s]
heat_frames = zeros((frames,N))
radius = zeros(frames)
central = zeros(frames)

"""lösning"""
heat_frames,radius,central,final_time = solve(time,u0hat,heat_frames,central,radius) 
print(f'rekommenderat frame: {ceil(final_time/snapshot)}')
print(f'smälttid: {final_time}s, {round(final_time/60)}min')

"""plott"""
t = linspace(0,final_time/60,len(radius))
fig,ax = subplots(2,1)

sca(ax[0])
plot(t,radius,label=f'Förändring hos radien',ls='dotted',c='w',lw=3)
xlabel(f'tid [min]'),ylabel('radie [cm]'),legend(loc='upper center')
title('3D (spektral metod)')

sca(ax[1])
plot(t,central,label=f'Centrala temperaturen',lw=3)
xlabel(f'tid [min]'),ylabel('temperatur [c]'),legend(loc='upper center')
show()


# In[183]:


"""animering"""
# plott grejer
fig = plt.figure()
ax = fig.add_subplot(111)
set_cmap('jet_r')
xlabel('rum [cm]'),ylabel('temperatur [c]')
ylim(T_ice,T_oil)
theta = linspace(0,2*pi,100)

def animate(t):
    ax.plot(s/2,heat_frames[t],            color=cm.jet(0.9*(frames-t)/frames))
    ax.plot(radius[t]*cos(theta),6+radius[t]*sin(theta)/2,            color=cm.jet(0.9*(frames-t)/frames))
    if t % (frames//50) == 0 and t != 0: # progressbar
        print(round(100*t/frames),end='% ')
    return fig,

anim = FuncAnimation(fig,animate,frames=frames,interval=100)
video = anim.to_html5_video() 
html = display.HTML(video) 
display.display(html) 
close() 


# In[77]:


# kurvanpassning
def f(t,a,b,c,d):
    return a + b*t + c*t**2 + d*t**3
s,cov = curve_fit(f,t,radius)
plot(t,f(t,s[0],s[1],s[2],s[3]),
     label=f'anpassad',lw=2,c='r')
for i in range(4):
    print(s[i])
print(f'stdev: {sqrt(diagonal(cov))}')


# In[ ]:


# fig = plt.figure()
# ax = fig.add_subplot(111) #, projection='3d'
# set_cmap('jet')

