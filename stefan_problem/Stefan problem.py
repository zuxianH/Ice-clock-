#!/usr/bin/env python
# coding: utf-8

# In[2]:


from numpy import * 
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
from numpy.fft import *
from numba import njit
from matplotlib.animation import *
from IPython import *
from scipy.optimize import curve_fit
from jupyterthemes import jtplot
from scipy.integrate import *
jtplot.style(fscale=1.5, gridlines='--',theme='monokai',ticks=True, grid=False, figsize=(14,14))


# In[6]:


"""2D heat equation"""
# konstanter [cm^2/w]
# verklighet:Python = 1cm:59element
factor = 59
R = round(factor*1) # radien hos isen: cm
L = round(2.2*R)  # lådans bredd: cm
T_ice = -5
T_oil = 20
a_water = 0.143e-2
a_oil = 0.15e-2
a_ice = 1.02e-2
ds = 1 # ds = 1pixel
dt = 1 # sekunder
c = a_ice*dt/ds**2
d = a_oil*dt/ds**2

# skapar lådan 
T0 = zeros((L,L)) + T_oil
T0 = flip(T0,axis=0)

@njit # skapar isen
def IceDisk(L,R,T0):
    for x in range(L):
        for y in range(L):
            if (x-L//2)**2 + (y-L//2)**2 <= R**2:
                T0[x,y] = T_ice
    return T0

@njit
def IceSquare(L,T0):
    for x in range(L):
        for y in range(L):
            if x < L//2 + L//4 and x > L//4             and y < L//2 + L//4 and y > L//4:
                    T0[x,y] = T_ice
    return T0

@njit # centrala algoritmen
def solve2d(heatmap,r,stefan,f=0): # beräknar temperaturförändringen
    ice = heatmap[0]
    ice_enthalpy = 0 # tillskottsenergi för fasövergång
    for t in range(total_time): # löser heat eq för varje sekund
        for x in range(L):
            for y in range(L):
                # beräknar det nya temperaturen baserad på omgivningens temperatur
                ice[x,y] += c*(ice[x+1,y]+ice[x-1,y]+ice[x,y+1]+ice[x,y-1]-4*ice[x,y])
                if ice[x,y] > 0 + ice_enthalpy and stefan: 
                    ice[x,y] = 2    # temperaturen omkring isen
        if t % snapshot == 0:           # lagrar 1 frame vart "snapshot" enhet
            heatmap[f] = ice        # uppdaterar temperaturen
            r[f] = count_nonzero( ice[:L//2,L//2] - ice[x,L//2] )
            f += 1
        if ice[L//2,L//2] > 0: # stoppvillkor
            break
    return heatmap, r/factor, t

# tid
snapshot = 5 # hur ofta varje frames lagras (normaliserad vid 1)   
frames = 1420     # totala antal frames
fps = 10
total_time = frames*snapshot # totala tiden som simuleras [s]

# skapar arrayer med frames
heat_frames = empty((frames,L,L)) 
heat_frames[0] = IceDisk(L,R,T0) # IceSquare(L,T0,S)
r = zeros(frames) # radien

"""lösningen till heat equation (i frames)"""
heat_frames, r, final_time = solve2d( heat_frames, r, stefan=True )
print(f'smälttid: {final_time:.0f}s, {final_time/60:.1f}min, {final_time/3600:.1f}h')
print(f'rekommenderat frames: {ceil(final_time/snapshot)}')

# plott grejer
t = linspace(0,final_time/60,len(r))
plot(t,r,label=f'data',c='w',ls='dotted',lw=3)

# kurvanpassning
def f(t,a,b):
    return a*t+b
s,cov = curve_fit(f,t,r)
plot(t,f(t,*s),label=f'anpassad',lw=2,c='r')
print('parametrar:',*s), print(f'stdev: {sqrt(diagonal(cov))}')

xlabel(f'tid [min]'),ylabel('radie [cm]'),legend(loc='upper center')
title('2D')
show()


# In[7]:


# animering
theta = linspace(0,2*pi,100)
fig,axes = subplots(1,2,figsize=(20,9))
left,right = axes
counter = left.text(1.0,1.05,'',fontsize=20,transform=left.transAxes)
line1, = left.plot([],[],lw=2,color=cm.jet(0.2))

# axelns egenskaper
left.set_xlabel('cm'), left.set_ylabel('cm')
left.set_xlim(0,L/factor), left.set_ylim(0,L/factor)
right.set_xlabel('pixlar'), right.set_ylabel('pixlar')
right.set_xlim(0,L), right.set_ylim(0,L)

def animate(f): 
    right.contourf(heat_frames[f],cmap='coolwarm'
                   ,vmin=T_ice, vmax=T_oil) # inferno coolwarm plasma
#     left.contourf(oil_frames[f],cmap='coolwarm',
#                   vmin=T_ice, vmax=T_oil) # inferno coolwarm plasma
    line1.set_data(r[f]*cos(theta)+L//2/factor,
                   r[f]*sin(theta)+L//2/factor)
    
    counter.set_text('t = {:0.2f}min'.format(f*snapshot/60)) # styr tiden 
    if f % (frames//100) == 0 and f != 0: # progressbar
        print(round(100*f/frames),end='% ')
    return fig,

anim = FuncAnimation(fig,animate,frames,interval=1000/fps)
name = 'stefan problem'
anim.save(name+'.gif',fps=fps,writer='pillow')


# In[ ]:


T_oil = 25
T_ice = -10 
L = 100
frames = 200
snapshot = 2
time = frames*snapshot

dt,dx = 1,1
s1 = 1
k = 0.001
a_ice = 0.01


# begynnelsevillkor
u = zeros(L) 
u[:40] = T_oil
u[60:] = T_oil
u[ u!=T_oil ] = T_ice
u_frames = zeros([frames,L])

for t in range(1,time):
    f = 0
    for x in range(L-1):
        
        s0 = s1
        s1 += dt*k*( u[x+dx] - u[x] ) / dx 
        
        # uppdaterar temperaturen
        u[x] += 0*x/s1* (s1-s0)/dt * (u[x+dx] - u[x])/dx +         a_ice * ( u[x+dx] + u[x-dx] - 2*u[x] ) / dx**2
        
        if t % snapshot == 0:
            u_frames[f] = u # lägger till frames
            f += 1
            
fig,ax = subplots(figsize=(14,14))

def animate(t):
    ax.plot(u_frames[t],c=cm.jet(0.01*t))
    if t % (frames//100) == 0 and t != 0: # progressbar
        print(round(100*t/frames),end='% ')
    return fig,

anim = FuncAnimation(fig,animate,frames,interval=1000/60)
name = 'stefan problem'
anim.save(name+'.gif',fps=60,writer='pillow')


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf,erfc
from scipy.optimize import fsolve
from scipy import sparse
from scipy.sparse.linalg import spsolve
import time

### Physical Constants ###
spy  = 60.*60.*24.*365.24           #Seconds per year
Lf = 3.335e5                        #Latent heat of fusion (J/kg)
rho = 1000.                        #Bulk density of water (kg/m3), density changes are ignored
Ks = spy*2.1                          #Conductivity of ice (J/mKs)
cs = 2009.                        #Heat capacity of ice (J/kgK) - ** Van der Veen uses 2097 but see Tr and Aschwanden 2012)
ks = Ks/(rho*cs)                #Cold ice diffusivity (m2/sec)
# Engineering Toolbox
Kl = spy*0.58                       #Conductivity of water (J/mKs)
cl = 4217.                      #Heat capacity of water (J/kgK)
kl = Kl/(rho*cl)

# Problem Constants
s0 = 0.
t0 = 0.
Tm = 0.0
T_ = -10.0
Tr = -16.33 

### Enthalpy Solution ###

# Test over several time step sizes
ax3 = plt.subplot(111)
for dt in [0.001,0.01,0.05,0.1,0.2,0.5]:
    N = 201
    l = 2.
    t = 0.
    dt /= 365
    dx=l/N
    xs = np.arange(0,l,dx)
    r = ks*dt/(dx**2)

    H = ((Tm-Tr)*cs+Lf)*np.ones(N)
    H[np.where(xs<=s0)] = (T_-Tr)*cs
    T = H/cs + Tr
    T[T>Tm]=Tm

    A = sparse.lil_matrix((N, N))
    A.setdiag((1+2*r)*np.ones(N))              # The diagonal
    A.setdiag(-r*np.ones(N-1),k=1)       # The fist upward off-diagonal.
    A.setdiag(-r*np.ones(N-1),k=-1)
    #Boundary Conditions
    A[0,:] = np.zeros(N) # zero out the first row
    A[0,0] = 1.0       # set diagonal in that row to 1.0
    A[-1,-1] = -2.*r
    A[-1,-2] = 2.*r
    # For performance, other sparse matrix formats are preferred.
    # Convert to "Compressed Row Format" CR
    A = A.tocsr()

    ts = np.array([t])
    PTB = np.array([np.min(xs[H>(0.0-Tr)*cs])])
    Mushy = [np.min(xs[H>(((Tm-Tr)*cs+Lf)+Tr*cs)*.95])]

    while t < 0.1:
        dT = spsolve(A,T)-T
        H += dT*cs
        T = H/cs + Tr
        T[T>Tm] = Tm
        t += dt
        ts = np.append(ts,[t])
        PTB = np.append(PTB,[np.min(xs[H>(0.0-Tr)*cs])])
        Mushy = np.append(Mushy,[np.min(xs[H>(((Tm-Tr)*cs+Lf)+Tr*cs)*.95])])

#     p1, = plt.plot(ts,PTB,'r',lw=2)

# Test over several spatial step sizes
for N in [21,51,101,201]:
    l = 2.
    dt = .01/365.#1000/spy
    t = 0.
    dx=l/N
    xs = np.arange(0,l,dx)
    r = ks*dt/(dx**2)

    H = ((Tm-Tr)*cs+Lf)*np.ones(N)
    H[np.where(xs<=s0)] = (T_-Tr)*cs
    T = H/cs + Tr
    T[T>Tm]=Tm

    A = sparse.lil_matrix((N, N))
    A.setdiag((1+2*r)*np.ones(N))              # The diagonal
    A.setdiag(-r*np.ones(N-1),k=1)       # The fist upward off-diagonal.
    A.setdiag(-r*np.ones(N-1),k=-1)
    #Boundary Conditions
    A[0,:] = np.zeros(N) # zero out the first row
    A[0,0] = 1.0       # set diagonal in that row to 1.0
    A[-1,-1] = -2.*r
    A[-1,-2] = 2.*r
    A = A.tocsr()

    ts = np.array([t])
    PTB = np.array(xs[np.min(np.where(H>(0.0-Tr)*cs))])
    
    while t < .1:
        dT = spsolve(A,T)-T
        H += dT*cs
        T = H/cs + Tr
        T[T>Tm]=Tm
        t += dt
        ts = np.append(ts,[t])
        PTB = np.append(PTB,xs[np.min(np.where(H>(0.0-Tr)*cs))])
    p2, = plt.plot(ts,PTB,'b',lw=2)

