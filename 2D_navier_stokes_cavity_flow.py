# The 2D convection, the coupled equations 
# and difference methods used to numerically solve are described in
# https://nbviewer.jupyter.org/github/barbagroup/CFDPython/blob/master/lessons/08_Step_6.ipynb

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# 3D support
from mpl_toolkits.mplot3d import Axes3D

nx = 41 # spacial points
ny = 41
nt = 200 #nt is the number of timesteps we want to calculate
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
sigma = 0.0099
nu = 0.01
dt = sigma * dx * dy / nu #dt is the amount of time each timestep covers (delta t)


# Initial Conditions
u = np.zeros((nx, ny))
un = np.zeros_like(u) # temporal to hold old data

v = np.zeros((nx, ny))
vn = np.zeros_like(u) # temporal to hold old data

p = np.zeros((nx, ny))
pn = np.zeros_like(u) # temporal to hold old data

# Boundary Conditions
u[0,:]  = 0
u[-1,:] = 0
u[:,0]  = 0
u[:,-1] = 1

v[0,:]  = 0
v[-1,:] = 0
v[:,0]  = 0
v[:,-1] = 0


# Set up graphing meshgrid
x = np.linspace(0,2,nx)
y = np.linspace(0,2,ny)
xx, yy = np.meshgrid(x, y)

# How to update the initial conditions based on the equation
def iterate():
        un = u.copy() # copy the old values
        vn = v.copy()

        # Vectorized | most of the time is still wasted plotting
        u[1:-1, 1:-1] = (un[1:-1, 1:-1] 
                        - un[1:-1, 1:-1] * dt/dx * (un[1:-1, 1:-1] - un[:-2,1:-1]) 
                        - vn[1:-1, 1:-1] * dt/dy * (un[1:-1, 1:-1] - un[1:-1, :-2]) 
                        + nu*dt/dx**2*(un[2:,1:-1] - 2*un[1:-1, 1:-1] + un[:-2,1:-1]) 
                        + nu*dt/dy**2 * (un[1:-1,2:] - 2*un[1:-1,1:-1] + un[1:-1,:-2]))
        
        v[1:-1, 1:-1] = (vn[1:-1, 1:-1] 
                        - vn[1:-1, 1:-1] * dt/dx * (vn[1:-1, 1:-1] - vn[:-2,1:-1]) 
                        - un[1:-1, 1:-1] * dt/dy * (vn[1:-1, 1:-1] - vn[1:-1, :-2])
                        + nu*dt/dx**2*(vn[2:,1:-1] - 2*vn[1:-1, 1:-1] + vn[:-2,1:-1]) 
                        + nu*dt/dy**2 * (vn[1:-1,2:] - 2*vn[1:-1,1:-1] + vn[1:-1,:-2]))

        # Boundary Conditions
        u[0,:]  = 0
        u[-1,:] = 0
        u[:,0]  = 0
        u[:,-1] = 1

        v[0,:]  = 0
        v[-1,:] = 0
        v[:,0]  = 0
        v[:,-1] = 0

        p[0,:]  = p[1,:]
        p[-1,:] = p[-2,:]
        p[:,0]  = p[:,1]
        p[:,-1] = 0


# set up the figure and axis
fig = plt.figure(figsize=(12,6))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d') 
surface = [ax1.plot_surface(xx, yy, u, color='0.75', rstride=1, cstride=1),
            ax2.plot_surface(xx, yy, v, color='0.75', rstride=1, cstride=1)]

# set axis parameters
ax1.set_xlim([0,2])
ax1.set_ylim([0,2])
ax1.set_zlim([1,2])

ax2.set_xlim([0,2])
ax2.set_ylim([0,2])
ax2.set_zlim([1,2])

ax1.set_title(r'$u$(x,y,T)')
ax2.set_title(r'$v$(x,y,T)')

def animate(i, surface):
    iterate()
    ax1.collections.clear()
    ax2.collections.clear()
    surface[0] = ax1.plot_surface(xx, yy, u, cmap="magma",edgecolor='none')
    surface[1] = ax2.plot_surface(xx, yy, v, cmap="magma",edgecolor='none')

# init_func=init, blit=True
ani = animation.FuncAnimation(
    fig, animate, interval=1000/25, save_count=nt, frames=nt, fargs=([surface]))

ani.save("2D_burgers.mp4",dpi=600)
# plt.show() # I need to change nx and ny to 21 to get the plots to render realtime smoothly
