# The 2D convection, the coupled equations 
# and difference methods used to numerically solve are described in
# https://nbviewer.jupyter.org/github/barbagroup/CFDPython/blob/master/lessons/08_Step_6.ipynb

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# 3D support
from mpl_toolkits.mplot3d import Axes3D

nx = 81 # spacial points
ny = 81
nt = 200 #nt is the number of timesteps we want to calculate
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
sigma = 0.2
dt = sigma * dx #dt is the amount of time each timestep covers (delta t)
c = 1

# Initial Conditions
u = np.ones((nx, ny))
un = np.ones_like(u) # temporal to hold old data
u[int(nx*3/8):int(nx*5/8), int(ny*3/8):int(ny*5/8)] = 2

v = np.ones((nx, ny))
vn = np.ones_like(u) # temporal to hold old data
v[int(nx*3/8):int(nx*5/8), int(ny*3/8):int(ny*5/8)] = 2


# Set up graphing meshgrid
x = np.linspace(0,2,nx)
y = np.linspace(0,2,ny)
xx, yy = np.meshgrid(x, y)

# How to update the initial conditions based on the equation
def iterate():
        un = u.copy() # copy the old values
        vn = v.copy()
        # row, col = u.shape
        # for i in range(1,row):
        #     for j in range(1, col):
        #         u[i, j] = un[i,j] - un[i,j] * dt / dx * (un[i,j] - un[i-1,j]) - vn[i,j] * dt / dy * (un[i,j] - un[i,j-1])
        #         v[i, j] = vn[i,j] - vn[i,j] * dt / dx * (vn[i,j] - vn[i-1,j]) - un[i,j] * dt / dy * (vn[i,j] - vn[i,j-1])

        # Vectorized | most of the time is still wasted plotting
        u[1:-1, 1:-1] = un[1:-1, 1:-1] - un[1:-1, 1:-1] * dt/dx * (un[1:-1, 1:-1] - un[:-2,1:-1]) - vn[1:-1, 1:-1] * dt/dy * (un[1:-1, 1:-1] - un[1:-1, :-2])
        v[1:-1, 1:-1] = vn[1:-1, 1:-1] - vn[1:-1, 1:-1] * dt/dx * (vn[1:-1, 1:-1] - vn[:-2,1:-1]) - un[1:-1, 1:-1] * dt/dy * (vn[1:-1, 1:-1] - vn[1:-1, :-2])

        u[0,:]  = 1
        u[-1,:] = 1
        u[:,0]  = 1
        u[:,-1] = 1

        v[0,:]  = 1
        v[-1,:] = 1
        v[:,0]  = 1
        v[:,-1] = 1


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

ani.save("2D_non_linear_convection.mp4",dpi=600)
# plt.show() # I need to change nx and ny to 21 to get the plots to render realtime smoothly
