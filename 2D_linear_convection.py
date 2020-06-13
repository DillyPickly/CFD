# The 2D wave equation 
# and difference methods used to numerically solve are described in
# https://nbviewer.jupyter.org/github/barbagroup/CFDPython/blob/master/lessons/07_Step_5.ipynb

import numpy as np
import time, sys
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

# Set up graphing meshgrid
x = np.linspace(0,2,nx)
y = np.linspace(0,2,ny)
xx, yy = np.meshgrid(x, y)

# How to update the initial conditions based on the equation
def iterate():
        un = u.copy() # copy the old values
        row, col = u.shape
        for i in range(1,row):
            for j in range(1, col):
                u[i,j] = un[i,j] - c*dt/dx * (un[i,j] - un[i-1,j]) - c*dt/dy * (un[i,j] - un[i,j-1])

        u[0,:]  = 1
        u[-1,:] = 1
        u[:,0]  = 1
        u[:,-1] = 1


# set up the figure and axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surface = [ax.plot_surface(xx, yy, u, color='0.75', rstride=1, cstride=1)]

# set axis parameters
ax.set_xlim([0,2])
ax.set_ylim([0,2])
ax.set_zlim([1,2])

def animate(i, surface):
    iterate()
    ax.collections.clear()
    surface = ax.plot_surface(xx, yy, u, cmap="magma",edgecolor='none')

# init_func=init, blit=True
ani = animation.FuncAnimation(
    fig, animate, interval=1000/25, save_count=nt, frames=nt, fargs=(surface))

ani.save("2D_linear_convection.mp4",dpi=600)
# plt.show()
