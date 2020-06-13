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
nu = 0.05
sigma = 0.25
dt = sigma * dx * dy / nu #dt is the amount of time each timestep covers (delta t)

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

        # Vectorized | most of the time is still wasted plotting
        u[1:-1, 1:-1] = un[1:-1,1:-1] + nu*dt/dx**2*(un[2:,1:-1] - 2*un[1:-1, 1:-1] + un[:-2,1:-1]) + nu*dt/dy**2 * (un[1:-1,2:] - 2*un[1:-1,1:-1] + un[1:-1,:-2])
        u[0,:]  = 1
        u[-1,:] = 1
        u[:,0]  = 1
        u[:,-1] = 1


# set up the figure and axis
fig = plt.figure(figsize=(9,9))
ax1 = fig.add_subplot(111, projection='3d') 
surface = [ax1.plot_surface(xx, yy, u, color='0.75', rstride=1, cstride=1)]

# set axis parameters
ax1.set_xlim([0,2])
ax1.set_ylim([0,2])
ax1.set_zlim([1,2])

ax1.set_title(r'$u$(x,y,T)')

def animate(i, surface):
    iterate()
    ax1.collections.clear()
    surface[0] = ax1.plot_surface(xx, yy, u, cmap="magma",edgecolor='none')

# init_func=init, blit=True
ani = animation.FuncAnimation(
    fig, animate, interval=1000/25, save_count=nt, frames=nt, fargs=([surface]))

ani.save("2D_diffusion.mp4",dpi=600)
# plt.show() # I need to change nx and ny to 21 to get the plots to render realtime smoothly
