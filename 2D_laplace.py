# The 2D laplace equation
# and difference methods used to numerically solve are described in
# https://nbviewer.jupyter.org/github/barbagroup/CFDPython/blob/master/lessons/12_Step_9.ipynb

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# 3D support
from mpl_toolkits.mplot3d import Axes3D

nx = 41 # spacial points
ny = 41
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)

# Set up graphing meshgrid
x = np.linspace(0,1,nx)
y = np.linspace(0,2,ny)

# Initial Conditions
u = np.zeros((nx, ny))
un = np.zeros_like(u) # temporal to hold old data

u[0,:]  = 0
u[-1,:] = y
u[:,0]  = u[:,1]
u[:,-1] = u[:,-2]



# How to update the initial conditions based on the equation
def laplace2D(u, un, dx, dy, tolerance):
    l1norm = 1

    while l1norm > tolerance:
        un = u.copy() # copy the old values

        # Vectorized | most of the time is still wasted plotting
        u[1:-1, 1:-1] = (dx**2*(un[2:,1:-1] + un[:-2,1:-1]) + dy**2*(un[1:-1,2:] + un[1:-1,:-2]))/(2*(dx**2+dy**2))

        u[0,:]  = 0
        u[-1,:] = y
        u[:,0]  = u[:,1]
        u[:,-1] = u[:,-2]

        l1norm = np.sum(np.abs(u - un))/np.sum(np.abs(un))
    
    return u


def plotter(x, y, u0, u1):
    xx, yy = np.meshgrid(x, y)

    # set up the figure and axis
    fig = plt.figure(figsize=(12,6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d') 
    ax1.plot_surface(xx, yy, u0, cmap="magma",edgecolor='none', color='0.75', rstride=1, cstride=1)
    ax2.plot_surface(xx, yy, u1, cmap="magma",edgecolor='none', color='0.75', rstride=1, cstride=1)

    # set axis parameters
    # ax1.set_xlim([0,2])
    # ax1.set_ylim([0,2])
    # ax1.set_zlim([1,2])

    # ax2.set_xlim([0,2])
    # ax2.set_ylim([0,2])
    # ax2.set_zlim([1,2])

    ax1.set_title(r'Initial Condition')
    ax2.set_title(r'Equilibrium Distribution')

    plt.savefig("2D_laplace.png",dpi=600)
    # plt.show() # I need to change nx and ny to 21 to get the plots to render realtime smoothly

u0 = u.copy()
u1 = laplace2D(u,un, dx, dy, 1e-5)
plotter(x, y, u0, u1)
