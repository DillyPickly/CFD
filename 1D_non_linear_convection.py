# The wave equation that we are looking into is du/dt + u*du/dx = 0
# Notice the wave speed is the solution.
# The numeric solution using foreward and backward difference method is:
# u_i,n+1 = u_i,n - u_i,n*(delta_t/delta_x)*(u_i,n - u_i-1,n)
# i is the spacial index while n is the temporal index 

import numpy as np
import time, sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation

nx = 41 # spacial points
dx = 2 / (nx - 1)
nt = 500 #nt is the number of timesteps we want to calculate
dt = 0.005 #dt is the amount of time each timestep covers (delta t)
c = 1 #assume wavespeed of c = 1

u = np.ones(nx)
u[int(.5 / dx):int(1 / dx+1)] = 2
un = np.ones_like(u) # temporal to hold old data

def iterate():
        un = u.copy() # copy the old values
        for i in range(1,nx):
            u[i] = un[i] - un[i] * dt / dx * (un[i] - un[i-1])


fig, ax = plt.subplots()
line, = ax.plot(np.linspace(0,2,nx),u)

def init():  # only required for blitting to give a clean slate.
    line.set_ydata([np.nan] * len(u))
    return line,

def animate(i):
    iterate()
    line.set_ydata(u)  # update the data.
    return line,

ani = animation.FuncAnimation(
    fig, animate, init_func=init, interval=6, blit=True, save_count=nt, frames=nt)

ani.save("1D_non_linear_convection.mp4")
# plt.show()