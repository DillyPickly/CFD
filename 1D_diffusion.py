# The wave equation that we are looking into is du/dt = v*d^2u/dx^2
# Notice the wave speed is the solution.
# The numeric solution using foreward and backward difference method we can arrive on the second derivative.
# u_i,n+1 = u_i,n + v*(delta_t/delta_x^2)*(u_i+1,n -2u_n,i + u_i-1,n)
# i is the spacial index while n is the temporal index 

import numpy as np
import time, sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation

nx = 41 # spacial points
dx = 2 / (nx - 1)
nt = 500 #nt is the number of timesteps we want to calculate
sigma = .2 #sigma is a parameter, we'll learn more about it later
nu = 0.3 # viscosity
dt = sigma * dx**2 / nu #dt is defined using sigma ... more later!



u = np.ones(nx)
u[int(nx/4):int(3/4*nx)] = 2
un = np.ones_like(u) # temporal to hold old data

def iterate():
        un = u.copy() # copy the old values
        for i in range(1,nx-1):
            u[i] = un[i] + nu * dt / dx**2 * (un[i+1] -2*un[i] + un[i-1])


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

ani.save("1D_diffusion.mp4")
# plt.show()