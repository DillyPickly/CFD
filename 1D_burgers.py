# The burgers equation comibines the diffusion and nonlinear wave equations. 
# du/dt + u*du/dx = v*d^2u/dx^2
# The numeric solution using foreward and backward difference method we can arrive on the second derivative.
#  after replacing all the derivatives with the discrete versions...
# u_i,n+1 = u_i,n - u_i,n *(delta_t/delta_x) * (u_i,n - u_i-1,n) + v*(delta_t/delta_x^2)*(u_i+1,n -2u_n,i + u_i-1,n)
# i is the spacial index while n is the temporal index 

# For this equation we will explore special Initial and Boundary conditions
# We will use the periodic boundary conditions u(0) = u(2*pi) (for all time)
# with the more complicated initial conditions which will form a saw tooth wave.
# u = -2v/phi*dphi/dx + 4
# phi = exp(-x^2/4v) + exp(-(x-2*pi)^2/4v)

# This inital condition has an analytical solution:
# u = -2v/phi*dphi/dx + 4
# phi = exp(-(x-4t)^2/4v(t+1)) + exp(-(x-4t-2*pi)^2/4v(t+1))

# This initial condition is a bit weildly to evaluate as we need to calculate the derivative.
# This is where sympy comes in

import numpy as np
import sympy
from sympy.utilities.lambdify import lambdify
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Creating the analytical solution.
x, nu, t = sympy.symbols('x nu t')
phi = (sympy.exp(-(x - 4 * t)**2 / (4 * nu * (t + 1))) +
       sympy.exp(-(x - 4 * t - 2 * sympy.pi)**2 / (4 * nu * (t + 1))))
phiprime = phi.diff(x)
u = -2 * nu * (phiprime / phi) + 4
ufunc = lambdify((t, x, nu), u)


nx = 101 # spacial points
dx = 2 * np.pi / (nx - 1)
nt = 500 #nt is the number of timesteps we want to calculate
nu = 0.07 # viscosity
dt = dx * nu 


# creating initial condition
x = np.linspace(0, 2 * np.pi, nx)
un = np.empty_like(x) # temporal to hold old data
t = 0
u = np.asarray([ufunc(t,x0, nu) for x0 in x])
u_analytical = np.copy(u)

def iterate(t):
    un = u.copy() # copy the old values
    for i in range(1,nx-1):
        u[i] = un[i] - un[i] * (dt/dx) * (un[i] - un[i-1]) + nu * dt / dx**2 * (un[i+1] -2*un[i] + un[i-1])
    u[0] = un[0] - un[0] * (dt/dx) * (un[0] - un[-2]) + nu * dt / dx**2 * (un[1] -2*un[0] + un[-2])
    u[-1] = u[0]

    u_analytical[:] = np.asarray([ufunc(t * dt, xi, nu) for xi in x])

fig, ax = plt.subplots()
line, = ax.plot(np.linspace(0,2*np.pi,nx),u)
line_a, = ax.plot(np.linspace(0,2*np.pi,nx),u)


def init():  # only required for blitting to give a clean slate.
    line.set_ydata([np.nan] * len(u))
    line_a.set_ydata([np.nan] * len(u))
    return line, line_a

def animate(i):
    iterate(i)
    line.set_ydata(u)  # update the data.
    line_a.set_ydata(u_analytical)
    return line, line_a

ani = animation.FuncAnimation(
    fig, animate, init_func=init, interval=6, blit=True, save_count=nt, frames=nt)

ani.save("1D_burgers.mp4")
# plt.show()