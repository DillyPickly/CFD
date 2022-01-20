# The 2D convection, the coupled equations 
# and difference methods used to numerically solve are described in
# https://nbviewer.jupyter.org/github/barbagroup/CFDPython/blob/master/lessons/08_Step_6.ipynb

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import pyplot, cm
# 3D support
from mpl_toolkits.mplot3d import Axes3D

nx = 81 # spacial points
ny = 81
nt = 100 #nt is the number of timesteps we want to calculate
nit = 100
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
rho = 1
nu = 0.1
F = 1
dt = .001 #dt is the amount of time each timestep covers (delta t)


# Initial Conditions
u = np.zeros((nx, ny))
un = np.zeros_like(u) # temporal to hold old data

v = np.zeros((nx, ny))
vn = np.zeros_like(u) # temporal to hold old data

p = np.zeros((nx, ny))
pn = np.zeros_like(u) # temporal to hold old data

b = np.zeros((nx, ny))

# Boundary Conditions
u[0,:]  = 0
u[-1,:] = 0
u[:,0]  = 0
u[:,-1] = 0

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
        pn = p.copy()

        # Vectorized | most of the time is still wasted plotting
        u[1:-1, 1:-1] = (un[1:-1, 1:-1]-
                        un[1:-1, 1:-1] * dt / dx *
                    (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                        vn[1:-1, 1:-1] * dt / dy *
                    (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                        dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                        nu * (dt / dx**2 *
                    (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                        dt / dy**2 *
                    (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])) + 
                        F*dt)
        
        v[1:-1,1:-1] = (vn[1:-1, 1:-1] -
                un[1:-1, 1:-1] * dt / dx *
                (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                vn[1:-1, 1:-1] * dt / dy *
                (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                nu * (dt / dx**2 *
                (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                dt / dy**2 *
                (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

        # Periodic BC u @ x = 2     
        u[1:-1, -1] = (un[1:-1, -1] - un[1:-1, -1] * dt / dx * 
                    (un[1:-1, -1] - un[1:-1, -2]) -
                    vn[1:-1, -1] * dt / dy * 
                    (un[1:-1, -1] - un[0:-2, -1]) -
                    dt / (2 * rho * dx) *
                    (p[1:-1, 0] - p[1:-1, -2]) + 
                    nu * (dt / dx**2 * 
                    (un[1:-1, 0] - 2 * un[1:-1,-1] + un[1:-1, -2]) +
                    dt / dy**2 * 
                    (un[2:, -1] - 2 * un[1:-1, -1] + un[0:-2, -1])) + F * dt)

        # Periodic BC u @ x = 0
        u[1:-1, 0] = (un[1:-1, 0] - un[1:-1, 0] * dt / dx *
                    (un[1:-1, 0] - un[1:-1, -1]) -
                    vn[1:-1, 0] * dt / dy * 
                    (un[1:-1, 0] - un[0:-2, 0]) - 
                    dt / (2 * rho * dx) * 
                    (p[1:-1, 1] - p[1:-1, -1]) + 
                    nu * (dt / dx**2 * 
                    (un[1:-1, 1] - 2 * un[1:-1, 0] + un[1:-1, -1]) +
                    dt / dy**2 *
                    (un[2:, 0] - 2 * un[1:-1, 0] + un[0:-2, 0])) + F * dt)

        # Periodic BC v @ x = 2
        v[1:-1, -1] = (vn[1:-1, -1] - un[1:-1, -1] * dt / dx *
                    (vn[1:-1, -1] - vn[1:-1, -2]) - 
                    vn[1:-1, -1] * dt / dy *
                    (vn[1:-1, -1] - vn[0:-2, -1]) -
                    dt / (2 * rho * dy) * 
                    (p[2:, -1] - p[0:-2, -1]) +
                    nu * (dt / dx**2 *
                    (vn[1:-1, 0] - 2 * vn[1:-1, -1] + vn[1:-1, -2]) +
                    dt / dy**2 *
                    (vn[2:, -1] - 2 * vn[1:-1, -1] + vn[0:-2, -1])))

        # Periodic BC v @ x = 0
        v[1:-1, 0] = (vn[1:-1, 0] - un[1:-1, 0] * dt / dx *
                    (vn[1:-1, 0] - vn[1:-1, -1]) -
                    vn[1:-1, 0] * dt / dy *
                    (vn[1:-1, 0] - vn[0:-2, 0]) -
                    dt / (2 * rho * dy) * 
                    (p[2:, 0] - p[0:-2, 0]) +
                    nu * (dt / dx**2 * 
                    (vn[1:-1, 1] - 2 * vn[1:-1, 0] + vn[1:-1, -1]) +
                    dt / dy**2 * 
                    (vn[2:, 0] - 2 * vn[1:-1, 0] + vn[0:-2, 0])))

        b[1:-1, 1:-1] = (rho * (1 / dt * 
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) / 
                     (2 * dx) + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
                      2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                           (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx))-
                          ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))

        # Periodic BC Pressure @ x = 2
        b[1:-1, -1] = (rho * (1 / dt * ((u[1:-1, 0] - u[1:-1,-2]) / (2 * dx) +
                                        (v[2:, -1] - v[0:-2, -1]) / (2 * dy)) -
                            ((u[1:-1, 0] - u[1:-1, -2]) / (2 * dx))**2 -
                            2 * ((u[2:, -1] - u[0:-2, -1]) / (2 * dy) *
                                (v[1:-1, 0] - v[1:-1, -2]) / (2 * dx)) -
                            ((v[2:, -1] - v[0:-2, -1]) / (2 * dy))**2))

        # Periodic BC Pressure @ x = 0
        b[1:-1, 0] = (rho * (1 / dt * ((u[1:-1, 1] - u[1:-1, -1]) / (2 * dx) +
                                    (v[2:, 0] - v[0:-2, 0]) / (2 * dy)) -
                            ((u[1:-1, 1] - u[1:-1, -1]) / (2 * dx))**2 -
                            2 * ((u[2:, 0] - u[0:-2, 0]) / (2 * dy) *
                                (v[1:-1, 1] - v[1:-1, -1]) / (2 * dx))-
                            ((v[2:, 0] - v[0:-2, 0]) / (2 * dy))**2))
        # Wall Boundary Conditions
        p[0, :] = p[1, :]   # dp/dy = 0 at y = 0
        p[-1, :] = p[-2,:]  # dp/dy = 0 at y = 2
        
        
        for q in range(nit):
            pn = p.copy()
            p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 + 
                            (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                            (2 * (dx**2 + dy**2)) -
                            dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * 
                            b[1:-1,1:-1])

            # Periodic BC Pressure @ x = 2
            p[1:-1, -1] = (((pn[1:-1, 0] + pn[1:-1, -2])* dy**2 +
                            (pn[2:, -1] + pn[0:-2, -1]) * dx**2) /
                        (2 * (dx**2 + dy**2)) -
                        dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, -1])

            # Periodic BC Pressure @ x = 0
            p[1:-1, 0] = (((pn[1:-1, 1] + pn[1:-1, -1])* dy**2 +
                        (pn[2:, 0] + pn[0:-2, 0]) * dx**2) /
                        (2 * (dx**2 + dy**2)) -
                        dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 0])

        # Wall Boundary Conditions
        u[0, :]  = 0
        u[-1, :] = 0   
        v[0, :]  = 0
        v[-1, :] = 0




        return u,v,p

# for i in range(nt):
#     iterate()


# set up the figure and axis
fig = plt.figure(figsize=(6,6))
ax1 = fig.add_subplot(111)
ax1.contourf(xx, yy, p, alpha=0.7) #, cmap=cm.viridis)
ax1.contour(xx, yy, p) #, cmap=cm.viridis)  
ax1.quiver(xx[::4, ::4], yy[::4, ::4], u[::4, ::4], v[::4, ::4])
surface = [ax1]
fig.colorbar(cm.ScalarMappable(norm=None, cmap=None))

# set axis parameters
ax1.set_xlim([0,2])
ax1.set_ylim([0,2])

ax1.set_title(r'')

def animate(i, surface):
    iterate()
    ax1.collections.clear()
    ax1.contourf(xx, yy, p, alpha=0.7) #, cmap=cm.viridis)
    ax1.contour(xx, yy, p) #, cmap=cm.viridis)  
    ax1.quiver(xx[::4, ::4], yy[::4, ::4], u[::4, ::4], v[::4, ::4])
    surface = [ax1]
    return surface
    
# init_func=init, blit=True
ani = animation.FuncAnimation(
    fig, animate, interval=1000/25, save_count=nt, frames=nt, fargs=([surface]))

ani.save("2D_navier_stokes_cavity_flow.mp4",dpi=600)
fig.colorbar(cm.ScalarMappable(norm=None, cmap=None))
# for ii in v:
#     print(ii)
# plt.show() # I need to change nx and ny to 21 to get the plots to render realtime smoothly
