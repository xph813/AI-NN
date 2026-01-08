import numpy as np
from matplotlib.widgets import Slider, Button

#!/usr/bin/env python3
"""
lorent_attractor.py

Simple 3D Lorenz attractor visualizer with interactive view controls (elevation & azimuth).
Run: python3 lorent_attractor.py
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # registers 3D projection

def lorenz_deriv(state, sigma=10.0, rho=28.0, beta=8/3):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz])

def integrate_lorenz(initial, dt=0.01, steps=20000, sigma=10.0, rho=28.0, beta=8/3):
    traj = np.empty((steps, 3))
    state = np.array(initial, dtype=float)
    traj[0] = state
    for i in range(1, steps):
        # RK4
        k1 = lorenz_deriv(state, sigma, rho, beta)
        k2 = lorenz_deriv(state + 0.5*dt*k1, sigma, rho, beta)
        k3 = lorenz_deriv(state + 0.5*dt*k2, sigma, rho, beta)
        k4 = lorenz_deriv(state + dt*k3, sigma, rho, beta)
        state = state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        traj[i] = state
    return traj

def main():
    # Parameters (you can change)
    initial = [0.0, 1.0, 1.05]
    dt = 0.01
    steps = 10000  # number of points (reduce if slow)
    sigma, rho, beta = 10.0, 28.0, 8.0/3.0

    traj = integrate_lorenz(initial, dt=dt, steps=steps, sigma=sigma, rho=rho, beta=beta)
    x, y, z = traj.T

    # Create figure and 3D axes
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    line, = ax.plot(x, y, z, lw=0.5, color='royalblue')

    # Autoscale and labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Lorenz Attractor")
    ax.auto_scale_xyz([x.min(), x.max()], [y.min(), y.max()], [z.min(), z.max()])

    # Initial view
    init_elev = 30
    init_azim = 45
    ax.view_init(elev=init_elev, azim=init_azim)

    # Slider axes
    slider_ax_az = fig.add_axes([0.15, 0.02, 0.7, 0.03], facecolor='lightgoldenrodyellow')
    slider_ax_el = fig.add_axes([0.15, 0.06, 0.7, 0.03], facecolor='lightgoldenrodyellow')

    slider_az = Slider(slider_ax_az, 'Azimuth', 0, 360, valinit=init_azim, valstep=1)
    slider_el = Slider(slider_ax_el, 'Elevation', -90, 90, valinit=init_elev, valstep=1)

    # Reset button
    reset_ax = fig.add_axes([0.88, 0.02, 0.09, 0.04])
    button_reset = Button(reset_ax, 'Reset', color='lightgrey', hovercolor='0.9')

    def update_view(val):
        az = slider_az.val
        el = slider_el.val
        ax.view_init(elev=el, azim=az)
        fig.canvas.draw_idle()

    def reset(event):
        slider_az.reset()
        slider_el.reset()

    slider_az.on_changed(update_view)
    slider_el.on_changed(update_view)
    button_reset.on_clicked(reset)

    plt.show()

if __name__ == "__main__":
    main()