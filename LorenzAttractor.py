import matplotlib.pyplot as plt
import numpy as np

def lorenz(xyz, *, s=10, r=28, b=2.667):
    """
    Calculate the next point in the Lorenz attractor given the current point.

    Parameters
    ----------
    xyz : array-like, shape (3,)
        The current point in three-dimensional space.
    s : float, optional
        The Prandtl number, controlling the strength of fluid viscosity.
    r : float, optional
        The Rayleigh number, controlling the intensity of thermal convection.
    b : float, optional
        The geometric factor, controlling the shape of the attractor.

    Returns
    -------
    xyz_dot : array, shape (3,)
        The values of the Lorenz attractor's partial derivatives at the current point.
        
    Notes
    -----
    The Lorenz system is a set of three ordinary differential equations that describe
    the behavior of a simplified model of atmospheric convection. It exhibits chaotic 
    behavior, meaning it is highly sensitive to initial conditions.
    """
    x, y, z = xyz
    x_dot = s * (y - x)
    y_dot = r * x - y - x * z
    z_dot = x * y - b * z
    return np.array([x_dot, y_dot, z_dot])


# Define parameters for simulation
dt = 0.01  # Step size for numerical integration
num_steps = 10000  # Number of steps for simulation

# Initialize array to store points in attractor
xyzs = np.empty((num_steps + 1, 3))  # Need one more for initial values

# Set initial values for the attractor
xyzs[0] = (0., 1., 1.05)  # Format: (x, y, z)

# Iterate through time steps to compute trajectory
for i in range(num_steps):
    # Calculate the next point in the attractor using the Lorenz equations
    xyzs[i + 1] = xyzs[i] + lorenz(xyzs[i]) * dt

# Plot the attractor
ax = plt.figure().add_subplot(projection="3d")  # Create a 3D subplot
ax.set_axis_off()  # Turn off the axis for better visualization

# Plot the trajectory of the attractor
ax.plot(*xyzs.T, color="black", linewidth=0.08)  # Plot the trajectory with thin lines

# Set background color of the plot
ax.patch.set_facecolor("white")  # Set the background color to white

# Optionally, save the plot as an image file
# plt.savefig("lorenz_attractor.png", dpi=1200)  # Save the plot as a PNG file with high resolution

# Show the plot
plt.show()

