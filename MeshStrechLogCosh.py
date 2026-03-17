import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# -------------------------------------------------
# Supporting functions
# -------------------------------------------------

# -------------------------------------------------
# STRETCHING TRANSFORMATION
# -------------------------------------------------
def stretch_transform(x, a, b, alpha, L):

    return (
        x
        + x/alpha * (
            np.log(np.cosh(alpha*(x-a)/L))
            + np.log(np.cosh(alpha*(x-b)/L))
            - 2*np.log(np.cosh(alpha*(b-a)/(2*L)))
        )
    )


# -------------------------------------------------
# NONLINEAR SYSTEM FOR AUXILIARY BOUNDARIES
# -------------------------------------------------
def boundary_system(X, lower, upper, a, b, alpha):

    xi, xf = X
    L = xf - xi

    eq1 = stretch_transform(xi, a, b, alpha, L) - lower
    eq2 = stretch_transform(xf, a, b, alpha, L) - upper

    return [eq1, eq2]

# -------------------------------------------------
# INPUT PARAMETERS
# -------------------------------------------------
# REFERENCE LENGTH
RL = 0.048

# droplet diameter
D0 = RL

# domain boundaries
lower = np.array([-6.25*RL, 0*RL])
upper = np.array([17*RL, 6*RL])

L = upper - lower

# droplet center
center = np.array([0.0,  0.0])

# number of elements
Nex = 1200
N = np.array([Nex,round(Nex * L[1] / L[0])], dtype=int)

# number of boundary points
Nb = N + 1

# stretching region
# negative direction
a = np.array([-0.6*RL, 0.0*RL])

# positive direction
b = np.array([0.6*RL, 0.6*RL])

# stretching smoothness
alpha = np.array([1e1, 1e0])

# -------------------------------------------------
# MESH GENERATION
# -------------------------------------------------
coords_aux = []
coords_stretched = []

for d in range(2):

    # solve auxiliary boundaries
    xi, xf = fsolve(
        boundary_system,
        [lower[d], upper[d]],
        args=(lower[d], upper[d], a[d], b[d], alpha[d])
    )

    L_aux = xf - xi

    # auxiliary uniform grid
    coord_aux = np.linspace(xi, xf, Nb[d])

    # apply stretching
    coord_stretched = stretch_transform(
        coord_aux,
        a[d],
        b[d],
        alpha[d],
        L_aux
    )

    coords_aux.append(coord_aux)
    coords_stretched.append(coord_stretched)


# -------------------------------------------------
# BUILD 2D MESH
# -------------------------------------------------
# auxiliary uniform
MX, MY = np.meshgrid(
    coords_aux[0],
    coords_aux[1]
)

# stretched
MXS, MYS = np.meshgrid(
    coords_stretched[0],
    coords_stretched[1]
)

# -------------------------------------------------
# DIAGNOSTICS
# -------------------------------------------------
dx = np.diff(coords_stretched[0])
dy = np.diff(coords_stretched[1])

print("difference between desired and obtained boundaries")
print("beginning:", abs(lower[0] - coords_stretched[0][0]))
print("end:", abs(upper[0] - coords_stretched[0][-1]))

print("min Δx / RL:", np.min(np.abs(dx)) / RL)
print("min Δy / RL:", np.min(np.abs(dy)) / RL)

# number of elements inside reference length
nOE = np.sum(
    (coords_stretched[0]/RL >= -1) &
    (coords_stretched[0]/RL <= 1)
)

print("elements inside RL:", nOE)

# -------------------------------------------------
# PARAMETERS FOR SOLVER INPUT
# -------------------------------------------------
print("\nInput the following on your file:")

print("Nx,Ny", N)

print("x_i/RL:", coords_aux[0][0] / RL)
print("x_f/RL:", coords_aux[0][-1] / RL)
print("y_i/RL:", coords_aux[1][0] / RL)
print("y_f/RL:", coords_aux[1][-1] / RL)

print("x_a/RL:", a[0] / RL)
print("x_b/RL:", b[0] / RL)

print("y_a/RL:", a[1] / RL)
print("y_b/RL:", b[1] / RL)

print("ax:", alpha[0])
print("ay:", alpha[1])

# -------------------------------------------------
# PLOT MESH
# -------------------------------------------------
# toggle plots
PF = True

if PF:
    ## plots ##
    # figures and axes
    fig, axes = plt.subplots(2, 1)

    # ------------------
    # Uniform mesh
    # ------------------
    axes[0].plot(MX / RL, MY / RL, 'k-', linewidth=0.2)
    axes[0].plot(MX.T / RL, MY.T / RL, 'k-', linewidth=0.2)

    axes[0].add_patch(plt.Circle((center[0] / RL, center[1] / RL), D0 / ( 2* RL ),
                    fill=False, linewidth=2))

    axes[0].set_xlabel("x/R_L []")
    axes[0].set_ylabel("y/R_L []")
    axes[0].set_title("Auxiliary Uniform")
    axes[0].set_aspect('equal')

    # ------------------
    # Stretched mesh
    # ------------------
    axes[1].plot(MXS / RL, MYS / RL, 'b-', linewidth=0.2)
    axes[1].plot(MXS.T / RL, MYS.T / RL, 'b-', linewidth=0.2)

    axes[1].add_patch(plt.Circle((center[0] / RL, center[1] / RL ) , D0 / ( 2 * RL ),
                    fill=False, linewidth=2))

    axes[1].set_xlabel("x/R_L []")
    axes[1].set_ylabel("y/R_L []")
    axes[1].set_title("Stretched")
    axes[1].set_aspect('equal')        
   
    plt.show()