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
# OUTPUTTING DIAGNOSTIGS FOR THE MESH
# -------------------------------------------------
def print_mesh_stats(deltas, RL, labels=('x','y')):
    for d, label in enumerate(labels):
        dabs = np.abs(deltas[d])
        print(f"{label}: max Δ/RL={np.max(dabs)/RL:.3e}, "
              f"min Δ/RL={np.min(dabs)/RL:.3e}, "
              f"ratio={np.max(dabs)/np.min(dabs):.3e}")

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
Nex = 1800
N = np.array([Nex,round(Nex * L[1] / L[0])], dtype=int)

# number of boundary points
Nb = N + 1

# stretching region
# negative direction
a = np.array([-0.6*RL, 0.0*RL])

# positive direction
b = np.array([0.6*RL, 0.6*RL])

# stretching smoothness
alpha = np.array([20.0e0, 1.3e0])

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
dS = [np.diff(c) for c in coords_stretched]

print("difference between desired and obtained boundaries")
print("beginning:", np.abs(lower - np.array([c[0] for c in coords_stretched])))
print("end:", np.abs(upper - np.array([c[-1] for c in coords_stretched])))

print_mesh_stats(dS, RL)
    
# number of elements inside reference length
nOE = np.sum(
    (center[0] - D0/2 <= coords_stretched[0]) &
    (coords_stretched[0] <= center[0] + D0/2)
) - 1

print("elements inside RL:", nOE)

# -------------------------------------------------
# PARAMETERS FOR SOLVER INPUT
# -------------------------------------------------
print("\nInput the following on your file:")

print("Nx,Ny", N)
print("ax, ay:", alpha)

bounds_if = np.array([
    [coords_aux[d][0], coords_aux[d][-1]]
    for d in range(2)
])

bounds_ab = np.column_stack((a, b))

print("[x,y]_[i,f]/RL:\n", bounds_if / RL)
print("[x,y]_[a,b]/RL:\n", bounds_ab / RL)

# -------------------------------------------------
# PLOT MESH
# -------------------------------------------------
# toggle plots
PF = True

if PF:
    ## plots ##
    # figures and axes
    fig, axes = plt.subplots(2, 2)

    # ------------------
    # Uniform mesh
    # ------------------
    axes[0][0].plot(MX / RL, MY / RL, 'k-', linewidth=0.2)
    axes[0][0].plot(MX.T / RL, MY.T / RL, 'k-', linewidth=0.2)

    axes[0][0].add_patch(plt.Circle((center[0] / RL, center[1] / RL), D0 / ( 2* RL ),
                    fill=False, linewidth=2))

    axes[0][0].set_xlabel("x/R_L []")
    axes[0][0].set_ylabel("y/R_L []")
    axes[0][0].set_title("Auxiliary Uniform")
    axes[0][0].set_aspect('equal')

    axes[0][1].plot(MX, MY, 'k-', linewidth=0.2)
    axes[0][1].plot(MX.T, MY.T, 'k-', linewidth=0.2)

    axes[0][1].add_patch(plt.Circle((center[0], center[1]), D0 / ( 2 ),
                    fill=False, linewidth=2))

    axes[0][1].set_xlabel("x [m]")
    axes[0][1].set_ylabel("y [m]")
    axes[0][1].set_title("Auxiliary Uniform")
    axes[0][1].set_aspect('equal')


    # ------------------
    # Stretched mesh
    # ------------------
    axes[1][0].plot(MXS / RL, MYS / RL, 'b-', linewidth=0.2)
    axes[1][0].plot(MXS.T / RL, MYS.T / RL, 'b-', linewidth=0.2)

    axes[1][0].add_patch(plt.Circle((center[0] / RL, center[1] / RL ) , D0 / ( 2 * RL ),
                    fill=False, linewidth=2))

    axes[1][0].set_xlabel("x/R_L []")
    axes[1][0].set_ylabel("y/R_L []")
    axes[1][0].set_title("Stretched")
    axes[1][0].set_aspect('equal')        
    
    axes[1][1].plot(MXS, MYS, 'b-', linewidth=0.2)
    axes[1][1].plot(MXS.T, MYS.T, 'b-', linewidth=0.2)

    axes[1][1].add_patch(plt.Circle((center[0], center[1]) , D0 / ( 2 ),
                    fill=False, linewidth=2))

    axes[1][1].set_xlabel("x [m]")
    axes[1][1].set_ylabel("y [m]")
    axes[1][1].set_title("Stretched")
    axes[1][1].set_aspect('equal')        
   
    plt.show()