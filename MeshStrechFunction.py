#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import scipy.optimize as ScpOpt

## Supporting functions ##

# solve geometric progression
def solve_geometric(a1, N, Sstr, q0=2.0, tol=1e-12):

    def f(q):
       return Sstr*(q - 1) - a1*(q**N - 1)
    #    return np.log( ( Sstr*(q - 1) + a1 ) / a1 ) / N - np.log( q )

    # if NO stretching is needed
    if N == 0 or Sstr == 0 or np.abs(Sstr - N*a1) < tol:
        return 1.0
    # if shrinking is needed
    elif (Sstr - N*a1) < 0:
        sol = ScpOpt.root_scalar(f, bracket=[0+tol,1-tol])
    # if stretching is needed
    elif (Sstr - N*a1) > 0:
        sol =  ScpOpt.root_scalar(f, bracket=[1+tol,1E5-tol])

    return sol.root

## main function ##

# ------------------
# Inputs
# ------------------

# reference length
RL = 30.0e-4

# number of dimensions
Ndims = 2

# Domain
x = np.array([0.0, 1.0]) * RL
y = np.array([0.0, 1.0]) * RL

# number of elements
# total
N = np.array([880, 111])          # [Nx, Ny]
# refined region
Nref = np.array([800, 11])        # refined elements

# droplet diameter
D0 = RL / 10

# droplet center location relative to the center of the domain
Drc = np.array([-0.5, 0.0]) * RL

# ------------------
# calculations
# ------------------

# upper and lower bounds
lower = np.array([x[0], y[0]])
upper = np.array([x[1], y[1]])

# auxiliary Length
L = upper - lower

# number of BOUNDARY points
# total
Nb = N + 1
# refined region
Nbref = Nref + 1

# equivalent dS for uniform grid
D_uniform = L / N

# droplet center
center = 0.5 * (upper + lower) + Drc

# coordinates of uniform mesh
coords_uniform = [
    np.linspace(lower[d], upper[d], Nb[d])
    for d in range(2)
]

# coordinates of stretched mesh
D_stretched = []
coords_stretched = []

## Stretchings
# On these matrices, the rows stand for the "negative" and "positive" directions, respectively, and the columns for x,y,z directions

# location at which stretchings will begin
# towards the "negative" direction
a = center - D0/2

# towards the "positive" direction
b = center + D0/2

# number of strecthed elements into the stretched region
Nstr = np.zeros((2,Ndims), dtype=int)

# strecthing ratio
q = np.zeros((2,Ndims))

# total length for the stretched regions
Sstr = np.zeros((2,Ndims))

for d in range(Ndims):

    Sstr[0,d] = a[d] - lower[d]
    Sstr[1,d] = upper[d] - b[d]

    # distribute remaining elements
    if Sstr[0,d] <= 0:
        Nstr[0,d] = 0
        Nstr[1,d] = N[d] - Nref[d]
        a[d] = lower[d]

    elif Sstr[1,d] <= 0:
        Nstr[0,d] = N[d] - Nref[d]
        Nstr[1,d] = 0
        b[d] = upper[d]

    else:
        M = np.array([[1, -Sstr[0,d]/Sstr[1,d]],
                      [1, 1]])
        RHS = np.array([0, N[d] - Nref[d]])
        Nstr[0,d], Nstr[1,d] = np.linalg.solve(M, RHS)

        Nstr[0,d] = int(round(Nstr[0,d]))
        Nstr[1,d] = int(round(Nstr[1,d]))

    # central uniform refined region
    Dc = np.full(Nref[d], (b[d] - a[d]) / Nref[d])

    a1 = Dc[0]

    for p in range(2):
        q[p,d] = solve_geometric(a1, Nstr[p,d], Sstr[p,d])

    # build spacings
    DL = a1 * q[0,d]**(Nstr[0,d] - 1 - np.arange(Nstr[0,d])) if Nstr[0,d] > 0 else np.array([])
    DR = a1 * q[1,d]**(np.arange(Nstr[1,d]))             if Nstr[1,d] > 0 else np.array([])

    D_total = np.concatenate([DL, Dc, DR])
    coord = np.cumsum(np.concatenate([[lower[d]], D_total]))

    D_stretched.append(D_total)
    coords_stretched.append(coord)

## outputting the variables ##

# location where stretching starts
print('1st, 2nd, and 3rd, refer to x,y,z')

print('minimum coordinates for the refined region')
print(a)

print('maximum coordinates for the refined region')
print(b)

# factors
print('stretching factors')
print(q)

# Number of elements into the strecthed region
print('Number of elements into the refined region')
print(Nref)

# Number of elements into the strecthed region
print('Number of elements into the stretched region')
print(Nstr)

## meshes ##

# unstretched
MX, MY = np.meshgrid(*coords_uniform)

# stretched
MXS, MYS = np.meshgrid(*coords_stretched)

# toggle plots
PF = False

if PF:   
    ## plots ##
    # figures and axes
    fig, axes = plt.subplots(3, 1)

    # ------------------
    # Uniform mesh
    # ------------------
    axes[0].plot(MX, MY, 'k-', linewidth=0.2)
    axes[0].plot(MX.T, MY.T, 'k-', linewidth=0.2)

    circle1 = Circle((center[0], center[1]), D0/2,
                    fill=False, linewidth=2)
    axes[0].add_patch(circle1)

    axes[0].set_title("Uniform")
    axes[0].set_aspect('equal')

    # ------------------
    # Stretched mesh
    # ------------------
    axes[1].plot(MXS, MYS, 'b-', linewidth=0.2)
    axes[1].plot(MXS.T, MYS.T, 'b-', linewidth=0.2)

    circle2 = Circle((center[0], center[1]), D0/2,
                    fill=False, linewidth=2)
    axes[1].add_patch(circle2)

    axes[1].set_title("Stretched")
    axes[1].set_aspect('equal')

    # ------------------
    # BOTH meshes
    # ------------------
    axes[2].plot(MX, MY, 'k-', linewidth=1.0)
    axes[2].plot(MXS, MYS, 'r-', linewidth=0.2)
    axes[2].set_aspect('equal')
        
    plt.tight_layout()
    plt.show()