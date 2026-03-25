#!/usr/bin/env python3
import math, json, numpy as np

def convert(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    elif isinstance(o, (np.floating,)):
        return float(o)
    elif isinstance(o, (np.ndarray,)):
        return o.tolist()
    else:
        raise TypeError

# ignoring NaN warnings from divisions
np.seterr(divide='ignore', invalid='ignore')

## FLUID PROPERTIES - 1 for liquid water, 2 for vapor, 3 for air
# pi infty
pi = np.array([1.00E9, 0.0E0])
# qv
qv = np.array([0.0E0, 0.0E0])
# qv'
qvp = np.array([0.0E0, 0.0E0])
# cv
cv = np.array([1000, 717.46])
# cp
cp = np.array([2.35 * cv[0], 1.4 * cv[1]])
# gamma
gama = cp / cv

# Reference Length
RL = 10E-04

# State Variables - 1 FOR ENVIRONMENT WATER, 2 PRESSURIZED BUBBLE
# PRESSURES - Pa
p0 = np.array([[1.5E6], [1.0E3]])

# TEMPERATURES - K
T0 = np.array( [[300], [300]] )

# density matrix
rho0 = ( p0 + pi ) / ( ( gama - 1.0 ) * cv * T0 )

# volume fraction matrix
alp_eps = 0.0E-05

alp = np.array( [[1.0E0 - 1 * alp_eps, 1 * alp_eps],
                 [1 * alp_eps, 1.0E0 - 1 * alp_eps]] )

# discretization parameters for the [x,y,z] directions, respectively. Rows represent the respective patch
vel = np.array( [[0.0E0, 0.0E0, 0.0E0],
                 [0.0E0, 0.0E0, 0.0E0],
                 [0.0E0, 0.0E0, 0.0E0]] )

# domain and patch boundaries - for patch k, that will be [[X_{k,i}, X_{k,f}], [X_{k+1,i}, X_{k+1,f}]
X = np.array( [ [ [0.0, 0.0, 0.0], [3.0*RL, 0.0, 0.0] ],
                [ [0.0, 0.0, 0.0], [1.5*RL, 0.0, 0.0] ], ] )

# Patch Centroids
PC = X.mean(axis=1)

# Patch Lengths
PL = np.diff(X, axis=1)[:, 0, :]
# speed of sound
c = ( gama * ( p0 + pi ) / rho0 ) ** ( 1 / 2 )

Ms = math.sqrt( ( gama[1] + 1. ) / ( 2. * gama[1] ) * ( p0[1,0]/p0[0,0] - 1. ) * ( p0[1,0] / ( p0[1,0] + pi[1] ) ) + 1.0 )

# shock speed
ss = Ms * c[0,1]

## SIMULATION PARAMETERS
# CFL
cfl = 0.50 / 10

# discretization parameters for the [x,y,z] directions, respectively. Rows represent beginning and end, respectively
# number of elements
N = np.array([3000, 0, 0])

# typical cell size, d[x,y,z]
dS = PL[0,:] / N

# time step
dt = 1E-9
# dt = cfl * dS[0] / ss

# save frequency = SF + 1 (because the initial state, 0.dat, is also saved)
SF = 1000

# making Nt divisible by SF
tendA = 1 * PL[1,0] / ss # s

# 1 - ensure NtA is sufficient to go a little beyond tendA
NtA = int( tendA//dt + 1 )

# Array of saves. it is the same as Nt/Sf = t_step_save
# AS = int( NtA // SF + 1 )

# Nt = total number of steps. Ensure Nt > NtA (so the total tendA is covered)
# Nt = AS * SF

# Total physical time
# tend = Nt * dt
tend = 1E-06 # s

Nt = 1000

AS = 1

# Configuring case dictionary ==================================================
print(
  json.dumps({
    # Logistics ================================================
    'run_time_info': 'T',
    # ==========================================================
    # Computational Domain Parameters ==========================
    'x_domain%beg' : X[0,0,0],        
    'x_domain%end' : X[0,1,0],        
    'm'            : N[0],        
    'n'            : N[1],        
    'p'            : N[2],         
    'dt'           : dt,
    't_step_start' : 0,
    't_step_stop'  : Nt,
    't_step_save'  : AS,
    # ==========================================================
    # Simulation Algorithm Parameters ==========================
    'num_patches'   : len(p0),
    'model_eqns'    : 3,
    'num_fluids'    : len(pi),
    'mpp_lim'       : 'T',
    'mixture_err'   : 'T',
    'relax'         : 'T',
    'relax_model'   : 4,
    'under_relax'   : 8.0E-1,
    'palpha_eps'    : 1.0E-08,
    'ptgalpha_eps'  : 1.0E-04,
    'time_stepper'  : 1,
    'recon_type'    : 2,
    'muscl_order'   : 1,
    'int_comp'      : 'T',
    'riemann_solver': 2,
    'wave_speeds'   : 1,
    'avg_state'     : 2,
    'bc_x%beg'      : -8, ## check if those are appropriate BC
    'bc_x%end'      : -7,
    # ==========================================================
    # Formatted Database Files Structure Parameters ============
    'format'       : 2,
    'precision'    : 2,
    'prim_vars_wrt': 'T',
    # 'cons_vars_wrt':'T',
    'parallel_io'  : 'T',
    # ==========================================================
    # Patch 1: High pressured water ============================
    'patch_icpp(1)%geometry'       : 1,
    'patch_icpp(1)%x_centroid'     : PC[0,0],
    'patch_icpp(1)%length_x'       : PL[0,0],
    'patch_icpp(1)%vel(1)'         : vel[0,0],   
    'patch_icpp(1)%pres'           : p0[0,0],  	
    'patch_icpp(1)%alpha_rho(1)'   : alp[0,0] * rho0[0,0],           	
    'patch_icpp(1)%alpha_rho(2)'   : alp[0,1] * rho0[0,1],           	
    'patch_icpp(1)%alpha(1)'       : alp[0,0],           	
    'patch_icpp(1)%alpha(2)'       : alp[0,1],           	
    # ==========================================================
    # Patch 2: (Vapor) Bubble ==================================
    'patch_icpp(2)%geometry'       : 1,
    'patch_icpp(2)%alter_patch(1)' : 'T',
    'patch_icpp(2)%x_centroid'     : PC[1,0],
    'patch_icpp(2)%length_x'       : PL[1,0],
    'patch_icpp(2)%vel(1)'         : vel[1,0],   	
    'patch_icpp(2)%pres'           : p0[1,0],  	
    'patch_icpp(2)%alpha_rho(1)'   : alp[1,0] * rho0[1,0],           	
    'patch_icpp(2)%alpha_rho(2)'   : alp[1,1] * rho0[1,1],           	
    'patch_icpp(2)%alpha(1)'       : alp[1,0],           	
    'patch_icpp(2)%alpha(2)'       : alp[1,1],           	
    # ==========================================================
    # Fluids Physical Parameters ===============================
    'fluid_pp(1)%gamma'            : 1.0E+00 / ( gama[0] - 1 ),       
    'fluid_pp(1)%pi_inf'           : gama[0] * pi[0] / ( gama[0] - 1 ),  
    'fluid_pp(1)%cv'          	   : cv[0],          
    'fluid_pp(1)%qv'        	   : qv[0],	
    'fluid_pp(1)%qvp'          	   : qvp[0],         
    'fluid_pp(2)%gamma'            : 1.0E+00 / ( gama[1] - 1 ),       
    'fluid_pp(2)%pi_inf'           : gama[1] * pi[1] / ( gama[1] - 1 ),  
    'fluid_pp(2)%cv'          	   : cv[1],          
    'fluid_pp(2)%qv'        	   : qv[1],  	
    'fluid_pp(2)%qvp'          	   : qvp[1],			
    # ==========================================================
}, default=convert, indent=2))