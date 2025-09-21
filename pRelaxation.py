import numpy as np

def softmax(z):
    ex = np.exp(z - np.max(z))
    return ex / np.sum(ex)

def p_of_alpha(alpha, b, c, E, Q):
    N = E - Q - b.dot(alpha)
    D = c.dot(alpha)
    return N / D

def grad_p_alpha(alpha, b, c, E, Q):
    N = E - Q - b.dot(alpha)
    D = c.dot(alpha)
    # grad wrt alpha
    return (-b * D - N * c) / (D*D)

def grad_ascent_softmax(b, c, E, Q, z0=None, lr=1e-2, maxit=5000, tol=1e-9):
    n = len(b)
    if z0 is None:
        z = np.zeros(n)
    else:
        z = z0.copy()
    history = []
    for it in range(maxit):
        alpha = softmax(z)
        p = p_of_alpha(alpha, b, c, E, Q)
        g_alpha = grad_p_alpha(alpha, b, c, E, Q)  # grad wrt alpha
        # grad wrt z: J^T g_alpha where J_ij = alpha_i (delta_ij - alpha_j)
        avg = np.dot(alpha, g_alpha)
        g_z = alpha * (g_alpha - avg)
        # simple adaptive step-size/backtracking can be added; here fixed lr
        z = z + lr * g_z
        history.append({'iter': it+1, 'p': p, 'alpha': alpha.copy(), 'norm_gz': np.linalg.norm(g_z)})
        if np.linalg.norm(g_z) < tol:
            break
    return p, alpha, history

# --------------------------
# Example usage


# pi infty
pi = np.array([1.0E09, 0.0E0])
# qv
qv = np.array([-1167E3, 0.0E0])
# qv'
qvp = np.array([0.0E0, 0.0E0])
# cv
cv = np.array([1816, 717.5])
# cp
cp = np.array([4267, 1006])
# gamma
gama = cp / cv

# PRESSURES - Pa
p0 = np.array([[1E+05], [5]])

# TEMPERATURES - K
T0 = np.array( [[498.15], [298.15]] )

# density matrix
rho0 = ( p0 + pi ) / ( ( gama - 1.0 ) * cv * T0 )

e0 = cv * T0 + pi / rho0 + qv

# volume fraction matrix
alp_eps = 1.0E-02
alp = np.array( [[1.0E0 - 1 * alp_eps, 1 * alp_eps],
                 [1 * alp_eps, 1.0E0 - 1 * alp_eps] ] )

Pi = gama * pi / ( gama - 1.0 )
Gam = 1 / (gama - 1.0)

Q = sum(alp[0,:] * rho0[0,:] * qv)
E = sum(alp[0,:] * rho0[0,:] * e0[0,:])

# print(E,Q)

p_star, alpha_star, hist = grad_ascent_softmax(Pi, Gam, E, Q)
print("Optimal p* =", p_star)
print("Optimal alpha* =", alpha_star)
print("Iterations:")
# for h in hist:
#     print(h)

# print( sum(alpha_star) )