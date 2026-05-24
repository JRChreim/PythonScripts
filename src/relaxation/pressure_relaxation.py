import numpy as np


def softmax(logits):
    exponentials = np.exp(logits - np.max(logits))
    return exponentials / np.sum(exponentials)


def pressure_of_alpha(alpha, pressure_term, gamma_term, total_energy, source_term):
    numerator = total_energy - source_term - pressure_term.dot(alpha)
    denominator = gamma_term.dot(alpha)
    return numerator / denominator


def grad_pressure_of_alpha(alpha, pressure_term, gamma_term, total_energy, source_term):
    numerator = total_energy - source_term - pressure_term.dot(alpha)
    denominator = gamma_term.dot(alpha)
    return (-pressure_term * denominator - numerator * gamma_term) / (denominator * denominator)


def grad_ascent_softmax(
    pressure_term,
    gamma_term,
    total_energy,
    source_term,
    z0=None,
    lr=1.0e-2,
    maxit=5000,
    tol=1.0e-12,
):
    if z0 is None:
        logits = np.zeros(len(pressure_term))
    else:
        logits = z0.copy()

    history = []
    for iteration in range(maxit):
        alpha = softmax(logits)
        pressure = pressure_of_alpha(alpha, pressure_term, gamma_term, total_energy, source_term)
        grad_alpha = grad_pressure_of_alpha(
            alpha,
            pressure_term,
            gamma_term,
            total_energy,
            source_term,
        )
        average_grad = np.dot(alpha, grad_alpha)
        grad_logits = alpha * (grad_alpha - average_grad)
        logits = logits + lr * grad_logits

        history.append(
            {
                "iter": iteration + 1,
                "p": pressure,
                "alpha": alpha.copy(),
                "norm_gz": np.linalg.norm(grad_logits),
            }
        )

        if np.linalg.norm(grad_logits) < tol:
            break

    return pressure, alpha, history


def build_example_problem():
    pi_infinity = np.array([1.0e09, 0.0])
    qv = np.array([-1167.0e3, 0.0])
    cv = np.array([1816.0, 717.5])
    cp = np.array([4267.0, 1006.0])
    gamma = cp / cv

    pressure_0 = np.array([[101325.0 * 10.0], [5.0]])
    temperature_0 = np.array([[298.15], [298.15]])
    rho_0 = (pressure_0 + pi_infinity) / ((gamma - 1.0) * cv * temperature_0)
    energy_0 = cv * temperature_0 + pi_infinity / rho_0 + qv

    alpha_eps = 1.0e-5
    alpha_0 = np.array(
        [
            [1.0 - alpha_eps, alpha_eps],
            [alpha_eps, 1.0 - alpha_eps],
        ]
    )

    pressure_term = alpha_0[0, :] * gamma * pi_infinity / (gamma - 1.0)
    gamma_term = alpha_0[0, :] / (gamma - 1.0)
    source_term = np.sum(alpha_0[0, :] * rho_0[0, :] * qv)
    total_energy = np.sum(alpha_0[0, :] * rho_0[0, :] * energy_0[0, :])

    return pressure_term, gamma_term, total_energy, source_term
