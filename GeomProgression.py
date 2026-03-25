import numpy as np
import matplotlib.pyplot as plt

# Original progression
q = 1.03
N = 36
a0 = 1.0

n = np.arange(N)
terms = a0 * q**n

# New initial term
b0 = terms[-1] * (q - 1) / (q**N - 1)

# Sum of new geometric progression
S_b = b0 * (q**N - 1) / (q - 1)

# Plot original terms
plt.figure()
plt.plot(n, terms, marker='o', label='Original progression')

# Plot the single point (N, Sum(b))
plt.scatter(N-1, S_b, color='red', zorder=3, label='Sum of new progression')

# Labels
plt.xlabel('n')
plt.ylabel('Value')
plt.title(f'Geometric Progression (q={q}, N={N})')

plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.show()

print(a0)
print(terms[-1])
print(b0)
print(S_b)