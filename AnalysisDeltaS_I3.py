#ΔS(q) Analysis of I₃ Molecular Structural Change via Debye Scattering Equation
import numpy as np
import matplotlib.pyplot as plt

# Generate q values from 0.01 to 8 Å⁻¹
q = np.linspace(0.01, 8, 1000)
s = q / (4 * np.pi)

# Scattering factor parameters for Iodine
a = np.array([20.1472, 18.9949, 7.5138, 2.2735])
b = np.array([4.3470, 0.3814, 27.7660, 67.8776])
c = 4.0712

# Atomic scattering factor f(q)
f_q = np.sum(a[:, None] * np.exp(-b[:, None] * s**2), axis=0) + c
f_q_squared = f_q**2

# Interatomic distances before the reaction (in Å)
r_12_a = 3.1
r_13_a = 5.54
r_23_a = 3.3

# Interatomic distances after the reaction (in Å)
r_12_b = 2.7
r_13_b = 5.4
r_23_b = 2.7

# Debye equation for a triatomic molecule
def S_three_atom(fq2, q, r12, r13, r23):
    return 3*fq2 + 2*fq2*(np.sin(q * r12) / (q * r12) + np.sin(q * r13) / (q * r13) +np.sin(q * r23) / (q * r23))

# Calculate scattering intensities for structure A and B
S_a = S_three_atom(f_q_squared, q, r_12_a, r_13_a, r_23_a)
S_b = S_three_atom(f_q_squared, q, r_12_b, r_13_b, r_23_b)
delta_S = S_b - S_a

# Plotting the results
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot ΔS(q)
axes[0].plot(q, delta_S, label=r'$\Delta S(q)$', color='DarkGreen', linewidth=2)
axes[0].set_xlabel(r'$q$ (Å$^{-1}$)', fontsize=12)
axes[0].set_ylabel(r'$\Delta S(q)$', fontsize=12)
axes[0].set_title(r'$\Delta S(q)$: Structure B - Structure A', fontsize=14)
axes[0].grid(True)
axes[0].legend()

# Plot f(q)
axes[1].plot(q, f_q, label=r'f(q)', color='Blue', linewidth=2)
axes[1].set_xlabel(r'q', fontsize=12)
axes[1].set_ylabel(r'f(q)', fontsize=12)
axes[1].set_title(r'q vs f(q)', fontsize=14)
axes[1].grid(True)
axes[1].legend()

# Plot S(q) for both structures
axes[2].plot(q, S_a, label='Sa', color='Red', linewidth=2)
axes[2].plot(q, S_b, label='Sb', color='Blue', linewidth=2)
axes[2].set_xlabel(r'q', fontsize=12)
axes[2].set_ylabel(r'S(q)', fontsize=12)
axes[2].set_title('q vs Sa and Sb', fontsize=14)
axes[2].grid(True)
axes[2].legend()

plt.tight_layout()
plt.show()

# Explanation:
# The necessary parameters for this problem are the interatomic distances before and after the reaction,
# and the scattering factor of iodine.
# By calculating the difference in scattering intensity between the two structural states,
# we obtain ΔS(q), which is plotted over q.
# Unlike Q1, here we explicitly use the Debye equation formulated for a three-atom molecule.