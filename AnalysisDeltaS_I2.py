#ΔS(q) Analysis of I2 Molecular Structural Change via Debye Scattering Equation
import numpy as np
import matplotlib.pyplot as plt

# Generate q values from 0.01 to 8 Å⁻¹
q = np.linspace(0.01, 8, 1000)
s = q / (4 * np.pi)

# Scattering factor coefficients for Iodine (from Cromer-Mann coefficients)
# Source: https://smilgies.github.io/dms79/x-rays/f0_CromerMann.txt
a = np.array([20.1472, 18.9949, 7.5138, 2.2735])
b = np.array([2.81262, 0.45418, 13.3225, 71.1309])
c = 4.0712

# Calculate atomic scattering factor f(q)
f_q = np.sum(a[:, None] * np.exp(-b[:, None] * s**2), axis=0) + c
f_q_squared = f_q**2

# Debye equation for a diatomic molecule
r1, r2 = 2.7, 3.3  # I-I distances before and after the reaction (in Å)
S1 = 2 * f_q_squared * (1 + np.sin(q * r1) / (q * r1))
S2 = 2 * f_q_squared * (1 + np.sin(q * r2) / (q * r2))
delta_S = S2 - S1

# Plot ΔS(q)
plt.figure(figsize=(8, 5))
plt.plot(q, delta_S, label=r'$\Delta S(q)$ for I$_2$', color='Green', linewidth=2)
plt.xlabel(r'$q$ (Å$^{-1}$)', fontsize=12)
plt.ylabel(r'$\Delta S(q)$', fontsize=12)
plt.title(r'$\Delta S(q)$ from Structural Change in I$_2$', fontsize=14)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Explanation:
# The required parameters for this problem include the I-I bond distance before and after the reaction,
# and the scattering factor of iodine.
# The difference in scattering intensity due to the bond length change is computed using the Debye equation,
# and plotted as a function of q.