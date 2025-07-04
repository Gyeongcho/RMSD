#Determination of I–I Bond Lengths in I₂ via Debye Scattering Curve Fitting

import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit

# Experimental data
q = np.array([0.1 * i for i in range(1, 81)])
deltaS = np.array([
    -0.7108, -2.7446, -5.8195, -9.5090, -13.3009, -16.6657, -19.1268, -20.3209, -20.0413, -18.2595,
    -15.1231, -10.9312, -6.0933, -1.0767, 3.6495, 7.6671, 10.6518, 12.4023, 12.8540, 12.0759,
    10.2532, 7.6587, 4.6172, 1.4674, -1.4749, -3.9464, -5.7592, -6.8123, -7.0927, -6.6679,
    -5.6689, -4.2698, -2.6637, -1.0400, 0.4360, 1.6381, 2.4876, 2.9559, 3.0605, 2.8566,
    2.4244, 1.8563, 1.2430, 0.6630, 0.1744, -0.1893, -0.4194, -0.5286, -0.5451, -0.5058,
    -0.4482, -0.4039, -0.3940, -0.4262, -0.4948, -0.5830, -0.6670, -0.7202, -0.7194, -0.6482,
    -0.5008, -0.2833, -0.0128, 0.2840, 0.5749, 0.8267, 1.0091, 1.0995, 1.0853, 0.9664,
    0.7545, 0.4718, 0.1485, -0.1812, -0.4833, -0.7276, -0.8914, -0.9617, -0.9361, -0.8228
])

# Atomic form factor parameters for iodine
# Source: https://smilgies.github.io/dms79/x-rays/f0_CromerMann.txt
a = np.array([20.1472, 18.9949, 7.5138, 2.2735])
b = np.array([4.3470, 0.3814, 27.7660, 67.8776])
c = 4.0712

# Function to calculate the atomic form factor f(q)
def form_factor(q):
    s = q / (4 * np.pi)
    return np.sum(a[:, None] * np.exp(-b[:, None] * s**2), axis=0) + c

# Debye equation for a diatomic molecule
def S_q(q, r):
    fq = form_factor(q)
    fq_sq = fq ** 2
    return 2 * fq_sq * (1 + np.sinc(q * r / np.pi))  # sinc(x) = sin(pi*x)/(pi*x)

# Theoretical difference scattering intensity function
def debye_model(q, r1, r2, scale):
    return scale * (S_q(q, r2) - S_q(q, r1))

# Define the chi-square function to minimize the difference between measured and theoretical data
def chi2(r1, r2, scale):
    return np.sum((deltaS - debye_model(q, r1, r2, scale)) ** 2)

# Set initial guesses and limits for fitting parameters
m = Minuit(chi2, r1=2.5, r2=3.0, scale=5.0)
m.limits = [(1.0, 5.0), (1.0, 5.0), (0.0001, 10)]
m.migrad()

# Extract optimized parameters from the fit
r1_fit, r2_fit, scale_fit = m.values["r1"], m.values["r2"], m.values["scale"]
deltaS_fit = debye_model(q, r1_fit, r2_fit, scale_fit)

# Plot experimental and fitted data
plt.plot(q, deltaS, 'o', label='Measured ΔS(q)')
plt.plot(q, deltaS_fit, '-', label=f'Fit: r1={r1_fit:.2f}, r2={r2_fit:.2f}, scale={scale_fit:.2f}')
plt.xlabel("q (Å⁻¹)")
plt.ylabel("ΔS(q)")
plt.title("Debye Equation Fit with Atomic Form Factor")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Explanation:
# By analyzing the given deltaS(q) data, we aim to determine the bond length changes of the I₂ molecule.
# This requires fitting the theoretical curve to the data by minimizing the difference (experimental - theoretical)^2.
# We use the chi-square minimization method (least squares fitting).
# The iminuit library performs the fitting, and setting appropriate initial guesses and parameter ranges is essential for convergence.